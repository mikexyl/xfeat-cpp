#include <iostream>

#ifdef HAVE_TENSORRT

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <boost/program_options.hpp>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xfeat-cpp/mono_depth/depth_anything_v3_trt.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct Quat {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double w = 1.0;
};

struct Pose {
  double timestamp = 0.0;
  Vec3 t;
  Quat q;
};

struct SelectedFrame {
  fs::path path;
  size_t image_index = 0;
  double image_timestamp = 0.0;
  Pose pose;
  double pose_dt = 0.0;
  double cumulative_distance = 0.0;
  double distance_from_previous_selected = 0.0;
};

using MapCloud = pcl::PointCloud<pcl::PointXYZRGB>;

std::string toLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool isImagePath(const fs::path& path) {
  const std::string ext = toLower(path.extension().string());
  return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

std::vector<fs::path> listImages(const fs::path& sequence_dir) {
  if (!fs::is_directory(sequence_dir)) {
    throw std::runtime_error("Sequence directory does not exist: " + sequence_dir.string());
  }

  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(sequence_dir)) {
    if (entry.is_regular_file() && isImagePath(entry.path())) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
  return images;
}

double imageTimestampSeconds(const fs::path& image_path) {
  const std::string stem = image_path.stem().string();
  const long double value = std::stold(stem);
  if (value > 1.0e12L) {
    return static_cast<double>(value * 1.0e-9L);
  }
  return static_cast<double>(value);
}

Quat normalized(Quat q) {
  const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (norm <= 0.0) {
    return {};
  }
  q.x /= norm;
  q.y /= norm;
  q.z /= norm;
  q.w /= norm;
  return q;
}

std::vector<Pose> readPoses(const fs::path& pose_path) {
  std::ifstream in(pose_path);
  if (!in) {
    throw std::runtime_error("Failed to open pose file: " + pose_path.string());
  }

  std::vector<Pose> poses;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream iss(line);
    Pose pose;
    if (!(iss >> pose.timestamp >> pose.t.x >> pose.t.y >> pose.t.z >> pose.q.x >> pose.q.y >> pose.q.z >> pose.q.w)) {
      continue;
    }
    pose.q = normalized(pose.q);
    poses.push_back(pose);
  }

  if (poses.empty()) {
    throw std::runtime_error("Pose file did not contain any valid poses: " + pose_path.string());
  }
  std::sort(poses.begin(), poses.end(), [](const Pose& a, const Pose& b) { return a.timestamp < b.timestamp; });
  return poses;
}

Pose nearestPose(const std::vector<Pose>& poses, double timestamp, double max_dt, double* dt_out) {
  const auto it = std::lower_bound(
      poses.begin(), poses.end(), timestamp, [](const Pose& pose, double t) { return pose.timestamp < t; });

  const Pose* best = nullptr;
  if (it != poses.end()) {
    best = &*it;
  }
  if (it != poses.begin()) {
    const Pose* prev = &*(it - 1);
    if (best == nullptr || std::abs(prev->timestamp - timestamp) < std::abs(best->timestamp - timestamp)) {
      best = prev;
    }
  }
  if (best == nullptr) {
    throw std::runtime_error("No pose available for timestamp " + std::to_string(timestamp));
  }

  const double dt = best->timestamp - timestamp;
  if (std::abs(dt) > max_dt) {
    throw std::runtime_error("Nearest pose is " + std::to_string(dt) + " seconds from image timestamp " +
                             std::to_string(timestamp) + ", above --pose-max-dt");
  }
  if (dt_out != nullptr) {
    *dt_out = dt;
  }
  return *best;
}

double translationDistance(const Pose& a, const Pose& b) {
  const double dx = a.t.x - b.t.x;
  const double dy = a.t.y - b.t.y;
  const double dz = a.t.z - b.t.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::vector<SelectedFrame> selectFrames(const std::vector<fs::path>& images,
                                        const std::vector<Pose>& poses,
                                        int start_index,
                                        int count,
                                        int interval,
                                        unsigned int seed,
                                        double pose_time_offset,
                                        double pose_max_dt) {
  if (count <= 0) {
    throw std::runtime_error("--count must be positive");
  }
  if (interval <= 0) {
    throw std::runtime_error("--interval must be positive");
  }
  const size_t required_span = static_cast<size_t>(interval) * static_cast<size_t>(count - 1);
  if (images.size() <= required_span) {
    throw std::runtime_error("Sequence has " + std::to_string(images.size()) + " images, but " +
                             std::to_string(required_span + 1) + " are needed");
  }

  const size_t max_start = images.size() - required_span - 1;
  size_t selected_start = 0;
  if (start_index >= 0) {
    selected_start = static_cast<size_t>(start_index);
    if (selected_start > max_start) {
      throw std::runtime_error("--start-index is too large for requested --count and --interval");
    }
  } else {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> distribution(0, max_start);
    selected_start = distribution(rng);
  }

  std::vector<SelectedFrame> selected;
  selected.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    const size_t image_index = selected_start + static_cast<size_t>(i * interval);
    const double image_timestamp = imageTimestampSeconds(images[image_index]);
    double pose_dt = 0.0;
    Pose pose = nearestPose(poses, image_timestamp + pose_time_offset, pose_max_dt, &pose_dt);
    selected.push_back(SelectedFrame{images[image_index], image_index, image_timestamp, pose, pose_dt});
  }
  return selected;
}

std::vector<SelectedFrame> selectFramesByDistance(const std::vector<fs::path>& images,
                                                  const std::vector<Pose>& poses,
                                                  int start_index,
                                                  int end_index,
                                                  double sample_distance,
                                                  bool include_last,
                                                  double pose_time_offset,
                                                  double pose_max_dt) {
  if (sample_distance <= 0.0) {
    throw std::runtime_error("--sample-distance must be positive");
  }
  if (images.empty()) {
    throw std::runtime_error("Sequence has no images");
  }

  const size_t start = start_index >= 0 ? static_cast<size_t>(start_index) : 0;
  const size_t end = end_index >= 0 ? static_cast<size_t>(end_index) : images.size() - 1;
  if (start >= images.size()) {
    throw std::runtime_error("--start-index is outside the image sequence");
  }
  if (end >= images.size()) {
    throw std::runtime_error("--end-index is outside the image sequence");
  }
  if (end < start) {
    throw std::runtime_error("--end-index must be greater than or equal to --start-index");
  }

  std::vector<SelectedFrame> selected;
  selected.reserve((end - start + 1) / 10 + 1);

  Pose previous_pose;
  bool have_previous_pose = false;
  double cumulative_distance = 0.0;
  double last_selected_distance = 0.0;
  double next_sample_distance = 0.0;

  for (size_t image_index = start; image_index <= end; ++image_index) {
    const double image_timestamp = imageTimestampSeconds(images[image_index]);
    double pose_dt = 0.0;
    Pose pose = nearestPose(poses, image_timestamp + pose_time_offset, pose_max_dt, &pose_dt);
    if (have_previous_pose) {
      cumulative_distance += translationDistance(previous_pose, pose);
    }
    previous_pose = pose;
    have_previous_pose = true;

    if (selected.empty() || cumulative_distance >= next_sample_distance) {
      const double distance_from_previous = selected.empty() ? 0.0 : cumulative_distance - last_selected_distance;
      selected.push_back(SelectedFrame{images[image_index],
                                       image_index,
                                       image_timestamp,
                                       pose,
                                       pose_dt,
                                       cumulative_distance,
                                       distance_from_previous});
      last_selected_distance = cumulative_distance;
      next_sample_distance = cumulative_distance + sample_distance;
    }
  }

  if (include_last && !selected.empty() && selected.back().image_index != end) {
    const double image_timestamp = imageTimestampSeconds(images[end]);
    double pose_dt = 0.0;
    Pose pose = nearestPose(poses, image_timestamp + pose_time_offset, pose_max_dt, &pose_dt);
    selected.push_back(SelectedFrame{images[end],
                                     end,
                                     image_timestamp,
                                     pose,
                                     pose_dt,
                                     cumulative_distance,
                                     cumulative_distance - last_selected_distance});
  }

  if (selected.empty()) {
    throw std::runtime_error("No frames were selected");
  }
  return selected;
}

std::vector<double> imageTimestamps(const std::vector<fs::path>& images) {
  std::vector<double> timestamps;
  timestamps.reserve(images.size());
  for (const auto& image : images) {
    timestamps.push_back(imageTimestampSeconds(image));
  }
  return timestamps;
}

std::optional<size_t> nearestImageIndex(const std::vector<double>& image_timestamps, double timestamp, double max_dt) {
  const auto it = std::lower_bound(image_timestamps.begin(), image_timestamps.end(), timestamp);

  size_t best_index = 0;
  bool have_best = false;
  if (it != image_timestamps.end()) {
    best_index = static_cast<size_t>(std::distance(image_timestamps.begin(), it));
    have_best = true;
  }
  if (it != image_timestamps.begin()) {
    const size_t previous_index = static_cast<size_t>(std::distance(image_timestamps.begin(), it - 1));
    if (!have_best ||
        std::abs(image_timestamps[previous_index] - timestamp) < std::abs(image_timestamps[best_index] - timestamp)) {
      best_index = previous_index;
      have_best = true;
    }
  }

  if (!have_best) {
    return std::nullopt;
  }

  const double dt = image_timestamps[best_index] - timestamp;
  if (std::abs(dt) > max_dt) {
    return std::nullopt;
  }
  return best_index;
}

std::vector<SelectedFrame> selectFramesAtPoseTimestamps(const std::vector<fs::path>& images,
                                                        const std::vector<Pose>& poses,
                                                        double pose_time_offset,
                                                        double pose_max_dt,
                                                        bool skip_unmatched_poses,
                                                        size_t* skipped_pose_count) {
  if (images.empty()) {
    throw std::runtime_error("Sequence has no images");
  }
  if (poses.empty()) {
    throw std::runtime_error("Pose file has no poses");
  }

  const std::vector<double> timestamps = imageTimestamps(images);
  std::vector<SelectedFrame> selected;
  selected.reserve(poses.size());
  size_t skipped = 0;

  Pose previous_pose;
  bool have_previous_pose = false;
  double cumulative_distance = 0.0;

  for (const Pose& pose : poses) {
    const double target_image_timestamp = pose.timestamp - pose_time_offset;
    const std::optional<size_t> image_index = nearestImageIndex(timestamps, target_image_timestamp, pose_max_dt);
    if (!image_index.has_value()) {
      if (!skip_unmatched_poses) {
        throw std::runtime_error("No image within --pose-max-dt for pose timestamp " + std::to_string(pose.timestamp));
      }
      ++skipped;
      continue;
    }
    const double image_timestamp = timestamps[*image_index];
    if (have_previous_pose) {
      cumulative_distance += translationDistance(previous_pose, pose);
    }
    previous_pose = pose;
    have_previous_pose = true;

    selected.push_back(SelectedFrame{images[*image_index],
                                     *image_index,
                                     image_timestamp,
                                     pose,
                                     pose.timestamp - (image_timestamp + pose_time_offset),
                                     cumulative_distance,
                                     selected.empty() ? 0.0 : translationDistance(selected.back().pose, pose)});
  }
  if (skipped_pose_count != nullptr) {
    *skipped_pose_count = skipped;
  }
  if (selected.empty()) {
    throw std::runtime_error("No pose timestamps had a matching image");
  }
  return selected;
}

xfeat::CameraIntrinsics makeIntrinsics(const cv::Size& size, double fx, double fy, double cx, double cy) {
  xfeat::CameraIntrinsics intrinsics;
  intrinsics.fx = fx;
  intrinsics.fy = fy;
  intrinsics.cx = cx > 0.0 ? cx : 0.5 * static_cast<double>(size.width - 1);
  intrinsics.cy = cy > 0.0 ? cy : 0.5 * static_cast<double>(size.height - 1);
  intrinsics.width = size.width;
  intrinsics.height = size.height;
  return intrinsics;
}

Eigen::Matrix4f poseToTransform(const Pose& pose, bool pose_is_world_to_body) {
  Eigen::Quaterniond q(pose.q.w, pose.q.x, pose.q.y, pose.q.z);
  q.normalize();

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = q.toRotationMatrix();
  transform(0, 3) = pose.t.x;
  transform(1, 3) = pose.t.y;
  transform(2, 3) = pose.t.z;
  if (pose_is_world_to_body) {
    transform = transform.inverse();
  }
  return transform.cast<float>();
}

Eigen::Matrix4f parseTransform(const std::string& text, const std::string& option_name) {
  if (text.empty()) {
    return Eigen::Matrix4f::Identity();
  }

  std::string cleaned = text;
  for (char& c : cleaned) {
    if (c == ',' || c == ';' || c == '[' || c == ']') {
      c = ' ';
    }
  }

  std::vector<double> values;
  std::istringstream iss(cleaned);
  double value = 0.0;
  while (iss >> value) {
    values.push_back(value);
  }

  if (values.size() != 16) {
    throw std::runtime_error(option_name + " expects 16 row-major values");
  }

  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      transform(r, c) = static_cast<float>(values[static_cast<size_t>(r * 4 + c)]);
    }
  }
  return transform;
}

cv::Matx44f toCvMatx44f(const Eigen::Matrix4f& matrix) {
  cv::Matx44f out;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      out(r, c) = matrix(r, c);
    }
  }
  return out;
}

Eigen::Matrix4f worldFromCamera(const Pose& pose, bool pose_is_world_to_body, const Eigen::Matrix4f& body_from_camera) {
  return poseToTransform(pose, pose_is_world_to_body) * body_from_camera;
}

MapCloud makeCameraCloud(const cv::Mat& image_bgr,
                         const xfeat::MonoDepthResult& result,
                         const xfeat::CameraIntrinsics& intrinsics,
                         int stride,
                         float max_depth,
                         float min_confidence,
                         bool include_sky) {
  if (image_bgr.empty() || result.depth.empty()) {
    return {};
  }
  if (image_bgr.type() != CV_8UC3 || result.depth.type() != CV_32FC1 || image_bgr.size() != result.depth.size()) {
    throw std::runtime_error("Depth/image shape mismatch while building map point cloud");
  }
  if (!result.confidence.empty() &&
      (result.confidence.type() != CV_32FC1 || result.confidence.size() != result.depth.size())) {
    throw std::runtime_error("Depth confidence map must be CV_32FC1 and match depth size");
  }

  MapCloud local_cloud;
  local_cloud.points.reserve(static_cast<size_t>((result.depth.rows + stride - 1) / stride) *
                             static_cast<size_t>((result.depth.cols + stride - 1) / stride));
  local_cloud.is_dense = false;

  for (int v = 0; v < result.depth.rows; v += stride) {
    const float* depth_row = result.depth.ptr<float>(v);
    const float* confidence_row = result.confidence.empty() ? nullptr : result.confidence.ptr<float>(v);
    const cv::Vec3b* color_row = image_bgr.ptr<cv::Vec3b>(v);
    const uint8_t* sky_row = result.sky_mask.empty() ? nullptr : result.sky_mask.ptr<uint8_t>(v);
    for (int u = 0; u < result.depth.cols; u += stride) {
      const float z = depth_row[u];
      if (!std::isfinite(z) || z <= 0.0f || z > max_depth) {
        continue;
      }
      if (!include_sky && sky_row != nullptr && sky_row[u] != 0) {
        continue;
      }
      if (min_confidence > 0.0f && confidence_row != nullptr) {
        const float confidence = confidence_row[u];
        if (!std::isfinite(confidence) || confidence < min_confidence) {
          continue;
        }
      }

      const cv::Vec3b bgr = color_row[u];

      pcl::PointXYZRGB point;
      point.x = static_cast<float>((static_cast<double>(u) - intrinsics.cx) * static_cast<double>(z) / intrinsics.fx);
      point.y = static_cast<float>((static_cast<double>(v) - intrinsics.cy) * static_cast<double>(z) / intrinsics.fy);
      point.z = z;
      point.r = bgr[2];
      point.g = bgr[1];
      point.b = bgr[0];
      local_cloud.points.push_back(point);
    }
  }

  local_cloud.width = static_cast<uint32_t>(local_cloud.points.size());
  local_cloud.height = 1;
  local_cloud.is_dense = false;
  return local_cloud;
}

size_t appendCameraCloudToMap(const MapCloud& local_cloud,
                              const Eigen::Matrix4f& world_from_camera,
                              MapCloud* map_cloud) {
  if (local_cloud.points.empty()) {
    return 0;
  }
  MapCloud transformed_cloud;
  pcl::transformPointCloud(local_cloud, transformed_cloud, world_from_camera);
  *map_cloud += transformed_cloud;
  map_cloud->width = static_cast<uint32_t>(map_cloud->points.size());
  map_cloud->height = 1;
  map_cloud->is_dense = false;
  return transformed_cloud.points.size();
}

void downsample(MapCloud* cloud, size_t max_points, unsigned int seed) {
  if (max_points == 0 || cloud->points.size() <= max_points) {
    return;
  }
  std::mt19937 rng(seed);
  std::shuffle(cloud->points.begin(), cloud->points.end(), rng);
  cloud->points.resize(max_points);
  cloud->width = static_cast<uint32_t>(cloud->points.size());
  cloud->height = 1;
  cloud->is_dense = false;
}

void voxelDownsample(MapCloud* cloud, float leaf_size) {
  if (leaf_size <= 0.0f || cloud->points.empty()) {
    return;
  }

  MapCloud::Ptr input(new MapCloud(*cloud));
  MapCloud filtered;
  pcl::VoxelGrid<pcl::PointXYZRGB> voxel;
  voxel.setInputCloud(input);
  voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
  voxel.filter(filtered);
  *cloud = std::move(filtered);
  cloud->width = static_cast<uint32_t>(cloud->points.size());
  cloud->height = 1;
  cloud->is_dense = false;
}

void writeBinaryPly(const fs::path& path, const MapCloud& cloud) {
  if (pcl::io::savePLYFileBinary(path.string(), cloud) < 0) {
    throw std::runtime_error("Failed to write map output: " + path.string());
  }
}

void writeComponentPose(size_t component_index,
                        const SelectedFrame& frame,
                        const Eigen::Matrix4f& world_from_camera,
                        const fs::path& cloud_path,
                        size_t point_count,
                        std::ofstream* poses_tum,
                        std::ofstream* poses_matrices,
                        std::ofstream* component_index_stream) {
  const Eigen::Matrix3f rotation = world_from_camera.block<3, 3>(0, 0);
  Eigen::Quaternionf q(rotation);
  q.normalize();
  const Eigen::Vector3f t = world_from_camera.block<3, 1>(0, 3);
  const std::string cloud_file = cloud_path.filename().string();

  *poses_tum << std::fixed << std::setprecision(9) << frame.image_timestamp << " " << t.x() << " " << t.y() << " "
             << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";

  *poses_matrices << component_index << " " << std::fixed << std::setprecision(9) << frame.image_timestamp << " "
                  << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " "
                  << q.w();
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      *poses_matrices << " " << world_from_camera(r, c);
    }
  }
  *poses_matrices << " " << cloud_file << "\n";

  *component_index_stream << component_index << "\t" << frame.image_index << "\t" << std::fixed << std::setprecision(9)
                          << frame.image_timestamp << "\t" << frame.pose.timestamp << "\t" << frame.pose_dt << "\t"
                          << frame.cumulative_distance << "\t" << frame.distance_from_previous_selected << "\t"
                          << point_count << "\t1\t" << cloud_file << "\t" << frame.path.string() << "\n";
}

void writeSkippedComponentPose(size_t component_index,
                               const SelectedFrame& frame,
                               std::ofstream* component_index_stream) {
  *component_index_stream << component_index << "\t" << frame.image_index << "\t" << std::fixed << std::setprecision(9)
                          << frame.image_timestamp << "\t" << frame.pose.timestamp << "\t" << frame.pose_dt << "\t"
                          << frame.cumulative_distance << "\t" << frame.distance_from_previous_selected << "\t0\t0\t\t"
                          << frame.path.string() << "\n";
}

void writeSelectedFrames(const fs::path& path, const std::vector<SelectedFrame>& selected) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to write selected frame list: " + path.string());
  }
  out << "# view image_index image_timestamp pose_timestamp pose_dt cumulative_distance "
         "distance_from_previous_selected path\n";
  out << std::fixed << std::setprecision(9);
  for (size_t i = 0; i < selected.size(); ++i) {
    out << i << " " << selected[i].image_index << " " << selected[i].image_timestamp << " "
        << selected[i].pose.timestamp << " " << selected[i].pose_dt << " " << selected[i].cumulative_distance << " "
        << selected[i].distance_from_previous_selected << " " << selected[i].path.string() << "\n";
  }
}

cv::Mat colorizeDepth(const cv::Mat& depth) {
  cv::Mat mask = depth > 0.0f;
  double min_depth = 0.0;
  double max_depth = 0.0;
  cv::minMaxLoc(depth, &min_depth, &max_depth, nullptr, nullptr, mask);

  cv::Mat normalized = cv::Mat::zeros(depth.size(), CV_8UC1);
  if (max_depth > min_depth) {
    depth.convertTo(normalized, CV_8UC1, 255.0 / (max_depth - min_depth), -min_depth * 255.0 / (max_depth - min_depth));
    normalized.setTo(0, mask == 0);
  }

  cv::Mat color;
  cv::applyColorMap(normalized, color, cv::COLORMAP_TURBO);
  color.setTo(cv::Scalar(0, 0, 0), mask == 0);
  return color;
}

std::string outputPrefix(size_t view_index, const fs::path& path) {
  std::ostringstream oss;
  oss << "view_" << std::setw(2) << std::setfill('0') << view_index << "_" << path.stem().string();
  return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::string engine_path;
  std::string sequence_dir = "/data/graco/ground-03_images/camera_left_image_raw";
  std::string pose_path = "/data/graco/ground-03.txt";
  std::string out_dir = "output/mono_depth_graco_ground03_pose_map";
  int start_index = -1;
  int end_index = -1;
  int count = 12;
  int interval = 2;
  double sample_distance = 0.0;
  unsigned int seed = 7;
  double fx = 940.862825677534;
  double fy = 938.554923506332;
  double cx = 799.1626975233576;
  double cy = 559.295406893583;
  double pose_time_offset = 0.0;
  double pose_max_dt = 0.02;
  std::string t_body_camera;
  int cloud_stride = 4;
  float max_depth = 200.0f;
  float min_confidence = 0.0f;
  int multi_view_size = 3;
  float voxel_size = 0.0f;
  size_t max_map_points = 0;
  bool include_sky = false;
  bool include_last = false;
  bool use_pose_timestamps = false;
  bool skip_unmatched_poses = false;
  bool poses_are_world_to_body = false;
  bool poses_are_world_to_camera = false;
  bool save_component_clouds = false;
  bool write_intermediate = false;
  bool verbose = false;

  po::options_description desc("Build a DA3 depth point-cloud map from timestamped poses");
  desc.add_options()("help,h", "Show this help message")(
      "engine,e", po::value<std::string>(&engine_path)->required(), "Path to a DA3 TensorRT .engine file")(
      "sequence-dir", po::value<std::string>(&sequence_dir)->default_value(sequence_dir), "Sorted image sequence")(
      "poses",
      po::value<std::string>(&pose_path)->default_value(pose_path),
      "Pose file with timestamp tx ty tz qx qy qz qw rows")(
      "out-dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "Directory for map outputs")(
      "start-index",
      po::value<int>(&start_index)->default_value(start_index),
      "Sorted image index to start from; negative selects a random nearby window in fixed-count mode or the first "
      "image in distance mode")("end-index",
                                po::value<int>(&end_index)->default_value(end_index),
                                "Sorted image index to end at in distance mode; negative uses the last image")(
      "count", po::value<int>(&count)->default_value(count), "Number of nearby frames to sample")(
      "interval", po::value<int>(&interval)->default_value(interval), "Frame interval between samples")(
      "sample-distance",
      po::value<double>(&sample_distance)->default_value(sample_distance),
      "Select one frame each time pose path length advances by this many meters; 0 uses fixed count/interval mode")(
      "seed", po::value<unsigned int>(&seed)->default_value(seed), "Random seed used when --start-index is negative")(
      "fx", po::value<double>(&fx)->default_value(fx), "Camera fx in pixels")(
      "fy", po::value<double>(&fy)->default_value(fy), "Camera fy in pixels")(
      "cx", po::value<double>(&cx)->default_value(cx), "Camera cx in pixels")(
      "cy", po::value<double>(&cy)->default_value(cy), "Camera cy in pixels")(
      "pose-time-offset",
      po::value<double>(&pose_time_offset)->default_value(pose_time_offset),
      "Seconds added to image timestamps before pose lookup")(
      "pose-max-dt",
      po::value<double>(&pose_max_dt)->default_value(pose_max_dt),
      "Maximum allowed absolute nearest-pose time difference in seconds")(
      "t-body-camera",
      po::value<std::string>(&t_body_camera)->default_value(t_body_camera),
      "Row-major 4x4 T_body_camera matrix as comma- or space-separated values; default is identity")(
      "cloud-stride",
      po::value<int>(&cloud_stride)->default_value(cloud_stride),
      "Pixel stride for point cloud sampling")(
      "max-depth", po::value<float>(&max_depth)->default_value(max_depth), "Maximum depth kept in the map")(
      "min-confidence",
      po::value<float>(&min_confidence)->default_value(min_confidence),
      "Drop points below this absolute depth-confidence value when the engine exposes confidence; 0 disables")(
      "multi-view-size",
      po::value<int>(&multi_view_size)->default_value(multi_view_size),
      "Number of selected frames per grouped inference for multi-view DA3 engines")(
      "voxel-size",
      po::value<float>(&voxel_size)->default_value(voxel_size),
      "PCL voxel leaf size in map units; 0 disables voxel downsampling")(
      "max-map-points",
      po::value<size_t>(&max_map_points)->default_value(max_map_points),
      "Randomly downsample fused map to this many points; 0 keeps all")(
      "include-sky", po::bool_switch(&include_sky), "Keep sky-mask pixels in the map")(
      "include-last", po::bool_switch(&include_last), "Also include the final frame in distance mode")(
      "use-pose-timestamps",
      po::bool_switch(&use_pose_timestamps),
      "Use every timestamped pose row and select the nearest image for each pose")(
      "skip-unmatched-poses",
      po::bool_switch(&skip_unmatched_poses),
      "In --use-pose-timestamps mode, skip pose rows that have no image within --pose-max-dt")(
      "poses-are-world-to-body",
      po::bool_switch(&poses_are_world_to_body),
      "Interpret poses as T_body_world and invert them before mapping")(
      "poses-are-world-to-camera", po::bool_switch(&poses_are_world_to_camera), "Alias for --poses-are-world-to-body")(
      "save-component-clouds",
      po::bool_switch(&save_component_clouds),
      "Write each filtered camera-frame component cloud plus T_world_camera pose files for ICP")(
      "write-intermediate", po::bool_switch(&write_intermediate), "Write per-frame depth visualizations")(
      "verbose,v", po::bool_switch(&verbose), "Enable TensorRT metadata logging");

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << "Argument error: " << e.what() << "\n\n" << desc << std::endl;
    return 2;
  }

  if (fx <= 0.0 || fy <= 0.0) {
    std::cerr << "--fx and --fy must be positive" << std::endl;
    return 2;
  }
  if (cloud_stride <= 0) {
    std::cerr << "--cloud-stride must be positive" << std::endl;
    return 2;
  }
  if (max_depth <= 0.0f) {
    std::cerr << "--max-depth must be positive" << std::endl;
    return 2;
  }
  if (!std::isfinite(min_confidence) || min_confidence < 0.0f) {
    std::cerr << "--min-confidence must be finite and non-negative" << std::endl;
    return 2;
  }
  if (multi_view_size <= 0) {
    std::cerr << "--multi-view-size must be positive" << std::endl;
    return 2;
  }
  if (sample_distance < 0.0) {
    std::cerr << "--sample-distance must be non-negative" << std::endl;
    return 2;
  }
  if (voxel_size < 0.0f) {
    std::cerr << "--voxel-size must be non-negative" << std::endl;
    return 2;
  }

  try {
    const auto images = listImages(sequence_dir);
    const auto poses = readPoses(pose_path);
    size_t skipped_pose_count = 0;
    const auto selected =
        use_pose_timestamps
            ? selectFramesAtPoseTimestamps(
                  images, poses, pose_time_offset, pose_max_dt, skip_unmatched_poses, &skipped_pose_count)
        : sample_distance > 0.0
            ? selectFramesByDistance(
                  images, poses, start_index, end_index, sample_distance, include_last, pose_time_offset, pose_max_dt)
            : selectFrames(images, poses, start_index, count, interval, seed, pose_time_offset, pose_max_dt);
    const Eigen::Matrix4f body_from_camera = parseTransform(t_body_camera, "--t-body-camera");
    const bool pose_is_world_to_body = poses_are_world_to_body || poses_are_world_to_camera;

    fs::create_directories(out_dir);
    writeSelectedFrames(fs::path(out_dir) / "selected_frames.txt", selected);
    const fs::path component_dir = fs::path(out_dir) / "component_clouds";
    std::ofstream component_poses_tum;
    std::ofstream component_poses_matrices;
    std::ofstream component_index_stream;
    size_t written_component_clouds = 0;
    if (save_component_clouds) {
      fs::create_directories(component_dir);
      component_poses_tum.open(component_dir / "poses_tum.txt");
      component_poses_matrices.open(component_dir / "poses_matrices.txt");
      component_index_stream.open(component_dir / "component_index.tsv");
      if (!component_poses_tum || !component_poses_matrices || !component_index_stream) {
        throw std::runtime_error("Failed to open component cloud pose outputs in " + component_dir.string());
      }
      component_poses_tum << "# timestamp tx ty tz qx qy qz qw\n";
      component_poses_tum << "# pose is T_world_camera for camera-frame component_clouds/*.ply\n";
      component_poses_matrices << "# component timestamp tx ty tz qx qy qz qw m00 m01 m02 m03 m10 m11 m12 m13 "
                                  "m20 m21 m22 m23 m30 m31 m32 m33 cloud_file\n";
      component_poses_matrices << "# matrix is row-major T_world_camera\n";
      component_index_stream << "# clouds are filtered camera-frame PLY files; pose files store T_world_camera\n";
      component_index_stream << "component\timage_index\timage_timestamp\tpose_timestamp\tpose_dt\t"
                                "cumulative_distance\tdistance_from_previous_selected\tpoint_count\twritten\t"
                                "cloud_file\timage_path\n";
    }
    std::cout << "selected_frames=" << selected.size();
    if (use_pose_timestamps) {
      std::cout << ", use_pose_timestamps=true, poses=" << poses.size() << ", skipped_poses=" << skipped_pose_count;
    } else if (sample_distance > 0.0) {
      std::cout << ", sample_distance=" << sample_distance << " m";
    }
    std::cout << std::endl;

    xfeat::DepthAnythingV3TRT::Params params;
    params.engine_path = engine_path;
    params.verbose = verbose;
    xfeat::DepthAnythingV3TRT model(params);

    MapCloud map_cloud;
    map_cloud.is_dense = false;
    auto write_intermediate_outputs =
        [&](size_t view_index, const SelectedFrame& frame, const xfeat::MonoDepthResult& result) {
          if (!write_intermediate) {
            return;
          }
          const std::string prefix = outputPrefix(view_index, frame.path);
          cv::imwrite((fs::path(out_dir) / (prefix + "_depth_vis.png")).string(), colorizeDepth(result.depth));
          if (!result.sky_mask.empty()) {
            cv::imwrite((fs::path(out_dir) / (prefix + "_sky_mask.png")).string(), result.sky_mask);
          }
        };
    auto write_component_outputs = [&](size_t view_index,
                                       const SelectedFrame& frame,
                                       const MapCloud& local_cloud,
                                       const Eigen::Matrix4f& world_from_camera) {
      if (!save_component_clouds) {
        return;
      }
      if (local_cloud.points.empty()) {
        writeSkippedComponentPose(view_index, frame, &component_index_stream);
        return;
      }

      const std::string prefix = outputPrefix(view_index, frame.path);
      const fs::path cloud_path = component_dir / (prefix + "_points_camera.ply");
      writeBinaryPly(cloud_path, local_cloud);
      writeComponentPose(view_index,
                         frame,
                         world_from_camera,
                         cloud_path,
                         local_cloud.points.size(),
                         &component_poses_tum,
                         &component_poses_matrices,
                         &component_index_stream);
      ++written_component_clouds;
    };

    if (model.has_camera_inputs()) {
      if (selected.size() < static_cast<size_t>(multi_view_size)) {
        throw std::runtime_error("Pose-conditioned multi-view engine needs at least --multi-view-size selected frames");
      }
      std::cout << "inference_mode=multi_view, multi_view_size=" << multi_view_size;
      if (min_confidence > 0.0f) {
        std::cout << ", min_confidence=" << min_confidence;
      }
      std::cout << std::endl;

      size_t next_append_index = 0;
      while (next_append_index < selected.size()) {
        size_t group_start = next_append_index;
        size_t append_start = next_append_index;
        if (group_start + static_cast<size_t>(multi_view_size) > selected.size()) {
          group_start = selected.size() - static_cast<size_t>(multi_view_size);
        }

        std::vector<cv::Mat> group_images;
        std::vector<xfeat::CameraIntrinsics> group_intrinsics;
        std::vector<Eigen::Matrix4f> group_world_from_camera;
        std::vector<cv::Matx44f> group_world_to_camera;
        group_images.reserve(static_cast<size_t>(multi_view_size));
        group_intrinsics.reserve(static_cast<size_t>(multi_view_size));
        group_world_from_camera.reserve(static_cast<size_t>(multi_view_size));
        group_world_to_camera.reserve(static_cast<size_t>(multi_view_size));

        for (int local = 0; local < multi_view_size; ++local) {
          const size_t frame_index = group_start + static_cast<size_t>(local);
          cv::Mat image = cv::imread(selected[frame_index].path.string(), cv::IMREAD_COLOR);
          if (image.empty()) {
            throw std::runtime_error("Failed to read image: " + selected[frame_index].path.string());
          }
          group_intrinsics.push_back(makeIntrinsics(image.size(), fx, fy, cx, cy));
          const Eigen::Matrix4f world_from_camera =
              worldFromCamera(selected[frame_index].pose, pose_is_world_to_body, body_from_camera);
          group_world_from_camera.push_back(world_from_camera);
          group_world_to_camera.push_back(toCvMatx44f(world_from_camera.inverse()));
          group_images.push_back(std::move(image));
        }

        const auto results = model.infer_multi_view(group_images, group_intrinsics, group_world_to_camera);
        for (int local = 0; local < multi_view_size; ++local) {
          const size_t frame_index = group_start + static_cast<size_t>(local);
          if (frame_index < append_start) {
            continue;
          }
          const MapCloud local_cloud = makeCameraCloud(group_images[static_cast<size_t>(local)],
                                                       results[static_cast<size_t>(local)],
                                                       group_intrinsics[static_cast<size_t>(local)],
                                                       cloud_stride,
                                                       max_depth,
                                                       min_confidence,
                                                       include_sky);
          const Eigen::Matrix4f& world_from_camera = group_world_from_camera[static_cast<size_t>(local)];
          const size_t added_points = appendCameraCloudToMap(local_cloud, world_from_camera, &map_cloud);
          write_component_outputs(frame_index, selected[frame_index], local_cloud, world_from_camera);
          write_intermediate_outputs(frame_index, selected[frame_index], results[static_cast<size_t>(local)]);
          std::cout << "view " << frame_index << ": image_index=" << selected[frame_index].image_index
                    << ", group_start=" << group_start << ", distance=" << selected[frame_index].cumulative_distance
                    << ", pose_dt=" << selected[frame_index].pose_dt << ", added_points=" << added_points << std::endl;
        }

        if (group_start + static_cast<size_t>(multi_view_size) >= selected.size()) {
          break;
        }
        next_append_index = group_start + static_cast<size_t>(multi_view_size);
      }
    } else {
      std::cout << "inference_mode=single_view";
      if (min_confidence > 0.0f) {
        std::cout << ", min_confidence=" << min_confidence;
      }
      std::cout << std::endl;

      for (size_t i = 0; i < selected.size(); ++i) {
        cv::Mat image = cv::imread(selected[i].path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
          throw std::runtime_error("Failed to read image: " + selected[i].path.string());
        }

        const auto intrinsics = makeIntrinsics(image.size(), fx, fy, cx, cy);
        const auto result = model.infer(image, intrinsics);
        const MapCloud local_cloud =
            makeCameraCloud(image, result, intrinsics, cloud_stride, max_depth, min_confidence, include_sky);
        const Eigen::Matrix4f world_from_camera =
            worldFromCamera(selected[i].pose, pose_is_world_to_body, body_from_camera);
        const size_t added_points = appendCameraCloudToMap(local_cloud, world_from_camera, &map_cloud);
        write_component_outputs(i, selected[i], local_cloud, world_from_camera);

        write_intermediate_outputs(i, selected[i], result);
        std::cout << "view " << i << ": image_index=" << selected[i].image_index
                  << ", distance=" << selected[i].cumulative_distance << ", pose_dt=" << selected[i].pose_dt
                  << ", added_points=" << added_points << std::endl;
      }
    }

    const size_t points_before_voxel = map_cloud.points.size();
    voxelDownsample(&map_cloud, voxel_size);
    if (voxel_size > 0.0f) {
      std::cout << "voxel_size=" << voxel_size << ", points_before_voxel=" << points_before_voxel
                << ", points_after_voxel=" << map_cloud.points.size() << std::endl;
    }
    downsample(&map_cloud, max_map_points, seed);
    const fs::path map_path = fs::path(out_dir) / "map_points.ply";
    writeBinaryPly(map_path, map_cloud);

    std::cout << "map_points=" << map_cloud.points.size() << std::endl;
    std::cout << "map=" << map_path << std::endl;
    if (save_component_clouds) {
      std::cout << "component_clouds=" << written_component_clouds << std::endl;
      std::cout << "component_dir=" << component_dir << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "Pose map demo failed: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

#else

int main() {
  std::cerr << "mono_depth_pose_map_example requires TensorRT support." << std::endl;
  return 1;
}

#endif  // HAVE_TENSORRT
