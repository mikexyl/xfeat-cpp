#include <iostream>

#ifdef HAVE_TENSORRT

#include <Eigen/Geometry>
#include <algorithm>
#include <boost/program_options.hpp>
#include <cctype>
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

struct CloudPoint {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
};

struct ConfidenceFilterOptions {
  float min_confidence = 1.0f;

  bool enabled() const { return std::isfinite(min_confidence) && min_confidence > 0.0f; }
};

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

cv::Vec3b turboColor(float value, float min_value, float max_value) {
  const float range = std::max(max_value - min_value, std::numeric_limits<float>::epsilon());
  const float normalized = std::clamp((value - min_value) / range, 0.0f, 1.0f);
  cv::Mat gray(1, 1, CV_8UC1, cv::Scalar(static_cast<int>(std::lround(normalized * 255.0f))));
  cv::Mat color;
  cv::applyColorMap(gray, color, cv::COLORMAP_TURBO);
  return color.at<cv::Vec3b>(0, 0);
}

cv::Mat colorizeConfidence(const cv::Mat& confidence, float marker_value) {
  if (confidence.empty()) {
    return {};
  }
  if (confidence.type() != CV_32FC1) {
    throw std::runtime_error("Confidence image must be CV_32FC1");
  }

  constexpr float kMinConfidenceVis = 0.0f;
  constexpr float kMaxConfidenceVis = 4.0f;
  cv::Mat normalized = cv::Mat::zeros(confidence.size(), CV_8UC1);
  for (int y = 0; y < confidence.rows; ++y) {
    const float* src = confidence.ptr<float>(y);
    uint8_t* dst = normalized.ptr<uint8_t>(y);
    for (int x = 0; x < confidence.cols; ++x) {
      if (!std::isfinite(src[x])) {
        continue;
      }
      const float value =
          std::clamp((src[x] - kMinConfidenceVis) / (kMaxConfidenceVis - kMinConfidenceVis), 0.0f, 1.0f);
      dst[x] = static_cast<uint8_t>(std::lround(value * 255.0f));
    }
  }

  cv::Mat color;
  cv::applyColorMap(normalized, color, cv::COLORMAP_TURBO);
  color.setTo(cv::Scalar(0, 0, 0), normalized == 0);

  const int legend_width = 120;
  const int bar_width = 24;
  const int margin = std::clamp(color.rows / 20, 14, 36);
  cv::Mat output(color.rows, color.cols + legend_width, CV_8UC3, cv::Scalar(24, 24, 24));
  color.copyTo(output(cv::Rect(0, 0, color.cols, color.rows)));

  const int bar_x = color.cols + 18;
  const int bar_y0 = margin;
  const int bar_y1 = color.rows - margin - 1;
  const int bar_height = std::max(1, bar_y1 - bar_y0 + 1);
  for (int y = bar_y0; y <= bar_y1; ++y) {
    const float t = 1.0f - static_cast<float>(y - bar_y0) / static_cast<float>(std::max(1, bar_height - 1));
    const float value = kMinConfidenceVis + t * (kMaxConfidenceVis - kMinConfidenceVis);
    output(cv::Rect(bar_x, y, bar_width, 1)).setTo(turboColor(value, kMinConfidenceVis, kMaxConfidenceVis));
  }
  cv::rectangle(output, cv::Rect(bar_x, bar_y0, bar_width, bar_height), cv::Scalar(235, 235, 235), 1);

  cv::putText(output,
              "conf",
              cv::Point(bar_x, std::max(12, bar_y0 - 6)),
              cv::FONT_HERSHEY_SIMPLEX,
              0.45,
              cv::Scalar(235, 235, 235),
              1,
              cv::LINE_AA);
  for (int tick = 0; tick <= 4; ++tick) {
    const float value = static_cast<float>(tick);
    const float t = (value - kMinConfidenceVis) / (kMaxConfidenceVis - kMinConfidenceVis);
    const int y = bar_y1 - static_cast<int>(std::lround(t * static_cast<float>(bar_height - 1)));
    cv::line(
        output, cv::Point(bar_x + bar_width, y), cv::Point(bar_x + bar_width + 6, y), cv::Scalar(235, 235, 235), 1);
    std::ostringstream label;
    label << tick;
    cv::putText(output,
                label.str(),
                cv::Point(bar_x + bar_width + 10, y + 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                cv::Scalar(235, 235, 235),
                1,
                cv::LINE_AA);
  }

  if (std::isfinite(marker_value) && marker_value > kMinConfidenceVis && marker_value < kMaxConfidenceVis) {
    const float t = (marker_value - kMinConfidenceVis) / (kMaxConfidenceVis - kMinConfidenceVis);
    const int y = bar_y1 - static_cast<int>(std::lround(t * static_cast<float>(bar_height - 1)));
    cv::line(output, cv::Point(bar_x - 6, y), cv::Point(bar_x + bar_width + 6, y), cv::Scalar(255, 255, 255), 2);
    std::ostringstream label;
    label << std::fixed << std::setprecision(1) << marker_value;
    cv::putText(output,
                label.str(),
                cv::Point(bar_x + bar_width + 32, y + 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                cv::Scalar(255, 255, 255),
                1,
                cv::LINE_AA);
  }

  return output;
}

cv::Mat makeConfidenceFilterMask(const cv::Mat& confidence, const ConfidenceFilterOptions& confidence_filter) {
  if (confidence.empty() || !confidence_filter.enabled()) {
    return {};
  }
  if (confidence.type() != CV_32FC1) {
    throw std::runtime_error("Confidence image must be CV_32FC1");
  }

  cv::Mat mask = cv::Mat::zeros(confidence.size(), CV_8UC1);
  for (int y = 0; y < confidence.rows; ++y) {
    const float* src = confidence.ptr<float>(y);
    uint8_t* dst = mask.ptr<uint8_t>(y);
    for (int x = 0; x < confidence.cols; ++x) {
      if (std::isfinite(src[x]) && src[x] >= confidence_filter.min_confidence) {
        dst[x] = 255;
      }
    }
  }
  return mask;
}

std::string toLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool isImagePath(const fs::path& path) {
  const std::string ext = toLower(path.extension().string());
  return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

double imageTimestampSeconds(const fs::path& image_path) {
  const std::string stem = image_path.stem().string();
  const long double value = std::stold(stem);
  if (value > 1.0e12L) {
    return static_cast<double>(value * 1.0e-9L);
  }
  return static_cast<double>(value);
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
    const Pose* previous = &*(it - 1);
    if (best == nullptr || std::abs(previous->timestamp - timestamp) < std::abs(best->timestamp - timestamp)) {
      best = previous;
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

std::string outputPrefix(int view_index, const fs::path& image_path) {
  std::ostringstream oss;
  oss << "view_" << std::setw(2) << std::setfill('0') << view_index << "_" << image_path.stem().string();
  return oss.str();
}

std::optional<xfeat::CameraIntrinsics> makeIntrinsics(const cv::Size& size,
                                                      bool enabled,
                                                      double fx,
                                                      double fy,
                                                      double cx,
                                                      double cy) {
  if (!enabled) {
    return std::nullopt;
  }

  xfeat::CameraIntrinsics intrinsics;
  intrinsics.fx = fx;
  intrinsics.fy = fy;
  intrinsics.cx = cx > 0.0 ? cx : 0.5 * static_cast<double>(size.width - 1);
  intrinsics.cy = cy > 0.0 ? cy : 0.5 * static_cast<double>(size.height - 1);
  intrinsics.width = size.width;
  intrinsics.height = size.height;
  return intrinsics;
}

Eigen::Matrix4d poseToTransform(const Pose& pose, bool pose_is_world_to_body) {
  Eigen::Quaterniond rotation(pose.q.w, pose.q.x, pose.q.y, pose.q.z);
  rotation.normalize();

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = rotation.toRotationMatrix();
  transform(0, 3) = pose.t.x;
  transform(1, 3) = pose.t.y;
  transform(2, 3) = pose.t.z;
  if (pose_is_world_to_body) {
    transform = transform.inverse();
  }
  return transform;
}

Eigen::Matrix4d parseTransform(const std::string& text, const std::string& option_name) {
  if (text.empty()) {
    return Eigen::Matrix4d::Identity();
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

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      transform(r, c) = values[static_cast<size_t>(r * 4 + c)];
    }
  }
  return transform;
}

CloudPoint transformPoint(const CloudPoint& point, const Eigen::Matrix4d& transform) {
  CloudPoint out = point;
  const Eigen::Vector4d transformed = transform * Eigen::Vector4d(point.x, point.y, point.z, 1.0);
  out.x = static_cast<float>(transformed.x());
  out.y = static_cast<float>(transformed.y());
  out.z = static_cast<float>(transformed.z());
  return out;
}

cv::Matx44f toCvMatx44f(const Eigen::Matrix4d& matrix) {
  cv::Matx44f out;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      out(r, c) = static_cast<float>(matrix(r, c));
    }
  }
  return out;
}

size_t appendDepthCloud(const cv::Mat& image_bgr,
                        const xfeat::MonoDepthResult& result,
                        const xfeat::CameraIntrinsics& intrinsics,
                        const Eigen::Matrix4d& output_from_camera,
                        int stride,
                        float max_depth,
                        bool include_sky,
                        const ConfidenceFilterOptions& confidence_filter,
                        std::vector<CloudPoint>* cloud) {
  if (image_bgr.empty() || result.depth.empty()) {
    return 0;
  }
  if (image_bgr.type() != CV_8UC3 || result.depth.type() != CV_32FC1 || image_bgr.size() != result.depth.size()) {
    throw std::runtime_error("Depth/image shape mismatch while building point cloud");
  }
  if (!result.confidence.empty() &&
      (result.confidence.type() != CV_32FC1 || result.confidence.size() != result.depth.size())) {
    throw std::runtime_error("Depth confidence map must be CV_32FC1 and match depth size");
  }

  const size_t before = cloud->size();
  const int cloud_stride = std::max(1, stride);
  const float min_confidence = confidence_filter.min_confidence;
  for (int v = 0; v < result.depth.rows; v += cloud_stride) {
    const float* depth_row = result.depth.ptr<float>(v);
    const cv::Vec3b* color_row = image_bgr.ptr<cv::Vec3b>(v);
    const uint8_t* sky_row = result.sky_mask.empty() ? nullptr : result.sky_mask.ptr<uint8_t>(v);
    const float* confidence_row = result.confidence.empty() ? nullptr : result.confidence.ptr<float>(v);
    for (int u = 0; u < result.depth.cols; u += cloud_stride) {
      const float z = depth_row[u];
      if (!std::isfinite(z) || z <= 0.0f || z > max_depth) {
        continue;
      }
      if (!include_sky && sky_row != nullptr && sky_row[u] != 0) {
        continue;
      }
      if (confidence_filter.enabled() && confidence_row != nullptr) {
        const float confidence = confidence_row[u];
        if (!std::isfinite(confidence) || confidence < min_confidence) {
          continue;
        }
      }

      const cv::Vec3b bgr = color_row[u];
      CloudPoint point;
      point.x = static_cast<float>((static_cast<double>(u) - intrinsics.cx) * static_cast<double>(z) / intrinsics.fx);
      point.y = static_cast<float>((static_cast<double>(v) - intrinsics.cy) * static_cast<double>(z) / intrinsics.fy);
      point.z = z;
      point.r = bgr[2];
      point.g = bgr[1];
      point.b = bgr[0];
      cloud->push_back(transformPoint(point, output_from_camera));
    }
  }
  return cloud->size() - before;
}

std::vector<CloudPoint> makeCameraCloud(const cv::Mat& image_bgr,
                                        const xfeat::MonoDepthResult& result,
                                        const xfeat::CameraIntrinsics& intrinsics,
                                        int stride,
                                        float max_depth,
                                        bool include_sky,
                                        const ConfidenceFilterOptions& confidence_filter) {
  std::vector<CloudPoint> cloud;
  appendDepthCloud(image_bgr,
                   result,
                   intrinsics,
                   Eigen::Matrix4d::Identity(),
                   stride,
                   max_depth,
                   include_sky,
                   confidence_filter,
                   &cloud);
  return cloud;
}

size_t appendTransformedCloud(const cv::Mat& image_bgr,
                              const xfeat::MonoDepthResult& result,
                              const xfeat::CameraIntrinsics& intrinsics,
                              const Pose& pose,
                              bool pose_is_world_to_body,
                              const Eigen::Matrix4d& body_from_camera,
                              int stride,
                              float max_depth,
                              bool include_sky,
                              const ConfidenceFilterOptions& confidence_filter,
                              std::vector<CloudPoint>* map_cloud) {
  const Eigen::Matrix4d world_from_camera = poseToTransform(pose, pose_is_world_to_body) * body_from_camera;
  return appendDepthCloud(
      image_bgr, result, intrinsics, world_from_camera, stride, max_depth, include_sky, confidence_filter, map_cloud);
}

void downsample(std::vector<CloudPoint>* points, size_t max_points, unsigned int seed) {
  if (max_points == 0 || points->size() <= max_points) {
    return;
  }
  std::mt19937 rng(seed);
  std::shuffle(points->begin(), points->end(), rng);
  points->resize(max_points);
}

void writePly(const fs::path& path, const std::vector<CloudPoint>& points) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to open point cloud output: " + path.string());
  }

  out << "ply\n";
  out << "format ascii 1.0\n";
  out << "element vertex " << points.size() << "\n";
  out << "property float x\n";
  out << "property float y\n";
  out << "property float z\n";
  out << "property uchar red\n";
  out << "property uchar green\n";
  out << "property uchar blue\n";
  out << "end_header\n";
  out << std::fixed << std::setprecision(5);
  for (const auto& point : points) {
    out << point.x << " " << point.y << " " << point.z << " " << static_cast<int>(point.r) << " "
        << static_cast<int>(point.g) << " " << static_cast<int>(point.b) << "\n";
  }
}

std::vector<fs::path> chooseSubsequence(const std::vector<fs::path>& images,
                                        int view_count,
                                        int interval,
                                        unsigned int seed,
                                        size_t* start_index) {
  if (view_count <= 0) {
    throw std::runtime_error("--views must be positive");
  }
  if (interval <= 0) {
    throw std::runtime_error("--interval must be positive");
  }
  const size_t required_span = static_cast<size_t>(interval) * static_cast<size_t>(view_count - 1);
  if (images.size() <= required_span) {
    throw std::runtime_error("Sequence has " + std::to_string(images.size()) + " images, but " +
                             std::to_string(required_span + 1) + " are needed for the requested subsequence");
  }

  const size_t max_start = images.size() - required_span - 1;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<size_t> distribution(0, max_start);
  *start_index = distribution(rng);

  std::vector<fs::path> selected;
  selected.reserve(static_cast<size_t>(view_count));
  for (int view = 0; view < view_count; ++view) {
    selected.push_back(images[*start_index + static_cast<size_t>(view * interval)]);
  }
  return selected;
}

void writeSelectedPaths(const fs::path& out_dir,
                        const std::vector<fs::path>& selected,
                        size_t start_index,
                        int interval,
                        unsigned int seed) {
  std::ofstream out(out_dir / "selected_views.txt");
  if (!out) {
    throw std::runtime_error("Failed to write selected view list");
  }

  out << "seed " << seed << "\n";
  out << "start_index " << start_index << "\n";
  out << "interval " << interval << "\n";
  for (size_t i = 0; i < selected.size(); ++i) {
    out << i << " " << selected[i].string() << "\n";
  }
}

void writeFloatImage(const fs::path& path, const cv::Mat& image, const std::string& label) {
  if (image.empty()) {
    return;
  }
  if (!cv::imwrite(path.string(), image)) {
    throw std::runtime_error("Failed to write " + label + ": " + path.string());
  }
}

void writeIntrinsics(const fs::path& path, const xfeat::CameraIntrinsics& intrinsics) {
  cv::Mat K =
      (cv::Mat_<double>(3, 3) << intrinsics.fx, 0.0, intrinsics.cx, 0.0, intrinsics.fy, intrinsics.cy, 0.0, 0.0, 1.0);

  cv::FileStorage storage(path.string(), cv::FileStorage::WRITE);
  if (!storage.isOpened()) {
    throw std::runtime_error("Failed to write intrinsics: " + path.string());
  }
  storage << "width" << intrinsics.width;
  storage << "height" << intrinsics.height;
  storage << "fx" << intrinsics.fx;
  storage << "fy" << intrinsics.fy;
  storage << "cx" << intrinsics.cx;
  storage << "cy" << intrinsics.cy;
  storage << "K" << K;
  storage.release();
}

void writeResult(const fs::path& out_dir,
                 int view_index,
                 const fs::path& image_path,
                 const cv::Mat& image,
                 const xfeat::MonoDepthResult& result,
                 const std::optional<xfeat::CameraIntrinsics>& intrinsics,
                 bool write_view_cloud,
                 int cloud_stride,
                 float max_depth,
                 bool include_sky,
                 const ConfidenceFilterOptions& confidence_filter,
                 size_t* view_cloud_points,
                 std::vector<cv::Mat>* summary_tiles) {
  const std::string prefix = outputPrefix(view_index, image_path);
  const fs::path view_out_dir = out_dir / "point_clouds" / prefix;
  fs::create_directories(view_out_dir);
  const cv::Mat depth_vis = colorizeDepth(result.depth);
  const float confidence_marker =
      confidence_filter.enabled() ? confidence_filter.min_confidence : std::numeric_limits<float>::quiet_NaN();
  const cv::Mat confidence_vis = colorizeConfidence(result.confidence, confidence_marker);
  const cv::Mat raw_confidence_vis = colorizeConfidence(result.raw_confidence, confidence_marker);
  const cv::Mat confidence_filter_mask = makeConfidenceFilterMask(result.confidence, confidence_filter);
  const cv::Mat raw_confidence_filter_mask = makeConfidenceFilterMask(result.raw_confidence, confidence_filter);

  writeFloatImage(view_out_dir / "depth.tiff", result.depth, "depth image");
  writeFloatImage(view_out_dir / "raw_depth.tiff", result.raw_depth, "raw depth image");
  writeFloatImage(view_out_dir / "confidence.tiff", result.confidence, "depth confidence image");
  writeFloatImage(view_out_dir / "raw_confidence.tiff", result.raw_confidence, "raw depth confidence image");
  if (intrinsics.has_value()) {
    writeIntrinsics(view_out_dir / "intrinsics.yml", *intrinsics);
  }
  if (view_cloud_points != nullptr) {
    *view_cloud_points = 0;
  }
  if (write_view_cloud && intrinsics.has_value()) {
    const auto cloud =
        makeCameraCloud(image, result, *intrinsics, cloud_stride, max_depth, include_sky, confidence_filter);
    writePly(view_out_dir / "points_camera.ply", cloud);
    if (view_cloud_points != nullptr) {
      *view_cloud_points = cloud.size();
    }
  }

  cv::FileStorage storage((view_out_dir / "metadata.yml").string(), cv::FileStorage::WRITE);
  storage << "focal_scale" << result.metadata.focal_scale;
  storage << "pose_scale" << result.metadata.pose_scale;
  if (!result.confidence.empty() && confidence_filter.enabled()) {
    storage << "min_confidence" << confidence_filter.min_confidence;
  }
  storage << "sky_fill_value" << result.metadata.sky_fill_value;
  storage.release();

  cv::imwrite((view_out_dir / "image.png").string(), image);
  cv::imwrite((view_out_dir / "depth_vis.png").string(), depth_vis);
  if (!confidence_vis.empty()) {
    cv::imwrite((view_out_dir / "confidence_vis.png").string(), confidence_vis);
  }
  if (!raw_confidence_vis.empty()) {
    cv::imwrite((view_out_dir / "raw_confidence_vis.png").string(), raw_confidence_vis);
  }
  if (!confidence_filter_mask.empty()) {
    cv::imwrite((view_out_dir / "confidence_filter_mask.png").string(), confidence_filter_mask);
  }
  if (!raw_confidence_filter_mask.empty()) {
    cv::imwrite((view_out_dir / "raw_confidence_filter_mask.png").string(), raw_confidence_filter_mask);
  }
  if (!result.sky_mask.empty()) {
    cv::imwrite((view_out_dir / "sky_mask.png").string(), result.sky_mask);
  }

  cv::Mat image_tile;
  cv::Mat depth_tile;
  cv::resize(image, image_tile, cv::Size(400, 275), 0.0, 0.0, cv::INTER_AREA);
  cv::resize(depth_vis, depth_tile, cv::Size(400, 275), 0.0, 0.0, cv::INTER_AREA);
  cv::putText(image_tile,
              "view " + std::to_string(view_index),
              cv::Point(12, 28),
              cv::FONT_HERSHEY_SIMPLEX,
              0.8,
              cv::Scalar(255, 255, 255),
              2,
              cv::LINE_AA);
  cv::putText(depth_tile,
              "depth " + std::to_string(view_index),
              cv::Point(12, 28),
              cv::FONT_HERSHEY_SIMPLEX,
              0.8,
              cv::Scalar(255, 255, 255),
              2,
              cv::LINE_AA);
  summary_tiles->push_back(std::move(image_tile));
  summary_tiles->push_back(std::move(depth_tile));
}

void writeSummary(const fs::path& out_dir, const std::vector<cv::Mat>& tiles) {
  if (tiles.empty()) {
    return;
  }

  std::vector<cv::Mat> rows;
  rows.reserve((tiles.size() + 1) / 2);
  for (size_t i = 0; i < tiles.size(); i += 2) {
    if (i + 1 >= tiles.size()) {
      rows.push_back(tiles[i]);
      continue;
    }

    cv::Mat row;
    cv::hconcat(std::vector<cv::Mat>{tiles[i], tiles[i + 1]}, row);
    rows.push_back(std::move(row));
  }

  cv::Mat summary;
  cv::vconcat(rows, summary);
  cv::imwrite((out_dir / "multiview_summary.png").string(), summary);
}

}  // namespace

int main(int argc, char** argv) {
  std::string engine_path;
  std::string sequence_dir = "/data/graco/ground-03_images/camera_left_image_raw";
  std::string pose_path = "/data/graco/ground-03.txt";
  std::string out_dir = "output/mono_depth_graco_ground03_multiview";
  int views = 5;
  int interval = 10;
  unsigned int seed = std::random_device{}();
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  double pose_time_offset = 0.0;
  double pose_max_dt = 0.02;
  std::string t_body_camera;
  int cloud_stride = 4;
  float max_depth = 200.0f;
  double min_confidence = 1.0;
  size_t max_map_points = 0;
  bool include_sky = false;
  bool poses_are_world_to_body = false;
  bool poses_are_world_to_camera = false;
  bool no_view_clouds = false;
  bool no_map = false;
  bool verbose = false;

  po::options_description desc("Random strided sequence example for grouped DA3 TensorRT inference");
  desc.add_options()("help,h", "Show this help message")(
      "engine,e", po::value<std::string>(&engine_path)->required(), "Path to a grouped-view DA3 TensorRT .engine file")(
      "sequence-dir", po::value<std::string>(&sequence_dir)->default_value(sequence_dir), "Sorted image sequence")(
      "poses",
      po::value<std::string>(&pose_path)->default_value(pose_path),
      "Optional pose file with timestamp tx ty tz qx qy qz qw rows for fused map_points.ply")(
      "out-dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "Directory for result files")(
      "views", po::value<int>(&views)->default_value(views), "Number of views in the grouped inference")(
      "interval", po::value<int>(&interval)->default_value(interval), "Frame interval between selected views")(
      "seed", po::value<unsigned int>(&seed)->default_value(seed), "Random seed for the start index")(
      "fx", po::value<double>(&fx), "Camera fx in pixels")("fy", po::value<double>(&fy), "Camera fy in pixels")(
      "cx", po::value<double>(&cx), "Camera cx in pixels (defaults to image center when fx/fy are set)")(
      "cy", po::value<double>(&cy), "Camera cy in pixels (defaults to image center when fx/fy are set)")(
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
      "Pixel stride for fused point cloud sampling")(
      "max-depth", po::value<float>(&max_depth)->default_value(max_depth), "Maximum depth kept in the fused map")(
      "min-confidence",
      po::value<double>(&min_confidence)->default_value(min_confidence),
      "Drop points below this absolute depth-confidence value; 0 disables confidence filtering")(
      "max-map-points",
      po::value<size_t>(&max_map_points)->default_value(max_map_points),
      "Randomly downsample fused map to this many points; 0 keeps all")(
      "include-sky", po::bool_switch(&include_sky), "Keep sky-mask pixels in the fused map")(
      "poses-are-world-to-body",
      po::bool_switch(&poses_are_world_to_body),
      "Interpret poses as T_body_world and invert them before mapping")(
      "poses-are-world-to-camera", po::bool_switch(&poses_are_world_to_camera), "Alias for --poses-are-world-to-body")(
      "no-view-clouds", po::bool_switch(&no_view_clouds), "Do not write per-view points_camera.ply files")(
      "no-map", po::bool_switch(&no_map), "Do not write fused map_points.ply")(
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

  const bool has_intrinsics = vm.count("fx") > 0 || vm.count("fy") > 0 || vm.count("cx") > 0 || vm.count("cy") > 0;
  if (has_intrinsics && (fx <= 0.0 || fy <= 0.0)) {
    std::cerr << "Both --fx and --fy must be positive when intrinsics are provided" << std::endl;
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
  if (!std::isfinite(min_confidence) || min_confidence < 0.0) {
    std::cerr << "--min-confidence must be finite and non-negative" << std::endl;
    return 2;
  }
  if (pose_max_dt <= 0.0) {
    std::cerr << "--pose-max-dt must be positive" << std::endl;
    return 2;
  }

  ConfidenceFilterOptions confidence_filter;
  confidence_filter.min_confidence = static_cast<float>(min_confidence);

  try {
    const auto all_images = listImages(sequence_dir);
    size_t start_index = 0;
    const auto selected = chooseSubsequence(all_images, views, interval, seed, &start_index);
    fs::create_directories(out_dir);
    writeSelectedPaths(out_dir, selected, start_index, interval, seed);

    std::vector<cv::Mat> images;
    images.reserve(selected.size());
    for (const auto& path : selected) {
      cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
      if (image.empty()) {
        throw std::runtime_error("Failed to read image: " + path.string());
      }
      images.push_back(std::move(image));
    }

    std::vector<xfeat::CameraIntrinsics> intrinsics;
    if (has_intrinsics) {
      intrinsics.reserve(images.size());
      for (const auto& image : images) {
        intrinsics.push_back(*makeIntrinsics(image.size(), true, fx, fy, cx, cy));
      }
    }

    const bool pose_is_world_to_body = poses_are_world_to_body || poses_are_world_to_camera;
    std::vector<Pose> poses;
    Eigen::Matrix4d body_from_camera = Eigen::Matrix4d::Identity();

    xfeat::DepthAnythingV3TRT::Params params;
    params.engine_path = engine_path;
    params.verbose = verbose;
    xfeat::DepthAnythingV3TRT model(params);

    if (!pose_path.empty() && (!no_map || model.has_camera_inputs())) {
      poses = readPoses(pose_path);
      body_from_camera = parseTransform(t_body_camera, "--t-body-camera");
    }

    std::vector<xfeat::MonoDepthResult> results;
    if (model.has_camera_inputs()) {
      if (!has_intrinsics) {
        throw std::runtime_error("Pose-conditioned DA3 engines require --fx and --fy");
      }
      if (poses.empty()) {
        throw std::runtime_error("Pose-conditioned DA3 engines require --poses");
      }

      std::vector<cv::Matx44f> world_to_camera_extrinsics;
      world_to_camera_extrinsics.reserve(selected.size());
      for (const auto& path : selected) {
        const Pose pose = nearestPose(poses, imageTimestampSeconds(path) + pose_time_offset, pose_max_dt, nullptr);
        const Eigen::Matrix4d world_from_camera = poseToTransform(pose, pose_is_world_to_body) * body_from_camera;
        world_to_camera_extrinsics.push_back(toCvMatx44f(world_from_camera.inverse()));
      }
      results = model.infer_multi_view(images, intrinsics, world_to_camera_extrinsics);
      std::cout << "Using input intrinsics/extrinsics for DA3 camera conditioning" << std::endl;
    } else {
      results = model.infer_multi_view(images, intrinsics);
    }

    const bool build_map = !no_map && has_intrinsics && !poses.empty();
    std::vector<CloudPoint> map_points;
    if (!build_map && !no_map) {
      if (!has_intrinsics) {
        std::cout << "Skipping fused map_points.ply: provide --fx and --fy to enable point-cloud projection"
                  << std::endl;
      } else if (pose_path.empty()) {
        std::cout << "Skipping fused map_points.ply: provide --poses to place per-view depth clouds in one frame"
                  << std::endl;
      }
    }
    const bool write_view_clouds = !no_view_clouds && has_intrinsics;
    if (!write_view_clouds && !no_view_clouds) {
      std::cout << "Skipping per-view points_camera.ply files: provide --fx and --fy to enable point-cloud projection"
                << std::endl;
    }

    std::vector<cv::Mat> summary_tiles;
    summary_tiles.reserve(results.size() * 2);
    for (size_t i = 0; i < results.size(); ++i) {
      const std::optional<xfeat::CameraIntrinsics> view_intrinsics =
          has_intrinsics ? std::optional<xfeat::CameraIntrinsics>(intrinsics[i]) : std::nullopt;
      size_t view_cloud_points = 0;
      writeResult(out_dir,
                  static_cast<int>(i),
                  selected[i],
                  images[i],
                  results[i],
                  view_intrinsics,
                  write_view_clouds,
                  cloud_stride,
                  max_depth,
                  include_sky,
                  confidence_filter,
                  &view_cloud_points,
                  &summary_tiles);
      std::cout << "view " << i << ": " << selected[i] << ", depth " << results[i].depth.cols << "x"
                << results[i].depth.rows << ", raw " << results[i].raw_depth.cols << "x" << results[i].raw_depth.rows
                << ", focal_scale=" << results[i].metadata.focal_scale;
      if (results[i].metadata.pose_scaled) {
        std::cout << ", pose_scale=" << results[i].metadata.pose_scale;
      }
      if (!results[i].confidence.empty() && confidence_filter.enabled()) {
        std::cout << ", min_confidence=" << confidence_filter.min_confidence;
      }
      if (!results[i].sky_mask.empty()) {
        std::cout << ", sky_fill=" << results[i].metadata.sky_fill_value;
      }
      if (write_view_clouds) {
        std::cout << ", view_cloud_points=" << view_cloud_points;
      }
      if (build_map) {
        double pose_dt = 0.0;
        const Pose pose =
            nearestPose(poses, imageTimestampSeconds(selected[i]) + pose_time_offset, pose_max_dt, &pose_dt);
        const size_t added_points = appendTransformedCloud(images[i],
                                                           results[i],
                                                           intrinsics[i],
                                                           pose,
                                                           pose_is_world_to_body,
                                                           body_from_camera,
                                                           cloud_stride,
                                                           max_depth,
                                                           include_sky,
                                                           confidence_filter,
                                                           &map_points);
        std::cout << ", pose_dt=" << pose_dt << ", map_points_added=" << added_points;
      }
      std::cout << std::endl;
    }

    writeSummary(out_dir, summary_tiles);
    if (build_map) {
      const size_t points_before_downsample = map_points.size();
      downsample(&map_points, max_map_points, seed);
      const fs::path map_path = fs::path(out_dir) / "map_points.ply";
      writePly(map_path, map_points);
      std::cout << "map_points=" << map_points.size();
      if (points_before_downsample != map_points.size()) {
        std::cout << " (downsampled from " << points_before_downsample << ")";
      }
      std::cout << std::endl;
      std::cout << "map=" << map_path << std::endl;
    }
    std::cout << "selected_start_index=" << start_index << ", interval=" << interval << ", seed=" << seed << std::endl;
    std::cout << "Results written to " << out_dir << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Multi-view depth example failed: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

#else

int main() {
  std::cerr << "mono_depth_multiview_sequence_example requires TensorRT support." << std::endl;
  return 1;
}

#endif  // HAVE_TENSORRT
