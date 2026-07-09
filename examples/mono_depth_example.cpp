#include <iostream>

#ifdef HAVE_TENSORRT

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xfeat-cpp/mono_depth/depth_anything_v3_trt.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

struct PointCloudOptions {
  bool enabled = true;
  int stride = 4;
  float max_depth = 200.0f;
  bool include_sky = false;
};

struct CloudPoint {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
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

std::string outputPrefix(const fs::path& image_path, int index) {
  std::ostringstream oss;
  oss << std::setw(2) << std::setfill('0') << index << "_" << image_path.stem().string();
  return oss.str();
}

xfeat::CameraIntrinsics intrinsicsForPointCloud(const cv::Size& image_size,
                                                const std::optional<xfeat::CameraIntrinsics>& intrinsics,
                                                bool* approximated) {
  if (approximated != nullptr) {
    *approximated = false;
  }
  if (intrinsics.has_value() && intrinsics->fx > 0.0 && intrinsics->fy > 0.0) {
    xfeat::CameraIntrinsics out = *intrinsics;
    if (out.cx <= 0.0) {
      out.cx = 0.5 * static_cast<double>(image_size.width - 1);
    }
    if (out.cy <= 0.0) {
      out.cy = 0.5 * static_cast<double>(image_size.height - 1);
    }
    out.width = image_size.width;
    out.height = image_size.height;
    return out;
  }

  if (approximated != nullptr) {
    *approximated = true;
  }
  xfeat::CameraIntrinsics out;
  const double focal = static_cast<double>(std::max(image_size.width, image_size.height));
  out.fx = focal;
  out.fy = focal;
  out.cx = 0.5 * static_cast<double>(image_size.width - 1);
  out.cy = 0.5 * static_cast<double>(image_size.height - 1);
  out.width = image_size.width;
  out.height = image_size.height;
  return out;
}

std::vector<CloudPoint> depthToPointCloud(const cv::Mat& image_bgr,
                                          const cv::Mat& depth,
                                          const cv::Mat& sky_mask,
                                          const xfeat::CameraIntrinsics& intrinsics,
                                          const PointCloudOptions& options) {
  if (image_bgr.empty() || depth.empty()) {
    return {};
  }
  if (image_bgr.type() != CV_8UC3 || depth.type() != CV_32FC1 || image_bgr.size() != depth.size()) {
    throw std::invalid_argument(
        "Point cloud projection expects BGR CV_8UC3 image and CV_32FC1 depth with matching sizes");
  }

  const int stride = std::max(1, options.stride);
  std::vector<CloudPoint> points;
  points.reserve(static_cast<size_t>((depth.rows + stride - 1) / stride) *
                 static_cast<size_t>((depth.cols + stride - 1) / stride));

  for (int v = 0; v < depth.rows; v += stride) {
    const float* depth_row = depth.ptr<float>(v);
    const cv::Vec3b* color_row = image_bgr.ptr<cv::Vec3b>(v);
    const uint8_t* sky_row = sky_mask.empty() ? nullptr : sky_mask.ptr<uint8_t>(v);
    for (int u = 0; u < depth.cols; u += stride) {
      const float z = depth_row[u];
      if (!std::isfinite(z) || z <= 0.0f || z > options.max_depth) {
        continue;
      }
      if (!options.include_sky && sky_row != nullptr && sky_row[u] != 0) {
        continue;
      }

      const cv::Vec3b bgr = color_row[u];
      CloudPoint point;
      point.x = static_cast<float>((static_cast<double>(u) - intrinsics.cx) * static_cast<double>(z) / intrinsics.fx);
      point.y = static_cast<float>((static_cast<double>(v) - intrinsics.cy) * static_cast<double>(z) / intrinsics.fy);
      point.z = z;
      point.r = bgr[2];
      point.g = bgr[1];
      point.b = bgr[0];
      points.push_back(point);
    }
  }
  return points;
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

cv::Mat renderPointCloudPreview(const std::vector<CloudPoint>& points,
                                const cv::Size& preview_size = cv::Size(1200, 900)) {
  cv::Mat preview(preview_size, CV_8UC3, cv::Scalar(18, 18, 18));
  if (points.empty()) {
    return preview;
  }

  static constexpr float kYaw = -0.45f;
  static constexpr float kPitch = 0.35f;
  const float cos_yaw = std::cos(kYaw);
  const float sin_yaw = std::sin(kYaw);
  const float cos_pitch = std::cos(kPitch);
  const float sin_pitch = std::sin(kPitch);

  struct ProjectedPoint {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    cv::Vec3b color;
  };

  std::vector<ProjectedPoint> projected;
  projected.reserve(points.size());
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();

  for (const auto& point : points) {
    const float x0 = point.x;
    const float y0 = -point.y;
    const float z0 = point.z;
    const float x1 = cos_yaw * x0 + sin_yaw * z0;
    const float z1 = -sin_yaw * x0 + cos_yaw * z0;
    const float y1 = cos_pitch * y0 - sin_pitch * z1;
    const float z2 = sin_pitch * y0 + cos_pitch * z1;

    min_x = std::min(min_x, x1);
    max_x = std::max(max_x, x1);
    min_y = std::min(min_y, y1);
    max_y = std::max(max_y, y1);
    projected.push_back(ProjectedPoint{x1, y1, z2, cv::Vec3b(point.b, point.g, point.r)});
  }

  const float range_x = std::max(max_x - min_x, 1e-3f);
  const float range_y = std::max(max_y - min_y, 1e-3f);
  const float margin = 24.0f;
  const float scale = std::min((static_cast<float>(preview_size.width) - 2.0f * margin) / range_x,
                               (static_cast<float>(preview_size.height) - 2.0f * margin) / range_y);
  std::vector<float> z_buffer(static_cast<size_t>(preview_size.width * preview_size.height),
                              std::numeric_limits<float>::max());

  for (const auto& point : projected) {
    const int px = static_cast<int>((point.x - min_x) * scale + margin);
    const int py = static_cast<int>(static_cast<float>(preview_size.height) - ((point.y - min_y) * scale + margin));
    if (px < 0 || px >= preview_size.width || py < 0 || py >= preview_size.height) {
      continue;
    }
    const size_t index = static_cast<size_t>(py * preview_size.width + px);
    if (point.z < z_buffer[index]) {
      z_buffer[index] = point.z;
      preview.at<cv::Vec3b>(py, px) = point.color;
    }
  }

  cv::Mat dilated;
  cv::dilate(preview, dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));
  return dilated;
}

void writeIntrinsics(const fs::path& path, const xfeat::CameraIntrinsics& intrinsics, bool approximate) {
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
  storage << "approximate" << static_cast<int>(approximate);
  storage.release();
}

void writePointCloudOutputs(const fs::path& out_dir,
                            const std::string& prefix,
                            const cv::Mat& image,
                            const xfeat::MonoDepthResult& result,
                            const std::optional<xfeat::CameraIntrinsics>& intrinsics,
                            const PointCloudOptions& options) {
  if (!options.enabled) {
    return;
  }

  bool approximated_intrinsics = false;
  const auto cloud_intrinsics = intrinsicsForPointCloud(image.size(), intrinsics, &approximated_intrinsics);
  writeIntrinsics(out_dir / (prefix + "_intrinsics.yml"), cloud_intrinsics, approximated_intrinsics);
  const auto points = depthToPointCloud(image, result.depth, result.sky_mask, cloud_intrinsics, options);
  if (points.empty()) {
    std::cerr << "Skipping point cloud for " << prefix << ": no valid depth samples" << std::endl;
    return;
  }

  const fs::path ply_path = out_dir / (prefix + "_cloud.ply");
  const fs::path preview_path = out_dir / (prefix + "_cloud_preview.png");
  writePly(ply_path, points);
  cv::imwrite(preview_path.string(), renderPointCloudPreview(points));

  std::cout << "  point_cloud=" << ply_path << " (" << points.size() << " points)";
  if (approximated_intrinsics) {
    std::cout << " using approximate intrinsics";
  }
  std::cout << std::endl;
  std::cout << "  point_cloud_preview=" << preview_path << std::endl;
}

void writeFloatImage(const fs::path& path, const cv::Mat& image, const std::string& label) {
  if (image.empty()) {
    return;
  }
  if (!cv::imwrite(path.string(), image)) {
    throw std::runtime_error("Failed to write " + label + ": " + path.string());
  }
}

void writeResult(const fs::path& out_dir,
                 const fs::path& image_path,
                 int index,
                 const cv::Mat& image,
                 const xfeat::MonoDepthResult& result,
                 const std::optional<xfeat::CameraIntrinsics>& intrinsics,
                 const PointCloudOptions& cloud_options) {
  const std::string prefix = outputPrefix(image_path, index);

  writeFloatImage(out_dir / (prefix + "_depth.tiff"), result.depth, "depth image");
  writeFloatImage(out_dir / (prefix + "_raw_depth.tiff"), result.raw_depth, "raw depth image");

  cv::FileStorage storage((out_dir / (prefix + "_metadata.yml")).string(), cv::FileStorage::WRITE);
  storage << "focal_scale" << result.metadata.focal_scale;
  storage << "sky_fill_value" << result.metadata.sky_fill_value;
  storage.release();
  if (!cloud_options.enabled && intrinsics.has_value()) {
    writeIntrinsics(out_dir / (prefix + "_intrinsics.yml"), *intrinsics, false);
  }

  cv::imwrite((out_dir / (prefix + "_depth_vis.png")).string(), colorizeDepth(result.depth));
  if (!result.sky_mask.empty()) {
    cv::imwrite((out_dir / (prefix + "_sky_mask.png")).string(), result.sky_mask);
  }
  writePointCloudOutputs(out_dir, prefix, image, result, intrinsics, cloud_options);
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

}  // namespace

int main(int argc, char** argv) {
  std::string engine_path;
  std::string out_dir = "output/mono_depth";
  std::vector<std::string> image_paths;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  bool verbose = false;
  bool no_cloud = false;
  bool cloud_include_sky = false;
  int cloud_stride = 4;
  float cloud_max_depth = 200.0f;

  po::options_description desc("Depth Anything V3 TensorRT mono depth example");
  desc.add_options()("help,h", "Show this help message")("engine,e",
                                                         po::value<std::string>(&engine_path)->required(),
                                                         "Path to a Depth Anything V3 TensorRT .engine file")(
      "out-dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "Directory for result files")(
      "fx", po::value<double>(&fx), "Camera fx in pixels")("fy", po::value<double>(&fy), "Camera fy in pixels")(
      "cx", po::value<double>(&cx), "Camera cx in pixels (defaults to image center when fx/fy are set)")(
      "cy", po::value<double>(&cy), "Camera cy in pixels (defaults to image center when fx/fy are set)")(
      "verbose,v", po::bool_switch(&verbose), "Enable TensorRT metadata logging")(
      "no-cloud", po::bool_switch(&no_cloud), "Skip colored point cloud PLY and preview outputs")(
      "cloud-stride",
      po::value<int>(&cloud_stride)->default_value(cloud_stride),
      "Pixel stride for point cloud sampling")("cloud-max-depth",
                                               po::value<float>(&cloud_max_depth)->default_value(cloud_max_depth),
                                               "Maximum depth kept in point cloud")(
      "cloud-include-sky", po::bool_switch(&cloud_include_sky), "Include sky-mask pixels in point cloud output")(
      "images", po::value<std::vector<std::string>>(&image_paths), "Input image path(s)");

  po::positional_options_description positional;
  positional.add("images", -1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(positional).run(), vm);
    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << "Argument error: " << e.what() << "\n\n" << desc << std::endl;
    return 2;
  }

  if (image_paths.empty()) {
    std::cerr << "At least one input image is required\n\n" << desc << std::endl;
    return 2;
  }

  bool has_intrinsics = vm.count("fx") > 0 || vm.count("fy") > 0 || vm.count("cx") > 0 || vm.count("cy") > 0;
  if (has_intrinsics && (fx <= 0.0 || fy <= 0.0)) {
    std::cerr << "Both --fx and --fy must be positive when intrinsics are provided" << std::endl;
    return 2;
  }
  if (cloud_stride <= 0) {
    std::cerr << "--cloud-stride must be positive" << std::endl;
    return 2;
  }
  if (cloud_max_depth <= 0.0f) {
    std::cerr << "--cloud-max-depth must be positive" << std::endl;
    return 2;
  }

  PointCloudOptions cloud_options;
  cloud_options.enabled = !no_cloud;
  cloud_options.stride = cloud_stride;
  cloud_options.max_depth = cloud_max_depth;
  cloud_options.include_sky = cloud_include_sky;

  std::vector<cv::Mat> images;
  images.reserve(image_paths.size());
  for (const auto& path : image_paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
      std::cerr << "Failed to read image: " << path << std::endl;
      return 1;
    }
    images.push_back(std::move(image));
  }

  std::vector<std::optional<xfeat::CameraIntrinsics>> image_intrinsics;
  image_intrinsics.reserve(images.size());
  for (const auto& image : images) {
    image_intrinsics.push_back(makeIntrinsics(image.size(), has_intrinsics, fx, fy, cx, cy));
  }

  fs::create_directories(out_dir);

  try {
    xfeat::DepthAnythingV3TRT::Params params;
    params.engine_path = engine_path;
    params.verbose = verbose;
    xfeat::DepthAnythingV3TRT model(params);

    std::vector<xfeat::MonoDepthResult> results;
    if (images.size() == 1) {
      results.push_back(model.infer(images.front(), image_intrinsics.front()));
    } else {
      std::vector<xfeat::CameraIntrinsics> intrinsics;
      if (has_intrinsics) {
        intrinsics.reserve(images.size());
        for (const auto& item : image_intrinsics) {
          intrinsics.push_back(*item);
        }
      }
      results = model.infer_multi_view(images, intrinsics);
    }

    for (size_t i = 0; i < results.size(); ++i) {
      writeResult(
          out_dir, image_paths[i], static_cast<int>(i), images[i], results[i], image_intrinsics[i], cloud_options);
      std::cout << image_paths[i] << ": depth " << results[i].depth.cols << "x" << results[i].depth.rows << ", raw "
                << results[i].raw_depth.cols << "x" << results[i].raw_depth.rows
                << ", focal_scale=" << results[i].metadata.focal_scale;
      if (!results[i].sky_mask.empty()) {
        std::cout << ", sky_fill=" << results[i].metadata.sky_fill_value;
      }
      std::cout << std::endl;
    }
    std::cout << "Results written to " << out_dir << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Depth inference failed: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

#else

int main() {
  std::cerr << "mono_depth_example requires TensorRT support." << std::endl;
  return 1;
}

#endif  // HAVE_TENSORRT
