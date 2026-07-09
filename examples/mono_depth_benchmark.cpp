#include <iostream>

#ifdef HAVE_TENSORRT

#include <Eigen/Geometry>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "xfeat-cpp/mono_depth/depth_anything_v3_trt.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

struct BenchmarkOptions {
  std::string mode = "single";
  std::string engine_path;
  std::vector<std::string> image_paths;
  std::string json_out;
  int warmup = 5;
  int runs = 50;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  std::string pose_path;
  double pose_time_offset = 0.0;
  double pose_max_dt = 0.02;
  std::string t_body_camera;
  bool poses_are_world_to_body = false;
  bool poses_are_world_to_camera = false;
  bool verbose = false;
};

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

struct MemorySnapshot {
  size_t free_bytes = 0;
  size_t total_bytes = 0;

  size_t usedBytes() const { return total_bytes >= free_bytes ? total_bytes - free_bytes : 0; }
};

struct TimingStats {
  double mean_ms = 0.0;
  double median_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double stddev_ms = 0.0;
};

struct BenchmarkResult {
  std::string mode;
  std::string engine_path;
  std::string pose_path;
  std::vector<std::string> image_paths;
  int warmup = 0;
  int runs = 0;
  int views_per_call = 1;
  TimingStats timing;
  MemorySnapshot before_model;
  MemorySnapshot after_model;
  MemorySnapshot after_warmup;
  MemorySnapshot after_benchmark;
  size_t peak_used_bytes = 0;
};

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

double imageTimestampSeconds(const fs::path& image_path) {
  const std::string stem = image_path.stem().string();
  const long double value = std::stold(stem);
  if (value > 1.0e12L) {
    return static_cast<double>(value * 1.0e-9L);
  }
  return static_cast<double>(value);
}

Pose nearestPose(const std::vector<Pose>& poses, double timestamp, double max_dt) {
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
  return *best;
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

cv::Matx44f toCvMatx44f(const Eigen::Matrix4d& matrix) {
  cv::Matx44f out;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      out(r, c) = static_cast<float>(matrix(r, c));
    }
  }
  return out;
}

void checkCuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

void synchronizeCuda() { checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

MemorySnapshot queryCudaMemory() {
  MemorySnapshot snapshot;
  checkCuda(cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes), "cudaMemGetInfo");
  return snapshot;
}

double bytesToMiB(size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

size_t positiveDelta(size_t value, size_t baseline) { return value >= baseline ? value - baseline : 0; }

class CudaMemorySampler {
 public:
  void start() {
    running_.store(true);
    peak_used_bytes_.store(queryCudaMemory().usedBytes());
    worker_ = std::thread([this]() {
      while (running_.load()) {
        const size_t used = queryCudaMemory().usedBytes();
        size_t current = peak_used_bytes_.load();
        while (used > current && !peak_used_bytes_.compare_exchange_weak(current, used)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
  }

  size_t stop() {
    const size_t used = queryCudaMemory().usedBytes();
    size_t current = peak_used_bytes_.load();
    while (used > current && !peak_used_bytes_.compare_exchange_weak(current, used)) {
    }

    running_.store(false);
    if (worker_.joinable()) {
      worker_.join();
    }
    return peak_used_bytes_.load();
  }

 private:
  std::atomic<bool> running_{false};
  std::atomic<size_t> peak_used_bytes_{0};
  std::thread worker_;
};

TimingStats computeTimingStats(std::vector<double> values) {
  if (values.empty()) {
    return {};
  }

  TimingStats stats;
  stats.min_ms = *std::min_element(values.begin(), values.end());
  stats.max_ms = *std::max_element(values.begin(), values.end());
  stats.mean_ms = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());

  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  if (values.size() % 2 == 0) {
    stats.median_ms = 0.5 * (values[middle - 1] + values[middle]);
  } else {
    stats.median_ms = values[middle];
  }

  double variance = 0.0;
  for (double value : values) {
    const double diff = value - stats.mean_ms;
    variance += diff * diff;
  }
  variance /= static_cast<double>(values.size());
  stats.stddev_ms = std::sqrt(variance);
  return stats;
}

std::string jsonEscape(const std::string& value) {
  std::ostringstream oss;
  for (char c : value) {
    switch (c) {
      case '\\':
        oss << "\\\\";
        break;
      case '"':
        oss << "\\\"";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        oss << c;
        break;
    }
  }
  return oss.str();
}

std::vector<cv::Mat> loadImages(const std::vector<std::string>& image_paths) {
  std::vector<cv::Mat> images;
  images.reserve(image_paths.size());
  for (const auto& image_path : image_paths) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("Failed to read image: " + image_path);
    }
    images.push_back(std::move(image));
  }
  return images;
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

std::vector<xfeat::CameraIntrinsics> makeIntrinsicsForImages(const std::vector<cv::Mat>& images,
                                                             bool enabled,
                                                             double fx,
                                                             double fy,
                                                             double cx,
                                                             double cy) {
  std::vector<xfeat::CameraIntrinsics> intrinsics;
  if (!enabled) {
    return intrinsics;
  }

  intrinsics.reserve(images.size());
  for (const auto& image : images) {
    intrinsics.push_back(*makeIntrinsics(image.size(), true, fx, fy, cx, cy));
  }
  return intrinsics;
}

std::vector<cv::Matx44f> makeWorldToCameraExtrinsics(const std::vector<std::string>& image_paths,
                                                     const std::string& pose_path,
                                                     double pose_time_offset,
                                                     double pose_max_dt,
                                                     bool pose_is_world_to_body,
                                                     const std::string& t_body_camera_text) {
  if (pose_path.empty()) {
    return {};
  }

  const auto poses = readPoses(pose_path);
  const Eigen::Matrix4d body_from_camera = parseTransform(t_body_camera_text, "--t-body-camera");

  std::vector<cv::Matx44f> extrinsics;
  extrinsics.reserve(image_paths.size());
  for (const auto& image_path : image_paths) {
    const Pose pose =
        nearestPose(poses, imageTimestampSeconds(fs::path(image_path)) + pose_time_offset, pose_max_dt);
    const Eigen::Matrix4d world_from_camera = poseToTransform(pose, pose_is_world_to_body) * body_from_camera;
    extrinsics.push_back(toCvMatx44f(world_from_camera.inverse()));
  }
  return extrinsics;
}

template <typename Fn>
TimingStats timeInference(int runs, Fn&& fn) {
  std::vector<double> times_ms;
  times_ms.reserve(static_cast<size_t>(runs));
  for (int i = 0; i < runs; ++i) {
    synchronizeCuda();
    const auto start = std::chrono::steady_clock::now();
    fn();
    synchronizeCuda();
    const auto end = std::chrono::steady_clock::now();
    times_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  return computeTimingStats(std::move(times_ms));
}

void printMemoryLine(const std::string& label, const MemorySnapshot& snapshot, const MemorySnapshot& baseline) {
  const double used = bytesToMiB(snapshot.usedBytes());
  const double delta = bytesToMiB(positiveDelta(snapshot.usedBytes(), baseline.usedBytes()));
  std::cout << "  " << std::left << std::setw(18) << label << std::right << std::fixed << std::setprecision(1) << used
            << " MiB used";
  if (snapshot.usedBytes() >= baseline.usedBytes()) {
    std::cout << " (+" << delta << " MiB)";
  }
  std::cout << std::endl;
}

void printBenchmarkResult(const BenchmarkResult& result) {
  const double calls_per_second = result.timing.mean_ms > 0.0 ? 1000.0 / result.timing.mean_ms : 0.0;
  const double views_per_second = calls_per_second * static_cast<double>(result.views_per_call);
  const double peak_delta_mib = result.peak_used_bytes >= result.before_model.usedBytes()
                                    ? bytesToMiB(result.peak_used_bytes - result.before_model.usedBytes())
                                    : 0.0;

  std::cout << "\nBenchmark result" << std::endl;
  std::cout << "  mode=" << result.mode << std::endl;
  std::cout << "  views_per_call=" << result.views_per_call << ", warmup=" << result.warmup << ", runs=" << result.runs
            << std::endl;
  std::cout << "  runtime_ms: mean=" << std::fixed << std::setprecision(3) << result.timing.mean_ms
            << ", median=" << result.timing.median_ms << ", min=" << result.timing.min_ms
            << ", max=" << result.timing.max_ms << ", stddev=" << result.timing.stddev_ms << std::endl;
  std::cout << "  throughput: " << std::fixed << std::setprecision(2) << calls_per_second << " calls/s, "
            << views_per_second << " views/s" << std::endl;
  std::cout << "  vram:" << std::endl;
  printMemoryLine("before_model", result.before_model, result.before_model);
  printMemoryLine("after_model", result.after_model, result.before_model);
  printMemoryLine("after_warmup", result.after_warmup, result.before_model);
  printMemoryLine("after_benchmark", result.after_benchmark, result.before_model);
  std::cout << "  " << std::left << std::setw(18) << "peak_sampled" << std::right << std::fixed << std::setprecision(1)
            << bytesToMiB(result.peak_used_bytes) << " MiB used (+" << peak_delta_mib << " MiB)" << std::endl;
}

void writeBenchmarkJson(const fs::path& path, const BenchmarkResult& result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to write benchmark JSON: " + path.string());
  }

  const auto write_memory = [&out](const char* name, const MemorySnapshot& snapshot, bool comma) {
    out << "  \"" << name << "\": {"
        << "\"used_mib\": " << bytesToMiB(snapshot.usedBytes()) << ", "
        << "\"free_mib\": " << bytesToMiB(snapshot.free_bytes) << ", "
        << "\"total_mib\": " << bytesToMiB(snapshot.total_bytes) << "}";
    if (comma) {
      out << ",";
    }
    out << "\n";
  };

  out << std::fixed << std::setprecision(6);
  out << "{\n";
  out << "  \"mode\": \"" << jsonEscape(result.mode) << "\",\n";
  out << "  \"engine_path\": \"" << jsonEscape(result.engine_path) << "\",\n";
  if (!result.pose_path.empty()) {
    out << "  \"pose_path\": \"" << jsonEscape(result.pose_path) << "\",\n";
  }
  out << "  \"warmup\": " << result.warmup << ",\n";
  out << "  \"runs\": " << result.runs << ",\n";
  out << "  \"views_per_call\": " << result.views_per_call << ",\n";
  out << "  \"runtime_ms\": {"
      << "\"mean\": " << result.timing.mean_ms << ", "
      << "\"median\": " << result.timing.median_ms << ", "
      << "\"min\": " << result.timing.min_ms << ", "
      << "\"max\": " << result.timing.max_ms << ", "
      << "\"stddev\": " << result.timing.stddev_ms << "},\n";
  out << "  \"image_paths\": [";
  for (size_t i = 0; i < result.image_paths.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << "\"" << jsonEscape(result.image_paths[i]) << "\"";
  }
  out << "],\n";
  write_memory("before_model", result.before_model, true);
  write_memory("after_model", result.after_model, true);
  write_memory("after_warmup", result.after_warmup, true);
  write_memory("after_benchmark", result.after_benchmark, true);
  out << "  \"peak_sampled_used_mib\": " << bytesToMiB(result.peak_used_bytes) << ",\n";
  out << "  \"peak_sampled_delta_mib\": "
      << bytesToMiB(positiveDelta(result.peak_used_bytes, result.before_model.usedBytes())) << "\n";
  out << "}\n";
}

BenchmarkResult runBenchmark(const BenchmarkOptions& options) {
  if (options.mode != "single" && options.mode != "batch") {
    throw std::runtime_error("--mode must be either 'single' or 'batch'");
  }
  if (options.image_paths.empty()) {
    throw std::runtime_error("At least one image path is required");
  }
  if (options.mode == "single" && options.image_paths.size() != 1) {
    throw std::runtime_error("--mode single expects exactly one image path");
  }
  if (options.warmup < 0) {
    throw std::runtime_error("--warmup must be non-negative");
  }
  if (options.runs <= 0) {
    throw std::runtime_error("--runs must be positive");
  }
  if (options.pose_max_dt <= 0.0) {
    throw std::runtime_error("--pose-max-dt must be positive");
  }
  const bool has_intrinsics = options.fx > 0.0 || options.fy > 0.0 || options.cx > 0.0 || options.cy > 0.0;
  if (has_intrinsics && (options.fx <= 0.0 || options.fy <= 0.0)) {
    throw std::runtime_error("Both --fx and --fy must be positive when intrinsics are provided");
  }

  const auto images = loadImages(options.image_paths);
  const auto single_intrinsics =
      makeIntrinsics(images.front().size(), has_intrinsics, options.fx, options.fy, options.cx, options.cy);
  const auto batch_intrinsics =
      makeIntrinsicsForImages(images, has_intrinsics, options.fx, options.fy, options.cx, options.cy);
  const bool pose_is_world_to_body = options.poses_are_world_to_body || options.poses_are_world_to_camera;
  const auto batch_extrinsics = makeWorldToCameraExtrinsics(options.image_paths,
                                                            options.pose_path,
                                                            options.pose_time_offset,
                                                            options.pose_max_dt,
                                                            pose_is_world_to_body,
                                                            options.t_body_camera);

  BenchmarkResult result;
  result.mode = options.mode;
  result.engine_path = options.engine_path;
  result.pose_path = options.pose_path;
  result.image_paths = options.image_paths;
  result.warmup = options.warmup;
  result.runs = options.runs;
  result.views_per_call = options.mode == "single" ? 1 : static_cast<int>(images.size());

  synchronizeCuda();
  result.before_model = queryCudaMemory();

  xfeat::DepthAnythingV3TRT::Params params;
  params.engine_path = options.engine_path;
  params.verbose = options.verbose;
  xfeat::DepthAnythingV3TRT model(params);
  if (model.has_camera_inputs()) {
    if (options.mode != "batch") {
      throw std::runtime_error("Pose-conditioned DA3 engines require --mode batch");
    }
    if (!has_intrinsics) {
      throw std::runtime_error("Pose-conditioned DA3 engines require --fx and --fy");
    }
    if (batch_extrinsics.empty()) {
      throw std::runtime_error("Pose-conditioned DA3 engines require --poses");
    }
  }

  synchronizeCuda();
  result.after_model = queryCudaMemory();

  const auto call_inference = [&]() {
    if (options.mode == "single") {
      volatile int width = model.infer(images.front(), single_intrinsics).depth.cols;
      (void)width;
      return;
    }
    const auto results = model.has_camera_inputs() ? model.infer_multi_view(images, batch_intrinsics, batch_extrinsics)
                                                   : model.infer_multi_view(images, batch_intrinsics);
    volatile size_t count = results.size();
    (void)count;
  };

  for (int i = 0; i < options.warmup; ++i) {
    call_inference();
  }
  synchronizeCuda();
  result.after_warmup = queryCudaMemory();

  CudaMemorySampler sampler;
  sampler.start();
  result.timing = timeInference(options.runs, call_inference);
  result.peak_used_bytes = sampler.stop();
  synchronizeCuda();
  result.after_benchmark = queryCudaMemory();
  result.peak_used_bytes = std::max(result.peak_used_bytes, result.after_benchmark.usedBytes());
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  BenchmarkOptions options;

  po::options_description visible("Benchmark DA3 TensorRT mono-depth inference");
  visible.add_options()("help,h", "Show this help message")(
      "engine,e", po::value<std::string>(&options.engine_path)->required(), "Path to a DA3 TensorRT .engine file")(
      "mode", po::value<std::string>(&options.mode)->default_value(options.mode), "Benchmark mode: single or batch")(
      "warmup", po::value<int>(&options.warmup)->default_value(options.warmup), "Warmup iterations")(
      "runs", po::value<int>(&options.runs)->default_value(options.runs), "Measured iterations")(
      "fx", po::value<double>(&options.fx), "Camera fx in pixels")(
      "fy", po::value<double>(&options.fy), "Camera fy in pixels")(
      "cx", po::value<double>(&options.cx), "Camera cx in pixels")(
      "cy", po::value<double>(&options.cy), "Camera cy in pixels")(
      "poses",
      po::value<std::string>(&options.pose_path),
      "Optional pose file with timestamp tx ty tz qx qy qz qw rows for pose-conditioned engines")(
      "pose-time-offset",
      po::value<double>(&options.pose_time_offset)->default_value(options.pose_time_offset),
      "Seconds added to image timestamps before pose lookup")(
      "pose-max-dt",
      po::value<double>(&options.pose_max_dt)->default_value(options.pose_max_dt),
      "Maximum allowed absolute nearest-pose time difference in seconds")(
      "t-body-camera",
      po::value<std::string>(&options.t_body_camera)->default_value(options.t_body_camera),
      "Row-major 4x4 T_body_camera matrix as comma- or space-separated values; default is identity")(
      "poses-are-world-to-body",
      po::bool_switch(&options.poses_are_world_to_body),
      "Interpret poses as T_body_world and invert them before benchmarking")(
      "poses-are-world-to-camera",
      po::bool_switch(&options.poses_are_world_to_camera),
      "Alias for --poses-are-world-to-body")(
      "json-out", po::value<std::string>(&options.json_out), "Optional benchmark JSON output path")(
      "verbose,v", po::bool_switch(&options.verbose), "Enable TensorRT metadata logging");

  po::options_description hidden("Hidden options");
  hidden.add_options()("images", po::value<std::vector<std::string>>(&options.image_paths), "Image paths");

  po::options_description all;
  all.add(visible).add(hidden);
  po::positional_options_description positional;
  positional.add("images", -1);

  try {
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(all).positional(positional).run(), vm);
    if (vm.count("help")) {
      std::cout << visible << "\n\nUsage:\n"
                << "  " << argv[0] << " --engine da3.engine --mode single image.png\n"
                << "  " << argv[0] << " --engine da3_v5.engine --mode batch view0.png view1.png ...\n"
                << "  " << argv[0]
                << " --engine da3_pose.engine --mode batch --poses poses.txt --fx 940 --fy 938 view0.png view1.png "
                   "view2.png\n";
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << "Argument error: " << e.what() << "\n\n" << visible << std::endl;
    return 2;
  }

  try {
    const auto result = runBenchmark(options);
    printBenchmarkResult(result);
    if (!options.json_out.empty()) {
      writeBenchmarkJson(options.json_out, result);
      std::cout << "  json=" << options.json_out << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "Mono-depth benchmark failed: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

#else

int main() {
  std::cerr << "mono_depth_benchmark requires TensorRT support." << std::endl;
  return 1;
}

#endif  // HAVE_TENSORRT
