#include <iostream>

#ifdef HAVE_TENSORRT

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
  bool verbose = false;
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
  const bool has_intrinsics = options.fx > 0.0 || options.fy > 0.0 || options.cx > 0.0 || options.cy > 0.0;
  if (has_intrinsics && (options.fx <= 0.0 || options.fy <= 0.0)) {
    throw std::runtime_error("Both --fx and --fy must be positive when intrinsics are provided");
  }

  const auto images = loadImages(options.image_paths);
  const auto single_intrinsics =
      makeIntrinsics(images.front().size(), has_intrinsics, options.fx, options.fy, options.cx, options.cy);
  const auto batch_intrinsics =
      makeIntrinsicsForImages(images, has_intrinsics, options.fx, options.fy, options.cx, options.cy);

  BenchmarkResult result;
  result.mode = options.mode;
  result.engine_path = options.engine_path;
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

  synchronizeCuda();
  result.after_model = queryCudaMemory();

  const auto call_inference = [&]() {
    if (options.mode == "single") {
      volatile int width = model.infer(images.front(), single_intrinsics).depth.cols;
      (void)width;
      return;
    }
    volatile size_t count = model.infer_multi_view(images, batch_intrinsics).size();
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
                << "  " << argv[0] << " --engine da3_v5.engine --mode batch view0.png view1.png ...\n";
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
