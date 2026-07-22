#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "xfeat-cpp/lighterglue_trt.h"
#include "xfeat-cpp/place_recognition/jist_trt.h"
#include "xfeat-cpp/place_recognition/mixvpr_trt.h"
#include "xfeat-cpp/tensorrt/detail/trt_engine.h"
#include "xfeat-cpp/xfeat_trt.h"

#ifdef HAVE_TENSORRT

namespace {

struct Options {
  std::string xfeat_engine;
  std::string lighterglue_engine;
  std::string jist_engine;
  std::string mixvpr_engine;
  std::vector<std::string> image_paths;
  std::string json_out;
  int top_k = 500;
  int warmup = 20;
  int runs = 200;
  bool verbose = false;
};

struct TimingStats {
  double mean_ms = 0.0;
  double median_ms = 0.0;
  double p90_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double stddev_ms = 0.0;
};

struct JistBreakdownStats {
  TimingStats preprocess;
  TimingStats engine_wrapper;
  TimingStats postprocess;
  TimingStats total;
};

struct MemorySnapshot {
  size_t free_bytes = 0;
  size_t total_bytes = 0;

  size_t usedBytes() const { return total_bytes - free_bytes; }
};

void printUsage(const char* executable) {
  std::cout << "XFeat/LighterGlue TensorRT benchmark\n\n"
            << "Usage:\n  " << executable
            << " --xfeat-engine xfeat.engine --lighterglue-engine lighterglue.engine [options] image0.jpg "
               "image1.jpg\n\n"
            << "Options:\n"
            << "  --top-k N       Maximum XFeat keypoints per image (default: 500)\n"
            << "  --warmup N      Warmup calls for each stage (default: 20)\n"
            << "  --runs N        Measured calls for each stage (default: 200)\n"
            << "  --json-out PATH Optional benchmark JSON output path\n"
            << "  --jist-engine PATH  Also benchmark a JIST TensorRT engine\n"
            << "  --mixvpr-engine PATH  Also benchmark a MixVPR TensorRT engine\n"
            << "  --verbose       Enable TensorRT metadata logging\n";
}

std::string requireValue(int argc, char** argv, int& index, const std::string& option) {
  if (++index >= argc) throw std::invalid_argument("Missing value for " + option);
  return argv[index];
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    } else if (argument == "--xfeat-engine") {
      options.xfeat_engine = requireValue(argc, argv, index, argument);
    } else if (argument == "--lighterglue-engine") {
      options.lighterglue_engine = requireValue(argc, argv, index, argument);
    } else if (argument == "--jist-engine") {
      options.jist_engine = requireValue(argc, argv, index, argument);
    } else if (argument == "--mixvpr-engine") {
      options.mixvpr_engine = requireValue(argc, argv, index, argument);
    } else if (argument == "--top-k") {
      options.top_k = std::stoi(requireValue(argc, argv, index, argument));
    } else if (argument == "--warmup") {
      options.warmup = std::stoi(requireValue(argc, argv, index, argument));
    } else if (argument == "--runs") {
      options.runs = std::stoi(requireValue(argc, argv, index, argument));
    } else if (argument == "--json-out") {
      options.json_out = requireValue(argc, argv, index, argument);
    } else if (argument == "--verbose" || argument == "-v") {
      options.verbose = true;
    } else if (!argument.empty() && argument.front() == '-') {
      throw std::invalid_argument("Unknown option: " + argument);
    } else {
      options.image_paths.push_back(argument);
    }
  }
  if (options.xfeat_engine.empty()) throw std::invalid_argument("--xfeat-engine is required");
  if (options.lighterglue_engine.empty()) throw std::invalid_argument("--lighterglue-engine is required");
  return options;
}

void checkCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void synchronizeCuda() { checkCuda(cudaDeviceSynchronize(), "CUDA synchronization failed"); }

MemorySnapshot queryCudaMemory() {
  MemorySnapshot result;
  checkCuda(cudaMemGetInfo(&result.free_bytes, &result.total_bytes), "CUDA memory query failed");
  return result;
}

double percentile(const std::vector<double>& sorted, double fraction) {
  if (sorted.empty()) return 0.0;
  const double index = fraction * static_cast<double>(sorted.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(index));
  const size_t upper = static_cast<size_t>(std::ceil(index));
  const double weight = index - static_cast<double>(lower);
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

TimingStats computeStats(std::vector<double> times_ms) {
  if (times_ms.empty()) throw std::invalid_argument("Cannot compute timing statistics without samples");
  TimingStats stats;
  stats.mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
  double variance = 0.0;
  for (const double value : times_ms) variance += (value - stats.mean_ms) * (value - stats.mean_ms);
  stats.stddev_ms = std::sqrt(variance / times_ms.size());
  std::sort(times_ms.begin(), times_ms.end());
  stats.min_ms = times_ms.front();
  stats.max_ms = times_ms.back();
  stats.median_ms = percentile(times_ms, 0.50);
  stats.p90_ms = percentile(times_ms, 0.90);
  stats.p95_ms = percentile(times_ms, 0.95);
  stats.p99_ms = percentile(times_ms, 0.99);
  return stats;
}

template <typename Fn>
TimingStats timeCalls(int runs, Fn&& call) {
  std::vector<double> times_ms;
  times_ms.reserve(static_cast<size_t>(runs));
  for (int i = 0; i < runs; ++i) {
    synchronizeCuda();
    const auto start = std::chrono::steady_clock::now();
    call(i);
    synchronizeCuda();
    const auto end = std::chrono::steady_clock::now();
    times_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  return computeStats(std::move(times_ms));
}

size_t matchCount(const std::vector<std::vector<int>>& matches) {
  size_t count = 0;
  for (const auto& indices : matches) count += indices.size();
  return count;
}

std::vector<float> prepareJistInput(const std::vector<cv::Mat>& sequence, int width, int height) {
  const size_t plane = static_cast<size_t>(height) * width;
  std::vector<float> tensor(sequence.size() * 3 * plane);
  for (size_t sequence_index = 0; sequence_index < sequence.size(); ++sequence_index) {
    cv::Mat color;
    cv::cvtColor(sequence[sequence_index], color, cv::COLOR_BGR2RGB);
    cv::resize(color, color, cv::Size(width, height));
    color.convertTo(color, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(color, channels);
    for (int channel = 0; channel < 3; ++channel) {
      const size_t offset = (sequence_index * 3 + static_cast<size_t>(channel)) * plane;
      std::memcpy(tensor.data() + offset, channels[channel].ptr<float>(), plane * sizeof(float));
    }
  }
  return tensor;
}

JistBreakdownStats timeJistBreakdown(int runs,
                                     const std::string& engine_path,
                                     const std::vector<cv::Mat>& sequence,
                                     int width,
                                     int height) {
  xfeat::trt_detail::Engine engine(engine_path);
  const std::string input_name = engine.input_names().front();
  const std::string output_name = engine.output_names().front();
  const std::vector<int64_t> input_shape = engine.tensor_shape(input_name);
  std::vector<double> preprocess_ms;
  std::vector<double> engine_ms;
  std::vector<double> postprocess_ms;
  std::vector<double> total_ms;
  preprocess_ms.reserve(static_cast<size_t>(runs));
  engine_ms.reserve(static_cast<size_t>(runs));
  postprocess_ms.reserve(static_cast<size_t>(runs));
  total_ms.reserve(static_cast<size_t>(runs));

  for (int run = 0; run < runs; ++run) {
    synchronizeCuda();
    const auto start = std::chrono::steady_clock::now();
    const std::vector<float> input = prepareJistInput(sequence, width, height);
    const auto after_preprocess = std::chrono::steady_clock::now();
    const auto outputs = engine.run({{input_name, input_shape, input.data(), input.size() * sizeof(float)}});
    const auto after_engine = std::chrono::steady_clock::now();
    const std::vector<float> values = xfeat::trt_detail::find_output(outputs, output_name).values<float>();
    cv::Mat descriptor(1, static_cast<int>(values.size()), CV_32F);
    std::memcpy(descriptor.ptr<float>(), values.data(), values.size() * sizeof(float));
    const double norm = cv::norm(descriptor, cv::NORM_L2);
    if (norm > 1e-8) descriptor /= norm;
    const auto end = std::chrono::steady_clock::now();

    preprocess_ms.push_back(std::chrono::duration<double, std::milli>(after_preprocess - start).count());
    engine_ms.push_back(std::chrono::duration<double, std::milli>(after_engine - after_preprocess).count());
    postprocess_ms.push_back(std::chrono::duration<double, std::milli>(end - after_engine).count());
    total_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }

  return {
      computeStats(std::move(preprocess_ms)),
      computeStats(std::move(engine_ms)),
      computeStats(std::move(postprocess_ms)),
      computeStats(std::move(total_ms)),
  };
}

double bytesToMiB(size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

double usedDeltaMiB(const MemorySnapshot& value, const MemorySnapshot& baseline) {
  return value.usedBytes() > baseline.usedBytes() ? bytesToMiB(value.usedBytes() - baseline.usedBytes()) : 0.0;
}

std::string jsonEscape(const std::string& value) {
  std::string escaped;
  for (const char character : value) {
    switch (character) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      default:
        escaped += character;
    }
  }
  return escaped;
}

void printStats(const char* label, const TimingStats& stats, const char* throughput_label, double work_per_call) {
  const double throughput = stats.mean_ms > 0.0 ? 1000.0 * work_per_call / stats.mean_ms : 0.0;
  std::cout << "  " << label << "_ms: mean=" << std::fixed << std::setprecision(3) << stats.mean_ms
            << ", median=" << stats.median_ms << ", p90=" << stats.p90_ms << ", p95=" << stats.p95_ms
            << ", p99=" << stats.p99_ms << ", min=" << stats.min_ms << ", max=" << stats.max_ms
            << ", stddev=" << stats.stddev_ms << '\n';
  std::cout << "  " << throughput_label << '=' << std::fixed << std::setprecision(2) << throughput << '\n';
}

void writeStatsJson(std::ostream& out, const char* name, const TimingStats& stats, bool comma) {
  out << "  \"" << name << "\": {"
      << "\"mean\": " << stats.mean_ms << ", "
      << "\"median\": " << stats.median_ms << ", "
      << "\"p90\": " << stats.p90_ms << ", "
      << "\"p95\": " << stats.p95_ms << ", "
      << "\"p99\": " << stats.p99_ms << ", "
      << "\"min\": " << stats.min_ms << ", "
      << "\"max\": " << stats.max_ms << ", "
      << "\"stddev\": " << stats.stddev_ms << '}';
  if (comma) out << ',';
  out << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  try {
    options = parseOptions(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "Argument error: " << error.what() << "\n\n";
    printUsage(argv[0]);
    return 2;
  }

  try {
    if (options.image_paths.size() != 2) throw std::invalid_argument("Exactly two input images are required");
    if (options.top_k <= 0) throw std::invalid_argument("--top-k must be positive");
    if (options.warmup < 0) throw std::invalid_argument("--warmup must be non-negative");
    if (options.runs <= 0) throw std::invalid_argument("--runs must be positive");

    const cv::Mat image0 = cv::imread(options.image_paths[0], cv::IMREAD_COLOR);
    const cv::Mat image1 = cv::imread(options.image_paths[1], cv::IMREAD_COLOR);
    if (image0.empty() || image1.empty()) throw std::runtime_error("Failed to read one or both input images");
    const std::array<float, 2> size0 = {static_cast<float>(image0.cols), static_cast<float>(image0.rows)};
    const std::array<float, 2> size1 = {static_cast<float>(image1.cols), static_cast<float>(image1.rows)};

    synchronizeCuda();
    const MemorySnapshot before_models = queryCudaMemory();
    const auto xfeat_load_start = std::chrono::steady_clock::now();
    xfeat::XFeatTRT extractor({.engine_path = options.xfeat_engine, .verbose = options.verbose});
    synchronizeCuda();
    const auto xfeat_load_end = std::chrono::steady_clock::now();
    const auto lighterglue_load_start = std::chrono::steady_clock::now();
    xfeat::LighterGlueTRT matcher(options.lighterglue_engine, options.verbose);
    synchronizeCuda();
    const auto lighterglue_load_end = std::chrono::steady_clock::now();
    std::unique_ptr<xfeat::JistTRT> jist;
    double jist_load_ms = 0.0;
    if (!options.jist_engine.empty()) {
      xfeat::JistTRT::Params params;
      params.model_path = options.jist_engine;
      params.verbose = options.verbose;
      const auto jist_load_start = std::chrono::steady_clock::now();
      jist = std::make_unique<xfeat::JistTRT>(params);
      synchronizeCuda();
      const auto jist_load_end = std::chrono::steady_clock::now();
      jist_load_ms = std::chrono::duration<double, std::milli>(jist_load_end - jist_load_start).count();
    }
    std::unique_ptr<xfeat::MixVPRTRT> mixvpr;
    double mixvpr_load_ms = 0.0;
    if (!options.mixvpr_engine.empty()) {
      xfeat::MixVPRTRT::Params params;
      params.model_path = options.mixvpr_engine;
      params.verbose = options.verbose;
      const auto mixvpr_load_start = std::chrono::steady_clock::now();
      mixvpr = std::make_unique<xfeat::MixVPRTRT>(params);
      synchronizeCuda();
      const auto mixvpr_load_end = std::chrono::steady_clock::now();
      mixvpr_load_ms = std::chrono::duration<double, std::milli>(mixvpr_load_end - mixvpr_load_start).count();
    }
    const MemorySnapshot after_models = queryCudaMemory();
    const double xfeat_load_ms = std::chrono::duration<double, std::milli>(xfeat_load_end - xfeat_load_start).count();
    const double lighterglue_load_ms =
        std::chrono::duration<double, std::milli>(lighterglue_load_end - lighterglue_load_start).count();

    xfeat::DetectionResult detection0 = extractor.detect_and_compute(image0, options.top_k);
    xfeat::DetectionResult detection1 = extractor.detect_and_compute(image1, options.top_k);
    std::vector<std::vector<int>> matches = matcher.match(detection0, size0, detection1, size1);
    const size_t canonical_matches = matchCount(matches);
    std::vector<cv::Mat> jist_sequence;
    cv::Mat jist_descriptor;
    if (jist) {
      jist_sequence.reserve(static_cast<size_t>(jist->get_seq_length()));
      for (int index = 0; index < jist->get_seq_length(); ++index) {
        jist_sequence.push_back(index % 2 == 0 ? image0 : image1);
      }
      jist_descriptor = jist->infer(jist_sequence);
    }
    cv::Mat mixvpr_descriptor;
    if (mixvpr) mixvpr_descriptor = mixvpr->infer(image0);

    for (int i = 0; i < options.warmup; ++i) {
      const cv::Mat& image = (i % 2 == 0) ? image0 : image1;
      detection0 = extractor.detect_and_compute(image, options.top_k);
    }
    detection0 = extractor.detect_and_compute(image0, options.top_k);
    detection1 = extractor.detect_and_compute(image1, options.top_k);
    for (int i = 0; i < options.warmup; ++i) {
      matches = matcher.match(detection0, size0, detection1, size1);
    }
    for (int i = 0; i < options.warmup; ++i) {
      auto first = extractor.detect_and_compute(image0, options.top_k);
      auto second = extractor.detect_and_compute(image1, options.top_k);
      matches = matcher.match(first, size0, second, size1);
    }
    if (jist) {
      for (int i = 0; i < options.warmup; ++i) jist_descriptor = jist->infer(jist_sequence);
    }
    if (mixvpr) {
      for (int i = 0; i < options.warmup; ++i) {
        mixvpr_descriptor = mixvpr->infer(i % 2 == 0 ? image0 : image1);
      }
    }
    synchronizeCuda();
    const MemorySnapshot after_warmup = queryCudaMemory();

    const TimingStats xfeat_timing = timeCalls(options.runs, [&](int iteration) {
      const cv::Mat& image = (iteration % 2 == 0) ? image0 : image1;
      detection0 = extractor.detect_and_compute(image, options.top_k);
    });
    detection0 = extractor.detect_and_compute(image0, options.top_k);
    detection1 = extractor.detect_and_compute(image1, options.top_k);
    const TimingStats lighterglue_timing =
        timeCalls(options.runs, [&](int) { matches = matcher.match(detection0, size0, detection1, size1); });
    const TimingStats pipeline_timing = timeCalls(options.runs, [&](int) {
      auto first = extractor.detect_and_compute(image0, options.top_k);
      auto second = extractor.detect_and_compute(image1, options.top_k);
      matches = matcher.match(first, size0, second, size1);
    });
    std::optional<TimingStats> jist_timing;
    std::optional<JistBreakdownStats> jist_breakdown;
    if (jist) {
      jist_timing = timeCalls(options.runs, [&](int) { jist_descriptor = jist->infer(jist_sequence); });
      jist_breakdown = timeJistBreakdown(
          options.runs, options.jist_engine, jist_sequence, jist->get_img_width(), jist->get_img_height());
    }
    std::optional<TimingStats> mixvpr_timing;
    if (mixvpr) {
      mixvpr_timing = timeCalls(options.runs, [&](int iteration) {
        mixvpr_descriptor = mixvpr->infer(iteration % 2 == 0 ? image0 : image1);
      });
    }
    synchronizeCuda();
    const MemorySnapshot after_benchmark = queryCudaMemory();

    std::cout << "\nXFeat/LighterGlue" << (jist ? "/JIST" : "") << (mixvpr ? "/MixVPR" : "") << " TensorRT benchmark\n";
    std::cout << "  images=" << image0.cols << 'x' << image0.rows << ", " << image1.cols << 'x' << image1.rows
              << "; model_input=" << extractor.input_width() << 'x' << extractor.input_height() << '\n';
    std::cout << "  top_k=" << options.top_k << ", keypoints=" << detection0.keypoints.rows << '/'
              << detection1.keypoints.rows << ", matches=" << canonical_matches << '\n';
    std::cout << "  warmup=" << options.warmup << ", runs=" << options.runs << '\n';
    std::cout << "  engine_load_ms: xfeat=" << std::fixed << std::setprecision(3) << xfeat_load_ms
              << ", lighterglue=" << lighterglue_load_ms;
    if (jist) std::cout << ", jist=" << jist_load_ms;
    if (mixvpr) std::cout << ", mixvpr=" << mixvpr_load_ms;
    std::cout << '\n';
    printStats("xfeat_per_image", xfeat_timing, "images_per_second", 1.0);
    printStats("lighterglue_per_pair", lighterglue_timing, "pairs_per_second", 1.0);
    printStats("pipeline_two_images_plus_match", pipeline_timing, "pairs_per_second", 1.0);
    if (jist_timing) {
      std::cout << "  jist_sequence=" << jist_sequence.size() << " frames, descriptor_dim=" << jist_descriptor.cols
                << ", descriptor_l2=" << cv::norm(jist_descriptor, cv::NORM_L2) << '\n';
      printStats("jist_per_sequence", *jist_timing, "sequences_per_second", 1.0);
      printStats("jist_breakdown_preprocess", jist_breakdown->preprocess, "sequences_per_second", 1.0);
      printStats("jist_breakdown_engine_wrapper", jist_breakdown->engine_wrapper, "sequences_per_second", 1.0);
      printStats("jist_breakdown_postprocess", jist_breakdown->postprocess, "sequences_per_second", 1.0);
      printStats("jist_breakdown_total", jist_breakdown->total, "sequences_per_second", 1.0);
    }
    if (mixvpr_timing) {
      std::cout << "  mixvpr_descriptor_dim=" << mixvpr_descriptor.cols
                << ", descriptor_l2=" << cv::norm(mixvpr_descriptor, cv::NORM_L2) << '\n';
      printStats("mixvpr_per_image", *mixvpr_timing, "images_per_second", 1.0);
    }
    std::cout << "  cuda_memory_mib: before_models=" << std::fixed << std::setprecision(1)
              << bytesToMiB(before_models.usedBytes()) << ", after_models=" << bytesToMiB(after_models.usedBytes())
              << " (+" << usedDeltaMiB(after_models, before_models)
              << "), after_warmup=" << bytesToMiB(after_warmup.usedBytes())
              << ", after_benchmark=" << bytesToMiB(after_benchmark.usedBytes()) << '\n';

    if (!options.json_out.empty()) {
      std::ofstream out(options.json_out);
      if (!out) throw std::runtime_error("Failed to write benchmark JSON: " + options.json_out);
      out << std::fixed << std::setprecision(6);
      out << "{\n";
      out << "  \"xfeat_engine\": \"" << jsonEscape(options.xfeat_engine) << "\",\n";
      out << "  \"lighterglue_engine\": \"" << jsonEscape(options.lighterglue_engine) << "\",\n";
      if (jist) out << "  \"jist_engine\": \"" << jsonEscape(options.jist_engine) << "\",\n";
      if (mixvpr) out << "  \"mixvpr_engine\": \"" << jsonEscape(options.mixvpr_engine) << "\",\n";
      out << "  \"images\": [\"" << jsonEscape(options.image_paths[0]) << "\", \"" << jsonEscape(options.image_paths[1])
          << "\"],\n";
      out << "  \"model_input\": [" << extractor.input_width() << ", " << extractor.input_height() << "],\n";
      out << "  \"top_k\": " << options.top_k << ",\n";
      out << "  \"keypoints\": [" << detection0.keypoints.rows << ", " << detection1.keypoints.rows << "],\n";
      out << "  \"matches\": " << canonical_matches << ",\n";
      out << "  \"warmup\": " << options.warmup << ",\n";
      out << "  \"runs\": " << options.runs << ",\n";
      out << "  \"engine_load_ms\": {\"xfeat\": " << xfeat_load_ms << ", \"lighterglue\": " << lighterglue_load_ms;
      if (jist) out << ", \"jist\": " << jist_load_ms;
      if (mixvpr) out << ", \"mixvpr\": " << mixvpr_load_ms;
      out << "},\n";
      writeStatsJson(out, "xfeat_per_image_ms", xfeat_timing, true);
      writeStatsJson(out, "lighterglue_per_pair_ms", lighterglue_timing, true);
      writeStatsJson(out, "pipeline_two_images_plus_match_ms", pipeline_timing, true);
      if (jist_timing) {
        out << "  \"jist_sequence_length\": " << jist_sequence.size() << ",\n";
        out << "  \"jist_descriptor_dim\": " << jist_descriptor.cols << ",\n";
        out << "  \"jist_descriptor_l2\": " << cv::norm(jist_descriptor, cv::NORM_L2) << ",\n";
        writeStatsJson(out, "jist_per_sequence_ms", *jist_timing, true);
        writeStatsJson(out, "jist_breakdown_preprocess_ms", jist_breakdown->preprocess, true);
        writeStatsJson(out, "jist_breakdown_engine_wrapper_ms", jist_breakdown->engine_wrapper, true);
        writeStatsJson(out, "jist_breakdown_postprocess_ms", jist_breakdown->postprocess, true);
        writeStatsJson(out, "jist_breakdown_total_ms", jist_breakdown->total, true);
      }
      if (mixvpr_timing) {
        out << "  \"mixvpr_descriptor_dim\": " << mixvpr_descriptor.cols << ",\n";
        out << "  \"mixvpr_descriptor_l2\": " << cv::norm(mixvpr_descriptor, cv::NORM_L2) << ",\n";
        writeStatsJson(out, "mixvpr_per_image_ms", *mixvpr_timing, true);
      }
      out << "  \"cuda_memory_mib\": {\"before_models\": " << bytesToMiB(before_models.usedBytes())
          << ", \"after_models\": " << bytesToMiB(after_models.usedBytes())
          << ", \"after_warmup\": " << bytesToMiB(after_warmup.usedBytes())
          << ", \"after_benchmark\": " << bytesToMiB(after_benchmark.usedBytes()) << "}\n";
      out << "}\n";
      std::cout << "  json=" << options.json_out << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << "TensorRT feature benchmark failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

#else

int main() {
  std::cerr << "tensorrt_features_benchmark requires TensorRT support.\n";
  return 1;
}

#endif  // HAVE_TENSORRT
