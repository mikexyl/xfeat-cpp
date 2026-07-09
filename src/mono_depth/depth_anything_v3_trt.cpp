#include "xfeat-cpp/mono_depth/depth_anything_v3_trt.h"

#ifdef HAVE_TENSORRT

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "xfeat-cpp/mono_depth/detail/depth_anything_v3_postprocess.h"
#include "xfeat-cpp/mono_depth/detail/depth_anything_v3_tensor.h"

namespace xfeat {
namespace {

class TrtLogger : public nvinfer1::ILogger {
 public:
  explicit TrtLogger(bool verbose) : verbose_(verbose) {}

  void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
    if (severity <= nvinfer1::ILogger::Severity::kWARNING ||
        (verbose_ && severity <= nvinfer1::ILogger::Severity::kINFO)) {
      std::cerr << "[TensorRT] " << msg << std::endl;
    }
  }

 private:
  bool verbose_ = false;
};

void checkCuda(cudaError_t status, const std::string& what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(what + ": " + cudaGetErrorString(status));
  }
}

std::string toLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool containsCaseInsensitive(const std::string& value, const std::string& needle) {
  return toLower(value).find(toLower(needle)) != std::string::npos;
}

int depthOutputScore(const std::string& name) {
  const std::string lower = toLower(name);
  const size_t separator = lower.find_last_of("/.:");
  const std::string base = separator == std::string::npos ? lower : lower.substr(separator + 1);
  if (base == "depth") {
    return 100;
  }
  if (lower.find("depth") != std::string::npos && lower.find("conf") == std::string::npos) {
    return 50;
  }
  if (lower.find("depth") != std::string::npos) {
    return 10;
  }
  return std::numeric_limits<int>::min();
}

int confidenceOutputScore(const std::string& name) {
  const std::string lower = toLower(name);
  const size_t separator = lower.find_last_of("/.:");
  const std::string base = separator == std::string::npos ? lower : lower.substr(separator + 1);
  if (base == "depth_conf" || base == "confidence" || base == "conf") {
    return 100;
  }
  if (lower.find("depth") != std::string::npos && lower.find("conf") != std::string::npos) {
    return 80;
  }
  if (lower.find("conf") != std::string::npos) {
    return 50;
  }
  return std::numeric_limits<int>::min();
}

std::string dimsToString(const nvinfer1::Dims& dims) {
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < dims.nbDims; ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << dims.d[i];
  }
  oss << "]";
  return oss.str();
}

size_t elementSize(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return sizeof(float);
    case nvinfer1::DataType::kHALF:
      return sizeof(__half);
    case nvinfer1::DataType::kINT32:
      return sizeof(int32_t);
    case nvinfer1::DataType::kINT8:
      return sizeof(int8_t);
    case nvinfer1::DataType::kUINT8:
      return sizeof(uint8_t);
    case nvinfer1::DataType::kBOOL:
      return sizeof(bool);
    default:
      throw std::runtime_error("Unsupported TensorRT tensor data type");
  }
}

bool isFloatLike(nvinfer1::DataType dtype) {
  return dtype == nvinfer1::DataType::kFLOAT || dtype == nvinfer1::DataType::kHALF;
}

size_t volume(const nvinfer1::Dims& dims) {
  if (dims.nbDims <= 0) {
    throw std::runtime_error("Invalid tensor dimensions: " + dimsToString(dims));
  }

  size_t count = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) {
      throw std::runtime_error("Tensor dimensions are not fully specified: " + dimsToString(dims));
    }
    count *= static_cast<size_t>(dims.d[i]);
  }
  return count;
}

struct DeviceBuffer {
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), bytes(other.bytes) {
    other.ptr = nullptr;
    other.bytes = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      ptr = other.ptr;
      bytes = other.bytes;
      other.ptr = nullptr;
      other.bytes = 0;
    }
    return *this;
  }

  ~DeviceBuffer() { release(); }

  void resize(size_t requested_bytes) {
    if (requested_bytes <= bytes) {
      return;
    }
    release();
    checkCuda(cudaMalloc(&ptr, requested_bytes), "Failed to allocate CUDA buffer");
    bytes = requested_bytes;
  }

  void release() {
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
      bytes = 0;
    }
  }

  void* ptr = nullptr;
  size_t bytes = 0;
};

struct TensorBinding {
  std::string name;
  nvinfer1::TensorIOMode mode = nvinfer1::TensorIOMode::kNONE;
  nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
  nvinfer1::Dims engine_dims{};
  nvinfer1::Dims runtime_dims{};
  DeviceBuffer device;
  std::vector<uint8_t> host;
  size_t bytes = 0;
};

bool dimMatches(int64_t dim, int64_t expected) { return dim <= 0 || dim == expected; }

bool dimsMatchMatrixInput(const nvinfer1::Dims& dims, int matrix_size) {
  if (dims.nbDims == 4) {
    return dimMatches(dims.d[0], 1) && dimMatches(dims.d[2], matrix_size) &&
           dimMatches(dims.d[3], matrix_size);
  }
  if (dims.nbDims == 3) {
    return dimMatches(dims.d[1], matrix_size) && dimMatches(dims.d[2], matrix_size);
  }
  return false;
}

int matrixInputScore(const TensorBinding& binding, int matrix_size, const std::string& name_hint) {
  if (!isFloatLike(binding.dtype) || !dimsMatchMatrixInput(binding.engine_dims, matrix_size)) {
    return std::numeric_limits<int>::min();
  }

  int score = 20;
  if (containsCaseInsensitive(binding.name, name_hint)) {
    score += 100;
  }
  return score;
}

bool isLikelyMatrixInput(const TensorBinding& binding) {
  return dimsMatchMatrixInput(binding.engine_dims, 3) || dimsMatchMatrixInput(binding.engine_dims, 4);
}

cv::Point3d cameraCenterFromWorldToCamera(const cv::Matx44f& extrinsic) {
  const double tx = extrinsic(0, 3);
  const double ty = extrinsic(1, 3);
  const double tz = extrinsic(2, 3);
  return cv::Point3d(-(extrinsic(0, 0) * tx + extrinsic(1, 0) * ty + extrinsic(2, 0) * tz),
                     -(extrinsic(0, 1) * tx + extrinsic(1, 1) * ty + extrinsic(2, 1) * tz),
                     -(extrinsic(0, 2) * tx + extrinsic(1, 2) * ty + extrinsic(2, 2) * tz));
}

double squaredDistance(const cv::Point3d& a, const cv::Point3d& b) {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

double estimateInputToPredictedPoseScale(const std::vector<cv::Matx44f>& predicted_world_to_camera,
                                         const std::vector<cv::Matx44f>& input_world_to_camera) {
  if (predicted_world_to_camera.size() != input_world_to_camera.size() || input_world_to_camera.size() < 2) {
    return 1.0;
  }

  std::vector<cv::Point3d> predicted_centers;
  std::vector<cv::Point3d> input_centers;
  predicted_centers.reserve(predicted_world_to_camera.size());
  input_centers.reserve(input_world_to_camera.size());
  for (size_t i = 0; i < input_world_to_camera.size(); ++i) {
    predicted_centers.push_back(cameraCenterFromWorldToCamera(predicted_world_to_camera[i]));
    input_centers.push_back(cameraCenterFromWorldToCamera(input_world_to_camera[i]));
  }

  double predicted_sum = 0.0;
  double input_sum = 0.0;
  int pairs = 0;
  for (size_t i = 0; i < input_centers.size(); ++i) {
    for (size_t j = i + 1; j < input_centers.size(); ++j) {
      const double predicted_d2 = squaredDistance(predicted_centers[i], predicted_centers[j]);
      const double input_d2 = squaredDistance(input_centers[i], input_centers[j]);
      if (std::isfinite(predicted_d2) && std::isfinite(input_d2) && predicted_d2 > 1e-12 && input_d2 > 1e-12) {
        predicted_sum += predicted_d2;
        input_sum += input_d2;
        ++pairs;
      }
    }
  }

  if (pairs == 0 || input_sum <= 1e-12) {
    return 1.0;
  }
  const double scale = std::sqrt(predicted_sum / input_sum);
  return std::isfinite(scale) && scale > 1e-12 ? scale : 1.0;
}

cv::Mat prepareConfidenceMap(const cv::Mat& raw_confidence, const cv::Size& original_size) {
  if (raw_confidence.empty()) {
    return {};
  }
  if (raw_confidence.channels() != 1) {
    throw std::runtime_error("Depth confidence output must be single-channel");
  }

  cv::Mat confidence;
  raw_confidence.convertTo(confidence, CV_32FC1);
  for (int y = 0; y < confidence.rows; ++y) {
    float* row = confidence.ptr<float>(y);
    for (int x = 0; x < confidence.cols; ++x) {
      if (!std::isfinite(row[x])) {
        row[x] = 0.0f;
      }
    }
  }

  if (original_size.width > 0 && original_size.height > 0 && confidence.size() != original_size) {
    cv::Mat resized;
    cv::resize(confidence, resized, original_size, 0.0, 0.0, cv::INTER_LINEAR);
    return resized;
  }
  return confidence;
}

enum class InputLayout {
  kVCHW,
  kBVCHW,
};

}  // namespace

class DepthAnythingV3TRT::Impl {
 public:
  explicit Impl(Params params) : params_(std::move(params)), logger_(params_.verbose) {
    validateParams();
    checkCuda(cudaSetDevice(params_.device_id), "Failed to set CUDA device");
    checkCuda(cudaStreamCreate(&stream_), "Failed to create CUDA stream");
    loadEngineData();

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (runtime_ == nullptr) {
      throw std::runtime_error("Failed to create TensorRT runtime");
    }

    engine_ = runtime_->deserializeCudaEngine(engine_data_.data(), engine_data_.size());
    if (engine_ == nullptr) {
      throw std::runtime_error("Failed to deserialize TensorRT engine: " + params_.engine_path);
    }

    context_ = engine_->createExecutionContext();
    if (context_ == nullptr) {
      throw std::runtime_error("Failed to create TensorRT execution context");
    }

    inspectBindings();
    configureInputShape(initialViewCount());
    refreshRuntimeBindings();

    if (params_.verbose) {
      std::cerr << "DepthAnythingV3TRT loaded " << params_.engine_path << std::endl;
      std::cerr << "  input tensor: " << input().name << " " << dimsToString(input().engine_dims) << std::endl;
      if (hasExtrinsicsInput()) {
        std::cerr << "  extrinsics tensor: " << extrinsicsInput().name << " "
                  << dimsToString(extrinsicsInput().engine_dims) << std::endl;
      }
      if (hasIntrinsicsInput()) {
        std::cerr << "  intrinsics tensor: " << intrinsicsInput().name << " "
                  << dimsToString(intrinsicsInput().engine_dims) << std::endl;
      }
      std::cerr << "  depth tensor: " << depth().name << " " << dimsToString(depth().engine_dims) << std::endl;
      if (hasConfidence()) {
        std::cerr << "  confidence tensor: " << confidence().name << " " << dimsToString(confidence().engine_dims)
                  << std::endl;
      }
      if (hasPoseOutput()) {
        std::cerr << "  output extrinsics tensor: " << poseOutput().name << " "
                  << dimsToString(poseOutput().engine_dims) << std::endl;
      }
      if (hasSky()) {
        std::cerr << "  sky tensor: " << sky().name << " " << dimsToString(sky().engine_dims) << std::endl;
      }
      std::cerr << "  model input: " << input_size_.width << "x" << input_size_.height << std::endl;
    }
  }

  ~Impl() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
    delete context_;
    delete engine_;
    delete runtime_;
  }

  const Params& params() const { return params_; }
  cv::Size inputSize() const { return input_size_; }
  bool hasCameraInputs() const { return hasExtrinsicsInput() || hasIntrinsicsInput(); }

  MonoDepthResult infer(const cv::Mat& image, const std::optional<CameraIntrinsics>& intrinsics) {
    std::vector<cv::Mat> views{image};
    std::vector<std::optional<CameraIntrinsics>> view_intrinsics;
    if (intrinsics.has_value()) {
      view_intrinsics.push_back(intrinsics);
    }
    auto results = inferViews(views, view_intrinsics, {});
    return std::move(results.front());
  }

  std::vector<MonoDepthResult> inferViews(const std::vector<cv::Mat>& views,
                                          const std::vector<std::optional<CameraIntrinsics>>& intrinsics,
                                          const std::vector<cv::Matx44f>& world_to_camera_extrinsics) {
    if (views.empty()) {
      throw std::invalid_argument("DepthAnythingV3TRT requires at least one input view");
    }
    if (!intrinsics.empty() && intrinsics.size() != views.size()) {
      throw std::invalid_argument("Intrinsics vector must be empty or match the number of views");
    }
    if (!world_to_camera_extrinsics.empty() && world_to_camera_extrinsics.size() != views.size()) {
      throw std::invalid_argument("Extrinsics vector must be empty or match the number of views");
    }
    if (hasIntrinsicsInput() && intrinsics.empty()) {
      throw std::invalid_argument("This DepthAnythingV3TRT engine requires per-view intrinsics");
    }
    if (hasExtrinsicsInput() && world_to_camera_extrinsics.empty()) {
      throw std::invalid_argument("This DepthAnythingV3TRT engine requires per-view world-to-camera extrinsics");
    }

    const int view_count = static_cast<int>(views.size());
    configureInputShape(view_count);
    refreshRuntimeBindings();
    copyInputToDevice(preprocess(views));
    if (hasIntrinsicsInput()) {
      copyIntrinsicsToDevice(makeIntrinsicsInput(views, intrinsics));
    }
    if (hasExtrinsicsInput()) {
      copyExtrinsicsToDevice(makeExtrinsicsInput(world_to_camera_extrinsics));
    }

    if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error("TensorRT enqueueV3 failed for Depth Anything V3");
    }

    copyOutputToHost(depth_index_);
    if (hasConfidence()) {
      copyOutputToHost(confidence_index_);
    }
    if (hasSky()) {
      copyOutputToHost(sky_index_);
    }
    if (hasPoseOutput()) {
      copyOutputToHost(pose_output_index_);
    }
    checkCuda(cudaStreamSynchronize(stream_), "Failed to synchronize CUDA stream after Depth Anything V3 inference");

    const auto raw_depths = extractOutputPlanes(depth(), view_count, "depth");
    const auto raw_confidences =
        hasConfidence() ? extractOutputPlanes(confidence(), view_count, "depth confidence") : std::vector<cv::Mat>{};
    const auto sky_preds = hasSky() ? extractOutputPlanes(sky(), view_count, "sky") : std::vector<cv::Mat>{};
    const auto predicted_extrinsics =
        hasPoseOutput() ? extractPoseOutputExtrinsics(poseOutput(), view_count) : std::vector<cv::Matx44f>{};
    const double pose_scale =
        (!predicted_extrinsics.empty() && !world_to_camera_extrinsics.empty())
            ? estimateInputToPredictedPoseScale(predicted_extrinsics, world_to_camera_extrinsics)
            : 1.0;
    const bool pose_scaled = std::isfinite(pose_scale) && pose_scale > 1e-12 && std::abs(pose_scale - 1.0) > 1e-6;

    std::vector<MonoDepthResult> results;
    results.reserve(views.size());
    for (size_t i = 0; i < views.size(); ++i) {
      mono_depth_detail::DepthAnythingPostprocessOptions options;
      options.original_size = views[i].size();
      options.sky_threshold = params_.sky_threshold;
      options.sky_depth_cap = params_.sky_depth_cap;
      options.model_focal_pixels = params_.model_focal_pixels;
      if (!intrinsics.empty()) {
        options.intrinsics = intrinsics[i];
      }

      const cv::Mat sky_pred = sky_preds.empty() ? cv::Mat{} : sky_preds[i];
      auto post = mono_depth_detail::postprocessDepthAnything(raw_depths[i], sky_pred, options);
      if (pose_scaled) {
        post.model_depth /= static_cast<float>(pose_scale);
        post.depth /= static_cast<float>(pose_scale);
      }

      MonoDepthResult result;
      result.depth = std::move(post.depth);
      result.raw_depth = std::move(post.raw_depth);
      if (!raw_confidences.empty()) {
        raw_confidences[i].convertTo(result.raw_confidence, CV_32FC1);
        result.confidence = prepareConfidenceMap(raw_confidences[i], views[i].size());
      }
      result.sky_mask = std::move(post.sky_mask);
      result.metadata.original_size = views[i].size();
      result.metadata.model_size = raw_depths[i].size();
      result.metadata.scale_x = static_cast<double>(raw_depths[i].cols) / static_cast<double>(views[i].cols);
      result.metadata.scale_y = static_cast<double>(raw_depths[i].rows) / static_cast<double>(views[i].rows);
      result.metadata.focal_scale = post.focal_scale;
      result.metadata.focal_scaled = post.focal_scaled;
      result.metadata.pose_scale = pose_scale;
      result.metadata.pose_scaled = pose_scaled;
      result.metadata.sky_filled = post.sky_filled;
      result.metadata.sky_fill_value = post.sky_fill_value;
      result.metadata.view_index = static_cast<int>(i);
      result.metadata.view_count = view_count;
      result.metadata.depth_tensor_name = depth().name;
      result.metadata.confidence_tensor_name = hasConfidence() ? confidence().name : std::string{};
      result.metadata.sky_tensor_name = hasSky() ? sky().name : std::string{};
      results.push_back(std::move(result));
    }
    return results;
  }

 private:
  void validateParams() const {
    if (params_.engine_path.empty()) {
      throw std::invalid_argument("DepthAnythingV3TRT engine_path cannot be empty");
    }
    if (params_.mean.size() != 3 || params_.std.size() != 3) {
      throw std::invalid_argument("DepthAnythingV3TRT mean and std must have three RGB values");
    }
    if (params_.fallback_input_size.width <= 0 || params_.fallback_input_size.height <= 0) {
      throw std::invalid_argument("DepthAnythingV3TRT fallback_input_size must be positive");
    }
    if (params_.sky_depth_cap <= 0.0f) {
      throw std::invalid_argument("DepthAnythingV3TRT sky_depth_cap must be positive");
    }
  }

  void loadEngineData() {
    std::ifstream file(params_.engine_path, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Failed to open TensorRT engine: " + params_.engine_path);
    }

    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size <= 0) {
      throw std::runtime_error("TensorRT engine is empty: " + params_.engine_path);
    }
    file.seekg(0, std::ios::beg);

    engine_data_.resize(static_cast<size_t>(size));
    file.read(engine_data_.data(), size);
    if (!file) {
      throw std::runtime_error("Failed to read TensorRT engine: " + params_.engine_path);
    }
  }

  void inspectBindings() {
    const int tensor_count = engine_->getNbIOTensors();
    bindings_.reserve(static_cast<size_t>(tensor_count));

    int first_output = -1;
    int first_non_sky_output = -1;
    int best_input_score = std::numeric_limits<int>::min();
    int best_extrinsics_score = std::numeric_limits<int>::min();
    int best_intrinsics_score = std::numeric_limits<int>::min();
    int best_depth_score = std::numeric_limits<int>::min();
    int best_confidence_score = std::numeric_limits<int>::min();
    int best_pose_output_score = std::numeric_limits<int>::min();
    for (int i = 0; i < tensor_count; ++i) {
      const char* name = engine_->getIOTensorName(i);
      if (name == nullptr) {
        continue;
      }

      TensorBinding binding;
      binding.name = name;
      binding.mode = engine_->getTensorIOMode(name);
      binding.dtype = engine_->getTensorDataType(name);
      binding.engine_dims = engine_->getTensorShape(name);
      bindings_.push_back(std::move(binding));
      const int index = static_cast<int>(bindings_.size() - 1);

      if (bindings_[index].mode == nvinfer1::TensorIOMode::kINPUT) {
        const int score = imageInputScore(bindings_[index]);
        if (score > best_input_score) {
          best_input_score = score;
          input_index_ = index;
        }
        const int extrinsics_score = matrixInputScore(bindings_[index], 4, "extr");
        if (extrinsics_score > best_extrinsics_score) {
          best_extrinsics_score = extrinsics_score;
          extrinsics_input_index_ = index;
        }
        const int intrinsics_score = matrixInputScore(bindings_[index], 3, "intr");
        if (intrinsics_score > best_intrinsics_score) {
          best_intrinsics_score = intrinsics_score;
          intrinsics_input_index_ = index;
        }
      } else if (bindings_[index].mode == nvinfer1::TensorIOMode::kOUTPUT) {
        if (first_output < 0) {
          first_output = index;
        }
        const bool name_has_sky = containsCaseInsensitive(bindings_[index].name, "sky");
        if (!name_has_sky && first_non_sky_output < 0) {
          first_non_sky_output = index;
        }
        const int score = depthOutputScore(bindings_[index].name);
        if (score > best_depth_score) {
          best_depth_score = score;
          depth_index_ = index;
        } else if (name_has_sky) {
          sky_index_ = index;
        }
        const int confidence_score = confidenceOutputScore(bindings_[index].name);
        if (confidence_score > best_confidence_score) {
          best_confidence_score = confidence_score;
          confidence_index_ = index;
        }
        const int pose_output_score = poseOutputScore(bindings_[index]);
        if (pose_output_score > best_pose_output_score) {
          best_pose_output_score = pose_output_score;
          pose_output_index_ = index;
        }
      }
    }

    if (input_index_ < 0) {
      throw std::runtime_error("TensorRT engine has no image input tensor");
    }
    if (extrinsics_input_index_ == input_index_) {
      extrinsics_input_index_ = -1;
    }
    if (intrinsics_input_index_ == input_index_) {
      intrinsics_input_index_ = -1;
    }
    for (int i = 0; i < static_cast<int>(bindings_.size()); ++i) {
      if (bindings_[i].mode != nvinfer1::TensorIOMode::kINPUT) {
        continue;
      }
      if (i == input_index_ || i == extrinsics_input_index_ || i == intrinsics_input_index_) {
        continue;
      }
      throw std::runtime_error("Unsupported DepthAnythingV3TRT input tensor " + bindings_[i].name + " " +
                               dimsToString(bindings_[i].engine_dims));
    }
    if (depth_index_ < 0) {
      depth_index_ = first_non_sky_output >= 0 ? first_non_sky_output : first_output;
    }
    if (depth_index_ < 0 || depth_index_ == sky_index_) {
      throw std::runtime_error("TensorRT engine has no output tensor for depth");
    }
    if (confidence_index_ == depth_index_ || confidence_index_ == sky_index_) {
      confidence_index_ = -1;
    }
    if (!isFloatLike(input().dtype)) {
      throw std::runtime_error("Depth Anything V3 input tensor must be float or half");
    }
    if (hasExtrinsicsInput() && !isFloatLike(extrinsicsInput().dtype)) {
      throw std::runtime_error("Depth Anything V3 extrinsics input tensor must be float or half");
    }
    if (hasIntrinsicsInput() && !isFloatLike(intrinsicsInput().dtype)) {
      throw std::runtime_error("Depth Anything V3 intrinsics input tensor must be float or half");
    }
    if (!isFloatLike(depth().dtype)) {
      throw std::runtime_error("Depth output tensor must be float or half");
    }
    if (hasConfidence() && !isFloatLike(confidence().dtype)) {
      throw std::runtime_error("Depth confidence output tensor must be float or half");
    }
    if (hasSky() && !isFloatLike(sky().dtype)) {
      throw std::runtime_error("Sky output tensor must be float or half");
    }

    inferInputLayout();
  }

  int imageInputScore(const TensorBinding& binding) const {
    if (!isFloatLike(binding.dtype)) {
      return -100;
    }
    if (isLikelyMatrixInput(binding)) {
      return std::numeric_limits<int>::min();
    }
    if (binding.engine_dims.nbDims != 4 && binding.engine_dims.nbDims != 5) {
      return 0;
    }

    const int name_score = containsCaseInsensitive(binding.name, "image") ? 100 : 0;
    if (binding.engine_dims.nbDims == 4 && (binding.engine_dims.d[1] == 3 || binding.engine_dims.d[1] <= 0) &&
        (binding.engine_dims.d[2] > 8 || binding.engine_dims.d[2] <= 0) &&
        (binding.engine_dims.d[3] > 8 || binding.engine_dims.d[3] <= 0)) {
      return 10 + name_score;
    }
    if (binding.engine_dims.nbDims == 5 && (binding.engine_dims.d[2] == 3 || binding.engine_dims.d[2] <= 0) &&
        (binding.engine_dims.d[3] > 8 || binding.engine_dims.d[3] <= 0) &&
        (binding.engine_dims.d[4] > 8 || binding.engine_dims.d[4] <= 0)) {
      return 10 + name_score;
    }
    return 1;
  }

  int poseOutputScore(const TensorBinding& binding) const {
    if (!isFloatLike(binding.dtype)) {
      return std::numeric_limits<int>::min();
    }
    const int name_score = containsCaseInsensitive(binding.name, "extr") ? 100 : 0;
    const auto dims = binding.engine_dims;
    if (dims.nbDims == 4 && dimMatches(dims.d[0], 1) && dimMatches(dims.d[2], 3) && dimMatches(dims.d[3], 4)) {
      return 20 + name_score;
    }
    if (dims.nbDims == 3 && dimMatches(dims.d[1], 3) && dimMatches(dims.d[2], 4)) {
      return 20 + name_score;
    }
    return std::numeric_limits<int>::min();
  }

  void inferInputLayout() {
    const auto resolved = resolveDynamicInputDims(input().engine_dims, 1);
    if (resolved.nbDims == 4 && resolved.d[1] == 3) {
      input_layout_ = InputLayout::kVCHW;
      input_size_ = cv::Size(resolved.d[3], resolved.d[2]);
      return;
    }
    if (resolved.nbDims == 5 && resolved.d[0] == 1 && resolved.d[2] == 3) {
      input_layout_ = InputLayout::kBVCHW;
      input_size_ = cv::Size(resolved.d[4], resolved.d[3]);
      return;
    }

    throw std::runtime_error("Unsupported Depth Anything V3 input shape " + dimsToString(input().engine_dims) +
                             ". Expected [V,3,H,W] or [1,V,3,H,W]");
  }

  nvinfer1::Dims profileDims(nvinfer1::OptProfileSelector selector) const {
    auto dims = engine_->getProfileShape(input().name.c_str(), 0, selector);
    if (dims.nbDims != input().engine_dims.nbDims) {
      dims.nbDims = -1;
    }
    return dims;
  }

  nvinfer1::Dims resolveDynamicInputDims(nvinfer1::Dims dims, int view_count) const {
    const auto opt_dims = profileDims(nvinfer1::OptProfileSelector::kOPT);
    for (int i = 0; i < dims.nbDims; ++i) {
      if (dims.d[i] <= 0) {
        dims.d[i] = (opt_dims.nbDims == dims.nbDims && opt_dims.d[i] > 0) ? opt_dims.d[i] : fallbackDim(dims.nbDims, i);
      }
    }

    if (dims.nbDims == 4) {
      dims.d[0] = view_count;
      dims.d[1] = 3;
      dims.d[2] = input_size_.height > 0 ? input_size_.height : dims.d[2];
      dims.d[3] = input_size_.width > 0 ? input_size_.width : dims.d[3];
    } else if (dims.nbDims == 5) {
      dims.d[0] = 1;
      dims.d[1] = view_count;
      dims.d[2] = 3;
      dims.d[3] = input_size_.height > 0 ? input_size_.height : dims.d[3];
      dims.d[4] = input_size_.width > 0 ? input_size_.width : dims.d[4];
    }
    return dims;
  }

  int fallbackDim(int nb_dims, int index) const {
    if (nb_dims == 4) {
      if (index == 0) return 1;
      if (index == 1) return 3;
      if (index == 2) return params_.fallback_input_size.height;
      if (index == 3) return params_.fallback_input_size.width;
    }
    if (nb_dims == 5) {
      if (index == 0) return 1;
      if (index == 1) return 1;
      if (index == 2) return 3;
      if (index == 3) return params_.fallback_input_size.height;
      if (index == 4) return params_.fallback_input_size.width;
    }
    return 1;
  }

  int viewDim() const { return input_layout_ == InputLayout::kVCHW ? 0 : 1; }

  int initialViewCount() const {
    const int axis = viewDim();
    if (input().engine_dims.d[axis] > 0) {
      return input().engine_dims.d[axis];
    }
    const auto opt_dims = profileDims(nvinfer1::OptProfileSelector::kOPT);
    if (opt_dims.nbDims == input().engine_dims.nbDims && opt_dims.d[axis] > 0) {
      return opt_dims.d[axis];
    }
    return 1;
  }

  void validateViewCount(int view_count) const {
    const int axis = viewDim();
    const int static_views = input().engine_dims.d[axis];
    if (static_views > 0 && static_views != view_count) {
      throw std::runtime_error("TensorRT engine input " + input().name + " expects " + std::to_string(static_views) +
                               " view(s), but " + std::to_string(view_count) + " were requested");
    }

    const auto min_dims = profileDims(nvinfer1::OptProfileSelector::kMIN);
    const auto max_dims = profileDims(nvinfer1::OptProfileSelector::kMAX);
    if (min_dims.nbDims == input().engine_dims.nbDims && max_dims.nbDims == input().engine_dims.nbDims &&
        min_dims.d[axis] > 0 && max_dims.d[axis] > 0 &&
        (view_count < min_dims.d[axis] || view_count > max_dims.d[axis])) {
      throw std::runtime_error("TensorRT engine input " + input().name + " supports view counts in [" +
                               std::to_string(min_dims.d[axis]) + ", " + std::to_string(max_dims.d[axis]) + "], but " +
                               std::to_string(view_count) + " were requested");
    }
  }

  nvinfer1::Dims resolveDynamicMatrixInputDims(nvinfer1::Dims dims, int view_count, int matrix_size) const {
    if (dims.nbDims == 4) {
      dims.d[0] = 1;
      dims.d[1] = view_count;
      dims.d[2] = matrix_size;
      dims.d[3] = matrix_size;
      return dims;
    }
    if (dims.nbDims == 3) {
      dims.d[0] = view_count;
      dims.d[1] = matrix_size;
      dims.d[2] = matrix_size;
      return dims;
    }
    throw std::runtime_error("Unsupported camera input shape " + dimsToString(dims));
  }

  void validateMatrixInputViewCount(const TensorBinding& binding, int view_count, int matrix_size) const {
    if (!dimsMatchMatrixInput(binding.engine_dims, matrix_size)) {
      throw std::runtime_error("Unsupported camera input shape for " + binding.name + ": " +
                               dimsToString(binding.engine_dims));
    }

    const int axis = binding.engine_dims.nbDims == 4 ? 1 : 0;
    const int static_views = binding.engine_dims.d[axis];
    if (static_views > 0 && static_views != view_count) {
      throw std::runtime_error("TensorRT engine input " + binding.name + " expects " + std::to_string(static_views) +
                               " view(s), but " + std::to_string(view_count) + " were requested");
    }
  }

  void configureMatrixInputShape(const TensorBinding& binding, int view_count, int matrix_size) {
    validateMatrixInputViewCount(binding, view_count, matrix_size);
    auto shape = resolveDynamicMatrixInputDims(binding.engine_dims, view_count, matrix_size);
    if (!context_->setInputShape(binding.name.c_str(), shape)) {
      throw std::runtime_error("Failed to set TensorRT input shape for " + binding.name + " to " +
                               dimsToString(shape));
    }
  }

  void configureInputShape(int view_count) {
    validateViewCount(view_count);
    auto shape = resolveDynamicInputDims(input().engine_dims, view_count);
    if (!context_->setInputShape(input().name.c_str(), shape)) {
      throw std::runtime_error("Failed to set TensorRT input shape for " + input().name + " to " + dimsToString(shape));
    }
    if (hasExtrinsicsInput()) {
      configureMatrixInputShape(extrinsicsInput(), view_count, 4);
    }
    if (hasIntrinsicsInput()) {
      configureMatrixInputShape(intrinsicsInput(), view_count, 3);
    }
  }

  void refreshRuntimeBindings() {
    for (auto& binding : bindings_) {
      binding.runtime_dims = context_->getTensorShape(binding.name.c_str());
      const size_t bytes = volume(binding.runtime_dims) * elementSize(binding.dtype);
      binding.device.resize(bytes);
      binding.bytes = bytes;
      if (!context_->setTensorAddress(binding.name.c_str(), binding.device.ptr)) {
        throw std::runtime_error("Failed to set TensorRT tensor address for " + binding.name);
      }
    }

    const auto in = input().runtime_dims;
    if (input_layout_ == InputLayout::kVCHW) {
      input_size_ = cv::Size(in.d[3], in.d[2]);
    } else {
      input_size_ = cv::Size(in.d[4], in.d[3]);
    }
  }

  std::vector<float> preprocess(const std::vector<cv::Mat>& views) const {
    const int h = input_size_.height;
    const int w = input_size_.width;
    const size_t plane = static_cast<size_t>(h) * static_cast<size_t>(w);
    std::vector<float> nchw(views.size() * 3 * plane);

    for (size_t view = 0; view < views.size(); ++view) {
      const cv::Mat& image = views[view];
      if (image.empty()) {
        throw std::invalid_argument("Input image " + std::to_string(view) + " is empty");
      }
      if (image.type() != CV_8UC3) {
        throw std::invalid_argument("DepthAnythingV3TRT expects BGR CV_8UC3 input images");
      }

      cv::Mat resized;
      cv::resize(image, resized, input_size_, 0.0, 0.0, cv::INTER_CUBIC);
      cv::Mat rgb;
      cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
      cv::Mat rgb_float;
      rgb.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);

      const size_t view_offset = view * 3 * plane;
      for (int y = 0; y < h; ++y) {
        const auto* row = rgb_float.ptr<cv::Vec3f>(y);
        for (int x = 0; x < w; ++x) {
          for (int c = 0; c < 3; ++c) {
            const float value = (row[x][c] - params_.mean[c]) / params_.std[c];
            nchw[view_offset + static_cast<size_t>(c) * plane + static_cast<size_t>(y) * w + x] = value;
          }
        }
      }
    }
    return nchw;
  }

  std::vector<float> makeIntrinsicsInput(const std::vector<cv::Mat>& views,
                                         const std::vector<std::optional<CameraIntrinsics>>& intrinsics) const {
    if (intrinsics.size() != views.size()) {
      throw std::invalid_argument("Camera-conditioned DA3 inference requires intrinsics for every view");
    }

    std::vector<float> values(views.size() * 3 * 3, 0.0f);
    for (size_t view = 0; view < views.size(); ++view) {
      if (!intrinsics[view].has_value()) {
        throw std::invalid_argument("Camera-conditioned DA3 inference requires intrinsics for every view");
      }
      if (views[view].empty()) {
        throw std::invalid_argument("Cannot scale intrinsics for an empty image");
      }
      const auto& item = *intrinsics[view];
      if (item.fx <= 0.0 || item.fy <= 0.0) {
        throw std::invalid_argument("Camera-conditioned DA3 inference requires positive fx and fy");
      }

      const double scale_x = static_cast<double>(input_size_.width) / static_cast<double>(views[view].cols);
      const double scale_y = static_cast<double>(input_size_.height) / static_cast<double>(views[view].rows);
      const size_t offset = view * 9;
      values[offset + 0] = static_cast<float>(item.fx * scale_x);
      values[offset + 1] = 0.0f;
      values[offset + 2] = static_cast<float>(item.cx * scale_x);
      values[offset + 3] = 0.0f;
      values[offset + 4] = static_cast<float>(item.fy * scale_y);
      values[offset + 5] = static_cast<float>(item.cy * scale_y);
      values[offset + 6] = 0.0f;
      values[offset + 7] = 0.0f;
      values[offset + 8] = 1.0f;
    }
    return values;
  }

  std::vector<float> makeExtrinsicsInput(const std::vector<cv::Matx44f>& world_to_camera_extrinsics) const {
    std::vector<float> values(world_to_camera_extrinsics.size() * 4 * 4, 0.0f);
    for (size_t view = 0; view < world_to_camera_extrinsics.size(); ++view) {
      const auto& matrix = world_to_camera_extrinsics[view];
      const size_t offset = view * 16;
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          values[offset + static_cast<size_t>(r * 4 + c)] = matrix(r, c);
        }
      }
    }
    return values;
  }

  void copyFloatVectorToDevice(TensorBinding& binding, const std::vector<float>& values, const std::string& label) {
    if (binding.dtype == nvinfer1::DataType::kFLOAT) {
      const size_t bytes = values.size() * sizeof(float);
      if (bytes > binding.bytes) {
        throw std::runtime_error(label + " is larger than the TensorRT input buffer");
      }
      checkCuda(cudaMemcpyAsync(binding.device.ptr, values.data(), bytes, cudaMemcpyHostToDevice, stream_),
                "Failed to copy " + label + " to CUDA");
      return;
    }

    std::vector<__half> input_half(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      input_half[i] = __float2half(values[i]);
    }
    const size_t bytes = input_half.size() * sizeof(__half);
    if (bytes > binding.bytes) {
      throw std::runtime_error(label + " is larger than the TensorRT input buffer");
    }
    checkCuda(cudaMemcpyAsync(binding.device.ptr, input_half.data(), bytes, cudaMemcpyHostToDevice, stream_),
              "Failed to copy " + label + " half data to CUDA");
  }

  void copyInputToDevice(const std::vector<float>& input_float) {
    copyFloatVectorToDevice(input(), input_float, "Depth Anything V3 image input");
  }

  void copyIntrinsicsToDevice(const std::vector<float>& intrinsics_float) {
    copyFloatVectorToDevice(intrinsicsInput(), intrinsics_float, "Depth Anything V3 intrinsics input");
  }

  void copyExtrinsicsToDevice(const std::vector<float>& extrinsics_float) {
    copyFloatVectorToDevice(extrinsicsInput(), extrinsics_float, "Depth Anything V3 extrinsics input");
  }

  void copyOutputToHost(int binding_index) {
    auto& binding = bindings_.at(static_cast<size_t>(binding_index));
    binding.host.resize(binding.bytes);
    checkCuda(cudaMemcpyAsync(binding.host.data(), binding.device.ptr, binding.bytes, cudaMemcpyDeviceToHost, stream_),
              "Failed to copy TensorRT output " + binding.name + " to host");
  }

  std::vector<float> hostAsFloat(const TensorBinding& binding) const {
    const size_t count = volume(binding.runtime_dims);
    std::vector<float> output(count);

    if (binding.dtype == nvinfer1::DataType::kFLOAT) {
      std::memcpy(output.data(), binding.host.data(), count * sizeof(float));
      return output;
    }

    const auto* half_data = reinterpret_cast<const __half*>(binding.host.data());
    for (size_t i = 0; i < count; ++i) {
      output[i] = __half2float(half_data[i]);
    }
    return output;
  }

  std::vector<cv::Mat> extractOutputPlanes(const TensorBinding& binding,
                                           int requested_views,
                                           const std::string& label) const {
    const auto values = hostAsFloat(binding);
    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(binding.runtime_dims.nbDims));
    for (int i = 0; i < binding.runtime_dims.nbDims; ++i) {
      dims.push_back(binding.runtime_dims.d[i]);
    }
    return mono_depth_detail::extractDepthAnythingTensorPlanes(values, dims, requested_views, label);
  }

  std::vector<cv::Matx44f> extractPoseOutputExtrinsics(const TensorBinding& binding, int requested_views) const {
    const auto values = hostAsFloat(binding);
    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(binding.runtime_dims.nbDims));
    for (int i = 0; i < binding.runtime_dims.nbDims; ++i) {
      dims.push_back(binding.runtime_dims.d[i]);
    }
    if (mono_depth_detail::checkedTensorVolume(dims, "extrinsics") != values.size()) {
      throw std::runtime_error("TensorRT extrinsics output shape is inconsistent with its buffer size");
    }

    int64_t rows = 0;
    int64_t cols = 0;
    size_t first_view_offset = 0;
    size_t view_stride = 0;
    if (dims.size() == 4 && dims[0] == 1 && dims[1] == requested_views) {
      rows = dims[2];
      cols = dims[3];
      view_stride = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    } else if (dims.size() == 3 && dims[0] == requested_views) {
      rows = dims[1];
      cols = dims[2];
      view_stride = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    } else {
      throw std::runtime_error("Unsupported extrinsics output shape " + mono_depth_detail::tensorShapeToString(dims));
    }
    if (!((rows == 3 || rows == 4) && cols == 4)) {
      throw std::runtime_error("Unsupported extrinsics output shape " + mono_depth_detail::tensorShapeToString(dims));
    }

    std::vector<cv::Matx44f> extrinsics;
    extrinsics.reserve(static_cast<size_t>(requested_views));
    for (int view = 0; view < requested_views; ++view) {
      cv::Matx44f matrix = cv::Matx44f::eye();
      const size_t offset = first_view_offset + static_cast<size_t>(view) * view_stride;
      for (int r = 0; r < static_cast<int>(rows); ++r) {
        for (int c = 0; c < 4; ++c) {
          matrix(r, c) = values[offset + static_cast<size_t>(r * 4 + c)];
        }
      }
      extrinsics.push_back(matrix);
    }
    return extrinsics;
  }

  TensorBinding& input() { return bindings_.at(static_cast<size_t>(input_index_)); }
  const TensorBinding& input() const { return bindings_.at(static_cast<size_t>(input_index_)); }
  TensorBinding& extrinsicsInput() { return bindings_.at(static_cast<size_t>(extrinsics_input_index_)); }
  const TensorBinding& extrinsicsInput() const { return bindings_.at(static_cast<size_t>(extrinsics_input_index_)); }
  TensorBinding& intrinsicsInput() { return bindings_.at(static_cast<size_t>(intrinsics_input_index_)); }
  const TensorBinding& intrinsicsInput() const { return bindings_.at(static_cast<size_t>(intrinsics_input_index_)); }
  TensorBinding& depth() { return bindings_.at(static_cast<size_t>(depth_index_)); }
  const TensorBinding& depth() const { return bindings_.at(static_cast<size_t>(depth_index_)); }
  TensorBinding& confidence() { return bindings_.at(static_cast<size_t>(confidence_index_)); }
  const TensorBinding& confidence() const { return bindings_.at(static_cast<size_t>(confidence_index_)); }
  TensorBinding& poseOutput() { return bindings_.at(static_cast<size_t>(pose_output_index_)); }
  const TensorBinding& poseOutput() const { return bindings_.at(static_cast<size_t>(pose_output_index_)); }
  TensorBinding& sky() { return bindings_.at(static_cast<size_t>(sky_index_)); }
  const TensorBinding& sky() const { return bindings_.at(static_cast<size_t>(sky_index_)); }
  bool hasExtrinsicsInput() const { return extrinsics_input_index_ >= 0; }
  bool hasIntrinsicsInput() const { return intrinsics_input_index_ >= 0; }
  bool hasConfidence() const { return confidence_index_ >= 0; }
  bool hasPoseOutput() const { return pose_output_index_ >= 0; }
  bool hasSky() const { return sky_index_ >= 0; }

  Params params_;
  TrtLogger logger_;
  std::vector<char> engine_data_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  cudaStream_t stream_ = nullptr;
  std::vector<TensorBinding> bindings_;
  int input_index_ = -1;
  int extrinsics_input_index_ = -1;
  int intrinsics_input_index_ = -1;
  int depth_index_ = -1;
  int confidence_index_ = -1;
  int pose_output_index_ = -1;
  int sky_index_ = -1;
  InputLayout input_layout_ = InputLayout::kVCHW;
  cv::Size input_size_;
};

DepthAnythingV3TRT::DepthAnythingV3TRT(const Params& params) : impl_(std::make_unique<Impl>(params)) {}

DepthAnythingV3TRT::~DepthAnythingV3TRT() = default;

MonoDepthResult DepthAnythingV3TRT::infer(const cv::Mat& image, const std::optional<CameraIntrinsics>& intrinsics) {
  return impl_->infer(image, intrinsics);
}

std::vector<MonoDepthResult> DepthAnythingV3TRT::infer_multi_view(const std::vector<cv::Mat>& views,
                                                                  const std::vector<CameraIntrinsics>& intrinsics) {
  std::vector<std::optional<CameraIntrinsics>> optional_intrinsics;
  if (!intrinsics.empty()) {
    optional_intrinsics.reserve(intrinsics.size());
    for (const auto& item : intrinsics) {
      optional_intrinsics.emplace_back(item);
    }
  }
  return impl_->inferViews(views, optional_intrinsics, {});
}

std::vector<MonoDepthResult> DepthAnythingV3TRT::infer_multi_view(
    const std::vector<cv::Mat>& views,
    const std::vector<CameraIntrinsics>& intrinsics,
    const std::vector<cv::Matx44f>& world_to_camera_extrinsics) {
  std::vector<std::optional<CameraIntrinsics>> optional_intrinsics;
  if (!intrinsics.empty()) {
    optional_intrinsics.reserve(intrinsics.size());
    for (const auto& item : intrinsics) {
      optional_intrinsics.emplace_back(item);
    }
  }
  return impl_->inferViews(views, optional_intrinsics, world_to_camera_extrinsics);
}

cv::Size DepthAnythingV3TRT::input_size() const { return impl_->inputSize(); }

bool DepthAnythingV3TRT::has_camera_inputs() const { return impl_->hasCameraInputs(); }

const DepthAnythingV3TRT::Params& DepthAnythingV3TRT::params() const { return impl_->params(); }

}  // namespace xfeat

#endif  // HAVE_TENSORRT
