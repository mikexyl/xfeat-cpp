#include "xfeat-cpp/mono_depth/depth_anything_v3_trt.h"

#ifdef HAVE_TENSORRT

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
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
      std::cerr << "  depth tensor: " << depth().name << " " << dimsToString(depth().engine_dims) << std::endl;
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

  MonoDepthResult infer(const cv::Mat& image, const std::optional<CameraIntrinsics>& intrinsics) {
    std::vector<cv::Mat> views{image};
    std::vector<std::optional<CameraIntrinsics>> view_intrinsics;
    if (intrinsics.has_value()) {
      view_intrinsics.push_back(intrinsics);
    }
    auto results = inferViews(views, view_intrinsics);
    return std::move(results.front());
  }

  std::vector<MonoDepthResult> inferViews(const std::vector<cv::Mat>& views,
                                          const std::vector<std::optional<CameraIntrinsics>>& intrinsics) {
    if (views.empty()) {
      throw std::invalid_argument("DepthAnythingV3TRT requires at least one input view");
    }
    if (!intrinsics.empty() && intrinsics.size() != views.size()) {
      throw std::invalid_argument("Intrinsics vector must be empty or match the number of views");
    }

    const int view_count = static_cast<int>(views.size());
    configureInputShape(view_count);
    refreshRuntimeBindings();
    copyInputToDevice(preprocess(views));

    if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error("TensorRT enqueueV3 failed for Depth Anything V3");
    }

    copyOutputToHost(depth_index_);
    if (hasSky()) {
      copyOutputToHost(sky_index_);
    }
    checkCuda(cudaStreamSynchronize(stream_), "Failed to synchronize CUDA stream after Depth Anything V3 inference");

    const auto raw_depths = extractOutputPlanes(depth(), view_count, "depth");
    const auto sky_preds = hasSky() ? extractOutputPlanes(sky(), view_count, "sky") : std::vector<cv::Mat>{};

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

      MonoDepthResult result;
      result.depth = std::move(post.depth);
      result.raw_depth = std::move(post.raw_depth);
      result.sky_mask = std::move(post.sky_mask);
      result.metadata.original_size = views[i].size();
      result.metadata.model_size = raw_depths[i].size();
      result.metadata.scale_x = static_cast<double>(raw_depths[i].cols) / static_cast<double>(views[i].cols);
      result.metadata.scale_y = static_cast<double>(raw_depths[i].rows) / static_cast<double>(views[i].rows);
      result.metadata.focal_scale = post.focal_scale;
      result.metadata.focal_scaled = post.focal_scaled;
      result.metadata.sky_filled = post.sky_filled;
      result.metadata.sky_fill_value = post.sky_fill_value;
      result.metadata.view_index = static_cast<int>(i);
      result.metadata.view_count = view_count;
      result.metadata.depth_tensor_name = depth().name;
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
    int input_count = 0;
    int best_input_score = std::numeric_limits<int>::min();
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
        ++input_count;
        const int score = imageInputScore(bindings_[index]);
        if (score > best_input_score) {
          best_input_score = score;
          input_index_ = index;
        }
      } else if (bindings_[index].mode == nvinfer1::TensorIOMode::kOUTPUT) {
        if (first_output < 0) {
          first_output = index;
        }
        const bool name_has_sky = containsCaseInsensitive(bindings_[index].name, "sky");
        if (!name_has_sky && first_non_sky_output < 0) {
          first_non_sky_output = index;
        }
        if (containsCaseInsensitive(bindings_[index].name, "depth")) {
          depth_index_ = index;
        } else if (name_has_sky) {
          sky_index_ = index;
        }
      }
    }

    if (input_index_ < 0) {
      throw std::runtime_error("TensorRT engine has no image input tensor");
    }
    if (input_count != 1) {
      throw std::runtime_error(
          "DepthAnythingV3TRT v1 supports exactly one image input tensor; pose or auxiliary inputs are not supported");
    }
    if (depth_index_ < 0) {
      depth_index_ = first_non_sky_output >= 0 ? first_non_sky_output : first_output;
    }
    if (depth_index_ < 0 || depth_index_ == sky_index_) {
      throw std::runtime_error("TensorRT engine has no output tensor for depth");
    }
    if (!isFloatLike(input().dtype)) {
      throw std::runtime_error("Depth Anything V3 input tensor must be float or half");
    }
    if (!isFloatLike(depth().dtype)) {
      throw std::runtime_error("Depth output tensor must be float or half");
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
    if (binding.engine_dims.nbDims != 4 && binding.engine_dims.nbDims != 5) {
      return 0;
    }

    if (binding.engine_dims.nbDims == 4 && (binding.engine_dims.d[1] == 3 || binding.engine_dims.d[1] <= 0)) {
      return 10;
    }
    if (binding.engine_dims.nbDims == 5 && (binding.engine_dims.d[2] == 3 || binding.engine_dims.d[2] <= 0)) {
      return 10;
    }
    return 1;
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

  void configureInputShape(int view_count) {
    validateViewCount(view_count);
    auto shape = resolveDynamicInputDims(input().engine_dims, view_count);
    if (!context_->setInputShape(input().name.c_str(), shape)) {
      throw std::runtime_error("Failed to set TensorRT input shape for " + input().name + " to " + dimsToString(shape));
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

  void copyInputToDevice(const std::vector<float>& input_float) {
    auto& input_binding = input();
    if (input_binding.dtype == nvinfer1::DataType::kFLOAT) {
      const size_t bytes = input_float.size() * sizeof(float);
      if (bytes > input_binding.bytes) {
        throw std::runtime_error("Preprocessed input is larger than the TensorRT input buffer");
      }
      checkCuda(cudaMemcpyAsync(input_binding.device.ptr, input_float.data(), bytes, cudaMemcpyHostToDevice, stream_),
                "Failed to copy Depth Anything V3 input to CUDA");
      return;
    }

    std::vector<__half> input_half(input_float.size());
    for (size_t i = 0; i < input_float.size(); ++i) {
      input_half[i] = __float2half(input_float[i]);
    }
    const size_t bytes = input_half.size() * sizeof(__half);
    if (bytes > input_binding.bytes) {
      throw std::runtime_error("Preprocessed input is larger than the TensorRT input buffer");
    }
    checkCuda(cudaMemcpyAsync(input_binding.device.ptr, input_half.data(), bytes, cudaMemcpyHostToDevice, stream_),
              "Failed to copy Depth Anything V3 half input to CUDA");
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
    const auto dims = binding.runtime_dims;
    std::vector<cv::Mat> planes;
    planes.reserve(static_cast<size_t>(requested_views));

    auto copyPlane = [&](size_t offset, int h, int w) {
      if (offset + static_cast<size_t>(h) * static_cast<size_t>(w) > values.size()) {
        throw std::runtime_error("TensorRT " + label + " output shape is inconsistent with its buffer size");
      }
      cv::Mat plane(h, w, CV_32FC1);
      std::memcpy(plane.data, values.data() + offset, static_cast<size_t>(h) * static_cast<size_t>(w) * sizeof(float));
      planes.push_back(std::move(plane));
    };

    if (dims.nbDims == 5) {
      const int batch = dims.d[0];
      const int views = dims.d[1];
      const int channels = dims.d[2];
      const int h = dims.d[3];
      const int w = dims.d[4];
      if (batch != 1 || channels < 1 || views != requested_views) {
        throw std::runtime_error("Unsupported " + label + " output shape " + dimsToString(dims));
      }
      const size_t view_stride = static_cast<size_t>(channels) * h * w;
      for (int view = 0; view < requested_views; ++view) {
        copyPlane(static_cast<size_t>(view) * view_stride, h, w);
      }
    } else if (dims.nbDims == 4) {
      const int views = dims.d[0];
      const int channels = dims.d[1];
      const int h = dims.d[2];
      const int w = dims.d[3];
      if (channels < 1 || views != requested_views) {
        throw std::runtime_error("Unsupported " + label + " output shape " + dimsToString(dims));
      }
      const size_t view_stride = static_cast<size_t>(channels) * h * w;
      for (int view = 0; view < requested_views; ++view) {
        copyPlane(static_cast<size_t>(view) * view_stride, h, w);
      }
    } else if (dims.nbDims == 3) {
      const int views = dims.d[0];
      const int h = dims.d[1];
      const int w = dims.d[2];
      if (views != requested_views) {
        throw std::runtime_error("Unsupported " + label + " output shape " + dimsToString(dims));
      }
      const size_t view_stride = static_cast<size_t>(h) * w;
      for (int view = 0; view < requested_views; ++view) {
        copyPlane(static_cast<size_t>(view) * view_stride, h, w);
      }
    } else if (dims.nbDims == 2 && requested_views == 1) {
      copyPlane(0, dims.d[0], dims.d[1]);
    } else {
      throw std::runtime_error("Unsupported " + label + " output shape " + dimsToString(dims));
    }

    return planes;
  }

  TensorBinding& input() { return bindings_.at(static_cast<size_t>(input_index_)); }
  const TensorBinding& input() const { return bindings_.at(static_cast<size_t>(input_index_)); }
  TensorBinding& depth() { return bindings_.at(static_cast<size_t>(depth_index_)); }
  const TensorBinding& depth() const { return bindings_.at(static_cast<size_t>(depth_index_)); }
  TensorBinding& sky() { return bindings_.at(static_cast<size_t>(sky_index_)); }
  const TensorBinding& sky() const { return bindings_.at(static_cast<size_t>(sky_index_)); }
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
  int depth_index_ = -1;
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
  return impl_->inferViews(views, optional_intrinsics);
}

cv::Size DepthAnythingV3TRT::input_size() const { return impl_->inputSize(); }

const DepthAnythingV3TRT::Params& DepthAnythingV3TRT::params() const { return impl_->params(); }

}  // namespace xfeat

#endif  // HAVE_TENSORRT
