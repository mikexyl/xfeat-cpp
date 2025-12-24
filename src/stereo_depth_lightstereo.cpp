#include "xfeat-cpp/stereo_depth_lightstereo.h"

// Only compile if TensorRT is available
#ifdef HAVE_TENSORRT

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace xfeat {

// Forward declaration helpers
namespace {

size_t volume(const nvinfer1::Dims& dims) {
  size_t vol = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    vol *= dims.d[i];
  }
  return vol;
}

size_t getElementSize(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return sizeof(float);
    case nvinfer1::DataType::kHALF:
      return sizeof(uint16_t);
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

class Logger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
    if (severity <= nvinfer1::ILogger::Severity::kWARNING) {
      std::cout << msg << std::endl;
    }
  }
};

}  // anonymous namespace

// InferenceEngine implementation
class LightStereoDepth::InferenceEngine {
 public:
  explicit InferenceEngine(const std::string& engine_file, bool verbose);
  ~InferenceEngine();

  std::map<std::string, cv::Mat> run(const std::map<std::string, cv::Mat>& sample);

 private:
  nvinfer1::IRuntime* runtime_;
  nvinfer1::ICudaEngine* engine_;
  nvinfer1::IExecutionContext* context_;
  cudaGraph_t graph_;
  cudaGraphExec_t instance_;
  cudaStream_t stream_;

  std::vector<void*> buffers_;
  std::vector<char> engine_data_;
  nvinfer1::Dims output_dims_;
  nvinfer1::DataType output_dtype_;

  Logger logger_;
  bool verbose_;

  void loadEngine(const std::string& engine_file);
  void allocateBuffers();
  void preprocess(const std::map<std::string, cv::Mat>& sample);
  std::map<std::string, cv::Mat> postprocess();
};

LightStereoDepth::InferenceEngine::InferenceEngine(const std::string& engine_file, bool verbose)
    : runtime_(nullptr),
      engine_(nullptr),
      context_(nullptr),
      graph_(nullptr),
      instance_(nullptr),
      stream_(nullptr),
      verbose_(verbose) {
  // Set CUDA device
  cudaError_t status = cudaSetDevice(0);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("Error setting CUDA device: ") + cudaGetErrorString(status));
  }

  loadEngine(engine_file);

  runtime_ = nvinfer1::createInferRuntime(logger_);
  if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime");

  engine_ = runtime_->deserializeCudaEngine(engine_data_.data(), engine_data_.size());
  if (!engine_) throw std::runtime_error("Failed to create TensorRT engine");

  context_ = engine_->createExecutionContext();
  if (!context_) throw std::runtime_error("Failed to create TensorRT context");

  allocateBuffers();

  // Set tensor addresses
  context_->setTensorAddress("left_img", buffers_[0]);
  context_->setTensorAddress("right_img", buffers_[1]);
  context_->setTensorAddress("disp_pred", buffers_[2]);

  // Create CUDA stream
  cudaStreamCreate(&stream_);
}

LightStereoDepth::InferenceEngine::~InferenceEngine() {
  if (stream_) cudaStreamDestroy(stream_);
  if (graph_) cudaGraphDestroy(graph_);
  if (instance_) cudaGraphExecDestroy(instance_);

  for (void* buffer : buffers_) {
    if (buffer) cudaFree(buffer);
  }

  delete context_;
  delete engine_;
  delete runtime_;
}

void LightStereoDepth::InferenceEngine::loadEngine(const std::string& engine_file) {
  std::ifstream file(engine_file, std::ios::binary);
  if (!file) throw std::runtime_error("Failed to open engine file: " + engine_file);

  file.seekg(0, std::ios::end);
  const size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  engine_data_.resize(size);
  file.read(engine_data_.data(), size);
  file.close();
}

void LightStereoDepth::InferenceEngine::allocateBuffers() {
  int nbIOTensors = engine_->getNbIOTensors();
  buffers_.resize(nbIOTensors);

  for (int i = 0; i < nbIOTensors; ++i) {
    auto dims = engine_->getTensorShape(engine_->getIOTensorName(i));
    auto dtype = engine_->getTensorDataType(engine_->getIOTensorName(i));
    size_t size = volume(dims) * getElementSize(dtype);

    if (cudaMalloc(&buffers_[i], size) != cudaSuccess) {
      throw std::runtime_error("Failed to allocate GPU buffer");
    }

    // Store output dimensions for postprocessing
    if (engine_->getTensorIOMode(engine_->getIOTensorName(i)) == nvinfer1::TensorIOMode::kOUTPUT) {
      output_dims_ = dims;
      output_dtype_ = dtype;
    }
  }
}

void LightStereoDepth::InferenceEngine::preprocess(const std::map<std::string, cv::Mat>& sample) {
  for (const auto& [key, image] : sample) {
    void* buffer = nullptr;
    if (key == "left_img") {
      buffer = buffers_[0];
    } else if (key == "right_img") {
      buffer = buffers_[1];
    }

    if (buffer && !image.empty()) {
      cudaMemcpyAsync(buffer, image.data, image.total() * image.elemSize(), cudaMemcpyHostToDevice, stream_);
    }
  }
}

std::map<std::string, cv::Mat> LightStereoDepth::InferenceEngine::postprocess() {
  std::map<std::string, cv::Mat> output;

  // Get output dimensions
  int height = output_dims_.d[output_dims_.nbDims - 2];
  int width = output_dims_.d[output_dims_.nbDims - 1];

  cv::Mat disp_pred(height, width, CV_32FC1);

  // Copy from device to host
  cudaMemcpyAsync(
      disp_pred.data, buffers_[2], disp_pred.total() * disp_pred.elemSize(), cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  output["disp_pred"] = disp_pred;

  if (disp_pred.empty()) {
    throw std::runtime_error("Error: disp_pred cv::Mat is empty!");
  }

  // Create normalized visualization
  double minVal, maxVal;
  cv::minMaxLoc(disp_pred, &minVal, &maxVal);
  cv::Mat normalized_disp_pred;
  disp_pred.convertTo(normalized_disp_pred, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
  output["normalized_disp_pred"] = normalized_disp_pred;

  // Create color visualization
  cv::Mat color_normalized_disp_pred;
  cv::applyColorMap(normalized_disp_pred, color_normalized_disp_pred, cv::COLORMAP_JET);
  output["color_normalized_disp_pred"] = color_normalized_disp_pred;

  return output;
}

std::map<std::string, cv::Mat> LightStereoDepth::InferenceEngine::run(const std::map<std::string, cv::Mat>& sample) {
  preprocess(sample);

  // Use CUDA Graph for better performance
  if (instance_ == nullptr) {
    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
    if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error("Failed to enqueue inference");
    }
    cudaStreamEndCapture(stream_, &graph_);
    cudaGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0);
  }

  cudaGraphLaunch(instance_, stream_);

  return postprocess();
}

// TransformPipeline implementation
class LightStereoDepth::TransformPipeline {
 public:
  explicit TransformPipeline(const cv::Size& target_size,
                             const std::vector<float>& mean,
                             const std::vector<float>& std);

  std::map<std::string, cv::Mat> operator()(const cv::Mat& left, const cv::Mat& right);

 private:
  cv::Size target_size_;
  cv::Scalar mean_;
  cv::Scalar std_;

  cv::Mat padImage(const cv::Mat& image);
  cv::Mat normalizeImage(const cv::Mat& image);
  cv::Mat transposeImage(const cv::Mat& image);
};

LightStereoDepth::TransformPipeline::TransformPipeline(const cv::Size& target_size,
                                                       const std::vector<float>& mean,
                                                       const std::vector<float>& std)
    : target_size_(target_size) {
  if (mean.size() != 3 || std.size() != 3) {
    throw std::invalid_argument("Mean and std must have 3 elements (RGB)");
  }
  mean_ = cv::Scalar(mean[0], mean[1], mean[2]);
  std_ = cv::Scalar(std[0], std[1], std[2]);
}

cv::Mat LightStereoDepth::TransformPipeline::padImage(const cv::Mat& image) {
  int pad_h = std::max(0, target_size_.height - image.rows);
  int pad_w = std::max(0, target_size_.width - image.cols);

  cv::Mat padded;
  cv::copyMakeBorder(image, padded, pad_h, 0, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  return padded;
}

cv::Mat LightStereoDepth::TransformPipeline::normalizeImage(const cv::Mat& image) {
  cv::Mat normalized;
  image.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

  // Apply normalization: (x - mean) / std
  std::vector<cv::Mat> channels(3);
  cv::split(normalized, channels);

  for (int i = 0; i < 3; ++i) {
    channels[i] = (channels[i] - mean_[i]) / std_[i];
  }

  cv::merge(channels, normalized);
  return normalized;
}

cv::Mat LightStereoDepth::TransformPipeline::transposeImage(const cv::Mat& image) {
  // Transpose from HWC to CHW format
  std::vector<cv::Mat> channels(3);
  cv::split(image, channels);

  cv::Mat result(image.rows * 3, image.cols, CV_32FC1);
  for (int i = 0; i < 3; ++i) {
    channels[i].copyTo(result(cv::Rect(0, i * image.rows, image.cols, image.rows)));
  }

  return result;
}

std::map<std::string, cv::Mat> LightStereoDepth::TransformPipeline::operator()(const cv::Mat& left,
                                                                               const cv::Mat& right) {
  std::map<std::string, cv::Mat> sample;

  // Convert BGR to RGB
  cv::Mat left_rgb, right_rgb;
  cv::cvtColor(left, left_rgb, cv::COLOR_BGR2RGB);
  cv::cvtColor(right, right_rgb, cv::COLOR_BGR2RGB);

  // Apply transformations
  cv::Mat left_padded = padImage(left_rgb);
  cv::Mat right_padded = padImage(right_rgb);

  cv::Mat left_normalized = normalizeImage(left_padded);
  cv::Mat right_normalized = normalizeImage(right_padded);

  cv::Mat left_transposed = transposeImage(left_normalized);
  cv::Mat right_transposed = transposeImage(right_normalized);

  sample["left_img"] = left_transposed;
  sample["right_img"] = right_transposed;

  return sample;
}

// LightStereoDepth implementation
LightStereoDepth::LightStereoDepth(const Params& params) : params_(params) {
  if (params_.engine_path.empty()) {
    throw std::invalid_argument("Engine path cannot be empty");
  }

  // Create inference engine
  engine_ = std::make_unique<InferenceEngine>(params_.engine_path, params_.verbose);

  // Create transform pipeline
  transform_ = std::make_unique<TransformPipeline>(params_.target_size, params_.mean, params_.std);
}

LightStereoDepth::~LightStereoDepth() = default;

void LightStereoDepth::compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity) {
  if (left.empty() || right.empty()) {
    throw std::invalid_argument("Input images are empty");
  }
  if (left.size() != right.size()) {
    throw std::invalid_argument("Left and right images must have the same size");
  }

  // Store original size for post-processing
  original_size_ = left.size();

  // Resize images if needed
  cv::Mat left_resized, right_resized;
  bool needs_resize = (left.size() != params_.target_size);

  if (needs_resize) {
    cv::resize(left, left_resized, params_.target_size, 0, 0, cv::INTER_LINEAR);
    cv::resize(right, right_resized, params_.target_size, 0, 0, cv::INTER_LINEAR);
  } else {
    left_resized = left;
    right_resized = right;
  }

  // Calculate scaling factors for disparity
  float scale_x = static_cast<float>(original_size_.width) / params_.target_size.width;
  float scale_y = static_cast<float>(original_size_.height) / params_.target_size.height;

  // Preprocess
  auto sample = (*transform_)(left_resized, right_resized);

  // Run inference
  auto output = engine_->run(sample);

  // Extract results
  raw_disparity_ = output["disp_pred"];
  color_disparity_ = output["color_normalized_disp_pred"];

  // Postprocess - crop padding if needed
  cv::Mat disparity_cropped;
  if (raw_disparity_.size() != params_.target_size) {
    cv::Rect roi(
        0, raw_disparity_.rows - params_.target_size.height, params_.target_size.width, params_.target_size.height);
    disparity_cropped = raw_disparity_(roi);
  } else {
    disparity_cropped = raw_disparity_;
  }

  // Scale disparity back to original size if needed
  if (needs_resize) {
    cv::Mat disparity_resized;
    cv::resize(disparity_cropped, disparity_resized, original_size_, 0, 0, cv::INTER_LINEAR);
    static constexpr float kMaxDisparityRatio = 0.05f;
    float max_disparity = kMaxDisparityRatio * params_.target_size.width;

    // Scale disparity values by the horizontal scaling factor
    disparity = disparity_resized * scale_x / 16;
    disparity.setTo(-1.0f, disparity_resized > max_disparity);
  } else {
    throw "should be resized";
    disparity = disparity_cropped.clone();
  }
}

void LightStereoDepth::warmup(const cv::Size& image_size) {
  // Create dummy images for warmup
  cv::Mat dummy_left = cv::Mat::zeros(image_size, CV_8UC3);
  cv::Mat dummy_right = cv::Mat::zeros(image_size, CV_8UC3);
  cv::Mat dummy_disparity;

  // Run warmup iterations
  for (int i = 0; i < params_.warmup_iterations; ++i) {
    try {
      compute(dummy_left, dummy_right, dummy_disparity);
    } catch (const std::exception& e) {
      // Ignore errors during warmup
    }
  }
}

}  // namespace xfeat

#endif  // HAVE_TENSORRT
