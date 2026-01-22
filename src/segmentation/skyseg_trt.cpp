#include "xfeat-cpp/segmentation/skyseg_trt.h"

// Only compile if TensorRT is available
#ifdef HAVE_TENSORRT

#include <NvInfer.h>
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
class SkySegTRT::InferenceEngine {
 public:
  explicit InferenceEngine(const std::string& engine_file, bool verbose);
  ~InferenceEngine();

  cv::Mat run(const cv::Mat& input);

  const nvinfer1::Dims& getInputDims() const { return input_dims_; }
  const nvinfer1::Dims& getOutputDims() const { return output_dims_; }

 private:
  nvinfer1::IRuntime* runtime_;
  nvinfer1::ICudaEngine* engine_;
  nvinfer1::IExecutionContext* context_;
  cudaStream_t stream_;

  void* d_input_;
  void* d_output_;

  std::vector<char> engine_data_;
  nvinfer1::Dims input_dims_;
  nvinfer1::Dims output_dims_;
  nvinfer1::DataType input_dtype_;
  nvinfer1::DataType output_dtype_;

  Logger logger_;
  bool verbose_;

  void loadEngine(const std::string& engine_file);
  void allocateBuffers();
};

SkySegTRT::InferenceEngine::InferenceEngine(const std::string& engine_file, bool verbose)
    : runtime_(nullptr),
      engine_(nullptr),
      context_(nullptr),
      stream_(nullptr),
      d_input_(nullptr),
      d_output_(nullptr),
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

  // Create CUDA stream
  cudaStreamCreate(&stream_);

  if (verbose_) {
    std::cout << "TensorRT Sky Segmentation engine loaded successfully" << std::endl;
    std::cout << "Input shape: [";
    for (int i = 0; i < input_dims_.nbDims; ++i) {
      std::cout << input_dims_.d[i];
      if (i < input_dims_.nbDims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Output shape: [";
    for (int i = 0; i < output_dims_.nbDims; ++i) {
      std::cout << output_dims_.d[i];
      if (i < output_dims_.nbDims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }
}

SkySegTRT::InferenceEngine::~InferenceEngine() {
  if (stream_) cudaStreamDestroy(stream_);
  if (d_input_) cudaFree(d_input_);
  if (d_output_) cudaFree(d_output_);

  delete context_;
  delete engine_;
  delete runtime_;
}

void SkySegTRT::InferenceEngine::loadEngine(const std::string& engine_file) {
  std::ifstream file(engine_file, std::ios::binary);
  if (!file) throw std::runtime_error("Failed to open engine file: " + engine_file);

  file.seekg(0, std::ios::end);
  const size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  engine_data_.resize(size);
  file.read(engine_data_.data(), size);
  file.close();
}

void SkySegTRT::InferenceEngine::allocateBuffers() {
  int nbIOTensors = engine_->getNbIOTensors();

  for (int i = 0; i < nbIOTensors; ++i) {
    const char* tensor_name = engine_->getIOTensorName(i);
    auto dims = engine_->getTensorShape(tensor_name);
    auto dtype = engine_->getTensorDataType(tensor_name);
    auto mode = engine_->getTensorIOMode(tensor_name);

    size_t size = volume(dims) * getElementSize(dtype);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      if (cudaMalloc(&d_input_, size) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU input buffer");
      }
      input_dims_ = dims;
      input_dtype_ = dtype;
      context_->setTensorAddress(tensor_name, d_input_);
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      if (cudaMalloc(&d_output_, size) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU output buffer");
      }
      output_dims_ = dims;
      output_dtype_ = dtype;
      context_->setTensorAddress(tensor_name, d_output_);
    }
  }
}

cv::Mat SkySegTRT::InferenceEngine::run(const cv::Mat& input) {
  // Copy input to device
  size_t input_size = input.total() * input.elemSize();
  cudaMemcpyAsync(d_input_, input.data, input_size, cudaMemcpyHostToDevice, stream_);

  // Run inference
  if (!context_->enqueueV3(stream_)) {
    throw std::runtime_error("Failed to enqueue inference");
  }

  // Get output dimensions
  int batch = output_dims_.d[0];
  int channels = output_dims_.d[1];
  int height = output_dims_.d[2];
  int width = output_dims_.d[3];

  // Allocate output buffer
  cv::Mat output(height, width, CV_32FC1);

  // Copy output from device
  size_t output_size = output.total() * output.elemSize();
  cudaMemcpyAsync(output.data, d_output_, output_size, cudaMemcpyDeviceToHost, stream_);

  // Synchronize
  cudaStreamSynchronize(stream_);

  return output;
}

// SkySegTRT implementation
SkySegTRT::SkySegTRT(const Params& params) : params_(params) {
  if (params_.engine_path.empty()) {
    throw std::invalid_argument("Engine path cannot be empty");
  }

  // Create inference engine
  engine_ = std::make_unique<InferenceEngine>(params_.engine_path, params_.verbose);
}

SkySegTRT::~SkySegTRT() = default;

cv::Mat SkySegTRT::preprocess(const cv::Mat& image) {
  // Resize to input size
  cv::Mat resized;
  cv::resize(image, resized, params_.input_size);

  // Convert BGR to RGB
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  // Convert to float and normalize [0, 1]
  cv::Mat normalized;
  rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);

  // Apply ImageNet normalization
  cv::Scalar mean(params_.mean[0], params_.mean[1], params_.mean[2]);
  cv::Scalar std(params_.std[0], params_.std[1], params_.std[2]);
  normalized = (normalized - mean) / std;

  // Convert HWC to CHW format
  std::vector<cv::Mat> channels(3);
  cv::split(normalized, channels);

  // Create output tensor in CHW format: (1, 3, H, W)
  cv::Mat chw(params_.input_size.height * 3, params_.input_size.width, CV_32FC1);
  for (int i = 0; i < 3; ++i) {
    channels[i].copyTo(
        chw(cv::Rect(0, i * params_.input_size.height, params_.input_size.width, params_.input_size.height)));
  }

  return chw;
}

void SkySegTRT::postprocess(const cv::Mat& raw_output, cv::Mat& mask) {
  // Normalize to 0-255
  double minVal, maxVal;
  cv::minMaxLoc(raw_output, &minVal, &maxVal);

  cv::Mat normalized;
  if (maxVal > minVal) {
    normalized = (raw_output - minVal) / (maxVal - minVal) * 255.0;
  } else {
    normalized = cv::Mat::zeros(raw_output.size(), CV_32F);
  }

  normalized.convertTo(mask, CV_8UC1);

  // Store raw mask
  raw_mask_ = mask.clone();
}

void SkySegTRT::segment(const cv::Mat& image, cv::Mat& mask) {
  if (image.empty()) {
    throw std::invalid_argument("Input image is empty");
  }

  // Store original size
  cv::Size original_size = image.size();

  // Preprocess
  cv::Mat input_tensor = preprocess(image);

  // Run inference
  cv::Mat raw_output = engine_->run(input_tensor);

  // Postprocess
  cv::Mat mask_resized;
  postprocess(raw_output, mask_resized);

  // Resize mask back to original size
  cv::resize(mask_resized, mask, original_size, 0, 0, cv::INTER_LINEAR);
}

void SkySegTRT::segmentWithVisualization(const cv::Mat& image,
                                         cv::Mat& mask,
                                         cv::Mat& overlay,
                                         float alpha,
                                         const cv::Scalar& color) {
  // Run segmentation
  segment(image, mask);

  // Create overlay
  overlay = image.clone();
  cv::Mat colored_mask = cv::Mat::zeros(image.size(), image.type());
  colored_mask.setTo(color, mask > params_.threshold);

  cv::addWeighted(image, 1.0 - alpha, colored_mask, alpha, 0, overlay);
}

void SkySegTRT::warmup(const cv::Size& image_size) {
  // Create dummy image for warmup
  cv::Mat dummy_image = cv::Mat::zeros(image_size, CV_8UC3);
  cv::Mat dummy_mask;

  // Run warmup iterations
  for (int i = 0; i < params_.warmup_iterations; ++i) {
    try {
      segment(dummy_image, dummy_mask);
    } catch (const std::exception& e) {
      // Ignore errors during warmup
      if (params_.verbose) {
        std::cerr << "Warmup iteration " << i << " failed: " << e.what() << std::endl;
      }
    }
  }

  if (params_.verbose) {
    std::cout << "Warmup completed with " << params_.warmup_iterations << " iterations" << std::endl;
  }
}

float SkySegTRT::calculateSkyPercentage(const cv::Mat& mask) {
  if (mask.empty()) {
    return 0.0f;
  }

  int sky_pixels = cv::countNonZero(mask > 127);
  int total_pixels = mask.rows * mask.cols;

  return (static_cast<float>(sky_pixels) / total_pixels) * 100.0f;
}

}  // namespace xfeat

#endif  // HAVE_TENSORRT
