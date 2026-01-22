#include "xfeat-cpp/stereo_depth_onnx.h"

#include <algorithm>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace xfeat {

OnnxStereoDepth::OnnxStereoDepth(const Params& params)
    : params_(params), memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  if (params_.model_path.empty()) {
    throw std::invalid_argument("Model path cannot be empty");
  }

  initializeSession();

  if (params_.verbose) {
    std::cout << "OnnxStereoDepth initialized with model: " << params_.model_path << std::endl;
    std::cout << "Input size: " << params_.input_size << std::endl;
    std::cout << "Max disparity: " << params_.max_disparity << std::endl;
  }
}

OnnxStereoDepth::~OnnxStereoDepth() = default;

void OnnxStereoDepth::initializeSession() {
  // Create ONNX Runtime environment
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OnnxStereoDepth");

  // Create session options
  session_options_ = std::make_unique<Ort::SessionOptions>();
  session_options_->SetIntraOpNumThreads(1);
  session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // Add CUDA execution provider if requested
  if (params_.use_cuda) {
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    cuda_options.arena_extend_strategy = 0;
    cuda_options.gpu_mem_limit = SIZE_MAX;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.do_copy_in_default_stream = 1;

    session_options_->AppendExecutionProvider_CUDA(cuda_options);
    if (params_.verbose) {
      std::cout << "Using CUDA execution provider" << std::endl;
    }
  }

  // Create session
  session_ = std::make_unique<Ort::Session>(*env_, params_.model_path.c_str(), *session_options_);

  // Get input names and shapes
  Ort::AllocatorWithDefaultOptions allocator;
  size_t num_input_nodes = session_->GetInputCount();

  for (size_t i = 0; i < num_input_nodes; i++) {
    auto input_name = session_->GetInputNameAllocated(i, allocator);
    input_names_storage_.push_back(std::string(input_name.get()));

    auto input_type_info = session_->GetInputTypeInfo(i);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape_ = tensor_info.GetShape();

    if (params_.verbose) {
      std::cout << "Input " << i << " name: " << input_names_storage_.back() << std::endl;
      std::cout << "Input shape: [";
      for (size_t j = 0; j < input_shape_.size(); j++) {
        std::cout << input_shape_[j];
        if (j < input_shape_.size() - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
    }
  }

  // Now populate input_names_ vector with pointers (after all strings are collected)
  for (const auto& name : input_names_storage_) {
    input_names_.push_back(name.c_str());
  }

  // Get output names and shapes
  size_t num_output_nodes = session_->GetOutputCount();
  for (size_t i = 0; i < num_output_nodes; i++) {
    auto output_name = session_->GetOutputNameAllocated(i, allocator);
    output_names_storage_.push_back(std::string(output_name.get()));

    auto output_type_info = session_->GetOutputTypeInfo(i);
    auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    output_shape_ = tensor_info.GetShape();

    if (params_.verbose && i == 0) {
      std::cout << "Output " << i << " name: " << output_names_storage_.back() << std::endl;
      std::cout << "Output shape: [";
      for (size_t j = 0; j < output_shape_.size(); j++) {
        std::cout << output_shape_[j];
        if (j < output_shape_.size() - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
    }
  }

  // Now populate output_names_ vector with pointers (after all strings are collected)
  for (const auto& name : output_names_storage_) {
    output_names_.push_back(name.c_str());
  }
}

void OnnxStereoDepth::preprocessImage(const cv::Mat& img, std::vector<float>& output) {
  cv::Mat rgb_img;
  if (img.channels() == 1) {
    cv::cvtColor(img, rgb_img, cv::COLOR_GRAY2RGB);
  } else if (img.channels() == 3) {
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
  } else {
    throw std::runtime_error("Unsupported number of channels: " + std::to_string(img.channels()));
  }

  // Resize to model input size
  cv::Mat resized;
  cv::resize(rgb_img, resized, params_.input_size, 0, 0, cv::INTER_AREA);

  // Convert to float and normalize
  resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

  // Normalize using mean and std
  std::vector<cv::Mat> channels(3);
  cv::split(resized, channels);

  for (int c = 0; c < 3; c++) {
    channels[c] = (channels[c] - params_.mean[c]) / params_.std[c];
  }

  // Convert to NCHW format (batch, channels, height, width)
  size_t image_size = params_.input_size.width * params_.input_size.height;
  output.resize(3 * image_size);

  for (int c = 0; c < 3; c++) {
    std::memcpy(output.data() + c * image_size, channels[c].data, image_size * sizeof(float));
  }
}

void OnnxStereoDepth::runInference(const std::vector<float>& left_tensor,
                                   const std::vector<float>& right_tensor,
                                   std::vector<float>& output) {
  // Create input tensors
  std::vector<int64_t> input_shape = {1, 3, params_.input_size.height, params_.input_size.width};

  auto left_tensor_obj = Ort::Value::CreateTensor<float>(
      memory_info_, const_cast<float*>(left_tensor.data()), left_tensor.size(), input_shape.data(), input_shape.size());

  auto right_tensor_obj = Ort::Value::CreateTensor<float>(memory_info_,
                                                          const_cast<float*>(right_tensor.data()),
                                                          right_tensor.size(),
                                                          input_shape.data(),
                                                          input_shape.size());

  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(std::move(left_tensor_obj));
  input_tensors.push_back(std::move(right_tensor_obj));

  // Run inference
  auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                      input_names_.data(),
                                      input_tensors.data(),
                                      input_tensors.size(),
                                      output_names_.data(),
                                      output_names_.size());

  // Extract output
  float* output_data = output_tensors[0].GetTensorMutableData<float>();
  auto output_shape_actual = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

  size_t output_size = 1;
  for (auto dim : output_shape_actual) {
    output_size *= dim;
  }

  output.resize(output_size);
  std::memcpy(output.data(), output_data, output_size * sizeof(float));
}

void OnnxStereoDepth::postprocessDisparity(const std::vector<float>& model_output, cv::Mat& disparity) {
  // Model output is typically [1, 1, H, W] or [1, H, W]
  int output_height = params_.input_size.height;
  int output_width = params_.input_size.width;

  // Create disparity map at model resolution
  raw_disparity_ = cv::Mat(output_height, output_width, CV_32F);
  std::memcpy(raw_disparity_.data, model_output.data(), output_height * output_width * sizeof(float));

  // Resize to original image size if different
  if (original_size_.width > 0 && original_size_.height > 0) {
    cv::resize(raw_disparity_, disparity, original_size_, 0, 0, cv::INTER_LINEAR);

    // Scale disparity values proportionally to the resize
    float scale_x = static_cast<float>(original_size_.width) / output_width;
    disparity *= scale_x;
  } else {
    disparity = raw_disparity_.clone();
  }
}

void OnnxStereoDepth::compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity) {
  if (left.empty() || right.empty()) {
    throw std::invalid_argument("Input images cannot be empty");
  }

  if (left.size() != right.size()) {
    throw std::invalid_argument("Left and right images must have the same size");
  }

  original_size_ = left.size();

  // Preprocess images
  std::vector<float> left_tensor, right_tensor;
  preprocessImage(left, left_tensor);
  preprocessImage(right, right_tensor);

  // Run inference
  std::vector<float> output;
  runInference(left_tensor, right_tensor, output);

  // Postprocess output
  postprocessDisparity(output, disparity);
}

void OnnxStereoDepth::computeDepthDirect(const cv::Mat& left, const cv::Mat& right, cv::Mat& depth) {
  if (params_.focal_length <= 0.0f || params_.baseline <= 0.0f) {
    throw std::runtime_error("Camera parameters (focal_length, baseline) must be set for depth computation");
  }

  cv::Mat disparity;
  compute(left, right, disparity);

  // Convert disparity to depth: depth = (focal_length * baseline) / disparity
  depth = cv::Mat::zeros(disparity.size(), CV_32F);
  for (int y = 0; y < disparity.rows; y++) {
    for (int x = 0; x < disparity.cols; x++) {
      float disp = disparity.at<float>(y, x);
      if (disp > 0.0f) {
        depth.at<float>(y, x) = (params_.focal_length * params_.baseline) / disp;
      }
    }
  }
}

void OnnxStereoDepth::warmup(const cv::Size& image_size) {
  if (params_.verbose) {
    std::cout << "Warming up ONNX stereo depth with " << params_.warmup_iterations << " iterations..." << std::endl;
  }

  cv::Mat dummy_left = cv::Mat::zeros(image_size, CV_8UC3);
  cv::Mat dummy_right = cv::Mat::zeros(image_size, CV_8UC3);
  cv::Mat dummy_disparity;

  for (int i = 0; i < params_.warmup_iterations; i++) {
    compute(dummy_left, dummy_right, dummy_disparity);
  }

  if (params_.verbose) {
    std::cout << "Warmup complete" << std::endl;
  }
}

cv::Mat OnnxStereoDepth::getColorDisparity() const {
  if (raw_disparity_.empty()) {
    return cv::Mat();
  }

  // Normalize disparity to 0-255 range
  cv::Mat normalized;
  double min_val, max_val;
  cv::minMaxLoc(raw_disparity_, &min_val, &max_val);

  if (max_val > min_val) {
    raw_disparity_.convertTo(normalized, CV_8U, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
  } else {
    normalized = cv::Mat::zeros(raw_disparity_.size(), CV_8U);
  }

  // Apply colormap
  cv::Mat color_disparity;
  cv::applyColorMap(normalized, color_disparity, cv::COLORMAP_MAGMA);

  // Resize to original size if needed
  if (original_size_.width > 0 && original_size_.height > 0) {
    cv::resize(color_disparity, color_disparity, original_size_, 0, 0, cv::INTER_LINEAR);
  }

  return color_disparity;
}

}  // namespace xfeat
