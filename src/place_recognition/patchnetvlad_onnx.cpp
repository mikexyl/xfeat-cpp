#include "xfeat-cpp/place_recognition/patchnetvlad_onnx.h"

#include <cuda_runtime.h>

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace xfeat {

PatchNetVLADONNX::PatchNetVLADONNX(Ort::Env& env, const Params& params)
    : session_(nullptr),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
      img_height_(params.img_height),
      img_width_(params.img_width),
      descriptor_dim_(params.descriptor_dim),
      normalize_output_(params.normalize_output) {
  // Configure session options
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

  if (params.use_gpu) {
    std::cout << "Attempting to use GPU for PatchNetVLAD ONNX Runtime." << std::endl;

    auto available_providers = Ort::GetAvailableProviders();
    bool cuda_available = false;

    std::cout << "Available ONNX Runtime providers: ";
    for (const auto& provider : available_providers) {
      std::cout << provider << " ";
      if (provider == "CUDAExecutionProvider") {
        cuda_available = true;
      }
    }
    std::cout << std::endl;

    if (!cuda_available) {
      std::cerr << "Warning: CUDAExecutionProvider not available. Falling back to CPU." << std::endl;
    } else {
      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id = 0;
      cuda_options.arena_extend_strategy = 0;
      cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;  // 2 GB
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
      cuda_options.do_copy_in_default_stream = 1;
      session_options_.AppendExecutionProvider_CUDA(cuda_options);
    }
  }

  // Load model
  session_ = Ort::Session(env, params.model_path.c_str(), session_options_);

  // Enumerate input/output names
  Ort::AllocatorWithDefaultOptions allocator;

  size_t num_inputs = session_.GetInputCount();
  input_name_strings_.resize(num_inputs);
  input_names_.resize(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    auto name = session_.GetInputNameAllocated(i, allocator);
    input_name_strings_[i] = std::string(name.get());
    input_names_[i] = input_name_strings_[i].c_str();
    std::cout << "Input " << i << ": " << input_names_[i] << std::endl;
  }

  size_t num_outputs = session_.GetOutputCount();
  output_name_strings_.resize(num_outputs);
  output_names_.resize(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    auto name = session_.GetOutputNameAllocated(i, allocator);
    output_name_strings_[i] = std::string(name.get());
    output_names_[i] = output_name_strings_[i].c_str();
    std::cout << "Output " << i << ": " << output_names_[i] << std::endl;
  }

  auto input_shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  std::cout << "PatchNetVLAD model loaded. Input shape: [" << input_shape[0] << ", " << input_shape[1] << ", "
            << input_shape[2] << ", " << input_shape[3] << "]" << std::endl;
  std::cout << "Descriptor dimension: " << descriptor_dim_ << std::endl;
}

cv::Mat PatchNetVLADONNX::preprocess_image(const cv::Mat& image) {
  if (image.empty()) {
    throw std::runtime_error("PatchNetVLADONNX: Input image is empty.");
  }

  cv::Mat processed;

  // Resize to target dimensions
  cv::resize(image, processed, cv::Size(img_width_, img_height_));

  // BGR → RGB
  cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

  // /255 → float32
  processed.convertTo(processed, CV_32F, 1.0 / 255.0);

  // ImageNet channel-wise normalization: (x - mean) / std
  static const float mean[3] = {0.485f, 0.456f, 0.406f};
  static const float std_dev[3] = {0.229f, 0.224f, 0.225f};

  std::vector<cv::Mat> channels(3);
  cv::split(processed, channels);
  for (int c = 0; c < 3; ++c) {
    channels[c] = (channels[c] - mean[c]) / std_dev[c];
  }
  cv::merge(channels, processed);

  return processed;
}

std::vector<float> PatchNetVLADONNX::prepare_input_tensor(const cv::Mat& preprocessed) {
  // Pack CV_32FC3 HWC mat into NCHW buffer [1, 3, H, W]
  std::vector<float> tensor(3 * img_height_ * img_width_);

  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < img_height_; ++h) {
      for (int w = 0; w < img_width_; ++w) {
        tensor[c * img_height_ * img_width_ + h * img_width_ + w] = preprocessed.at<cv::Vec3f>(h, w)[c];
      }
    }
  }

  return tensor;
}

void PatchNetVLADONNX::normalize_descriptor(cv::Mat& desc) {
  if (desc.empty()) return;

  double norm = cv::norm(desc, cv::NORM_L2);
  if (norm < 1e-8) {
    std::cerr << "Warning: descriptor has near-zero norm" << std::endl;
    return;
  }
  desc /= norm;
}

PatchNetVLADONNX::Features PatchNetVLADONNX::extract(const cv::Mat& image) {
  // Clear any pending CUDA errors from previous operations
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    cudaGetLastError();  // Clear error
  }

  cv::Mat preprocessed = preprocess_image(image);
  std::vector<float> input_data = prepare_input_tensor(preprocessed);

  std::vector<int64_t> input_shape = {1, 3, img_height_, img_width_};
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(std::move(input_tensor));

  auto output_tensors = session_.Run(Ort::RunOptions{nullptr},
                                     input_names_.data(),
                                     input_tensors.data(),
                                     input_tensors.size(),
                                     output_names_.data(),
                                     output_names_.size());

  Features features;

  // Output 0: global_feat [1, descriptor_dim]
  {
    float* global_data = output_tensors[0].GetTensorMutableData<float>();
    features.global_desc = cv::Mat(1, descriptor_dim_, CV_32F);
    std::memcpy(features.global_desc.data, global_data, descriptor_dim_ * sizeof(float));
    if (normalize_output_) {
      normalize_descriptor(features.global_desc);
    }
  }

  // Outputs 1..N: local_i [1, descriptor_dim, num_patches_i] → squeeze → [descriptor_dim, num_patches_i]
  for (size_t i = 1; i < output_tensors.size(); ++i) {
    auto shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
    // shape = [1, D, P]
    int D = static_cast<int>(shape[1]);
    int P = static_cast<int>(shape[2]);

    float* local_data = output_tensors[i].GetTensorMutableData<float>();
    cv::Mat local_desc(D, P, CV_32F);
    std::memcpy(local_desc.data, local_data, static_cast<size_t>(D) * P * sizeof(float));
    features.local_descs.push_back(local_desc);
  }

  return features;
}

cv::Mat PatchNetVLADONNX::infer(const cv::Mat& image) {
  return extract(image).global_desc;
}

cv::Mat PatchNetVLADONNX::infer(const std::vector<cv::Mat>& images) {
  if (images.size() != 1) {
    throw std::runtime_error("PatchNetVLADONNX::infer: expected exactly 1 image, got " +
                             std::to_string(images.size()));
  }
  return infer(images[0]);
}

}  // namespace xfeat
