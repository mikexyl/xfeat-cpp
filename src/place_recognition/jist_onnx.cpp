#include "xfeat-cpp/place_recognition/jist_onnx.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace xfeat {

JistONNX::JistONNX(Ort::Env& env, const Params& params)
    : session_(nullptr),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
      seq_length_(0),
      img_height_(params.img_height),
      img_width_(params.img_width),
      descriptor_dim_(0),
      normalize_output_(params.normalize_output) {
  // Configure session options
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
  session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

  if (params.use_gpu) {
    std::cout << "Attempting to use GPU for JIST ONNX Runtime." << std::endl;

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
      cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;  // 2GB
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
      cuda_options.do_copy_in_default_stream = 1;
      session_options_.AppendExecutionProvider_CUDA(cuda_options);
    }
  }

  // Load model
  session_ = Ort::Session(env, params.model_path.c_str(), session_options_);

  // Get input/output names
  Ort::AllocatorWithDefaultOptions allocator;

  // Input names - store strings first, then create const char* pointers
  size_t num_input_nodes = session_.GetInputCount();
  input_name_strings_.resize(num_input_nodes);
  input_names_.resize(num_input_nodes);
  for (size_t i = 0; i < num_input_nodes; i++) {
    auto input_name = session_.GetInputNameAllocated(i, allocator);
    input_name_strings_[i] = std::string(input_name.get());
    input_names_[i] = input_name_strings_[i].c_str();
    std::cout << "Input " << i << ": " << input_names_[i] << std::endl;
  }

  // Output names - store strings first, then create const char* pointers
  size_t num_output_nodes = session_.GetOutputCount();
  output_name_strings_.resize(num_output_nodes);
  output_names_.resize(num_output_nodes);
  for (size_t i = 0; i < num_output_nodes; i++) {
    auto output_name = session_.GetOutputNameAllocated(i, allocator);
    output_name_strings_[i] = std::string(output_name.get());
    output_names_[i] = output_name_strings_[i].c_str();
    std::cout << "Output " << i << ": " << output_names_[i] << std::endl;
  }

  // Read seq_length and descriptor_dim from model shapes
  auto input_shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  seq_length_ = static_cast<int>(input_shape[1]);

  auto output_shape = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  descriptor_dim_ = static_cast<int>(output_shape[1]);

  std::cout << "JIST model loaded successfully." << std::endl;
  std::cout << "Expected input shape: [batch=" << input_shape[0] << ", seq_length=" << input_shape[1]
            << ", channels=" << input_shape[2] << ", height=" << input_shape[3] << ", width=" << input_shape[4] << "]"
            << std::endl;
  std::cout << "Descriptor dimension: " << descriptor_dim_ << std::endl;
}

cv::Mat JistONNX::preprocess_image(const cv::Mat& image) {
  if (image.empty()) {
    throw std::runtime_error("JistONNX: Input image is empty.");
  }

  cv::Mat processed;

  // Resize to target dimensions
  cv::resize(image, processed, cv::Size(img_width_, img_height_));

  // Convert BGR to RGB
  cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

  // Convert to float and normalize to [0, 1]
  processed.convertTo(processed, CV_32F, 1.0 / 255.0);

  return processed;
}

std::vector<float> JistONNX::prepare_input_tensor(const std::vector<cv::Mat>& image_sequence) {
  if (image_sequence.size() != static_cast<size_t>(seq_length_)) {
    throw std::runtime_error("Image sequence size (" + std::to_string(image_sequence.size()) +
                             ") doesn't match expected seq_length (" + std::to_string(seq_length_) + ")");
  }

  // Allocate tensor data: (1, seq_length, 3, height, width)
  size_t total_size = 1 * seq_length_ * 3 * img_height_ * img_width_;
  std::vector<float> tensor_data(total_size);

  // Fill tensor in NCHW format for each frame in sequence
  for (int s = 0; s < seq_length_; ++s) {
    const cv::Mat& img = image_sequence[s];

    if (img.type() != CV_32FC3) {
      throw std::runtime_error("Image must be preprocessed (CV_32FC3)");
    }

    // Copy data in CHW format: R, G, B channels separately
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < img_height_; ++h) {
        for (int w = 0; w < img_width_; ++w) {
          // Tensor layout: [batch=0, sequence=s, channel=c, height=h, width=w]
          size_t tensor_idx = s * (3 * img_height_ * img_width_) + c * (img_height_ * img_width_) + h * img_width_ + w;
          tensor_data[tensor_idx] = img.at<cv::Vec3f>(h, w)[c];
        }
      }
    }
  }

  return tensor_data;
}

std::vector<float> JistONNX::prepare_batch_input_tensor(const std::vector<std::vector<cv::Mat>>& batch_sequences) {
  size_t batch_size = batch_sequences.size();

  // Allocate tensor data: (batch_size, seq_length, 3, height, width)
  size_t total_size = batch_size * seq_length_ * 3 * img_height_ * img_width_;
  std::vector<float> tensor_data(total_size);

  for (size_t b = 0; b < batch_size; ++b) {
    const auto& image_sequence = batch_sequences[b];

    if (image_sequence.size() != static_cast<size_t>(seq_length_)) {
      throw std::runtime_error("Sequence " + std::to_string(b) + " size mismatch");
    }

    for (int s = 0; s < seq_length_; ++s) {
      const cv::Mat& img = image_sequence[s];

      if (img.type() != CV_32FC3) {
        throw std::runtime_error("Image must be preprocessed (CV_32FC3)");
      }

      for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < img_height_; ++h) {
          for (int w = 0; w < img_width_; ++w) {
            size_t tensor_idx = b * (seq_length_ * 3 * img_height_ * img_width_) + s * (3 * img_height_ * img_width_) +
                                c * (img_height_ * img_width_) + h * img_width_ + w;
            tensor_data[tensor_idx] = img.at<cv::Vec3f>(h, w)[c];
          }
        }
      }
    }
  }

  return tensor_data;
}

void JistONNX::normalize_descriptor(cv::Mat& descriptor) {
  if (descriptor.empty()) return;

  // Compute L2 norm
  double norm = cv::norm(descriptor, cv::NORM_L2);

  // Avoid division by zero
  if (norm < 1e-8) {
    std::cerr << "Warning: descriptor has near-zero norm" << std::endl;
    return;
  }

  // Normalize in-place
  descriptor /= norm;
}

cv::Mat JistONNX::infer(const std::vector<cv::Mat>& image_sequence) {
  // Synchronize CUDA to ensure all previous operations complete
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    // Don't print warning for CPU-only mode
    cudaGetLastError();  // Clear error
  }

  // Preprocess all images
  std::vector<cv::Mat> preprocessed_images;
  preprocessed_images.reserve(image_sequence.size());
  for (const auto& img : image_sequence) {
    preprocessed_images.push_back(preprocess_image(img));
  }

  // Prepare input tensor
  std::vector<float> input_tensor_data = prepare_input_tensor(preprocessed_images);

  // Create input tensor shape: (1, seq_length, 3, height, width)
  std::vector<int64_t> input_shape = {1, seq_length_, 3, img_height_, img_width_};

  // Create ONNX tensor
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, input_tensor_data.data(), input_tensor_data.size(), input_shape.data(), input_shape.size());

  // Run inference
  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(std::move(input_tensor));

  auto output_tensors = session_.Run(Ort::RunOptions{nullptr},
                                     input_names_.data(),
                                     input_tensors.data(),
                                     input_tensors.size(),
                                     output_names_.data(),
                                     output_names_.size());

  // Extract output
  float* output_data = output_tensors[0].GetTensorMutableData<float>();
  auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

  if (output_shape.size() != 2 || output_shape[0] != 1) {
    throw std::runtime_error("Unexpected output shape");
  }

  // Create output matrix (1 x descriptor_dim)
  cv::Mat descriptor(1, static_cast<int>(output_shape[1]), CV_32F);
  std::memcpy(descriptor.data, output_data, output_shape[1] * sizeof(float));

  // Normalize if requested
  if (normalize_output_) {
    normalize_descriptor(descriptor);
  }

  return descriptor;
}

cv::Mat JistONNX::infer_batch(const std::vector<std::vector<cv::Mat>>& batch_sequences) {
  if (batch_sequences.empty()) {
    throw std::runtime_error("Empty batch");
  }

  size_t batch_size = batch_sequences.size();

  // Preprocess all images in all sequences
  std::vector<std::vector<cv::Mat>> preprocessed_batch;
  preprocessed_batch.reserve(batch_size);

  for (const auto& sequence : batch_sequences) {
    std::vector<cv::Mat> preprocessed_seq;
    preprocessed_seq.reserve(sequence.size());
    for (const auto& img : sequence) {
      preprocessed_seq.push_back(preprocess_image(img));
    }
    preprocessed_batch.push_back(preprocessed_seq);
  }

  // Prepare input tensor
  std::vector<float> input_tensor_data = prepare_batch_input_tensor(preprocessed_batch);

  // Create input tensor shape: (batch_size, seq_length, 3, height, width)
  std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), seq_length_, 3, img_height_, img_width_};

  // Create ONNX tensor
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, input_tensor_data.data(), input_tensor_data.size(), input_shape.data(), input_shape.size());

  // Run inference
  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(std::move(input_tensor));

  auto output_tensors = session_.Run(Ort::RunOptions{nullptr},
                                     input_names_.data(),
                                     input_tensors.data(),
                                     input_tensors.size(),
                                     output_names_.data(),
                                     output_names_.size());

  // Extract output
  float* output_data = output_tensors[0].GetTensorMutableData<float>();
  auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

  if (output_shape.size() != 2 || output_shape[0] != static_cast<int64_t>(batch_size)) {
    throw std::runtime_error("Unexpected output shape");
  }

  // Create output matrix (batch_size x descriptor_dim)
  cv::Mat descriptors(static_cast<int>(batch_size), static_cast<int>(output_shape[1]), CV_32F);
  std::memcpy(descriptors.data, output_data, batch_size * output_shape[1] * sizeof(float));

  // Normalize each descriptor if requested
  if (normalize_output_) {
    for (int i = 0; i < descriptors.rows; ++i) {
      cv::Mat descriptor_row = descriptors.row(i);
      normalize_descriptor(descriptor_row);
    }
  }

  return descriptors;
}

}  // namespace xfeat
