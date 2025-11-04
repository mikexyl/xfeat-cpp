#include "xfeat-cpp/netvlad_onnx.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace xfeat {
NetVLADONNX::NetVLADONNX(Ort::Env& env, const std::string& model_path, bool use_gpu, size_t height, size_t width)
    : session_options_(), session_(nullptr), height_(height), width_(width), input_size_(256 * height * width) {
  if (use_gpu) {
    OrtCUDAProviderOptions cuda_options{};
    session_options_.AppendExecutionProvider_CUDA(cuda_options);
  }
  session_ = Ort::Session(env, model_path.c_str(), session_options_);
}

std::vector<std::vector<float>> NetVLADONNX::infer(const std::vector<float>& input, size_t batch_size) {
  std::vector<int64_t> input_shape = {
      static_cast<int64_t>(batch_size), 256, static_cast<int64_t>(height_), static_cast<int64_t>(width_)};
  Ort::AllocatorWithDefaultOptions allocator;
  // Use GetInputNameAllocated and GetOutputNameAllocated
  auto input_name_alloc = session_.GetInputNameAllocated(0, allocator);
  auto output_name_alloc = session_.GetOutputNameAllocated(0, allocator);
  const char* input_name = input_name_alloc.get();
  const char* output_name = output_name_alloc.get();

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float*>(input.data()), input.size(), input_shape.data(), input_shape.size());

  auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);

  float* output_data = output_tensors.front().GetTensorMutableData<float>();
  auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
  size_t output_dim = output_shape[1];

  std::vector<std::vector<float>> result(batch_size, std::vector<float>(output_dim));
  for (size_t i = 0; i < batch_size; ++i) {
    std::copy(output_data + i * output_dim, output_data + (i + 1) * output_dim, result[i].begin());
  }
  return result;
}

std::vector<std::vector<float>> NetVLADONNX::infer(const cv::Mat& input) {
  // Expect input to be CV_32F, shape: [batch_size, 256, height, width]
  if (input.type() != CV_32F || input.dims != 4 || input.size[1] != 256 || input.size[2] != static_cast<int>(height_) ||
      input.size[3] != static_cast<int>(width_)) {
    std::string input_shape = std::string("Input shape: [") + std::to_string(input.size[0]) + ", " +
                              std::to_string(input.size[1]) + ", " + std::to_string(input.size[2]) + ", " +
                              std::to_string(input.size[3]) + "]";
    std::string required_shape = std::string("Required shape: [batch_size, 256, ") + std::to_string(height_) + ", " +
                                 std::to_string(width_) + "]";
    std::string msg = input_shape + "; " + required_shape;
    throw std::invalid_argument(std::string("Input must be CV_32F with shape [batch_size,256,height,width]: ") + msg);
  }
  size_t batch_size = input.size[0];
  std::vector<float> input_vec(input.begin<float>(), input.end<float>());
  return infer(input_vec, batch_size);
}

}  // namespace xfeat
