#include "xfeat-cpp/xfeat_netvlad_onnx.h"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

namespace xfeat {

HeadNetVLADONNX::HeadNetVLADONNX(Ort::Env& env, const std::string& model_path, bool use_gpu)
    : session_options_(), session_(nullptr) {
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
  session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  if (use_gpu) {
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    cuda_options.arena_extend_strategy = 0;  // kNextPowerOfTwo - preallocate to avoid fragmentation
    cuda_options.gpu_mem_limit = 1ULL * 1024 * 1024 * 1024;  // Limit to 1GB per instance
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;  // Use default, not exhaustive
    cuda_options.do_copy_in_default_stream = 1;  // Use default stream for multi-process safety
    session_options_.AppendExecutionProvider_CUDA(cuda_options);
  }
  session_ = Ort::Session(env, model_path.c_str(), session_options_);

  // Set input/output names
  input_names_ = {"input", "input.1"};
  output_names_ = {"output"};
}

cv::Mat HeadNetVLADONNX::run(const cv::Mat& M1, const cv::Mat& x_prep) {
  // Synchronize CUDA to ensure all previous operations complete
  // This is critical when mixing CUDA libraries (vilib + ONNX Runtime)
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    std::cerr << "CUDA synchronization error before HeadNetVLAD: " 
              << cudaGetErrorString(sync_err) << " (error " << sync_err << ")" << std::endl;
    // Clear the error state
    cudaGetLastError();
  }
  
  // Check input types
  if (M1.type() != CV_32F || x_prep.type() != CV_32F) {
    std::cerr << "M1 type: " << M1.type() << ", x_prep type: " << x_prep.type() << std::endl;
    std::cerr << "M1 size: " << M1.size() << ", x_prep size: " << x_prep.size() << std::endl;
    throw std::invalid_argument("Inputs must be CV_32F (float32)");
  }

  // Ensure inputs are continuous and cloned to avoid memory issues
  cv::Mat M1_continuous = M1.isContinuous() ? M1.clone() : M1.clone();
  cv::Mat x_prep_continuous = x_prep.isContinuous() ? x_prep.clone() : x_prep.clone();

  // Reshape M1 from [64,60,80] to [1,64,60,80] if needed
  cv::Mat M1_reshaped;
  if (M1_continuous.dims == 3 && M1_continuous.size[0] == 64 && M1_continuous.size[1] == 60 && M1_continuous.size[2] == 80) {
    int sizes[4] = {1, 64, 60, 80};
    M1_reshaped = cv::Mat(4, sizes, CV_32F, M1_continuous.data).clone();
  } else {
    M1_reshaped = M1_continuous.clone();
  }

  // Prepare input shapes
  std::vector<int64_t> input_shape = {
      M1_reshaped.size[0], M1_reshaped.size[1], M1_reshaped.size[2], M1_reshaped.size[3]};
  std::vector<int64_t> input1_shape = {x_prep_continuous.size[0], x_prep_continuous.size[1], x_prep_continuous.size[2], x_prep_continuous.size[3]};

  // Create Ort tensors with owned memory
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem_info, (float*)M1_reshaped.data, M1_reshaped.total(), input_shape.data(), input_shape.size());
  Ort::Value input1_tensor = Ort::Value::CreateTensor<float>(
      mem_info, (float*)x_prep_continuous.data, x_prep_continuous.total(), input1_shape.data(), input1_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_tensor));
  ort_inputs.push_back(std::move(input1_tensor));

  // Run inference
  auto output_tensors = session_.Run(
      Ort::RunOptions{nullptr}, input_names_.data(), ort_inputs.data(), ort_inputs.size(), output_names_.data(), 1);

  // Get output tensor
  float* output_data = output_tensors[0].GetTensorMutableData<float>();
  auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

  if (output_shape.size() == 0 || output_data == nullptr) {
    std::cerr << "ONNX output is empty or invalid!" << std::endl;
    return cv::Mat();
  }

  // Convert output_shape to int for cv::Mat
  std::vector<int> shape_int(output_shape.begin(), output_shape.end());
  int dims = shape_int.size();
  cv::Mat output(dims, shape_int.data(), CV_32F, output_data);
  return output.clone();  // clone to own the data
}

}  // namespace xfeat
