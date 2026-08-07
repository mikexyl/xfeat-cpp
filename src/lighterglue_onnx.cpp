#include "xfeat-cpp/lighterglue_onnx.h"

#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>

#include <array>
#include <chrono>
#include <iostream>
#include <vector>

namespace xfeat {

LighterGlueOnnx::LighterGlueOnnx(Ort::Env& env, const std::string& model_path, bool use_gpu)
    : session_options_(), session_(nullptr), input_names_(), output_names_() {
  std::cout << "Loading LighterGlue ONNX model from: " << model_path << std::endl;
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
  session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  if (use_gpu) {
    std::cout << "Attempting to use GPU for ONNX Runtime." << std::endl;

    auto available_providers = Ort::GetAvailableProviders();
    bool cuda_available = false;
    for (const auto& provider : available_providers) {
      if (provider == "CUDAExecutionProvider") {
        cuda_available = true;
        break;
      }
    }

    if (!cuda_available) {
      std::cerr << "Error: CUDAExecutionProvider is not available. Terminating." << std::endl;
      throw std::runtime_error("CUDAExecutionProvider not found.");
    }

    // const auto& api = Ort::GetApi();
    // OrtTensorRTProviderOptionsV2* tensorrt_options;
    // Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));

    // // Append the V2 TensorRT provider
    // session_options_.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);

    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    cuda_options.arena_extend_strategy = 0;  // kNextPowerOfTwo - preallocate to avoid fragmentation
    cuda_options.gpu_mem_limit = 1ULL * 1024 * 1024 * 1024;  // Limit to 1GB per instance to leave room for vilib
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
    cuda_options.do_copy_in_default_stream = 1;  // Use default stream for multi-process safety
    session_options_.AppendExecutionProvider_CUDA(cuda_options);
  }

  session_ = Ort::Session(env, model_path.c_str(), session_options_);
  input_names_ = {"mkpts0", "feats0", "image0_size", "mkpts1", "feats1", "image1_size"};
  output_names_ = {"matches", "scores"};
}

void LighterGlueOnnx::run(const std::vector<float>& mkpts0,
                          const std::vector<float>& feats0,
                          const std::array<float, 2>& image0_size,
                          const std::vector<float>& mkpts1,
                          const std::vector<float>& feats1,
                          const std::array<float, 2>& image1_size,
                          std::vector<std::array<int64_t, 2>>& matches,
                          std::vector<float>& scores) {
  // Clear any pre-existing CUDA errors from other libraries (for example vilib)
  cudaError_t prev_err = cudaGetLastError();
  if (prev_err != cudaSuccess) {
    std::cerr << "WARNING: Pre-existing CUDA error before LighterGlue: " 
              << cudaGetErrorString(prev_err) << " (error " << prev_err << ")" << std::endl;
    std::cerr << "Attempting to continue after clearing error state..." << std::endl;
  }
  
  // Now synchronize to ensure all previous operations complete
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    std::cerr << "ERROR: CUDA sync failed before LighterGlue: " 
              << cudaGetErrorString(sync_err) << " (error " << sync_err << ")" << std::endl;
    cudaGetLastError();  // Clear it again
    // Don't throw - try to continue
  }
  
  Ort::AllocatorWithDefaultOptions allocator;

  // 1) Derive N from BOTH sources and validate
  const int64_t n0_kp = static_cast<int64_t>(mkpts0.size() / 2);
  const int64_t n0_feat = static_cast<int64_t>(feats0.size() / 64);
  const int64_t n1_kp = static_cast<int64_t>(mkpts1.size() / 2);
  const int64_t n1_feat = static_cast<int64_t>(feats1.size() / 64);

  if (n0_kp != n0_feat) {
    std::string msg = "mkpts0 vs feats0 count mismatch: " + std::to_string(n0_kp) + " vs " + std::to_string(n0_feat);
    throw std::runtime_error(msg);
  }
  if (n1_kp != n1_feat) {
    std::string msg = "mkpts1 vs feats1 count mismatch: " + std::to_string(n1_kp) + " vs " + std::to_string(n1_feat);
    throw std::runtime_error(msg);
  }

  const int64_t n0 = n0_kp;
  const int64_t n1 = n1_kp;

  // 2) Build shapes from the validated counts
  std::array<int64_t, 3> dims_kp0 = {1, n0, 2};
  std::array<int64_t, 3> dims_feat0 = {1, n0, 64};
  std::array<int64_t, 3> dims_kp1 = {1, n1, 2};
  std::array<int64_t, 3> dims_feat1 = {1, n1, 64};
  std::array<int64_t, 1> dims_size = {2};  // use the same for both images

  // Create Ort tensors for inputs
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  Ort::Value in_mkpts0 =
      Ort::Value::CreateTensor<float>(mem, const_cast<float*>(mkpts0.data()), mkpts0.size(), dims_kp0.data(), 3);
  Ort::Value in_feats0 =
      Ort::Value::CreateTensor<float>(mem, const_cast<float*>(feats0.data()), feats0.size(), dims_feat0.data(), 3);
  Ort::Value in_img0 =
      Ort::Value::CreateTensor<float>(mem, const_cast<float*>(image0_size.data()), 2, dims_size.data(), 1);

  Ort::Value in_mkpts1 =
      Ort::Value::CreateTensor<float>(mem, const_cast<float*>(mkpts1.data()), mkpts1.size(), dims_kp1.data(), 3);
  Ort::Value in_feats1 =
      Ort::Value::CreateTensor<float>(mem, const_cast<float*>(feats1.data()), feats1.size(), dims_feat1.data(), 3);
  Ort::Value in_img1 =
      Ort::Value::CreateTensor<float>(mem, const_cast<float*>(image1_size.data()), 2, dims_size.data(), 1);

  // Bundle inputs
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.reserve(6);
  ort_inputs.emplace_back(std::move(in_mkpts0));
  ort_inputs.emplace_back(std::move(in_feats0));
  ort_inputs.emplace_back(std::move(in_img0));
  ort_inputs.emplace_back(std::move(in_mkpts1));
  ort_inputs.emplace_back(std::move(in_feats1));
  ort_inputs.emplace_back(std::move(in_img1));

  auto ro = Ort::RunOptions();
  ro.SetRunLogSeverityLevel(0);
  ro.SetRunLogVerbosityLevel(1);
  // Run inference
  auto output_tensors = session_.Run(
      ro, input_names_.data(), ort_inputs.data(), ort_inputs.size(), output_names_.data(), output_names_.size());

  // Extract matches
  auto& out_matches = output_tensors[0];
  int64_t* match_data = out_matches.GetTensorMutableData<int64_t>();
  auto match_info = out_matches.GetTensorTypeAndShapeInfo();
  auto match_shape = match_info.GetShape();
  int64_t num_matches = match_shape[0];
  matches.resize(num_matches);
  for (int64_t i = 0; i < num_matches; ++i) {
    matches[i] = {match_data[2 * i], match_data[2 * i + 1]};
  }

  // Extract scores
  auto& out_scores = output_tensors[1];
  float* score_data = out_scores.GetTensorMutableData<float>();
  auto score_info = out_scores.GetTensorTypeAndShapeInfo();
  auto score_shape = score_info.GetShape();
  int64_t num_scores = score_shape[0];
  scores.resize(num_scores);
  for (int64_t i = 0; i < num_scores; ++i) {
    scores[i] = score_data[i];
  }
}

}  // namespace xfeat
