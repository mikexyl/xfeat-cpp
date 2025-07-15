#include "xfeat-cpp/lighterglue_onnx.h"

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <array>
#include <iostream>
#include <vector>

namespace xfeat {

LighterGlueOnnx::LighterGlueOnnx(Ort::Env& env, const std::string& model_path, bool use_gpu)
    : session_options_(),
      session_(nullptr),
      input_names_(),
      output_names_() {
  std::cout << "Loading LighterGlue ONNX model from: " << model_path << std::endl;
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
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

    OrtCUDAProviderOptions cuda_options{};
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
                          std::vector<float>& scores,
                          int num_feat) {
  Ort::AllocatorWithDefaultOptions allocator;

  // Define tensor shapes
  std::array<int64_t, 3> dims_kp = {1, num_feat, 2};
  std::array<int64_t, 3> dims_feat = {1, num_feat, 64};
  std::array<int64_t, 1> dims_size = {2};

  // Create Ort tensors for inputs
  Ort::Value in_mkpts0 = Ort::Value::CreateTensor<float>(
      allocator.GetInfo(), const_cast<float*>(mkpts0.data()), mkpts0.size(), dims_kp.data(), dims_kp.size());
  Ort::Value in_feats0 = Ort::Value::CreateTensor<float>(
      allocator.GetInfo(), const_cast<float*>(feats0.data()), feats0.size(), dims_feat.data(), dims_feat.size());
  Ort::Value in_img0 = Ort::Value::CreateTensor<float>(allocator.GetInfo(),
                                                       const_cast<float*>(image0_size.data()),
                                                       image0_size.size(),
                                                       dims_size.data(),
                                                       dims_size.size());
  Ort::Value in_mkpts1 = Ort::Value::CreateTensor<float>(
      allocator.GetInfo(), const_cast<float*>(mkpts1.data()), mkpts1.size(), dims_kp.data(), dims_kp.size());
  Ort::Value in_feats1 = Ort::Value::CreateTensor<float>(
      allocator.GetInfo(), const_cast<float*>(feats1.data()), feats1.size(), dims_feat.data(), dims_feat.size());
  Ort::Value in_img1 = Ort::Value::CreateTensor<float>(allocator.GetInfo(),
                                                       const_cast<float*>(image1_size.data()),
                                                       image1_size.size(),
                                                       dims_size.data(),
                                                       dims_size.size());

  // Bundle inputs
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.reserve(6);
  ort_inputs.emplace_back(std::move(in_mkpts0));
  ort_inputs.emplace_back(std::move(in_feats0));
  ort_inputs.emplace_back(std::move(in_img0));
  ort_inputs.emplace_back(std::move(in_mkpts1));
  ort_inputs.emplace_back(std::move(in_feats1));
  ort_inputs.emplace_back(std::move(in_img1));

  // Run inference
  auto output_tensors = session_.Run(Ort::RunOptions{nullptr},
                                     input_names_.data(),
                                     ort_inputs.data(),
                                     ort_inputs.size(),
                                     output_names_.data(),
                                     output_names_.size());
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