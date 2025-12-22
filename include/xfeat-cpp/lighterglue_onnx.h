#pragma once

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "xfeat-cpp/types.h"

namespace xfeat {

class LighterGlueOnnx {
 public:
  // Load the ONNX model from file, optionally enable GPU
  LighterGlueOnnx(Ort::Env& env, const std::string& model_path, bool use_gpu = false);

  // Run inference: inputs are flattened row-major float arrays and 2-element image sizes
  void run(const std::vector<float>& mkpts0,
           const std::vector<float>& feats0,
           const std::array<float, 2>& image0_size,
           const std::vector<float>& mkpts1,
           const std::vector<float>& feats1,
           const std::array<float, 2>& image1_size,
           std::vector<std::array<int64_t, 2>>& matches,
           std::vector<float>& scores);

  // Convenience: run and return matches/scores as output
  std::pair<std::vector<std::array<int64_t, 2>>, std::vector<float>> match(const std::vector<float>& mkpts0,
                                                                           const std::vector<float>& feats0,
                                                                           const std::array<float, 2>& image0_size,
                                                                           const std::vector<float>& mkpts1,
                                                                           const std::vector<float>& feats1,
                                                                           const std::array<float, 2>& image1_size) {
    std::vector<std::array<int64_t, 2>> matches;
    std::vector<float> scores;
    run(mkpts0, feats0, image0_size, mkpts1, feats1, image1_size, matches, scores);
    return {matches, scores};
  }

  // Overload: match using DetectionResult for each image
  std::vector<std::vector<int>> match(const DetectionResult& det0,
                                      const std::array<float, 2>& image0_size,
                                      const DetectionResult& det1,
                                      const std::array<float, 2>& image1_size,
                                      float min_score = -1,
                                      std::vector<float>* scores_out = nullptr) {
    if (scores_out) {
      scores_out->assign(det0.keypoints.rows, std::numeric_limits<float>::lowest());
    }
    
    // Validate inputs
    if (det0.keypoints.empty() || det1.keypoints.empty()) {
      std::cerr << "Error: Empty keypoints in LighterGlueOnnx::match" << std::endl;
      return std::vector<std::vector<int>>(det0.keypoints.rows, std::vector<int>{});
    }
    if (det0.descriptors.empty() || det1.descriptors.empty()) {
      std::cerr << "Error: Empty descriptors in LighterGlueOnnx::match" << std::endl;
      return std::vector<std::vector<int>>(det0.keypoints.rows, std::vector<int>{});
    }
    
    // Assume det0.keypoints: CV_32FC2, det0.descriptors: CV_32FC1 or CV_32FC64
    std::vector<float> mkpts0, feats0, mkpts1, feats1;
    
    // Ensure continuous memory and flatten keypoints
    cv::Mat kpts0_continuous = det0.keypoints.isContinuous() ? det0.keypoints : det0.keypoints.clone();
    cv::Mat kpts1_continuous = det1.keypoints.isContinuous() ? det1.keypoints : det1.keypoints.clone();
    
    mkpts0.assign((float*)kpts0_continuous.data, (float*)kpts0_continuous.data + kpts0_continuous.total() * kpts0_continuous.channels());
    mkpts1.assign((float*)kpts1_continuous.data, (float*)kpts1_continuous.data + kpts1_continuous.total() * kpts1_continuous.channels());

    cv::Mat desc0_continuous = det0.descriptors.isContinuous() ? det0.descriptors : det0.descriptors.clone();
    cv::Mat desc1_continuous = det1.descriptors.isContinuous() ? det1.descriptors : det1.descriptors.clone();

    feats0.assign((float*)desc0_continuous.data, (float*)desc0_continuous.data + desc0_continuous.total() * desc0_continuous.channels());
    feats1.assign((float*)desc1_continuous.data, (float*)desc1_continuous.data + desc1_continuous.total() * desc1_continuous.channels());

    auto [matches, scores] = match(mkpts0, feats0, image0_size, mkpts1, feats1, image1_size);
    std::vector<std::vector<int>> idx(det0.keypoints.rows, std::vector<int>{});
    for (size_t i = 0; i < matches.size(); ++i) {
      if (min_score >= 0 && scores[i] < min_score) continue;  // Filter by score
      int idx0 = static_cast<int>(matches[i][0]);
      int idx1 = static_cast<int>(matches[i][1]);
      if (idx0 >= 0 && idx0 < det0.keypoints.rows && idx1 >= 0 && idx1 < det1.keypoints.rows) {
        idx[idx0].push_back(idx1);
        if (scores_out) {
          (*scores_out)[idx0] = std::max((*scores_out)[idx0], scores[i]);
        }
      } else {
        std::cerr << "Warning: match index out of bounds: " << idx0 << ", " << idx1 << std::endl;
      }
    }
    return idx;
  }

 private:
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  // Input and output names
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
};

}  // namespace xfeat
