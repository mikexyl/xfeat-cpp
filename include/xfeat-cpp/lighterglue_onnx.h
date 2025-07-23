#pragma once

#include <onnxruntime_cxx_api.h>

#include <array>
#include <iostream>
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
           std::vector<float>& scores,
           int num_feat = 500);

  // Convenience: run and return matches/scores as output
  std::pair<std::vector<std::array<int64_t, 2>>, std::vector<float>> match(const std::vector<float>& mkpts0,
                                                                           const std::vector<float>& feats0,
                                                                           const std::array<float, 2>& image0_size,
                                                                           const std::vector<float>& mkpts1,
                                                                           const std::vector<float>& feats1,
                                                                           const std::array<float, 2>& image1_size) {
    std::vector<std::array<int64_t, 2>> matches;
    std::vector<float> scores;
    run(mkpts0, feats0, image0_size, mkpts1, feats1, image1_size, matches, scores, mkpts0.size() / 2);
    return {matches, scores};
  }

  // Overload: match using DetectionResult for each image
  std::vector<std::vector<int>> match(const DetectionResult& det0,
                                      const std::array<float, 2>& image0_size,
                                      const DetectionResult& det1,
                                      const std::array<float, 2>& image1_size,
                                      float min_score = 0.5) {
    // Assume det0.keypoints: CV_32FC2, det0.descriptors: CV_32FC1 or CV_32FC64
    std::vector<float> mkpts0, feats0, mkpts1, feats1;
    // Flatten keypoints and descriptors
    mkpts0.assign((float*)det0.keypoints.datastart, (float*)det0.keypoints.dataend);
    feats0.assign((float*)det0.descriptors.datastart, (float*)det0.descriptors.dataend);
    mkpts1.assign((float*)det1.keypoints.datastart, (float*)det1.keypoints.dataend);
    feats1.assign((float*)det1.descriptors.datastart, (float*)det1.descriptors.dataend);

    auto [matches, scores] = match(mkpts0, feats0, image0_size, mkpts1, feats1, image1_size);
    std::vector<std::vector<int>> idx(det0.keypoints.rows, std::vector<int>{});
    for (size_t i = 0; i < matches.size(); ++i) {
      if (min_score >= 0 && scores[i] < min_score) continue;  // Filter by score
      int idx0 = static_cast<int>(matches[i][0]);
      int idx1 = static_cast<int>(matches[i][1]);
      if (idx0 >= 0 && idx0 < det0.keypoints.rows && idx1 >= 0 && idx1 < det1.keypoints.rows) {
        idx[idx0].push_back(idx1);
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