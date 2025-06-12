#pragma once

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

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
  LighterGlueOnnx(const std::string& model_path, bool use_gpu = false);

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
    // print matches
    std::cout << "LighterGlue matches: " << matches.size() << std::endl;
    for (const auto& match : matches) {
      std::cout << "Match: (" << match[0] << ", " << match[1] << ")" << std::endl;
    }
    return {matches, scores};
  }

  // Overload: match using DetectionResult for each image
  std::tuple<std::vector<int>, std::vector<int>> match(const DetectionResult& det0,
                                                       const std::array<float, 2>& image0_size,
                                                       const DetectionResult& det1,
                                                       const std::array<float, 2>& image1_size) {
    std::cout << "LighterGlueOnnx::match called with DetectionResult" << std::endl;
    // Assume det0.keypoints: CV_32FC2, det0.descriptors: CV_32FC1 or CV_32FC64
    std::vector<float> mkpts0, feats0, mkpts1, feats1;
    // Flatten keypoints and descriptors
    mkpts0.assign((float*)det0.keypoints.datastart, (float*)det0.keypoints.dataend);
    feats0.assign((float*)det0.descriptors.datastart, (float*)det0.descriptors.dataend);
    mkpts1.assign((float*)det1.keypoints.datastart, (float*)det1.keypoints.dataend);
    feats1.assign((float*)det1.descriptors.datastart, (float*)det1.descriptors.dataend);

    auto [matches, scores] = match(mkpts0, feats0, image0_size, mkpts1, feats1, image1_size);
    std::vector<int> idx0(matches.size()), idx1(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
      idx0[i] = matches[i][0];
      idx1[i] = matches[i][1];
    }
    return {idx0, idx1};
  }

 private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  // Input and output names
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
};

}  // namespace xfeat