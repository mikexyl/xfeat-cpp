#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>

#include "xfeat-cpp/lighterglue_onnx.h"
#include "xfeat-cpp/types.h"

namespace xfeat {

class LighterGlueCV {
 public:
  struct Params {
    std::string model_path;
    bool use_gpu = false;
    float min_score = 0.5f;
    int n_kpts = 500;  // Default number of keypoints to detect
  };

  explicit LighterGlueCV(Ort::Env& env, const Params& params)
      : env_(env), matcher_(env, params.model_path, params.use_gpu), params_(params), min_score_(params.min_score) {}

  static cv::Ptr<LighterGlueCV> create(Ort::Env& env, const Params& params) {
    return cv::Ptr<LighterGlueCV>(new LighterGlueCV(env, params));
  }

  // OpenCV-style: match keypoints and descriptors from two images
  void match(const DetectionResult& det0,
             const cv::Size& image0_size,
             const DetectionResult& det1,
             const cv::Size& image1_size,
             std::vector<cv::DMatch>& matches) /* not const */ {
    if (det0.keypoints.empty() || det1.keypoints.empty()) {
      matches.clear();
      return;  // No keypoints to match
    }
    if (det0.descriptors.empty() || det1.descriptors.empty()) {
      throw std::runtime_error("Descriptors are empty in one of the detection results.");
    }
    if (det0.keypoints.rows != det0.descriptors.rows || det1.keypoints.rows != det1.descriptors.rows) {
      throw std::runtime_error("Keypoints and descriptors row count mismatch.");
    }
    // check if number of keypoints EQUALS n_kpts
    if (det0.keypoints.rows != params_.n_kpts || det1.keypoints.rows != params_.n_kpts) {
      throw std::runtime_error("Number of keypoints does not match the expected n_kpts.");
    }

    std::array<float, 2> size0 = {static_cast<float>(image0_size.width), static_cast<float>(image0_size.height)};
    std::array<float, 2> size1 = {static_cast<float>(image1_size.width), static_cast<float>(image1_size.height)};
    auto result = matcher_.match(det0, size0, det1, size1, min_score_);
    const auto& idx0 = std::get<0>(result);
    const auto& idx1 = std::get<1>(result);
    matches.clear();
    for (size_t i = 0; i < idx0.size(); ++i) {
      matches.emplace_back(idx0[i], idx1[i], 0.f);  // Score not available in DMatch
    }
  }

  void knnMatch(const cv::Mat& curr_desc,
                const cv::Mat& ref_desc,
                std::vector<std::vector<cv::DMatch>>& matches,
                int k) {
    throw std::runtime_error("knnMatch is not implemented");
  }

 public:
  Params params_;

 private:
  Ort::Env& env_;
  LighterGlueOnnx matcher_;
  float min_score_;
};

}  // namespace xfeat
