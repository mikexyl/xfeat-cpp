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
    bool use_gpu = true;
    float min_score = -1.f;
    int n_kpts = 500;  // Default number of keypoints to detect
  };

  explicit LighterGlueCV(Ort::Env& env, const Params& params)
      : matcher_(env, params.model_path, params.use_gpu), params_(params), min_score_(params.min_score) {}

  static cv::Ptr<LighterGlueCV> create(Ort::Env& env, const Params& params) {
    return cv::Ptr<LighterGlueCV>(new LighterGlueCV(env, params));
  }

  // OpenCV-style: match keypoints and descriptors from two images
  void match(const DetectionResult& query_det,
             const cv::Size& image0_size,
             const DetectionResult& train_det,
             const cv::Size& image1_size,
             std::vector<cv::DMatch>& matches) /* not const */ {
    if (query_det.keypoints.empty() || train_det.keypoints.empty()) {
      matches.clear();
      return;  // No keypoints to match
    }
    if (query_det.descriptors.empty() || train_det.descriptors.empty()) {
      throw std::runtime_error("Descriptors are empty in one of the detection results.");
    }
    if (query_det.keypoints.rows != query_det.descriptors.rows ||
        train_det.keypoints.rows != train_det.descriptors.rows) {
      throw std::runtime_error("Keypoints and descriptors row count mismatch.");
    }
    // check if number of keypoints EQUALS n_kpts
    if (query_det.keypoints.rows != params_.n_kpts || train_det.keypoints.rows != params_.n_kpts) {
      throw std::runtime_error("Number of keypoints does not match the expected n_kpts.");
    }

    std::array<float, 2> size0 = {static_cast<float>(image0_size.width), static_cast<float>(image0_size.height)};
    std::array<float, 2> size1 = {static_cast<float>(image1_size.width), static_cast<float>(image1_size.height)};
    auto indexes = matcher_.match(query_det, size0, train_det, size1, min_score_);
    matches.clear();
    for (size_t i = 0; i < indexes.size(); ++i) {
      if (indexes[i].empty()) continue;                       // No matches for this keypoint
      matches.emplace_back(cv::DMatch(i, indexes[i][0], 0));  // Use first match only
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
  LighterGlueOnnx matcher_;
  float min_score_;
};

}  // namespace xfeat
