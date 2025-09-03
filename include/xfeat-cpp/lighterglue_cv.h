#pragma once

#include <algorithm>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <tuple>

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

  void warmup() {
    // create a dummy input to warm up the model
    cv::Mat random_kpts0(params_.n_kpts, 2, CV_32F);
    cv::randu(random_kpts0, 0, 100);  // Random keypoints
    cv::Mat random_desc0(params_.n_kpts, 128, CV_32F);
    cv::randu(random_desc0, 0, 255);  // Random descriptors
    cv::Mat random_kpts1(params_.n_kpts, 2, CV_32F);
    cv::randu(random_kpts1, 0, 100);  // Random keypoints
    cv::Mat random_desc1(params_.n_kpts, 128, CV_32F);
    cv::randu(random_desc1, 0, 255);
    DetectionResult dummy_det0{random_kpts0, {}, random_desc0};
    DetectionResult dummy_det1{random_kpts1, {}, random_desc1};
    std::vector<cv::DMatch> dummy_matches;
    match(dummy_det0, cv::Size(640, 480), dummy_det1, cv::Size(640, 480), dummy_matches);
  }

  // OpenCV-style: match keypoints and descriptors from two images
  void match(DetectionResult& query_det,
             const cv::Size& image0_size,
             DetectionResult& train_det,
             const cv::Size& image1_size,
             std::vector<cv::DMatch>& matches) /* not const */ {
    if (query_det.keypoints.empty() || train_det.keypoints.empty()) {
      matches.clear();
      return;  // No keypoints to match
    }
    if (query_det.descriptors.empty() || train_det.descriptors.empty()) {
      throw std::runtime_error("Descriptors are empty in one of the detection results.");
    }

    if (query_det.keypoints.rows != query_det.scores.rows) {
      throw std::runtime_error("Query keypoints and scores size mismatch: " + std::to_string(query_det.keypoints.rows) +
                               " vs " + std::to_string(query_det.scores.rows));
    }
    if (train_det.keypoints.rows != train_det.scores.rows) {
      throw std::runtime_error("Train keypoints and scores size mismatch: " + std::to_string(train_det.keypoints.rows) +
                               " vs " + std::to_string(train_det.scores.rows));
    }

    if (query_det.keypoints.rows != query_det.descriptors.rows ||
        train_det.keypoints.rows != train_det.descriptors.rows) {
      throw std::runtime_error("Keypoints and descriptors row count mismatch.");
    }
    std::vector<int> query_resampled_ids, train_resampled_ids;
    // check if number of keypoints EQUALS n_kpts
    if (query_det.keypoints.rows != params_.n_kpts || train_det.keypoints.rows != params_.n_kpts) {
      // if there are more keypoints than we need, take the highest scores top k points
      if (query_det.keypoints.rows > params_.n_kpts) {
        auto [resampled_kpts, resampled_scores, resampled_desc, original_ids] =
            resampleTopK(query_det.keypoints, query_det.scores, query_det.descriptors, params_.n_kpts);
        query_det.keypoints = resampled_kpts;
        query_det.scores = resampled_scores;
        query_det.descriptors = resampled_desc;
        query_resampled_ids = original_ids;
      }

      if (train_det.keypoints.rows > params_.n_kpts) {
        auto [resampled_kpts, resampled_scores, resampled_desc, original_ids] =
            resampleTopK(train_det.keypoints, train_det.scores, train_det.descriptors, params_.n_kpts);
        train_det.keypoints = resampled_kpts;
        train_det.scores = resampled_scores;
        train_det.descriptors = resampled_desc;
        train_resampled_ids = original_ids;
      }
    }

    std::array<float, 2> size0 = {static_cast<float>(image0_size.width), static_cast<float>(image0_size.height)};
    std::array<float, 2> size1 = {static_cast<float>(image1_size.width), static_cast<float>(image1_size.height)};
    auto indexes = matcher_.match(query_det, size0, train_det, size1, min_score_);
    matches.clear();
    for (size_t i = 0; i < indexes.size(); ++i) {
      if (indexes[i].empty()) continue;  // No matches for this keypoint

      // Map back to original indices if resampling was performed
      int query_idx = query_resampled_ids.empty() ? static_cast<int>(i) : query_resampled_ids[i];
      int train_idx = train_resampled_ids.empty() ? indexes[i][0] : train_resampled_ids[indexes[i][0]];

      matches.emplace_back(cv::DMatch(query_idx, train_idx, 0));  // Use first match only
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

  // Helper function to resample top K keypoints based on scores
  std::tuple<cv::Mat, cv::Mat, cv::Mat, std::vector<int>> resampleTopK(const cv::Mat& keypoints,
                                                                       const cv::Mat& scores,
                                                                       const cv::Mat& descriptors,
                                                                       int k) {
    if (keypoints.rows <= k) {
      throw std::runtime_error("Not enough keypoints to resample");
    }

    // Create pairs of (score, index) for sorting
    std::vector<std::pair<float, int>> score_index_pairs;
    score_index_pairs.reserve(scores.rows);

    for (int i = 0; i < scores.rows; ++i) {
      score_index_pairs.emplace_back(scores.at<float>(i), i);
    }

    // Sort by score in descending order (highest scores first)
    std::sort(score_index_pairs.begin(),
              score_index_pairs.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });

    // Extract top k indices
    std::vector<int> top_k_indices;
    top_k_indices.reserve(k);
    for (int i = 0; i < k; ++i) {
      top_k_indices.push_back(score_index_pairs[i].second);
    }

    // Create new matrices with top k elements
    cv::Mat new_keypoints(k, keypoints.cols, keypoints.type());
    cv::Mat new_scores(k, scores.cols, scores.type());
    cv::Mat new_descriptors(k, descriptors.cols, descriptors.type());

    for (int i = 0; i < k; ++i) {
      int original_idx = top_k_indices[i];
      keypoints.row(original_idx).copyTo(new_keypoints.row(i));
      scores.row(original_idx).copyTo(new_scores.row(i));
      descriptors.row(original_idx).copyTo(new_descriptors.row(i));
    }

    return {new_keypoints, new_scores, new_descriptors, top_k_indices};
  }
};

}  // namespace xfeat
