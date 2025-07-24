#pragma once

#include <cublas_v2.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <vector>

#include "xfeat-cpp/helpers.h"
#include "xfeat-cpp/types.h"

namespace xfeat {

// ------------------------------------------------ Matcher class (same as
// before)
class CuMatcher {
 public:
  CuMatcher() = default;
  ~CuMatcher() { destroy(); }

  void init(int maxN1, int maxN2, int D);

  void destroy();

  std::vector<cv::DMatch> match(DetectionResult& result1,
                                DetectionResult& result2,
                                float min_sim,
                                cv::Mat H,
                                int search_radius) {
    std::vector<cv::Point2f> keypoints1, keypoints2;
    for (int i = 0; i < result1.keypoints.rows; ++i) {
      keypoints1.emplace_back(result1.keypoints.at<float>(i, 0), result1.keypoints.at<float>(i, 1));
    }
    for (int i = 0; i < result2.keypoints.rows; ++i) {
      keypoints2.emplace_back(result2.keypoints.at<float>(i, 0), result2.keypoints.at<float>(i, 1));
    }

    std::vector<cv::Point2f> kpts1_warped;
    if (not H.empty()) {
      cv::perspectiveTransform(keypoints1, kpts1_warped, H);
    } else {
      kpts1_warped = keypoints1;  // no warp, use original
    }

    auto [indexes1, indexes2] = this->match_mkpts_local(
        result1.descriptors, result2.descriptors, kpts1_warped, keypoints2, search_radius, min_sim);

    size_t num_matched = indexes1.size();

    cv::Mat mkpts1(static_cast<int>(num_matched), 2, CV_32F);
    cv::Mat mkpts2(static_cast<int>(num_matched), 2, CV_32F);

    std::vector<int> matched_indices1;
    std::vector<int> matched_indices2;
    matched_indices1.reserve(num_matched);
    matched_indices2.reserve(num_matched);

    // --- fill them using a ‘row’ index that only counts valid matches ----------
    size_t row = 0;
    for (size_t i = 0; i < indexes1.size(); ++i) {
      int index_i = indexes1[i];
      int index_j = indexes2[i];
      if (index_j < 0 || index_j >= result2.keypoints.rows) continue;

      mkpts1.at<float>(row, 0) = result1.keypoints.at<float>(index_i, 0);
      mkpts1.at<float>(row, 1) = result1.keypoints.at<float>(index_i, 1);
      mkpts2.at<float>(row, 0) = result2.keypoints.at<float>(index_j, 0);
      mkpts2.at<float>(row, 1) = result2.keypoints.at<float>(index_j, 1);

      matched_indices1.push_back(index_i);
      matched_indices2.push_back(index_j);
      ++row;  // advance only on success
    }

    // (optional) shrink if we skipped any rows because ‘j’ was invalid
    if (row != num_matched) {
      mkpts1 = mkpts1.rowRange(0, static_cast<int>(row)).clone();
      mkpts2 = mkpts2.rowRange(0, static_cast<int>(row)).clone();
      matched_indices1.resize(row);
      matched_indices2.resize(row);
    }

    // Filter matches using homography (RANSAC)
    cv::Mat new_H;
    auto inliers = calc_warp_corners_and_matches(mkpts1, mkpts2, &H);

    std::vector<int> inlier_indices1, inlier_indices2;
    for (size_t i = 0; i < inliers.size(); ++i) {
      if (inliers[i] > 0) {  // Inlier
        inlier_indices1.push_back(matched_indices1[i]);
        inlier_indices2.push_back(matched_indices2[i]);
      }
    }

    std::vector<cv::DMatch> matches;
    if (not H.empty() and inlier_indices1.size() < result1.keypoints.rows * 0.8) {
      std::vector<cv::Point2f> kpts1_warped;
      cv::perspectiveTransform(keypoints1, kpts1_warped, H);
      auto [rematch_id1, rematch_id2] = this->match_mkpts_local(
          result1.descriptors, result2.descriptors, kpts1_warped, keypoints2, search_radius * 0.05, min_sim);

      for (int i = 0; i < rematch_id1.size(); ++i) {
        if (rematch_id1[i] >= 0 && rematch_id2[i] >= 0) {
          matches.emplace_back(rematch_id1[i], rematch_id2[i], 0.0f);
        }
      }
    } else {
      // populate matches with inliers
      for (size_t i = 0; i < inlier_indices1.size(); ++i) {
        matches.emplace_back(inlier_indices1[i], inlier_indices2[i],
                             0.0f);  // Assuming distance is not used here
      }
    }
    return matches;
  }

  std::vector<std::vector<int>> match_mkpts(const cv::Mat& desc1, const cv::Mat& desc2, float min_cossim);

  std::tuple<std::vector<int>, std::vector<int>> match_mkpts_local(const cv::Mat& desc1,
                                                                   const cv::Mat& desc2,
                                                                   const std::vector<cv::Point2f>& kp1,
                                                                   const std::vector<cv::Point2f>& kp2,
                                                                   float search_radius,
                                                                   float min_cossim = 0.f);

 private:
  float *d_scores = nullptr, *d_A = nullptr, *d_B = nullptr;
  float2 *d_pred = nullptr, *d_kp2 = nullptr;
  float4* d_param = nullptr;
  int* d_bestIdx = nullptr;
  float* d_bestScore = nullptr;
  int* d_bestIdxRow = nullptr;      // N1
  float* d_bestScoreRow = nullptr;  // N1
  int* d_bestIdxCol = nullptr;      // N2
  float* d_bestScoreCol = nullptr;  // N2

  int max_N1{}, max_N2{}, max_D{};
  cublasHandle_t handle = nullptr;
  static void CUBLAS_CHECK(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "cuBLAS error " << s << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
};

}  // namespace xfeat