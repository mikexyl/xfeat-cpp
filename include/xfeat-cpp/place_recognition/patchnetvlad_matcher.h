#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "xfeat-cpp/place_recognition/patchnetvlad_onnx.h"

namespace xfeat {

/**
 * @brief Local patch RANSAC re-ranker for PatchNetVLAD.
 *
 * Ports PatchMatcher.compare_two_ransac() from the Python reference implementation.
 *
 * Algorithm per scale:
 *   1. Compute dot-product matrix between query and DB local descriptors.
 *   2. Forward matching: for each DB patch, find nearest query patch (argmax dot).
 *   3. Backward matching: for each query patch, find nearest DB patch.
 *   4. Mutual-consistency filter.
 *   5. RANSAC homography on the mutual-match point pairs.
 *   6. Inlier ratio → per-scale score.
 *
 * Final score is a weighted sum of per-scale inlier ratios (higher = better match).
 *
 * Receptive field: VGG-16 conv5_3, rf=196, stride=16, padding=90.
 * For 480×640 input the feature map is 30×40.
 */
class PatchNetVLADMatcher {
 public:
  struct Params {
    int img_height = 480;
    int img_width = 640;
    std::vector<int> patch_sizes = {2, 5, 8};    // one per scale
    std::vector<int> strides = {1, 1, 1};         // patch strides (all 1 in default model)
    std::vector<float> patch_weights = {0.45f, 0.15f, 0.40f};  // weighted score combination
  };

  struct MatchResult {
    float score;                     // weighted inlier ratio (higher = better match)
    float global_sim;                // cosine similarity between global descriptors
    std::vector<int> inlier_counts;  // RANSAC inlier count per scale
  };

  PatchNetVLADMatcher();
  explicit PatchNetVLADMatcher(const Params& params);

  /**
   * @brief Re-rank a query-DB pair using local patch RANSAC.
   *
   * @param query Features from the query image (from PatchNetVLADONNX::extract)
   * @param db    Features from the database image
   * @return MatchResult with score, global similarity, and per-scale inlier counts
   */
  MatchResult match(const PatchNetVLADONNX::Features& query, const PatchNetVLADONNX::Features& db) const;

 private:
  Params params_;
  std::vector<cv::Mat> keypoints_;  // precomputed per scale, [2, num_patches] CV_32F (row 0=x, row 1=y)

  /**
   * @brief Compute receptive field boxes for all H×W feature positions.
   *
   * Returns [H*W, 4] CV_32F matrix; each row = [xmin, ymin, xmax, ymax]
   * using VGG-16 conv5_3 constants: rf=196, stride=16, padding=90.
   */
  cv::Mat compute_receptive_boxes(int H, int W) const;

  /**
   * @brief Compute patch center keypoints in image coordinates.
   *
   * Returns [2, Hout*Wout] CV_32F where row 0=x (horizontal), row 1=y (vertical).
   */
  cv::Mat compute_keypoints(int patch_size, int stride) const;

  /**
   * @brief Match one scale via mutual nearest-neighbor + RANSAC homography.
   *
   * @param q_local   Query local descriptors  [D, P_q]  CV_32F
   * @param db_local  DB local descriptors     [D, P_db] CV_32F
   * @param keypoints Patch centers            [2, P]    CV_32F  (same for both images)
   * @param stride    Patch stride (used for RANSAC threshold = 16 * stride * 1.5)
   * @return Negative inlier ratio (-inliers / P_q), or 0.0f if fewer than 4 mutual matches
   */
  float match_one_scale(const cv::Mat& q_local, const cv::Mat& db_local, const cv::Mat& keypoints,
                        int stride) const;
};

}  // namespace xfeat
