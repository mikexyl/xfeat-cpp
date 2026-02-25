#include "xfeat-cpp/place_recognition/patchnetvlad_matcher.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace xfeat {

// VGG-16 conv5_3 receptive field constants (hardcoded, same as Python reference)
static constexpr int kVggRf = 196;
static constexpr int kVggStride = 16;
static constexpr int kVggPadding = 90;

PatchNetVLADMatcher::PatchNetVLADMatcher() : PatchNetVLADMatcher(Params{}) {}

PatchNetVLADMatcher::PatchNetVLADMatcher(const Params& params) : params_(params) {
  if (params_.patch_sizes.size() != params_.strides.size() ||
      params_.patch_sizes.size() != params_.patch_weights.size()) {
    throw std::runtime_error("PatchNetVLADMatcher: patch_sizes, strides, and patch_weights must have the same length");
  }
  for (size_t i = 0; i < params_.patch_sizes.size(); ++i) {
    keypoints_.push_back(compute_keypoints(params_.patch_sizes[i], params_.strides[i]));
  }
}

cv::Mat PatchNetVLADMatcher::compute_receptive_boxes(int H, int W) const {
  // Returns [H*W, 4] CV_32F: each row = [xmin, ymin, xmax, ymax]
  // Box for feature at (row=h, col=w): x spans [kVggStride*w - kVggPadding, ..+kVggRf-1]
  cv::Mat boxes(H * W, 4, CV_32F);

  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      int idx = h * W + w;
      float xmin = static_cast<float>(kVggStride * w - kVggPadding);
      float ymin = static_cast<float>(kVggStride * h - kVggPadding);
      float xmax = xmin + kVggRf - 1;  // = 16w + 105
      float ymax = ymin + kVggRf - 1;  // = 16h + 105
      boxes.at<float>(idx, 0) = xmin;
      boxes.at<float>(idx, 1) = ymin;
      boxes.at<float>(idx, 2) = xmax;
      boxes.at<float>(idx, 3) = ymax;
    }
  }
  return boxes;
}

cv::Mat PatchNetVLADMatcher::compute_keypoints(int patch_size, int stride) const {
  int H = params_.img_height / kVggStride;  // feature map height (30 for 480px)
  int W = params_.img_width / kVggStride;   // feature map width  (40 for 640px)

  int Hout = (H - patch_size) / stride + 1;
  int Wout = (W - patch_size) / stride + 1;
  int num_patches = Hout * Wout;

  cv::Mat boxes = compute_receptive_boxes(H, W);

  // keypoints: [2, num_patches]  row 0 = x (horizontal), row 1 = y (vertical)
  cv::Mat kpts(2, num_patches, CV_32F);

  int k = 0;
  for (int i = 0; i < Hout; ++i) {
    for (int j = 0; j < Wout; ++j) {
      // x-extent: columns [j, j+patch_size-1] in the feature map
      // Due to VGG separability xmin depends only on w, xmax only on w.
      float xmin = boxes.at<float>(i * W + j, 0);
      float xmax = boxes.at<float>(i * W + (j + patch_size - 1), 2);

      // y-extent: rows [i, i+patch_size-1] in the feature map
      float ymin = boxes.at<float>(i * W + j, 1);
      float ymax = boxes.at<float>((i + patch_size - 1) * W + j, 3);

      kpts.at<float>(0, k) = (xmin + xmax) / 2.0f;
      kpts.at<float>(1, k) = (ymin + ymax) / 2.0f;
      ++k;
    }
  }
  return kpts;
}

float PatchNetVLADMatcher::match_one_scale(const cv::Mat& q_local, const cv::Mat& db_local,
                                           const cv::Mat& keypoints, int stride) const {
  int P_q = q_local.cols;
  int P_db = db_local.cols;

  // Dot-product matrix: mul[i,j] = <q_local[:,i], db_local[:,j]>
  // cv::gemm with GEMM_1_T: result = q_local^T * db_local → [P_q, P_db]
  cv::Mat mul;
  cv::gemm(q_local, db_local, 1.0, cv::Mat(), 0.0, mul, cv::GEMM_1_T);

  // Forward: fw_inds[j] = argmax_i mul(i,j)  (closest query patch to each db patch)
  std::vector<int> fw_inds(P_db);
  for (int j = 0; j < P_db; ++j) {
    float best = -std::numeric_limits<float>::infinity();
    int best_i = 0;
    for (int i = 0; i < P_q; ++i) {
      float v = mul.at<float>(i, j);
      if (v > best) {
        best = v;
        best_i = i;
      }
    }
    fw_inds[j] = best_i;
  }

  // Backward: bw_inds[i] = argmax_j mul(i,j)  (closest db patch to each query patch)
  std::vector<int> bw_inds(P_q);
  for (int i = 0; i < P_q; ++i) {
    float best = -std::numeric_limits<float>::infinity();
    int best_j = 0;
    for (int j = 0; j < P_db; ++j) {
      float v = mul.at<float>(i, j);
      if (v > best) {
        best = v;
        best_j = j;
      }
    }
    bw_inds[i] = best_j;
  }

  // Mutual matches: db patch j is a mutual match if bw_inds[fw_inds[j]] == j
  std::vector<int> mutuals;
  mutuals.reserve(std::min(P_q, P_db));
  for (int j = 0; j < P_db; ++j) {
    int qi = fw_inds[j];
    if (bw_inds[qi] == j) {
      mutuals.push_back(j);
    }
  }

  if (static_cast<int>(mutuals.size()) <= 3) {
    return 0.0f;
  }

  // Build point correspondences in image coordinates
  std::vector<cv::Point2f> q_pts, db_pts;
  q_pts.reserve(mutuals.size());
  db_pts.reserve(mutuals.size());

  for (int j : mutuals) {
    int qi = fw_inds[j];
    q_pts.emplace_back(keypoints.at<float>(0, qi), keypoints.at<float>(1, qi));
    db_pts.emplace_back(keypoints.at<float>(0, j), keypoints.at<float>(1, j));
  }

  // RANSAC homography: find H such that db_pts → q_pts
  double ransac_thresh = 16.0 * stride * 1.5;
  cv::Mat mask;
  cv::findHomography(db_pts, q_pts, cv::RANSAC, ransac_thresh, mask);

  int inlier_count = (mask.empty()) ? 0 : cv::countNonZero(mask);
  return -static_cast<float>(inlier_count) / static_cast<float>(P_q);
}

PatchNetVLADMatcher::MatchResult PatchNetVLADMatcher::match(const PatchNetVLADONNX::Features& query,
                                                             const PatchNetVLADONNX::Features& db) const {
  int num_scales = static_cast<int>(params_.patch_sizes.size());

  if (static_cast<int>(query.local_descs.size()) < num_scales ||
      static_cast<int>(db.local_descs.size()) < num_scales) {
    throw std::runtime_error("PatchNetVLADMatcher::match: insufficient local descriptor scales");
  }

  MatchResult result;
  result.score = 0.0f;
  result.inlier_counts.resize(num_scales, 0);

  // Global cosine similarity (both descriptors are L2-normalized)
  cv::Mat dot = query.global_desc * db.global_desc.t();
  result.global_sim = dot.at<float>(0, 0);

  for (int i = 0; i < num_scales; ++i) {
    float scale_score = match_one_scale(query.local_descs[i], db.local_descs[i], keypoints_[i], params_.strides[i]);

    // scale_score ≤ 0  (−inliers/P_q); −scale_score is the positive inlier ratio
    int P_q = query.local_descs[i].cols;
    result.inlier_counts[i] = static_cast<int>(-scale_score * P_q);
    result.score += params_.patch_weights[i] * (-scale_score);
  }

  return result;
}

}  // namespace xfeat
