#include "xfeat-cpp/lighterglue_trt.h"

#ifdef HAVE_TENSORRT

#include <algorithm>
#include <cstring>
#include <limits>
#include <opencv2/core.hpp>
#include <stdexcept>

#include "xfeat-cpp/tensorrt/detail/trt_engine.h"

namespace xfeat {
namespace {

void validateFeatureInputs(const std::vector<float>& keypoints,
                           const std::vector<float>& descriptors,
                           const std::string& label) {
  if (keypoints.empty() || descriptors.empty()) {
    throw std::invalid_argument("LighterGlueTRT " + label + " inputs are empty");
  }
  if (keypoints.size() % 2 != 0 || descriptors.size() % 64 != 0 || keypoints.size() / 2 != descriptors.size() / 64) {
    throw std::invalid_argument("LighterGlueTRT " + label + " keypoint/descriptor counts do not match");
  }
}

std::vector<float> flatten(const cv::Mat& input, int expected_columns, const std::string& label) {
  if (input.empty() || input.type() != CV_32F || input.cols != expected_columns) {
    throw std::invalid_argument("LighterGlueTRT " + label + " must be a non-empty CV_32F matrix with " +
                                std::to_string(expected_columns) + " columns");
  }
  const cv::Mat continuous = input.isContinuous() ? input : input.clone();
  const float* begin = continuous.ptr<float>();
  return std::vector<float>(begin, begin + continuous.total());
}

}  // namespace

LighterGlueTRT::LighterGlueTRT(const std::string& engine_path, bool verbose)
    : engine_(std::make_unique<trt_detail::Engine>(engine_path, verbose)) {
  const std::vector<std::string> expected_inputs = {
      "mkpts0", "feats0", "image0_size", "mkpts1", "feats1", "image1_size"};
  for (const auto& name : expected_inputs) {
    if (engine_->tensor_type(name) != trt_detail::DataType::kFloat32) {
      throw std::runtime_error("LighterGlueTRT input must be float32: " + name);
    }
  }
  if (engine_->tensor_type("matches") != trt_detail::DataType::kInt64 ||
      engine_->tensor_type("scores") != trt_detail::DataType::kFloat32) {
    throw std::runtime_error("LighterGlueTRT engine must expose INT64 matches and float32 scores");
  }
}

LighterGlueTRT::~LighterGlueTRT() = default;

void LighterGlueTRT::run(const std::vector<float>& mkpts0,
                         const std::vector<float>& feats0,
                         const std::array<float, 2>& image0_size,
                         const std::vector<float>& mkpts1,
                         const std::vector<float>& feats1,
                         const std::array<float, 2>& image1_size,
                         std::vector<std::array<int64_t, 2>>& matches,
                         std::vector<float>& scores) {
  validateFeatureInputs(mkpts0, feats0, "image 0");
  validateFeatureInputs(mkpts1, feats1, "image 1");
  const int64_t count0 = static_cast<int64_t>(mkpts0.size() / 2);
  const int64_t count1 = static_cast<int64_t>(mkpts1.size() / 2);

  const std::vector<trt_detail::InputTensor> inputs = {
      {"mkpts0", {1, count0, 2}, mkpts0.data(), mkpts0.size() * sizeof(float)},
      {"feats0", {1, count0, 64}, feats0.data(), feats0.size() * sizeof(float)},
      {"image0_size", {2}, image0_size.data(), image0_size.size() * sizeof(float)},
      {"mkpts1", {1, count1, 2}, mkpts1.data(), mkpts1.size() * sizeof(float)},
      {"feats1", {1, count1, 64}, feats1.data(), feats1.size() * sizeof(float)},
      {"image1_size", {2}, image1_size.data(), image1_size.size() * sizeof(float)},
  };
  const auto outputs = engine_->run(inputs);
  const auto& matches_output = trt_detail::find_output(outputs, "matches");
  const auto& scores_output = trt_detail::find_output(outputs, "scores");
  if (matches_output.shape.size() != 2 || matches_output.shape[1] != 2 || scores_output.shape.size() != 1 ||
      matches_output.shape[0] != scores_output.shape[0]) {
    throw std::runtime_error("LighterGlueTRT returned inconsistent match and score shapes");
  }

  const std::vector<int64_t> match_values = matches_output.values<int64_t>();
  const std::vector<float> score_values = scores_output.values<float>();
  matches.resize(score_values.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    matches[i] = {match_values[i * 2], match_values[i * 2 + 1]};
  }
  scores = score_values;
}

std::pair<std::vector<std::array<int64_t, 2>>, std::vector<float>> LighterGlueTRT::match(
    const std::vector<float>& mkpts0,
    const std::vector<float>& feats0,
    const std::array<float, 2>& image0_size,
    const std::vector<float>& mkpts1,
    const std::vector<float>& feats1,
    const std::array<float, 2>& image1_size) {
  std::vector<std::array<int64_t, 2>> matches;
  std::vector<float> scores;
  run(mkpts0, feats0, image0_size, mkpts1, feats1, image1_size, matches, scores);
  return {std::move(matches), std::move(scores)};
}

std::vector<std::vector<int>> LighterGlueTRT::match(const DetectionResult& det0,
                                                    const std::array<float, 2>& image0_size,
                                                    const DetectionResult& det1,
                                                    const std::array<float, 2>& image1_size,
                                                    float min_score,
                                                    std::vector<float>* scores_out) {
  if (scores_out != nullptr) {
    scores_out->assign(det0.keypoints.rows, std::numeric_limits<float>::lowest());
  }
  if (det0.keypoints.empty() || det1.keypoints.empty()) {
    return std::vector<std::vector<int>>(det0.keypoints.rows);
  }
  if (det0.keypoints.rows != det0.descriptors.rows || det1.keypoints.rows != det1.descriptors.rows) {
    throw std::invalid_argument("LighterGlueTRT keypoint and descriptor row counts do not match");
  }

  const std::vector<float> mkpts0 = flatten(det0.keypoints, 2, "image 0 keypoints");
  const std::vector<float> feats0 = flatten(det0.descriptors, 64, "image 0 descriptors");
  const std::vector<float> mkpts1 = flatten(det1.keypoints, 2, "image 1 keypoints");
  const std::vector<float> feats1 = flatten(det1.descriptors, 64, "image 1 descriptors");
  auto [raw_matches, raw_scores] = match(mkpts0, feats0, image0_size, mkpts1, feats1, image1_size);

  std::vector<std::vector<int>> indices(det0.keypoints.rows);
  for (size_t i = 0; i < raw_matches.size(); ++i) {
    if (min_score >= 0.0f && raw_scores[i] < min_score) continue;
    const int64_t index0 = raw_matches[i][0];
    const int64_t index1 = raw_matches[i][1];
    if (index0 < 0 || index0 >= det0.keypoints.rows || index1 < 0 || index1 >= det1.keypoints.rows) {
      throw std::runtime_error("LighterGlueTRT returned an out-of-range match index");
    }
    indices[static_cast<size_t>(index0)].push_back(static_cast<int>(index1));
    if (scores_out != nullptr) {
      (*scores_out)[static_cast<size_t>(index0)] = std::max((*scores_out)[static_cast<size_t>(index0)], raw_scores[i]);
    }
  }
  return indices;
}

}  // namespace xfeat

#endif  // HAVE_TENSORRT
