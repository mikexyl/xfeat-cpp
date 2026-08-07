#pragma once

#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace xfeat {
namespace vpi_detail {

inline cv::Mat makeValidityMask(const cv::Mat& disparity,
                                const cv::Mat& confidence,
                                const int min_valid_disparity,
                                const int confidence_threshold,
                                const int disparity_scale) {
  if (disparity.type() != CV_16S || confidence.type() != CV_16U || disparity.size() != confidence.size()) {
    throw std::invalid_argument("VPI disparity and confidence maps have incompatible formats");
  }

  // Use a strict lower bound: a result exactly on the configured acceptance
  // boundary may be a clamped correspondence, and must not become an
  // artificial constant-depth plane.
  return (confidence >= confidence_threshold) & (disparity > min_valid_disparity * disparity_scale);
}

inline void restoreDisparityWithValidity(const cv::Mat& working_disparity,
                                         const cv::Mat& working_valid,
                                         const cv::Size& output_size,
                                         const double horizontal_scale,
                                         const int16_t invalid_disparity,
                                         cv::Mat* output_disparity) {
  if (output_disparity == nullptr) {
    throw std::invalid_argument("VPI output disparity pointer is null");
  }
  if (working_disparity.type() != CV_16S || working_valid.type() != CV_8U ||
      working_disparity.size() != working_valid.size() || output_size.width <= 0 || output_size.height <= 0 ||
      !(horizontal_scale > 0.0)) {
    throw std::invalid_argument("Invalid VPI disparity restoration arguments");
  }

  // Interpolating a disparity image containing invalid sentinels biases
  // adjacent valid disparities toward the sentinel. Resize the weighted
  // disparity and its validity weight independently, then normalize.
  cv::Mat weighted_disparity;
  working_disparity.convertTo(weighted_disparity, CV_32F);
  weighted_disparity.setTo(0.0f, ~working_valid);

  cv::Mat validity_weight;
  working_valid.convertTo(validity_weight, CV_32F, 1.0 / 255.0);

  cv::Mat resized_weighted_disparity;
  cv::Mat resized_validity_weight;
  cv::resize(weighted_disparity, resized_weighted_disparity, output_size, 0.0, 0.0, cv::INTER_LINEAR);
  cv::resize(validity_weight, resized_validity_weight, output_size, 0.0, 0.0, cv::INTER_LINEAR);

  constexpr float kMinimumValidityWeight = 1e-6f;
  const cv::Mat has_valid_weight = resized_validity_weight > kMinimumValidityWeight;
  cv::Mat safe_validity_weight = resized_validity_weight.clone();
  safe_validity_weight.setTo(1.0f, ~has_valid_weight);

  cv::Mat normalized_disparity;
  cv::divide(resized_weighted_disparity, safe_validity_weight, normalized_disparity, horizontal_scale, CV_32F);
  normalized_disparity.convertTo(*output_disparity, CV_16S);

  // Preserve the original nearest-neighbor confidence support. Normalized
  // interpolation corrects values at its boundary but must not fill holes.
  cv::Mat output_valid;
  cv::resize(working_valid, output_valid, output_size, 0.0, 0.0, cv::INTER_NEAREST);
  output_valid &= has_valid_weight;
  output_disparity->setTo(invalid_disparity, ~output_valid);
}

}  // namespace vpi_detail
}  // namespace xfeat
