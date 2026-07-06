#pragma once

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "xfeat-cpp/mono_depth/mono_depth.h"

namespace xfeat {
namespace mono_depth_detail {

inline bool hasValidFocalScaleInputs(const std::optional<CameraIntrinsics>& intrinsics,
                                     const cv::Size& original_size,
                                     const cv::Size& model_size,
                                     double model_focal_pixels) {
  return intrinsics.has_value() && intrinsics->fx > 0.0 && intrinsics->fy > 0.0 && original_size.width > 0 &&
         original_size.height > 0 && model_size.width > 0 && model_size.height > 0 && model_focal_pixels > 0.0;
}

inline double computeFocalScale(const std::optional<CameraIntrinsics>& intrinsics,
                                const cv::Size& original_size,
                                const cv::Size& model_size,
                                double model_focal_pixels = 300.0) {
  if (!hasValidFocalScaleInputs(intrinsics, original_size, model_size, model_focal_pixels)) {
    return 1.0;
  }

  const double scale_x = static_cast<double>(model_size.width) / static_cast<double>(original_size.width);
  const double scale_y = static_cast<double>(model_size.height) / static_cast<double>(original_size.height);
  const double focal_pixels = 0.5 * (intrinsics->fx * scale_x + intrinsics->fy * scale_y);
  return focal_pixels > 0.0 ? focal_pixels / model_focal_pixels : 1.0;
}

inline cv::Mat clampInvalidDepth(const cv::Mat& depth) {
  if (depth.empty()) {
    throw std::invalid_argument("Depth map is empty");
  }
  if (depth.channels() != 1) {
    throw std::invalid_argument("Depth map must be single-channel");
  }

  cv::Mat clamped;
  if (depth.type() == CV_32FC1) {
    clamped = depth.clone();
  } else {
    depth.convertTo(clamped, CV_32FC1);
  }

  for (int y = 0; y < clamped.rows; ++y) {
    float* row = clamped.ptr<float>(y);
    for (int x = 0; x < clamped.cols; ++x) {
      const float value = row[x];
      if (!std::isfinite(value) || value < 0.0f) {
        row[x] = 0.0f;
      }
    }
  }
  return clamped;
}

inline cv::Mat skyMaskFromPrediction(const cv::Mat& sky_prediction, float sky_threshold) {
  if (sky_prediction.empty()) {
    return {};
  }
  if (sky_prediction.channels() != 1) {
    throw std::invalid_argument("Sky prediction must be single-channel");
  }

  cv::Mat sky_float;
  if (sky_prediction.type() == CV_32FC1) {
    sky_float = sky_prediction;
  } else {
    sky_prediction.convertTo(sky_float, CV_32FC1);
  }

  cv::Mat mask;
  cv::compare(sky_float, sky_threshold, mask, cv::CMP_GT);
  return mask;
}

inline float percentile(std::vector<float> values, double q) {
  if (values.empty()) {
    return 0.0f;
  }
  q = std::max(0.0, std::min(1.0, q));
  const auto idx = static_cast<size_t>(q * static_cast<double>(values.size() - 1));
  auto nth = values.begin() + static_cast<std::vector<float>::difference_type>(idx);
  std::nth_element(values.begin(), nth, values.end());
  return values[idx];
}

inline float fillSkyDepthWithPercentile(cv::Mat& depth, const cv::Mat& sky_mask, float sky_depth_cap) {
  if (depth.empty() || sky_mask.empty()) {
    return 0.0f;
  }
  if (depth.type() != CV_32FC1) {
    throw std::invalid_argument("Depth map must be CV_32FC1");
  }
  if (sky_mask.type() != CV_8UC1) {
    throw std::invalid_argument("Sky mask must be CV_8UC1");
  }
  if (depth.size() != sky_mask.size()) {
    throw std::invalid_argument("Depth map and sky mask sizes must match");
  }

  std::vector<float> non_sky_depths;
  non_sky_depths.reserve(static_cast<size_t>(depth.rows * depth.cols));
  for (int y = 0; y < depth.rows; ++y) {
    const float* depth_row = depth.ptr<float>(y);
    const uint8_t* mask_row = sky_mask.ptr<uint8_t>(y);
    for (int x = 0; x < depth.cols; ++x) {
      const float value = depth_row[x];
      if (mask_row[x] == 0 && std::isfinite(value) && value > 0.0f) {
        non_sky_depths.push_back(value);
      }
    }
  }

  if (non_sky_depths.empty()) {
    return 0.0f;
  }

  const float fill_value = std::min(percentile(std::move(non_sky_depths), 0.99), sky_depth_cap);
  depth.setTo(fill_value, sky_mask);
  return fill_value;
}

struct DepthAnythingPostprocessOptions {
  cv::Size original_size;
  std::optional<CameraIntrinsics> intrinsics;
  float sky_threshold = 0.3f;
  float sky_depth_cap = 200.0f;
  double model_focal_pixels = 300.0;
  int resize_interpolation = cv::INTER_CUBIC;
};

struct DepthAnythingPostprocessOutput {
  cv::Mat depth;
  cv::Mat raw_depth;
  cv::Mat model_depth;
  cv::Mat sky_mask;
  cv::Mat model_sky_mask;
  double focal_scale = 1.0;
  bool focal_scaled = false;
  bool sky_filled = false;
  float sky_fill_value = 0.0f;
};

inline DepthAnythingPostprocessOutput postprocessDepthAnything(const cv::Mat& raw_depth,
                                                               const cv::Mat& sky_prediction,
                                                               const DepthAnythingPostprocessOptions& options) {
  if (raw_depth.empty()) {
    throw std::invalid_argument("Raw depth output is empty");
  }
  if (raw_depth.channels() != 1) {
    throw std::invalid_argument("Raw depth output must be single-channel");
  }

  DepthAnythingPostprocessOutput output;
  raw_depth.convertTo(output.raw_depth, CV_32FC1);
  output.model_depth = clampInvalidDepth(output.raw_depth);

  output.focal_scaled = hasValidFocalScaleInputs(
      options.intrinsics, options.original_size, output.model_depth.size(), options.model_focal_pixels);
  output.focal_scale = computeFocalScale(
      options.intrinsics, options.original_size, output.model_depth.size(), options.model_focal_pixels);
  output.model_depth *= static_cast<float>(output.focal_scale);

  if (!sky_prediction.empty()) {
    cv::Mat sky_model;
    sky_prediction.convertTo(sky_model, CV_32FC1);
    if (sky_model.size() != output.model_depth.size()) {
      cv::resize(sky_model, sky_model, output.model_depth.size(), 0.0, 0.0, cv::INTER_LINEAR);
    }
    output.model_sky_mask = skyMaskFromPrediction(sky_model, options.sky_threshold);
    output.sky_fill_value =
        fillSkyDepthWithPercentile(output.model_depth, output.model_sky_mask, options.sky_depth_cap);
    output.sky_filled = output.sky_fill_value > 0.0f;
  }

  if (options.original_size.width > 0 && options.original_size.height > 0 &&
      output.model_depth.size() != options.original_size) {
    cv::resize(output.model_depth, output.depth, options.original_size, 0.0, 0.0, options.resize_interpolation);
    output.depth = clampInvalidDepth(output.depth);
    if (!output.model_sky_mask.empty()) {
      cv::resize(output.model_sky_mask, output.sky_mask, options.original_size, 0.0, 0.0, cv::INTER_NEAREST);
    }
  } else {
    output.depth = output.model_depth.clone();
    output.sky_mask = output.model_sky_mask.clone();
  }

  return output;
}

}  // namespace mono_depth_detail
}  // namespace xfeat
