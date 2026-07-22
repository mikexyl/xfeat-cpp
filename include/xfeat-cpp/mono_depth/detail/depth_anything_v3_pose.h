#pragma once

#include <cmath>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>

namespace xfeat {
namespace mono_depth_detail {

inline cv::Point3d cameraCenterFromWorldToCamera(const cv::Matx44f& extrinsic) {
  for (const float value : extrinsic.val) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("Camera extrinsic contains a non-finite value");
    }
  }
  const double tx = extrinsic(0, 3);
  const double ty = extrinsic(1, 3);
  const double tz = extrinsic(2, 3);
  return cv::Point3d(-(extrinsic(0, 0) * tx + extrinsic(1, 0) * ty + extrinsic(2, 0) * tz),
                     -(extrinsic(0, 1) * tx + extrinsic(1, 1) * ty + extrinsic(2, 1) * tz),
                     -(extrinsic(0, 2) * tx + extrinsic(1, 2) * ty + extrinsic(2, 2) * tz));
}

inline double squaredDistance(const cv::Point3d& a, const cv::Point3d& b) {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

inline double estimateInputToPredictedPoseScale(const std::vector<cv::Matx44f>& predicted_world_to_camera,
                                                const std::vector<cv::Matx44f>& input_world_to_camera) {
  if (predicted_world_to_camera.size() != input_world_to_camera.size() || input_world_to_camera.size() < 2) {
    throw std::invalid_argument("Cannot estimate pose scale from mismatched or insufficient camera poses");
  }

  std::vector<cv::Point3d> predicted_centers;
  std::vector<cv::Point3d> input_centers;
  predicted_centers.reserve(predicted_world_to_camera.size());
  input_centers.reserve(input_world_to_camera.size());
  for (size_t i = 0; i < input_world_to_camera.size(); ++i) {
    predicted_centers.push_back(cameraCenterFromWorldToCamera(predicted_world_to_camera[i]));
    input_centers.push_back(cameraCenterFromWorldToCamera(input_world_to_camera[i]));
  }

  double predicted_sum = 0.0;
  double input_sum = 0.0;
  int pairs = 0;
  for (size_t i = 0; i < input_centers.size(); ++i) {
    for (size_t j = i + 1; j < input_centers.size(); ++j) {
      const double predicted_d2 = squaredDistance(predicted_centers[i], predicted_centers[j]);
      const double input_d2 = squaredDistance(input_centers[i], input_centers[j]);
      if (predicted_d2 > 1e-12 && input_d2 > 1e-12) {
        predicted_sum += predicted_d2;
        input_sum += input_d2;
        ++pairs;
      }
    }
  }

  if (pairs == 0 || input_sum <= 1e-12) {
    throw std::runtime_error("Cannot estimate pose scale from degenerate camera baselines");
  }
  const double scale = std::sqrt(predicted_sum / input_sum);
  if (!std::isfinite(scale) || scale <= 1e-12) {
    throw std::runtime_error("Estimated pose scale is not finite and positive");
  }
  return scale;
}

}  // namespace mono_depth_detail
}  // namespace xfeat
