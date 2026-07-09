#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

namespace xfeat {

struct CameraIntrinsics {
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  int width = 0;
  int height = 0;
};

struct MonoDepthMetadata {
  cv::Size original_size;
  cv::Size model_size;
  double scale_x = 1.0;
  double scale_y = 1.0;
  double focal_scale = 1.0;
  bool focal_scaled = false;
  double pose_scale = 1.0;
  bool pose_scaled = false;
  bool sky_filled = false;
  float sky_fill_value = 0.0f;
  int view_index = 0;
  int view_count = 1;
  std::string depth_tensor_name;
  std::string confidence_tensor_name;
  std::string sky_tensor_name;
};

struct MonoDepthResult {
  cv::Mat depth;           // CV_32FC1, resized to the input image size.
  cv::Mat raw_depth;       // CV_32FC1, raw model output before scaling and sky fill.
  cv::Mat confidence;      // CV_32FC1, resized to the input image size, may be empty.
  cv::Mat raw_confidence;  // CV_32FC1, raw confidence output at model size, may be empty.
  cv::Mat sky_mask;        // CV_8UC1, non-zero pixels are sky, resized to the input image size.
  MonoDepthMetadata metadata;
};

class MonoDepth {
 public:
  virtual ~MonoDepth() = default;

  virtual MonoDepthResult infer(const cv::Mat& image,
                                const std::optional<CameraIntrinsics>& intrinsics = std::nullopt) = 0;

  virtual std::vector<MonoDepthResult> infer_multi_view(const std::vector<cv::Mat>& views,
                                                        const std::vector<CameraIntrinsics>& intrinsics = {}) = 0;
};

}  // namespace xfeat
