#pragma once

#include "xfeat-cpp/mono_depth/mono_depth.h"

#ifdef HAVE_TENSORRT

#include <memory>
#include <string>
#include <vector>

namespace xfeat {

class DepthAnythingV3TRT : public MonoDepth {
 public:
  struct Params {
    std::string engine_path;
    int device_id = 0;
    cv::Size fallback_input_size = cv::Size(504, 280);
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    float sky_threshold = 0.3f;
    float sky_depth_cap = 200.0f;
    double model_focal_pixels = 300.0;
    bool verbose = false;
  };

  explicit DepthAnythingV3TRT(const Params& params);
  ~DepthAnythingV3TRT() override;

  DepthAnythingV3TRT(const DepthAnythingV3TRT&) = delete;
  DepthAnythingV3TRT& operator=(const DepthAnythingV3TRT&) = delete;

  MonoDepthResult infer(const cv::Mat& image,
                        const std::optional<CameraIntrinsics>& intrinsics = std::nullopt) override;

  std::vector<MonoDepthResult> infer_multi_view(const std::vector<cv::Mat>& views,
                                                const std::vector<CameraIntrinsics>& intrinsics = {}) override;

  std::vector<MonoDepthResult> infer_multi_view(const std::vector<cv::Mat>& views,
                                                const std::vector<CameraIntrinsics>& intrinsics,
                                                const std::vector<cv::Matx44f>& world_to_camera_extrinsics);

  cv::Size input_size() const;
  bool has_camera_inputs() const;
  bool has_pose_inputs() const;
  const Params& params() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xfeat

#endif  // HAVE_TENSORRT
