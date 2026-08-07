#pragma once

#ifdef HAVE_VPI

#include <memory>
#include <opencv2/core.hpp>

#include "xfeat-cpp/stereo_depth/stereo_depth.h"

namespace xfeat {

/**
 * @brief NVIDIA VPI CUDA semi-global stereo matcher.
 *
 * VPI returns signed Q10.5 disparity. This class preserves that native
 * representation (CV_16S with scale 32), including when a smaller working
 * resolution is requested.
 */
class VPIStereoDepth final : public StereoDepth {
 public:
  static constexpr int kDisparityScale = 32;

  struct Params {
    int min_disparity = 0;
    // Minimum disparity accepted as a depth measurement. This is independent
    // of the search floor so far matches can be rejected instead of clamped.
    int min_valid_disparity = 0;
    int max_disparity = 128;
    int p1 = 3;
    int p2 = 48;
    // NVIDIA's CUDA stereo sample uses UINT16_MAX - 10000. The VPI API
    // default (32767) is too permissive for sparse VIO depth lookup.
    int confidence_threshold = 55535;
    float uniqueness = -1.0f;
    bool include_diagonals = true;
    cv::Size target_size;
  };

  VPIStereoDepth();
  explicit VPIStereoDepth(const Params& params);
  ~VPIStereoDepth() override;

  VPIStereoDepth(const VPIStereoDepth&) = delete;
  VPIStereoDepth& operator=(const VPIStereoDepth&) = delete;

  void compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity) override;
  void warmup(const cv::Size& image_size) override;

  int getDisparityScale() const override { return kDisparityScale; }
  int getMinDisparity() const override { return params_.min_disparity; }
  int getNumDisparities() const override { return params_.max_disparity - params_.min_disparity; }
  int getBlockSize() const override { return 7; }
  bool requiresGrayscale() const override { return true; }

 private:
  class Impl;

  Params params_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xfeat

#endif  // HAVE_VPI
