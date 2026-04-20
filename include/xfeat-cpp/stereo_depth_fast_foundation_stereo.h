#pragma once

// FastFoundationStereo requires TensorRT
#ifdef HAVE_TENSORRT

#include "xfeat-cpp/stereo_depth.h"
#include <opencv2/core.hpp>
#include <memory>
#include <string>

// Forward declarations to avoid including TensorRT headers
namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}  // namespace nvinfer1

namespace xfeat {

/**
 * @brief Fast-FoundationStereo stereo depth estimator using two TensorRT engines.
 *
 * Uses the two-stage TRT split from NVlabs/Fast-FoundationStereo:
 *   1. feature_runner.engine  – backbone feature extraction
 *   2. post_runner.engine     – GWC cost volume + iterative GRU refinement
 *
 * A CUDA kernel builds the GWC (Group-Wise Correlation) volume between the two
 * engine calls, matching the Python TrtRunner implementation exactly.
 *
 * Model weights and ONNX export: see thirdparty/Fast-FoundationStereo/scripts/make_onnx.py
 * TRT conversion: python/convert_fast_foundation_stereo_to_tensorrt.py
 */
class FastFoundationStereoDepth : public StereoDepth {
 public:
  struct Params {
    std::string feature_engine_path;  ///< Path to feature_runner.engine
    std::string post_engine_path;     ///< Path to post_runner.engine
    /// Input size the engines were built for (width × height, must be multiples of 32)
    cv::Size target_size = cv::Size(640, 448);
    int max_disp = 192;   ///< Maximum disparity (same as --max_disp used during export)
    int cv_group = 8;     ///< Number of groups for GWC volume (from model checkpoint)
    bool normalize_gwc = true;  ///< L2-normalize features before GWC dot product
    int warmup_iterations = 3;
    bool verbose = false;
  };

  explicit FastFoundationStereoDepth(const Params& params);
  ~FastFoundationStereoDepth() override;

  /**
   * @brief Compute disparity from a rectified stereo pair.
   * @param left   Left image (BGR, any size; resized internally to target_size)
   * @param right  Right image (BGR, same size as left)
   * @param disparity  Output float32 disparity at the original image resolution
   */
  void compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity) override;

  int getDisparityScale() const override { return 1; }
  int getMinDisparity() const override { return 0; }
  int getNumDisparities() const override { return params_.max_disp; }
  int getBlockSize() const override { return 0; }
  bool requiresGrayscale() const override { return false; }

  void warmup(const cv::Size& image_size) override;

 private:
  Params params_;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xfeat

#endif  // HAVE_TENSORRT
