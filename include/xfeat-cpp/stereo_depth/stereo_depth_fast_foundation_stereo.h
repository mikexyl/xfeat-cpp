#pragma once

// FastFoundationStereo requires TensorRT
#ifdef HAVE_TENSORRT

#include <memory>
#include <opencv2/core.hpp>
#include <string>

#include "xfeat-cpp/stereo_depth/stereo_depth.h"

namespace xfeat {
namespace trt_detail {
class Engine;
}

/**
 * @brief Fast-FoundationStereo stereo depth estimator using one TensorRT engine.
 *
 * Loads the official single-engine TensorRT export. The engine contains the
 * FFSGWCVolume plugin and is executed through xfeat-cpp's shared TensorRT
 * engine wrapper.
 *
 * Model weights and ONNX export:
 * thirdparty/Fast-FoundationStereo/scripts/make_plugin_onnx.py
 * TRT conversion: python/convert_fast_foundation_stereo_to_tensorrt.py
 */
class FastFoundationStereoDepth : public StereoDepth {
 public:
  struct Params {
    std::string engine_path;  ///< Path to fast_foundationstereo.engine
    /// Export-time disparity range, retained as StereoDepth API metadata.
    int max_disparity = 192;
    int warmup_iterations = 3;
    bool verbose = false;
  };

  explicit FastFoundationStereoDepth(const Params& params);
  ~FastFoundationStereoDepth() override;

  FastFoundationStereoDepth(const FastFoundationStereoDepth&) = delete;
  FastFoundationStereoDepth& operator=(const FastFoundationStereoDepth&) = delete;

  /**
   * @brief Compute disparity from a rectified stereo pair.
   * @param left   Left image (grayscale or BGR, resized and padded internally)
   * @param right  Right image (grayscale or BGR, same size as left)
   * @param disparity  Output float32 disparity at the original image resolution
   */
  void compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity) override;

  int getDisparityScale() const override { return 1; }
  int getMinDisparity() const override { return 0; }
  int getNumDisparities() const override { return params_.max_disparity; }
  int getBlockSize() const override { return 0; }
  bool requiresGrayscale() const override { return false; }

  void warmup(const cv::Size& image_size) override;

  cv::Size inputSize() const { return input_size_; }

 private:
  Params params_;
  std::unique_ptr<trt_detail::Engine> engine_;
  std::string left_input_name_;
  std::string right_input_name_;
  std::string output_name_;
  cv::Size input_size_;
};

}  // namespace xfeat

#endif  // HAVE_TENSORRT
