#pragma once

// LightStereo requires TensorRT
#ifdef HAVE_TENSORRT

#include "xfeat-cpp/stereo_depth.h"
#include <opencv2/core.hpp>
#include <string>
#include <memory>
#include <map>

// Forward declarations to avoid including TensorRT and CUDA headers
namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}

namespace xfeat {

/**
 * @brief LightStereo-based stereo depth estimator using TensorRT
 * 
 * This implementation uses a deep learning model (LightStereo) for stereo depth
 * estimation. It requires a pre-trained TensorRT engine file and runs on CUDA GPU.
 */
class LightStereoDepth : public StereoDepth {
 public:
  struct Params {
    std::string engine_path;              // Path to TensorRT engine file
    cv::Size target_size = cv::Size(1248, 384);  // Target input size for the model
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};  // Normalization mean (RGB)
    std::vector<float> std = {0.229f, 0.224f, 0.225f};   // Normalization std (RGB)
    bool verbose = false;                 // Enable verbose output
    int warmup_iterations = 10;           // Number of warmup iterations
    float max_disparity = 192.0f;         // Maximum disparity value for the model
  };

  explicit LightStereoDepth(const Params& params);
  ~LightStereoDepth() override;

  void compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity) override;

  int getDisparityScale() const override { return 1; }  // Output is already in float
  int getMinDisparity() const override { return 0; }
  int getNumDisparities() const override { return static_cast<int>(params_.max_disparity); }
  int getBlockSize() const override { return 0; }  // Not applicable for deep learning
  bool requiresGrayscale() const override { return false; }  // Uses RGB images

  void warmup(const cv::Size& image_size) override;

  /**
   * @brief Get the original disparity prediction before post-processing
   * 
   * @return The raw disparity prediction from the model
   */
  const cv::Mat& getRawDisparity() const { return raw_disparity_; }

  /**
   * @brief Get the color-normalized disparity for visualization
   * 
   * @return Color-mapped disparity for visualization
   */
  const cv::Mat& getColorDisparity() const { return color_disparity_; }

 private:
  Params params_;
  
  // TensorRT inference engine components
  class InferenceEngine;
  std::unique_ptr<InferenceEngine> engine_;
  
  // Transform pipeline
  class TransformPipeline;
  std::unique_ptr<TransformPipeline> transform_;
  
  // Output buffers
  cv::Mat raw_disparity_;
  cv::Mat color_disparity_;
  cv::Size original_size_;
  
  // Preprocessing
  void preprocess(const cv::Mat& left, const cv::Mat& right,
                 std::map<std::string, cv::Mat>& sample);
  
  // Postprocessing
  void postprocess(const std::map<std::string, cv::Mat>& output, 
                  cv::Mat& disparity);
};

}  // namespace xfeat

#endif  // HAVE_TENSORRT
