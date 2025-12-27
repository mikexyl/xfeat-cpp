#pragma once

#include "xfeat-cpp/stereo_depth.h"
#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <memory>
#include <vector>

namespace xfeat {

/**
 * @brief ONNX-based stereo depth estimator
 * 
 * This implementation uses ONNX Runtime to run deep learning models for stereo depth
 * estimation. It supports models like FastACVNet and other stereo matching networks
 * exported to ONNX format.
 */
class OnnxStereoDepth : public StereoDepth {
 public:
  struct Params {
    std::string model_path;                      // Path to ONNX model file
    cv::Size input_size = cv::Size(640, 480);    // Input size expected by the model
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};  // Normalization mean (RGB)
    std::vector<float> std = {0.229f, 0.224f, 0.225f};   // Normalization std (RGB)
    bool use_cuda = true;                        // Use CUDA execution provider
    bool verbose = false;                        // Enable verbose output
    int warmup_iterations = 3;                   // Number of warmup iterations
    float max_disparity = 192.0f;                // Maximum disparity value for the model
    
    // Camera parameters for depth conversion (optional)
    float focal_length = 0.0f;                   // Focal length in pixels (0 = not set)
    float baseline = 0.0f;                       // Baseline in meters (0 = not set)
  };

  explicit OnnxStereoDepth(const Params& params);
  ~OnnxStereoDepth() override;

  void compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity) override;

  int getDisparityScale() const override { return 1; }  // Output is already in float
  int getMinDisparity() const override { return 0; }
  int getNumDisparities() const override { return static_cast<int>(params_.max_disparity); }
  int getBlockSize() const override { return 0; }  // Not applicable for deep learning
  bool requiresGrayscale() const override { return false; }  // Uses RGB images

  void warmup(const cv::Size& image_size) override;

  /**
   * @brief Get the raw disparity prediction from the model
   * 
   * @return The disparity map at the model's output resolution
   */
  const cv::Mat& getRawDisparity() const { return raw_disparity_; }

  /**
   * @brief Get color-mapped disparity for visualization
   * 
   * @return Color-mapped disparity image
   */
  cv::Mat getColorDisparity() const;

  /**
   * @brief Compute depth map directly (if camera params are set)
   * 
   * @param left Left rectified image
   * @param right Right rectified image
   * @param depth Output depth map in meters (CV_32F)
   */
  void computeDepthDirect(const cv::Mat& left, const cv::Mat& right, cv::Mat& depth);

 private:
  Params params_;
  
  // ONNX Runtime components
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  Ort::MemoryInfo memory_info_;
  
  // Input/output tensor info
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  std::vector<std::string> input_names_storage_;
  std::vector<std::string> output_names_storage_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  
  // Processing buffers
  cv::Mat raw_disparity_;
  cv::Size original_size_;
  
  // Helper methods
  void initializeSession();
  void preprocessImage(const cv::Mat& img, std::vector<float>& output);
  void runInference(const std::vector<float>& left_tensor, 
                   const std::vector<float>& right_tensor,
                   std::vector<float>& output);
  void postprocessDisparity(const std::vector<float>& model_output, 
                           cv::Mat& disparity);
};

}  // namespace xfeat
