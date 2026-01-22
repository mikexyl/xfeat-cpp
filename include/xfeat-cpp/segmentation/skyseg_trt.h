#pragma once

// Sky Segmentation requires TensorRT
#ifdef HAVE_TENSORRT

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

// Forward declarations to avoid including TensorRT and CUDA headers
namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}  // namespace nvinfer1

namespace xfeat {

/**
 * @brief Sky Segmentation using TensorRT
 *
 * This implementation uses a deep learning model for sky segmentation.
 * It requires a pre-trained TensorRT engine file and runs on CUDA GPU.
 * The model outputs a binary mask indicating sky regions in the input image.
 */
class SkySegTRT {
 public:
  struct Params {
    std::string engine_path;                             // Path to TensorRT engine file
    cv::Size input_size = cv::Size(320, 320);            // Model input size
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};  // ImageNet normalization mean (RGB)
    std::vector<float> std = {0.229f, 0.224f, 0.225f};   // ImageNet normalization std (RGB)
    bool verbose = false;                                // Enable verbose output
    int warmup_iterations = 3;                           // Number of warmup iterations
    float threshold = 127.0f;                            // Threshold for binary mask (0-255)
  };

  explicit SkySegTRT(const Params& params);
  ~SkySegTRT();

  /**
   * @brief Segment sky regions in the input image
   *
   * @param image Input image (BGR format)
   * @param mask Output segmentation mask (CV_8UC1, 0-255)
   */
  void segment(const cv::Mat& image, cv::Mat& mask);

  /**
   * @brief Segment sky and create visualization overlay
   *
   * @param image Input image (BGR format)
   * @param mask Output segmentation mask (CV_8UC1, 0-255)
   * @param overlay Output visualization with sky overlay
   * @param alpha Transparency for overlay (0.0-1.0)
   * @param color Sky color for overlay (BGR)
   */
  void segmentWithVisualization(const cv::Mat& image,
                                cv::Mat& mask,
                                cv::Mat& overlay,
                                float alpha = 0.5f,
                                const cv::Scalar& color = cv::Scalar(255, 100, 0));

  /**
   * @brief Warmup the inference engine
   *
   * @param image_size Size of images to warmup with
   */
  void warmup(const cv::Size& image_size);

  /**
   * @brief Get the raw segmentation output before thresholding
   *
   * @return The raw segmentation prediction (CV_8UC1, 0-255)
   */
  const cv::Mat& getRawMask() const { return raw_mask_; }

  /**
   * @brief Calculate sky percentage in the image
   *
   * @param mask Segmentation mask
   * @return Percentage of sky pixels (0-100)
   */
  static float calculateSkyPercentage(const cv::Mat& mask);

  Params params_;

 private:
  // TensorRT inference engine
  class InferenceEngine;
  std::unique_ptr<InferenceEngine> engine_;

  // Output buffers
  cv::Mat raw_mask_;

  // Preprocessing
  cv::Mat preprocess(const cv::Mat& image);

  // Postprocessing
  void postprocess(const cv::Mat& raw_output, cv::Mat& mask);
};

}  // namespace xfeat

#endif  // HAVE_TENSORRT
