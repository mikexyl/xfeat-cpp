#pragma once

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "xfeat-cpp/place_recognition/place_recognizer.h"

namespace xfeat {

/**
 * @brief MixVPR ONNX Inference Wrapper
 *
 * MixVPR is a single-image visual place recognition model that produces
 * a global descriptor via a ResNet backbone with MLP mixer aggregation.
 *
 * Model architecture (mixvpr_resnet50_4096d.onnx):
 *   Input:  "image"      [1, 3, 320, 320]  float32 NCHW, ImageNet-normalized RGB
 *   Output: "descriptor" [1, 4096]          global descriptor
 *
 * Preprocessing: resize → BGR→RGB → /255 → ImageNet normalize
 *   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
 */
class MixVPRONNX : public PlaceRecognizer {
 public:
  struct Params : PlaceRecognizer::Params {
    Params() {
      img_height = 320;
      img_width = 320;
    }
  };

  /**
   * @brief Construct a new MixVPR ONNX wrapper.
   * @param env    ONNX Runtime environment
   * @param params Configuration parameters
   */
  MixVPRONNX(Ort::Env& env, const Params& params);

  /**
   * @brief Process a single image and return the global descriptor.
   * @param images Must contain exactly 1 image (BGR format)
   * @return cv::Mat Global descriptor (1 x descriptor_dim, CV_32F)
   */
  cv::Mat infer(const std::vector<cv::Mat>& images) override;

  /**
   * @brief Convenience single-image overload.
   * @param image Input image (BGR format)
   * @return cv::Mat Global descriptor (1 x descriptor_dim, CV_32F)
   */
  cv::Mat infer(const cv::Mat& image) override;

  int get_seq_length() const override { return 1; }
  int get_descriptor_dim() const override { return descriptor_dim_; }
  int get_img_height() const { return img_height_; }
  int get_img_width() const { return img_width_; }

 private:
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  Ort::MemoryInfo memory_info_;

  int img_height_, img_width_, descriptor_dim_;
  bool normalize_output_;

  std::vector<std::string> input_name_strings_, output_name_strings_;
  std::vector<const char*> input_names_, output_names_;

  /**
   * @brief Resize, BGR→RGB, /255, ImageNet normalize.
   * @param image Input image (BGR, any size)
   * @return Preprocessed image (RGB, float32, normalized), CV_32FC3
   */
  cv::Mat preprocess_image(const cv::Mat& image);

  /**
   * @brief Pack a CV_32FC3 mat into an NCHW float buffer [1, 3, H, W].
   * @param preprocessed Preprocessed image (CV_32FC3)
   * @return Flattened NCHW tensor data
   */
  std::vector<float> prepare_input_tensor(const cv::Mat& preprocessed);

  /**
   * @brief L2-normalize descriptor in-place.
   */
  void normalize_descriptor(cv::Mat& desc);
};

}  // namespace xfeat
