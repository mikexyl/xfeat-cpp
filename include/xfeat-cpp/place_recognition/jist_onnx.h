#pragma once

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "xfeat-cpp/place_recognition/place_recognizer.h"

namespace xfeat {

/**
 * @brief JIST (Joint Image Sequence Transformer) ONNX Inference Wrapper
 *
 * JIST is a sequence-based visual place recognition model that processes
 * multiple consecutive images (a sequence) to produce a single descriptor.
 *
 * The model architecture:
 * - Input: (batch, seq_length, 3, height, width) - sequence of RGB images
 * - Processing: ResNet backbone + sequence aggregation (SeqGeM by default)
 * - Output: (batch, descriptor_dim) - single descriptor per sequence
 *
 * Typical parameters:
 * - seq_length: 5 frames
 * - img_shape: [288, 512] (H x W)
 * - backbone: ResNet18
 * - fc_output_dim: 512 (frame descriptor)
 * - aggregation: seqgem (sequence descriptor same as frame)
 */
class JistONNX : public PlaceRecognizer {
 public:
  struct Params : PlaceRecognizer::Params {
    Params() {
      img_height = 288;
      img_width = 512;
    }
  };

  /**
   * @brief Construct a new JIST ONNX wrapper
   * @param env ONNX Runtime environment
   * @param params Configuration parameters
   */
  JistONNX(Ort::Env& env, const Params& params);

  /**
   * @brief Process a sequence of images and return a descriptor.
   *
   * @param image_sequence Vector of images (seq_length images, BGR format)
   * @return cv::Mat Single descriptor (1 x descriptor_dim, CV_32F)
   *
   * Images will be automatically resized to the configured dimensions.
   * The sequence should contain exactly seq_length images.
   */
  cv::Mat infer(const std::vector<cv::Mat>& image_sequence) override;

  /**
   * @brief Process a batch of sequences (efficient batched ONNX inference).
   *
   * @param batch_sequences Batch of image sequences, each with seq_length images
   * @return cv::Mat Descriptors (batch_size x descriptor_dim, CV_32F)
   */
  cv::Mat infer_batch(const std::vector<std::vector<cv::Mat>>& batch_sequences) override;

  // Accessors
  int get_seq_length() const override { return seq_length_; }
  int get_img_height() const { return img_height_; }
  int get_img_width() const { return img_width_; }
  int get_descriptor_dim() const override { return descriptor_dim_; }

 private:
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  Ort::MemoryInfo memory_info_;

  int seq_length_;
  int img_height_;
  int img_width_;
  int descriptor_dim_;
  bool normalize_output_;

  // Input/output names
  std::vector<std::string> input_name_strings_;
  std::vector<std::string> output_name_strings_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;

  /**
   * @brief Preprocess a single image
   *
   * - Resize to (img_height_, img_width_)
   * - Convert BGR to RGB
   * - Normalize to [0, 1] range
   * - Convert to float32
   *
   * @param image Input image (BGR, any size)
   * @return Preprocessed image (RGB, float32, normalized)
   */
  cv::Mat preprocess_image(const cv::Mat& image);

  /**
   * @brief Prepare input tensor from image sequence
   *
   * Combines multiple preprocessed images into a single tensor
   * of shape (1, seq_length, 3, height, width)
   *
   * @param image_sequence Preprocessed images
   * @return Flattened tensor data ready for ONNX
   */
  std::vector<float> prepare_input_tensor(const std::vector<cv::Mat>& image_sequence);

  /**
   * @brief Prepare batch input tensor
   *
   * @param batch_sequences Multiple sequences
   * @return Tensor data (batch_size, seq_length, 3, height, width)
   */
  std::vector<float> prepare_batch_input_tensor(const std::vector<std::vector<cv::Mat>>& batch_sequences);

  /**
   * @brief L2-normalize descriptor
   *
   * @param descriptor Input/output descriptor to normalize in-place
   */
  void normalize_descriptor(cv::Mat& descriptor);
};

}  // namespace xfeat
