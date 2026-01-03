#pragma once

#include <onnxruntime_cxx_api.h>

#include <deque>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

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
class JistONNX {
 public:
  struct Params {
    std::string model_path;        // Path to JIST ONNX model
    bool use_gpu = true;           // Use GPU for inference
    int seq_length = 5;            // Number of frames in sequence
    int img_height = 288;          // Input image height
    int img_width = 512;           // Input image width
    int descriptor_dim = 512;      // Output descriptor dimension
    bool normalize_output = true;  // L2-normalize output descriptors
  };

  /**
   * @brief Construct a new JIST ONNX wrapper
   * @param env ONNX Runtime environment
   * @param params Configuration parameters
   */
  JistONNX(Ort::Env& env, const Params& params);

  /**
   * @brief Process a sequence of images and return a descriptor
   *
   * @param image_sequence Vector of images (seq_length images, BGR format)
   * @return cv::Mat Single descriptor vector (1 x descriptor_dim, CV_32F)
   *
   * Images will be automatically resized to the configured dimensions.
   * The sequence should contain exactly seq_length images.
   */
  cv::Mat infer(const std::vector<cv::Mat>& image_sequence);

  /**
   * @brief Process a batch of sequences
   *
   * @param batch_sequences Batch of image sequences, each with seq_length images
   * @return cv::Mat Descriptors (batch_size x descriptor_dim, CV_32F)
   */
  cv::Mat infer_batch(const std::vector<std::vector<cv::Mat>>& batch_sequences);

  /**
   * @brief Add an image to the rolling buffer and return descriptor if ready
   *
   * This is useful for online/streaming scenarios where you process images
   * one at a time and want to maintain a rolling window.
   *
   * @param image New image to add (will be added to the end of buffer)
   * @param descriptor Output descriptor (only written when buffer is full)
   * @return true if a descriptor was computed (buffer is full), false otherwise
   *
   * Example usage:
   * @code
   * JistONNX jist(env, params);
   * for (const auto& frame : video_frames) {
   *     cv::Mat descriptor;
   *     if (jist.add_frame(frame, descriptor)) {
   *         // Process descriptor for this sequence
   *         database.add(descriptor);
   *     }
   * }
   * @endcode
   */
  bool add_frame(const cv::Mat& image, cv::Mat& descriptor);

  /**
   * @brief Reset the internal rolling buffer
   *
   * Call this when starting a new video sequence or when you want to
   * clear the accumulated frames.
   */
  void reset_buffer();

  /**
   * @brief Get the current buffer size
   * @return Number of frames currently in buffer
   */
  size_t get_buffer_size() const { return frame_buffer_.size(); }

  /**
   * @brief Check if the buffer is ready (full) for inference
   * @return true if buffer has seq_length frames
   */
  bool is_buffer_ready() const { return frame_buffer_.size() >= static_cast<size_t>(seq_length_); }

  // Accessors
  int get_seq_length() const { return seq_length_; }
  int get_img_height() const { return img_height_; }
  int get_img_width() const { return img_width_; }
  int get_descriptor_dim() const { return descriptor_dim_; }

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

  // Rolling buffer for streaming inference
  std::deque<cv::Mat> frame_buffer_;

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
