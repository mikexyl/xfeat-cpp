#pragma once

#include <deque>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace xfeat {

/**
 * @brief Abstract base class for Visual Place Recognition (VPR) models.
 *
 * Unifies the interface for both sequence-based models (seq_length > 1,
 * e.g. JIST) and single-image models (seq_length = 1).
 *
 * Derived classes must implement:
 *   - infer(const std::vector<cv::Mat>&)
 *   - get_seq_length()
 *   - get_descriptor_dim()
 *
 * The streaming API (add_frame / reset_buffer) is provided as a concrete
 * shared implementation that delegates to infer().
 */
class PlaceRecognizer {
 public:
  virtual ~PlaceRecognizer() = default;

  /**
   * @brief Common configuration parameters shared by all VPR models.
   *
   * Derived model Params structs inherit from this and set model-specific
   * defaults for img_height and img_width.
   */
  struct Params {
    std::string model_path;       // Path to the backend-specific model or engine file
    bool use_gpu = true;          // Use GPU (CUDA) for inference
    int img_height = 0;           // Input image height (set by derived model)
    int img_width = 0;            // Input image width  (set by derived model)
    bool normalize_output = true; // L2-normalize output descriptors
  };

  /**
   * @brief Process a sequence of images and return a descriptor.
   *
   * Sequence-based models receive seq_length images; single-image models
   * receive exactly 1. Images should be in BGR format.
   *
   * @param images Vector of images (seq_length images, BGR format)
   * @return cv::Mat Single descriptor (1 x descriptor_dim, CV_32F)
   */
  virtual cv::Mat infer(const std::vector<cv::Mat>& images) = 0;

  /**
   * @brief Convenience single-image overload.
   *
   * Default implementation wraps infer({image}). Single-image derived
   * classes may override for efficiency.
   * Calling this on a sequence model (seq_length > 1) will throw.
   *
   * @param image Input image (BGR format)
   * @return cv::Mat Single descriptor (1 x descriptor_dim, CV_32F)
   */
  virtual cv::Mat infer(const cv::Mat& image);

  /**
   * @brief Batch inference over multiple sequences.
   *
   * Default implementation loops over infer(); override for GPU batching.
   *
   * @param batch Vector of sequences, each with seq_length images
   * @return cv::Mat Descriptors (batch_size x descriptor_dim, CV_32F)
   */
  virtual cv::Mat infer_batch(const std::vector<std::vector<cv::Mat>>& batch);

  /**
   * @brief Add a single frame to the rolling buffer.
   *
   * When the buffer accumulates seq_length frames, runs inference and
   * writes the descriptor to the output parameter.
   *
   * @param image New image to add (BGR format)
   * @param descriptor Output descriptor (written only when buffer is full)
   * @return true if a descriptor was produced (buffer was full), false otherwise
   */
  bool add_frame(const cv::Mat& image, cv::Mat& descriptor);

  /**
   * @brief Clear the internal rolling buffer.
   *
   * Call when starting a new sequence or resetting streaming state.
   */
  void reset_buffer();

  /**
   * @brief Return the number of frames currently in the rolling buffer.
   */
  size_t get_buffer_size() const { return frame_buffer_.size(); }

  /**
   * @brief Return true if the buffer holds seq_length frames and is ready.
   */
  bool is_buffer_ready() const { return frame_buffer_.size() >= static_cast<size_t>(get_seq_length()); }

  /**
   * @brief Return the sequence length expected by this model.
   *
   * Single-image models return 1; sequence-based models return > 1.
   */
  virtual int get_seq_length() const = 0;

  /**
   * @brief Return the dimensionality of the output descriptor.
   */
  virtual int get_descriptor_dim() const = 0;

 protected:
  std::deque<cv::Mat> frame_buffer_;
};

}  // namespace xfeat
