#include "xfeat-cpp/place_recognition/place_recognizer.h"

#include <stdexcept>

namespace xfeat {

cv::Mat PlaceRecognizer::infer(const cv::Mat& image) {
  if (get_seq_length() > 1) {
    throw std::runtime_error(
        "PlaceRecognizer::infer(single image) called on a sequence-based model "
        "(seq_length=" +
        std::to_string(get_seq_length()) + "). Use infer(std::vector<cv::Mat>) instead.");
  }
  return infer(std::vector<cv::Mat>{image});
}

cv::Mat PlaceRecognizer::infer_batch(const std::vector<std::vector<cv::Mat>>& batch) {
  if (batch.empty()) {
    throw std::runtime_error("PlaceRecognizer::infer_batch: empty batch");
  }

  // Collect descriptors row by row
  cv::Mat result;
  for (const auto& sequence : batch) {
    cv::Mat descriptor = infer(sequence);
    result.push_back(descriptor);
  }
  return result;
}

bool PlaceRecognizer::add_frame(const cv::Mat& image, cv::Mat& descriptor) {
  frame_buffer_.push_back(image.clone());

  if (frame_buffer_.size() > static_cast<size_t>(get_seq_length())) {
    frame_buffer_.pop_front();
  }

  if (frame_buffer_.size() == static_cast<size_t>(get_seq_length())) {
    std::vector<cv::Mat> sequence(frame_buffer_.begin(), frame_buffer_.end());
    descriptor = infer(sequence);
    return true;
  }

  return false;
}

void PlaceRecognizer::reset_buffer() { frame_buffer_.clear(); }

}  // namespace xfeat
