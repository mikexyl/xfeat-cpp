#pragma once

#ifdef HAVE_TENSORRT

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "xfeat-cpp/place_recognition/place_recognizer.h"

namespace xfeat {
namespace trt_detail {
class Engine;
}

class JistTRT : public PlaceRecognizer {
 public:
  struct InferenceResult {
    cv::Mat sequence_descriptor;
    cv::Mat frame_descriptors;
  };

  struct Params : PlaceRecognizer::Params {
    Params() {
      img_height = 288;
      img_width = 512;
    }
    bool verbose = false;
  };

  explicit JistTRT(const Params& params);
  ~JistTRT() override;

  cv::Mat infer(const std::vector<cv::Mat>& image_sequence) override;
  InferenceResult infer_with_frame_descriptors(const std::vector<cv::Mat>& image_sequence);
  cv::Mat infer_batch(const std::vector<std::vector<cv::Mat>>& batch_sequences) override;

  int get_seq_length() const override { return seq_length_; }
  int get_img_height() const { return img_height_; }
  int get_img_width() const { return img_width_; }
  int get_descriptor_dim() const override { return descriptor_dim_; }
  bool has_frame_descriptors() const { return !frame_output_name_.empty(); }

 private:
  std::unique_ptr<trt_detail::Engine> engine_;
  std::string input_name_;
  std::string output_name_;
  std::string frame_output_name_;
  int seq_length_ = 0;
  int img_height_ = 0;
  int img_width_ = 0;
  int descriptor_dim_ = 0;
  bool normalize_output_ = true;

  cv::Mat preprocess_image(const cv::Mat& image) const;
  std::vector<float> prepare_input_tensor(const std::vector<cv::Mat>& image_sequence) const;
  void normalize_descriptor(cv::Mat& descriptor) const;
  InferenceResult infer_all(const std::vector<cv::Mat>& image_sequence);
};

}  // namespace xfeat

#endif  // HAVE_TENSORRT
