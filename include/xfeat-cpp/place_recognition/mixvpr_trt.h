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

/** Native TensorRT backend for the single-image ResNet-50 MixVPR model. */
class MixVPRTRT : public PlaceRecognizer {
 public:
  struct Params : PlaceRecognizer::Params {
    Params() {
      img_height = 320;
      img_width = 320;
    }
    bool verbose = false;
  };

  explicit MixVPRTRT(const Params& params);
  ~MixVPRTRT() override;

  cv::Mat infer(const cv::Mat& image) override;
  cv::Mat infer(const std::vector<cv::Mat>& images) override;

  int get_seq_length() const override { return 1; }
  int get_descriptor_dim() const override { return descriptor_dim_; }
  int get_img_height() const { return img_height_; }
  int get_img_width() const { return img_width_; }

 private:
  std::unique_ptr<trt_detail::Engine> engine_;
  std::string input_name_;
  std::string output_name_;
  int img_height_ = 0;
  int img_width_ = 0;
  int descriptor_dim_ = 0;
  bool normalize_output_ = true;

  std::vector<float> prepare_input_tensor(const cv::Mat& image) const;
  void normalize_descriptor(cv::Mat& descriptor) const;
};

}  // namespace xfeat

#endif  // HAVE_TENSORRT
