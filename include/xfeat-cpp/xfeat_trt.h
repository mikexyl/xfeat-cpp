#pragma once

#ifdef HAVE_TENSORRT

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

#include "xfeat-cpp/types.h"

namespace xfeat {
namespace trt_detail {
class Engine;
}

class XFeatTRT {
 public:
  struct Params {
    std::string engine_path;
    int nkpts = 4096;
    int anms = 0;
    int nkpts_before_anms = 0;
    int keypoint_detection = 0;
    bool verbose = false;
  };

  explicit XFeatTRT(const Params& params);
  ~XFeatTRT();

  DetectionResult detect_and_compute(cv::Mat image,
                                     int top_k = 4096,
                                     cv::Mat* heatmap = nullptr,
                                     cv::Mat* feature_map = nullptr,
                                     cv::Mat* preprocessed = nullptr,
                                     std::vector<cv::Vec2d>* uncertainty = nullptr,
                                     const std::vector<cv::KeyPoint>& keypoints = {},
                                     cv::Mat mask = {});

  int input_width() const { return input_width_; }
  int input_height() const { return input_height_; }

 private:
  std::unique_ptr<trt_detail::Engine> engine_;
  std::string input_name_;
  std::string feature_output_name_;
  std::string keypoint_output_name_;
  int input_width_ = 0;
  int input_height_ = 0;
  int anms_ = 0;
  int nkpts_before_anms_ = 0;
  int keypoint_detection_ = 0;

  std::vector<float> preprocess_image(const cv::Mat& image, float& resize_rate_w, float& resize_rate_h) const;
};

}  // namespace xfeat

#endif  // HAVE_TENSORRT
