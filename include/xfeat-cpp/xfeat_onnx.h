#pragma once

#include <map>
#include <opencv2/core/mat.hpp>
#include <string>

#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
namespace xfeat {

struct TimingStats : std::map<std::string, double> {};

class XFeatONNX {
 public:
  struct DetectionResult {
    cv::Mat keypoints;
    cv::Mat scores;
    cv::Mat descriptors;
  };

  XFeatONNX(const std::string& xfeat_path,
            const std::string& interp_bilinear_path,
            const std::string& interp_bicubic_path,
            const std::string& interp_nearest_path,
            bool use_gpu);

  std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> match(const cv::Mat& image1,
                                                       const cv::Mat& image2,
                                                       int top_k = 4096,
                                                       float min_cossim = -1.0f,
                                                       TimingStats* timing_stats = nullptr);

  std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> match(const xfeat::XFeatONNX::DetectionResult& result1,
                                                       const xfeat::XFeatONNX::DetectionResult& result2,
                                                       const cv::Mat& image1,
                                                       int top_k = 4096,
                                                       float min_cossim = -1.0f,
                                                       TimingStats* timing_stats = nullptr);

  std::tuple<cv::Mat, std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>, std::vector<cv::DMatch>>
  calc_warp_corners_and_matches(const cv::Mat& ref_points, const cv::Mat& dst_points, const cv::Mat& image1);

  DetectionResult detect_and_compute(const cv::Mat& image, int top_k = 4096);

 private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session xfeat_session_;
  Ort::Session interp_bilinear_session_;
  Ort::Session interp_bicubic_session_;
  Ort::Session interp_nearest_session_;

 public:  // Make accessible
  int input_width_;
  int input_height_;

 private:
  std::string interp_input_name1_;
  std::string interp_input_name2_;

  std::tuple<cv::Mat, float, float> preprocess_image(const cv::Mat& image);

  cv::Mat get_kpts_heatmap(const Ort::Value& kpts_tensor, float softmax_temp = 1.0f);

  cv::Mat nms(const Ort::Value& heatmap_tensor, float threshold = 0.05f, int kernel_size = 5);

  // NMS on upsampled heatmap
  cv::Mat nms(const cv::Mat& heatmap, float threshold = 0.05f, int kernel_size = 5);

  DetectionResult detect_and_compute(Ort::Session& session, const cv::Mat& image, int top_k = 4096);

  std::tuple<std::vector<int>, std::vector<int>> match_mkpts(const cv::Mat& feats1,
                                                             const cv::Mat& feats2,
                                                             float min_cossim = 0.82f);

  std::tuple<std::vector<int>, std::vector<int>> match_mkpts_flann(const cv::Mat& feats1,
                                                                   const cv::Mat& feats2,
                                                                   float min_cossim = 0.82f);
};
}  // namespace xfeat