#pragma once

#include <onnxruntime_cxx_api.h>

#include <map>
#include <opencv2/core/mat.hpp>
#include <string>

#include "xfeat-cpp/gpu_matcher.h"
#include "xfeat-cpp/lighterglue_onnx.h"
#include "xfeat-cpp/nms/anms/anms.h"
#include "xfeat-cpp/types.h"

namespace xfeat {

struct TimingStats : std::map<std::string, double> {};

class XFeatONNX {
 public:
  struct Params {
    std::string xfeat_path;
    std::string interp_bilinear_path;
    std::string interp_bicubic_path;
    std::string interp_nearest_path;
    bool use_gpu = false;
    int nkpts = 4096;            // Default number of keypoints
    int anms = 0;                // Default no anms
    int nkpts_before_anms = 0;   // Default no keypoints before ANMS
    int keypoint_detection = 0;  // Default xfeat keypoint detection
  };

  XFeatONNX(Ort::Env& env, const Params& params);
  XFeatONNX(Ort::Env& env,
            const std::string& xfeat_path,
            const std::string& interp_bilinear_path,
            const std::string& interp_bicubic_path,
            const std::string& interp_nearest_path,
            bool use_gpu,
            int nkpts,
            int anms,
            int nkpts_before_anms,
            int keypoint_detection);

  std::vector<cv::DMatch> match(cv::Mat image1,
                                cv::Mat image2,
                                int top_k = 4096,
                                float min_cossim = -1.0f,
                                cv::Mat* heatmap1 = nullptr,
                                cv::Mat* heatmap2 = nullptr,
                                TimingStats* timing_stats = nullptr);

  std::vector<cv::DMatch> match(const DetectionResult& result1,
                                const DetectionResult& result2,
                                cv::Mat image1,
                                float min_cossim = -1.0f,
                                TimingStats* timing_stats = nullptr);

  DetectionResult detect_and_compute(cv::Mat image,
                                     int top_k = 4096,
                                     cv::Mat* heatmap = nullptr,
                                     cv::Mat* M1 = nullptr,
                                     cv::Mat* x_prep = nullptr,
                                     std::vector<cv::Vec2d>* std = nullptr,
                                     const std::vector<cv::KeyPoint>& keypoints = {},
                                     cv::Mat mask = {});

 private:
  Ort::SessionOptions session_options_;
  Ort::Session xfeat_session_;
  Ort::Session interp_bilinear_session_;
  Ort::Session interp_bicubic_session_;
  Ort::Session interp_nearest_session_;
  int anms_ = 0;                // Default no anms
  int nkpts_before_anms_ = 0;   // Default no keypoints before ANMS
  int keypoint_detection_ = 0;  // Default xfeat keypoint detection

 public:
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

  DetectionResult detect_and_compute(Ort::Session& session,
                                     cv::Mat image,
                                     int top_k = 4096,
                                     cv::Mat* heatmap = nullptr,
                                     cv::Mat* M1 = nullptr,
                                     cv::Mat* x_prep = nullptr,
                                     std::vector<cv::Vec2d>* std = nullptr,
                                     int anms = 0,
                                     int nkpts_before_anms = 0,
                                     int keypoint_detection = 0,
                                     const std::vector<cv::KeyPoint>& keypoints = {},
                                     cv::Mat mask = {});
};
}  // namespace xfeat