#pragma once

#include <onnxruntime_cxx_api.h>

#include <map>
#include <opencv2/core/mat.hpp>
#include <string>

#include "xfeat-cpp/gpu_matcher.h"
#include "xfeat-cpp/lighterglue_onnx.h"
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
    int nkpts = 4096;                               // Default number of keypoints
    MatcherType matcher_type = MatcherType::FLANN;  // Default to BFMatcher
  };

  XFeatONNX(Ort::Env& env, const Params& params, std::unique_ptr<LighterGlueOnnx> lighterglue = nullptr);
  XFeatONNX(Ort::Env& env,
            const std::string& xfeat_path,
            const std::string& interp_bilinear_path,
            const std::string& interp_bicubic_path,
            const std::string& interp_nearest_path,
            bool use_gpu,
            int nkpts,
            MatcherType matcher_type,
            std::unique_ptr<LighterGlueOnnx> lighterglue = nullptr);

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

  std::vector<int> calc_warp_corners_and_matches(const cv::Mat& ref_points,
                                                 const cv::Mat& dst_points,
                                                 const cv::Mat& image1,
                                                 cv::Mat* H);

  DetectionResult detect_and_compute(cv::Mat image,
                                     int top_k = 4096,
                                     cv::Mat* heatmap = nullptr,
                                     cv::Mat* M1 = nullptr,
                                     cv::Mat* x_prep = nullptr,
                                     std::vector<cv::Vec2d>* std = nullptr);

 private:
  Ort::SessionOptions session_options_;
  Ort::Session xfeat_session_;
  Ort::Session interp_bilinear_session_;
  Ort::Session interp_bicubic_session_;
  Ort::Session interp_nearest_session_;

 public:
  int input_width_;
  int input_height_;
  const MatcherType matcher_type_;
  std::unique_ptr<LighterGlueOnnx> lighterglue_;
  std::unique_ptr<CuMatcher> gpu_matcher_;

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
                                     std::vector<cv::Vec2d>* std = nullptr);

  std::vector<std::vector<int>> match_mkpts_bf(const cv::Mat& feats1, const cv::Mat& feats2, float min_cossim = 0.82f);

  std::vector<std::vector<int>> match_mkpts_flann(const cv::Mat& feats1,
                                                  const cv::Mat& feats2,
                                                  float min_cossim = 0.82f);

  std::vector<std::vector<int>> match_mkpts_lg(const cv::Mat& feats1, const cv::Mat& feats2, float min_cossim = 0.82f);
};
}  // namespace xfeat