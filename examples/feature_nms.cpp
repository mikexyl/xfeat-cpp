#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <random>

#include "onnxruntime_cxx_api.h"
#include "xfeat-cpp/xfeat_onnx.h"

using namespace xfeat;

int main(int argc, char** argv) {
  const int N = 2000, K = 8;
  cv::Mat topkVals(N, K, CV_32F);
  std::vector<cv::Point2f> kpts(N);

  const std::string image_resolution = "640x480";
  constexpr int max_kpts = 2000;  // Default maximum keypoints to detect

  std::filesystem::path xfeat_model_folder = (argc > 2) ? argv[2] : "onnx_model";
  std::filesystem::path xfeat_model_path = xfeat_model_folder / ("xfeat_" + image_resolution + ".onnx");
  std::filesystem::path interp_bilinear_path =
      xfeat_model_folder / ("interpolator_bilinear_" + image_resolution + ".onnx");
  std::filesystem::path interp_bicubic_path =
      xfeat_model_folder / ("interpolator_bicubic_" + image_resolution + ".onnx");
  std::filesystem::path interp_nearest_path =
      xfeat_model_folder / ("interpolator_nearest_" + image_resolution + ".onnx");
  std::filesystem::path lighterglue_model_path =
      xfeat_model_folder / ("lg_" + image_resolution + "_" + std::to_string(max_kpts) + ".onnx");

  cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "xfeat-shared-env");

  XFeatONNX xfeat_onnx(env,
                       XFeatONNX::Params{
                           .xfeat_path = xfeat_model_path.string(),
                           .interp_bilinear_path = interp_bilinear_path.string(),
                           .interp_bicubic_path = interp_bicubic_path.string(),
                           .interp_nearest_path = interp_nearest_path.string(),
                           .use_gpu = true,
                           .nkpts = max_kpts,
                           .matcher_type = MatcherType::GPU_BF,
                       },
                       nullptr);

  xfeat_onnx.detect_and_compute(image, max_kpts, nullptr, {}, {}, nullptr);

  auto tic = cv::getTickCount();
  auto result = xfeat_onnx.detect_and_compute(image, max_kpts, nullptr, {}, {}, nullptr);
  auto toc = cv::getTickCount();
  std::cout << "Inference time: " << (toc - tic) / cv::getTickFrequency() << " seconds" << std::endl;
  std::cout << "Found " << result.keypoints.rows << " keypoints." << std::endl;

  // draw the points
  cv::Mat color_image;
  cv::cvtColor(image, color_image, cv::COLOR_GRAY2BGR);
  for (int i = 0; i < result.keypoints.rows; ++i) {
    cv::Point2f kp = result.keypoints.at<cv::Point2f>(i);
    cv::circle(image, kp, 1, cv::Scalar(255, 0, 0), -1);
  }

  cv::imshow("Keypoints", image);
  cv::waitKey(0);

  return 0;
}
