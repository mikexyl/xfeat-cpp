// filepath:
// /Users/mikexyl/Workspaces/onnx_ws/src/XFeat-Image-Matching-ONNX-Sample/main.cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "xfeat-cpp/xfeat_onnx.h"

using namespace xfeat;

int main(int argc, char* argv[]) {
  std::filesystem::path image_folder((argc > 1) ? argv[1] : "image");
  std::filesystem::path image1_path = image_folder / "sample1.png";
  std::filesystem::path image2_path = image_folder / "sample2.png";

  const std::string image_resolution = "640x352";
  constexpr int max_kpts = 1000;  // Default maximum keypoints to detect

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

  const float min_cos = (argc > 3) ? std::stof(argv[3]) : -1.0f;
  const int matcher_type_int = (argc > 4) ? std::stoi(argv[4]) : static_cast<int>(XFeatONNX::MatcherType::BF);
  std::cout << "Using matcher type: " << matcher_type_int << std::endl;

  auto matcher_type = static_cast<XFeatONNX::MatcherType>(matcher_type_int);

  cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_COLOR);
  cv::Mat image2 = cv::imread(image2_path, cv::IMREAD_COLOR);

  if (image1.empty() || image2.empty()) {
    std::cerr << "Error loading images! path: " << image1_path << " or " << image2_path << std::endl;
    return 1;
  }

  try {
    XFeatONNX xfeat_onnx(XFeatONNX::Params{.xfeat_path = xfeat_model_path.string(),
                                           .interp_bilinear_path = interp_bilinear_path.string(),
                                           .interp_bicubic_path = interp_bicubic_path.string(),
                                           .interp_nearest_path = interp_nearest_path.string(),
                                           .use_gpu = true,
                                           .matcher_type = matcher_type,
                                           .lighterglue_path = lighterglue_model_path.string()});

    xfeat_onnx.match(image1, image2, max_kpts);

    auto start = std::chrono::high_resolution_clock::now();
    TimingStats timing_stats;
    auto [mkpts0, mkpts1, kpts1, kpts2] = xfeat_onnx.match(image1, image2, max_kpts, min_cos, &timing_stats);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "xfeat_onnx detection+match+RANSAC on 2 images (size: " << image1.cols << "x" << image1.rows
              << ") took " << duration.count() << "ms." << std::endl;

    // print timing stats
    std::cout << "Timing Stats:" << std::endl;
    for (const auto& entry : timing_stats) {
      std::cout << entry.first << ": " << entry.second << " ms" << std::endl;
    }

    // Draw matches using OpenCV's drawMatches
    if (!mkpts0.empty() && !mkpts1.empty()) {
      cv::Mat img1 = cv::imread(image1_path, cv::IMREAD_COLOR);
      cv::Mat img2 = cv::imread(image2_path, cv::IMREAD_COLOR);
      std::vector<cv::KeyPoint> kpts1, kpts2;
      for (int i = 0; i < mkpts0.rows; ++i) {
        kpts1.emplace_back(mkpts0.at<float>(i, 0), mkpts0.at<float>(i, 1), 1.f);
        kpts2.emplace_back(mkpts1.at<float>(i, 0), mkpts1.at<float>(i, 1), 1.f);
      }
      // Create DMatch vector (1-to-1)
      std::vector<cv::DMatch> matches;
      for (int i = 0; i < mkpts0.rows; ++i) {
        matches.emplace_back(i, i, 0.f);
      }
      std::cout << "Number of matches: " << matches.size() << std::endl;
      cv::Mat out_img;
      cv::drawMatches(img1,
                      kpts1,
                      img2,
                      kpts2,
                      matches,
                      out_img,
                      cv::Scalar::all(-1),
                      cv::Scalar::all(-1),
                      std::vector<char>(),
                      cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
      cv::imshow("Matches", out_img);
      cv::waitKey(0);
    }

  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime Exception in main: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Standard Exception in main: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
