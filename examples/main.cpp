// filepath:
// /Users/mikexyl/Workspaces/onnx_ws/src/XFeat-Image-Matching-ONNX-Sample/main.cpp
#include <filesystem>
#include <iostream>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "xfeat-cpp/xfeat_onnx.h"

using namespace xfeat;

int main(int argc, char *argv[]) {
  std::filesystem::path image_folder((argc > 1) ? argv[1] : "image");
  std::filesystem::path image1_path = image_folder / "sample1.png";
  std::filesystem::path image2_path = image_folder / "sample2.png";

  std::filesystem::path xfeat_model_folder =
      (argc > 2) ? argv[2] : "onnx_model";
  std::filesystem::path xfeat_model_path =
      xfeat_model_folder / "xfeat_640x352.onnx";
  std::filesystem::path interp_bilinear_path =
      xfeat_model_folder / "interpolator_bilinear_640x352.onnx";
  std::filesystem::path interp_bicubic_path =
      xfeat_model_folder / "interpolator_bicubic_640x352.onnx";
  std::filesystem::path interp_nearest_path =
      xfeat_model_folder / "interpolator_nearest_640x352.onnx";

  cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_COLOR);
  cv::Mat image2 = cv::imread(image2_path, cv::IMREAD_COLOR);

  if (image1.empty() || image2.empty()) {
    std::cerr << "Error loading images! path: " << image1_path << " or "
              << image2_path << std::endl;
    return 1;
  }

  try {
    XFeatONNX xfeat_onnx(xfeat_model_path, interp_bilinear_path,
                         interp_bicubic_path, interp_nearest_path, true);

    auto [mkpts0, mkpts1, kpts1, kpts2] = xfeat_onnx.match(image1, image2);

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
      cv::drawMatches(img1, kpts1, img2, kpts2, matches, out_img,
                      cv::Scalar::all(-1), cv::Scalar::all(-1),
                      std::vector<char>(),
                      cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
      cv::imshow("Matches", out_img);
      cv::waitKey(0);
    }

  } catch (const Ort::Exception &e) {
    std::cerr << "ONNX Runtime Exception in main: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Standard Exception in main: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
