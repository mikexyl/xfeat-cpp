// filepath:
// /Users/mikexyl/Workspaces/onnx_ws/src/XFeat-Image-Matching-ONNX-Sample/main.cpp
#include <algorithm>
#include <iostream>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "xfeat_onnx.h"

using namespace xfeat;

int main(int argc, char *argv[]) {
  std::string image1_path = (argc > 1) ? argv[1] : "image/sample1.jpg";
  std::string image2_path = (argc > 2) ? argv[2] : "image/sample2.jpg";

  std::string xfeat_model_path =
      (argc > 3) ? argv[3] : "onnx_model/xfeat_256x256.onnx";
  std::string interp_bilinear_path =
      (argc > 4) ? argv[4] : "onnx_model/interpolator_bilinear_256x256.onnx";
  std::string interp_bicubic_path =
      (argc > 5) ? argv[5] : "onnx_model/interpolator_bicubic_256x256.onnx";
  std::string interp_nearest_path =
      (argc > 6) ? argv[6] : "onnx_model/interpolator_nearest_256x256.onnx";

  cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_COLOR);
  cv::Mat image2 = cv::imread(image2_path, cv::IMREAD_COLOR);

  if (image1.empty() || image2.empty()) {
    std::cerr << "Error loading images!" << std::endl;
    return 1;
  }
  std::cout << "Images loaded successfully." << std::endl;

  try {
    XFeatONNX xfeat_onnx(xfeat_model_path, interp_bilinear_path,
                         interp_bicubic_path, interp_nearest_path,
                         false // use_gpu
    );

    auto [mkpts0, mkpts1] = xfeat_onnx.match(image1, image2);

    std::cout << "Matching complete (partially implemented)." << std::endl;
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
