// filepath:
// /Users/mikexyl/Workspaces/onnx_ws/src/XFeat-Image-Matching-ONNX-Sample/main.cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "xfeat-cpp/helpers.h"
#include "xfeat-cpp/xfeat_onnx.h"

using namespace xfeat;

static constexpr bool kDrawHeatmap = true;

int main(int argc, char* argv[]) {
  std::filesystem::path image_folder((argc > 1) ? argv[1] : "image");
  std::filesystem::path image1_path = image_folder / "sample1.png";
  std::filesystem::path image2_path = image_folder / "sample2.png";

  const std::string image_resolution = "640x352";
  constexpr int max_kpts = 500;  // Default maximum keypoints to detect

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
  const int matcher_type_int = (argc > 4) ? std::stoi(argv[4]) : static_cast<int>(MatcherType::BF);
  std::cout << "Using matcher type: " << matcher_type_int << std::endl;

  auto matcher_type = static_cast<MatcherType>(matcher_type_int);

  cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);
  cv::Mat image2 = cv::imread(image2_path, cv::IMREAD_GRAYSCALE);

  if (image1.empty() || image2.empty()) {
    std::cerr << "Error loading images! path: " << image1_path << " or " << image2_path << std::endl;
    return 1;
  }

  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "xfeat-shared-env");
    auto lighterglue_model = std::make_unique<LighterGlueOnnx>(env, lighterglue_model_path.string(), true);
    XFeatONNX xfeat_onnx(env,
                         XFeatONNX::Params{
                             .xfeat_path = xfeat_model_path.string(),
                             .interp_bilinear_path = interp_bilinear_path.string(),
                             .interp_bicubic_path = interp_bicubic_path.string(),
                             .interp_nearest_path = interp_nearest_path.string(),
                             .use_gpu = true,
                             .matcher_type = matcher_type,
                         },
                         std::move(lighterglue_model));

    xfeat_onnx.match(image1, image2, max_kpts);

    auto start = std::chrono::high_resolution_clock::now();
    TimingStats timing_stats;
    cv::Mat heatmap1, heatmap2;
    auto result1 = xfeat_onnx.detect_and_compute(image1, max_kpts, &heatmap1);
    auto result2 = xfeat_onnx.detect_and_compute(image2, max_kpts, &heatmap2);
    auto [mkpts0, mkpts1, kpts1, kpts2] = xfeat_onnx.match(result1, result2, image1, min_cos, &timing_stats);
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

      // draw the heatmap
      if (!heatmap1.empty() and !heatmap2.empty() and kDrawHeatmap) {
        cv::Mat heatmap1_colored, heatmap2_colored;
        cv::Mat heatmap1_u8, heatmap2_u8, heatmap1_norm, heatmap2_norm;
        // Normalize heatmaps to 0-255 range
        cv::normalize(heatmap1, heatmap1_norm, 0, 255, cv::NORM_MINMAX, CV_32F);
        cv::normalize(heatmap2, heatmap2_norm, 0, 255, cv::NORM_MINMAX, CV_32F);
        heatmap1.convertTo(heatmap1_u8, CV_8U, 255.0 / cv::norm(heatmap1, cv::NORM_INF));
        heatmap2.convertTo(heatmap2_u8, CV_8U, 255.0 / cv::norm(heatmap2, cv::NORM_INF));
        cv::applyColorMap(heatmap1_u8, heatmap1_colored, cv::COLORMAP_JET);
        cv::applyColorMap(heatmap2_u8, heatmap2_colored, cv::COLORMAP_JET);
        cv::resize(heatmap1_colored, heatmap1_colored, img1.size());
        cv::resize(heatmap2_colored, heatmap2_colored, img2.size());
        // Concatenate the two heatmaps to match out_img size
        cv::Mat heatmap_combined;
        cv::hconcat(heatmap1_colored, heatmap2_colored, heatmap_combined);
        // Ensure heatmap_combined and out_img have the same size
        if (heatmap_combined.size() == out_img.size()) {
          cv::addWeighted(out_img, 0.5, heatmap_combined, 0.5, 0.0, out_img);
        } else {
          std::cerr << "Heatmap and output image sizes do not match!" << std::endl;
        }

        // Draw keypoints on the heatmap
        for (const auto& kp : kpts1) {
          cv::circle(out_img, kp.pt, 5, cv::Scalar(0, 255, 0), -1);
        }
        for (const auto& kp : kpts2) {
          cv::circle(out_img, kp.pt + cv::Point2f(img1.cols, 0), 5, cv::Scalar(0, 255, 0), -1);
        }

        auto t0_std = std::chrono::high_resolution_clock::now();
        cv::Mat debug_hm;
        auto std1 = computeUncertaintySobel(heatmap1, kpts1, 1e-2, 1e-12, &debug_hm);

        auto t1_std = std::chrono::high_resolution_clock::now();
        auto std2 = computeUncertaintySobel(heatmap2, kpts2, 1e-2, 1e-12, nullptr);
        std::cout << "Computed keypoint standard deviations in "
                  << std::chrono::duration<double, std::milli>(t1_std - t0_std).count() << " ms." << std::endl;

        // draw the two heatmap sides by side
        cv::imshow("Heatmap", heatmap_combined);

        cv::imshow("debug heatmap", debug_hm);

        // print the std
        std::cout << "Keypoint uncertainties (std):" << std::endl;
        for (size_t i = 0; i < std1.size(); ++i) {
          std::cout << std1.at(i) << " ";
        }

        std::cout << std::endl;
        for (size_t i = 0; i < std2.size(); ++i) {
          std::cout << std2.at(i) << " ";
        }
        std::cout << std::endl;

        // plot the keypoints with uncertainties
        for (size_t i = 0; i < kpts1.size(); ++i) {
          cv::circle(out_img, kpts1[i].pt, static_cast<int>(std1[i][0]), cv::Scalar(255, 0, 0), 1);
          cv::circle(
              out_img, kpts2[i].pt + cv::Point2f(img1.cols, 0), static_cast<int>(std2[i][0]), cv::Scalar(255, 0, 0), 1);
        }
      }

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
