#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "xfeat-cpp/helpers.h"
#include "xfeat-cpp/lighterglue_cv.h"
#include "xfeat-cpp/xfeat_onnx.h"

using namespace xfeat;

/**
 * Find, for each keypoint, the indices of visually–and–spatially similar neighbours.
 *
 * @param keypoints     N×2 matrix (CV_32F) of (x,y) image coordinates.
 * @param descriptors   N×D descriptor matrix (CV_8U for ORB; CV_32F for float nets).
 * @param neighbours    Output: neighbours[i] holds indices of keypoints similar to i.
 * @param k             Max. number of descriptor neighbours to consider (>=1).
 * @param maxDescDist   Hamming (for CV_8U) or L2 (for CV_32F) threshold.
 * @param maxPixDist    Maximum pixel distance between keypoints to be considered neighbours.
 */
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * Find visually + spatially similar neighbours for a *masked* subset of keypoints,
 * using an individual descriptor-distance threshold for each keypoint.
 *
 * @param keypoints        N×2  (CV_32F) matrix of (x,y) image coordinates.
 * @param descriptors      N×D  descriptor matrix (CV_8U binary OR CV_32F float).
 * @param neighbours       Output: size N.  For i with mask(i)=0 list is empty.
 * @param maxDescDistVec   Length-N vector: max. descriptor distance tolerated for each query kp.
 *                         • For binary descriptors ➜ Hamming threshold (int, 0–256)
 *                         • For float  descriptors ➜ L2 threshold      (float)
 * @param mask             Optional N×1 (CV_8U) column; non-zero ⇒ kp is considered. Pass cv::Mat() to use all.
 * @param k                How many nearest-descriptor candidates to test (≥1, default 3).
 * @param maxPixDist       Maximum pixel distance allowed between neighbouring keypoints.
 */
inline void findSimilarNeighbours(const cv::Mat& keypoints,
                                  const cv::Mat& descriptors,
                                  std::vector<std::vector<int>>& neighbours,
                                  const std::vector<float>& maxDescDistVec,  // NEW
                                  const cv::Mat& mask = cv::Mat(),
                                  int k = 3,
                                  float maxPixDist = 5.f) {
  CV_Assert(keypoints.rows == descriptors.rows);
  const int N = keypoints.rows;
  CV_Assert(static_cast<int>(maxDescDistVec.size()) == N);
  CV_Assert(keypoints.cols == 2);
  CV_Assert(descriptors.rows == N);

  /* ---------- prepare / sanity-check mask ---------- */
  cv::Mat useMask;
  if (mask.empty()) {
    useMask = cv::Mat::ones(N, 1, CV_8U);
  } else {
    CV_Assert(mask.rows == N && mask.cols == 1 && mask.type() == CV_8U);
    useMask = mask;
  }

  neighbours.assign(N, {});

  /* ---------- choose descriptor norm automatically ---------- */
  const bool binaryDesc = descriptors.type() == CV_8U || descriptors.type() == CV_8S;
  const int normType = binaryDesc ? cv::NORM_HAMMING : cv::NORM_L2;

  /* ---------- 1. K-NN in descriptor space (self-match included) ---------- */
  cv::BFMatcher matcher(normType, /*crossCheck=*/false);
  std::vector<std::vector<cv::DMatch>> knnMatches;
  matcher.knnMatch(descriptors, descriptors, knnMatches, k + 1);  // +1 for self

  /* ---------- 2. filter by mask, per-kp descriptor gate, pixel gate ---------- */
  for (int i = 0; i < N; ++i) {
    if (!useMask.at<uchar>(i))  // kp i isn't part of the subset
      continue;

    const float descGate_i = maxDescDistVec[i];
    const cv::Point2f pi = keypoints.at<cv::Point2f>(i);

    for (const auto& m : knnMatches[i]) {
      const int j = m.trainIdx;
      if (j == i)  // skip self
        continue;
      if (!useMask.at<uchar>(j))  // neighbour not in subset
        continue;
      if (m.distance > descGate_i)  // exceeds kp-specific descriptor gate
        continue;

      const cv::Point2f pj = keypoints.at<cv::Point2f>(j);
      if (cv::norm(pi - pj) > maxPixDist) continue;

      neighbours[i].push_back(j);
    }
  }
}

static constexpr bool kDrawHeatmap = true;

int main(int argc, char* argv[]) {
  std::filesystem::path image_folder((argc > 1) ? argv[1] : "image");
  std::filesystem::path image1_path = image_folder / "sample1.jpg";
  std::filesystem::path image2_path = image_folder / "sample2.jpg";

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

  const float min_cos = (argc > 3) ? std::stof(argv[3]) : -1.0f;
  const int matcher_type_int = (argc > 4) ? std::stoi(argv[4]) : static_cast<int>(MatcherType::BF);
  std::cout << "Using matcher type: " << matcher_type_int << std::endl;

  cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);
  cv::Mat image2 = cv::imread(image2_path, cv::IMREAD_GRAYSCALE);

  if (image1.empty() || image2.empty()) {
    std::cerr << "Error loading images! path: " << image1_path << " or " << image2_path << std::endl;
    return 1;
  }

  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "xfeat-shared-env");
    auto lighterglue = std::make_unique<LighterGlueOnnx>(env, lighterglue_model_path.string(),
                                                         true);  // Use GPU

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
                         std::move(lighterglue));

    xfeat::CuMatcher gpu_matcher;
    gpu_matcher.init(max_kpts, max_kpts, 64);

    // warm up the model
    auto result1 = xfeat_onnx.detect_and_compute(image1, max_kpts, nullptr, {}, {}, nullptr);
    auto result2 = xfeat_onnx.detect_and_compute(image2, max_kpts, nullptr, {}, {}, nullptr);
    xfeat_onnx.match(result1, result2, image1, min_cos, nullptr);

    auto start = std::chrono::high_resolution_clock::now();
    TimingStats timing_stats;
    cv::Mat heatmap1, heatmap2;
    std::vector<cv::Vec2d> std1;
    std::vector<cv::Vec2d> std2;
    result1 = xfeat_onnx.detect_and_compute(image1, max_kpts, &heatmap1, {}, {}, &std1);
    result2 = xfeat_onnx.detect_and_compute(image2, max_kpts, &heatmap2, {}, {}, &std2);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "xfeat_onnx detection on 2 images (size: " << image1.cols << "x" << image1.rows << ") took "
              << duration.count() << "ms." << std::endl;
    std::vector<std::vector<int>> self_neighbours1, self_neighbours2;

    // lighterglue->match(result1, image1.size(), result2, image2.size(), matches);

    xfeat::TimingStats match_timing_stats;
    auto matches = gpu_matcher.match(result1, result2, 0.4, cv::Mat());

    for (auto stats : match_timing_stats) {
      std::cout << "Match timing stats: " << stats.first << ": " << stats.second << " ms" << std::endl;
    }

    // print timing stats
    std::cout << "Timing Stats:" << std::endl;
    for (const auto& entry : timing_stats) {
      std::cout << entry.first << ": " << entry.second << " ms" << std::endl;
    }

    // Draw matches using OpenCV's drawMatches
    if (not matches.empty()) {
      cv::Mat img1 = cv::imread(image1_path, cv::IMREAD_COLOR);
      cv::Mat img2 = cv::imread(image2_path, cv::IMREAD_COLOR);

      // draw self matches
      for (size_t i = 0; i < self_neighbours1.size(); ++i) {
        for (int j : self_neighbours1[i]) {
          cv::line(img1,
                   result1.keypoints.at<cv::Point2f>(i),
                   result1.keypoints.at<cv::Point2f>(j),
                   cv::Scalar(255, 255, 0),
                   1);
        }
      }
      for (size_t i = 0; i < self_neighbours2.size(); ++i) {
        for (int j : self_neighbours2[i]) {
          cv::line(img2,
                   result2.keypoints.at<cv::Point2f>(i),
                   result2.keypoints.at<cv::Point2f>(j),
                   cv::Scalar(255, 255, 0),
                   1);
        }
      }

      std::cout << "Number of matches: " << matches.size() << std::endl;
      cv::Mat out_img;

      std::vector<cv::KeyPoint> kpts1, kpts2;
      for (int i = 0; i < result1.keypoints.rows; ++i) {
        kpts1.emplace_back(result1.keypoints.at<cv::Point2f>(i), 1);
      }
      for (int i = 0; i < result2.keypoints.rows; ++i) {
        kpts2.emplace_back(result2.keypoints.at<cv::Point2f>(i), 1);
      }

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
      cv::Mat track_img = img2.clone();
      // Draw lines for each match
      for (auto match : matches) {
        const cv::Point2f& pt1 = kpts1[match.queryIdx].pt;
        const cv::Point2f& pt2 = kpts2[match.trainIdx].pt;
        cv::line(track_img, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        // Optionally, draw circles at keypoints
        cv::circle(track_img, pt1, 4, cv::Scalar(0, 0, 255), -1);
        cv::circle(track_img, pt2, 4, cv::Scalar(255, 0, 0), -1);
      }
      cv::imshow("track_img", track_img);

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

        // draw the two heatmap sides by side
        cv::imshow("Heatmap", heatmap_combined);

        // plot the keypoints with uncertainties
        for (auto match : matches) {
          const cv::Point2f& pt1 = result1.keypoints.at<cv::Point2f>(match.queryIdx);
          const cv::Point2f& pt2 = result2.keypoints.at<cv::Point2f>(match.trainIdx) + cv::Point2f(img1.cols, 0);
          // draw keypoints
          // draw eclipse for uncertainty
          cv::ellipse(out_img,
                      pt1,
                      cv::Size(static_cast<int>(std1[match.queryIdx][0]), static_cast<int>(std1[match.queryIdx][1])),
                      0,
                      0,
                      360,
                      cv::Scalar(0, 255, 0, 50),
                      1);
          cv::ellipse(out_img,
                      pt2,
                      cv::Size(static_cast<int>(std2[match.trainIdx][0]), static_cast<int>(std2[match.trainIdx][1])),
                      0,
                      0,
                      360,
                      cv::Scalar(0, 255, 0, 50),
                      1);
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
