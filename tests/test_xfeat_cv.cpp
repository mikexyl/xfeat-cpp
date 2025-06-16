#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "xfeat-cpp/lighterglue_cv.h"
#include "xfeat-cpp/xfeat_cv.h"

using namespace xfeat;
using namespace cv;

// Helper to get test image path
std::string getTestImagePath(const std::string& name) { return std::string("/workspaces/src/xfeat-cpp/image/") + name; }

TEST(XFeatCVTest, DetectAndComputeGray) {
  XFeatCV::Params params;
  params.xfeat_path = "/workspaces/src/xfeat-cpp/onnx_model/xfeat_640x352.onnx";
  params.interp_bilinear_path = "/workspaces/src/xfeat-cpp/onnx_model/interpolator_bilinear_640x352.onnx";
  params.interp_bicubic_path = "/workspaces/src/xfeat-cpp/onnx_model/interpolator_bicubic_640x352.onnx";
  params.interp_nearest_path = "/workspaces/src/xfeat-cpp/onnx_model/interpolator_nearest_640x352.onnx";
  params.use_gpu = true;
  params.max_features = 500;
  auto xfeat = XFeatCV::create(params);

  Mat img = imread(getTestImagePath("sample1.png"), IMREAD_GRAYSCALE);
  ASSERT_FALSE(img.empty());

  std::vector<KeyPoint> keypoints;
  Mat descriptors;
  xfeat->detectAndCompute(img, noArray(), keypoints, descriptors);
  EXPECT_GT(keypoints.size(), 0);
  EXPECT_EQ(descriptors.rows, keypoints.size());
  EXPECT_EQ(descriptors.cols, xfeat->descriptorSize());
}

TEST(XFeatCVTest, DetectAndComputeColor) {
  XFeatCV::Params params;
  params.xfeat_path = "/workspaces/src/xfeat-cpp/onnx_model/xfeat_640x352.onnx";
  params.interp_bilinear_path = "/workspaces/src/xfeat-cpp/onnx_model/interpolator_bilinear_640x352.onnx";
  params.interp_bicubic_path = "/workspaces/src/xfeat-cpp/onnx_model/interpolator_bicubic_640x352.onnx";
  params.interp_nearest_path = "/workspaces/src/xfeat-cpp/onnx_model/interpolator_nearest_640x352.onnx";
  params.use_gpu = true;
  params.max_features = 500;
  auto xfeat = XFeatCV::create(params);

  Mat img = imread(getTestImagePath("sample2.png"), IMREAD_COLOR);
  ASSERT_FALSE(img.empty());

  std::vector<KeyPoint> keypoints;
  Mat descriptors;
  xfeat->detectAndCompute(img, noArray(), keypoints, descriptors);
  EXPECT_GT(keypoints.size(), 0);
  EXPECT_EQ(descriptors.rows, keypoints.size());
  EXPECT_EQ(descriptors.cols, xfeat->descriptorSize());
}

TEST(LighterGlueCVTest, MatchDetectionResults) {
  // Dummy model path (replace with actual ONNX model for real test)
  LighterGlueCV::Params params;
  params.model_path = "/workspaces/src/xfeat-cpp/onnx_model/lg_640x352_500.onnx";
  params.use_gpu = true;
  params.min_score = 0.0f;
  // Use OpenCV smart pointer to avoid abstract class instantiation
  auto matcher = LighterGlueCV::create(params);

  // Create dummy DetectionResult for two images
  DetectionResult det0, det1;
  int num_kpts = 500;
  det0.keypoints = cv::Mat(num_kpts, 2, CV_32F);
  det0.descriptors = cv::Mat(num_kpts, 64, CV_32F);
  det1.keypoints = cv::Mat(num_kpts, 2, CV_32F);
  det1.descriptors = cv::Mat(num_kpts, 64, CV_32F);
  cv::randu(det0.keypoints, 0, 100);
  cv::randu(det1.keypoints, 0, 100);
  cv::randu(det0.descriptors, 0, 1);
  cv::randu(det1.descriptors, 0, 1);

  // Image sizes
  cv::Size size0(640, 352), size1(640, 352);

  // Run match
  std::vector<cv::DMatch> matches;
  matcher->match(det0, size0, det1, size1, matches);
}
