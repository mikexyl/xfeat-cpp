#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

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
