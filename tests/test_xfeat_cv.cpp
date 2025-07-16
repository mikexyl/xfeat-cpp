#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "xfeat-cpp/lighterglue_cv.h"
#include "xfeat-cpp/netvlad_onnx.h"
#include "xfeat-cpp/xfeat_cv.h"
#include "xfeat-cpp/xfeat_netvlad_onnx.h"

using namespace xfeat;
using namespace cv;

// Helper to get test image path
std::string getTestImagePath(const std::string& name) { return std::string("/workspaces/src/xfeat-cpp/image/") + name; }

class XFeatFullTestFixture : public ::testing::Test {
 protected:
  cv::Ptr<XFeatCV> xfeat;
  std::unique_ptr<xfeat::HeadNetVLADONNX> head_netvlad;
  std::unique_ptr<xfeat::NetVLADONNX> netvlad;
  cv::Mat img_gray;
  cv::Mat img_color;
  std::shared_ptr<Ort::Env> env;
  void SetUp() override {
    XFeatCV::Params params;
    params.xfeat_path = "/workspaces/src/xfeat-cpp/onnx_model/xfeat_640x480.onnx";
    params.interp_bilinear_path = "/workspaces/src/xfeat-cpp/onnx_model/interpolator_bilinear_640x480.onnx";
    params.interp_bicubic_path = "/workspaces/src/xfeat-cpp/onnx_model/interpolator_bicubic_640x480.onnx";
    params.interp_nearest_path = "/workspaces/src/xfeat-cpp/onnx_model/interpolator_nearest_640x480.onnx";
    params.use_gpu = true;
    params.max_features = 500;
    env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "xfeat-shared-env");
    xfeat = XFeatCV::create(*env, params);
    head_netvlad = std::make_unique<xfeat::HeadNetVLADONNX>(*env, "/workspaces/src/xfeat-cpp/onnx_model/xfeat_nv.onnx");
    netvlad = std::make_unique<xfeat::NetVLADONNX>(*env, "/workspaces/src/xfeat-cpp/onnx_model/netvlad.onnx");
    img_gray = imread(getTestImagePath("sample1.png"), IMREAD_GRAYSCALE);
    img_color = imread(getTestImagePath("sample2.png"), IMREAD_COLOR);
  }
};

TEST_F(XFeatFullTestFixture, DetectAndComputeGray) {
  ASSERT_FALSE(img_gray.empty());
  std::vector<KeyPoint> keypoints;
  Mat descriptors;
  xfeat->detectAndCompute(img_gray, noArray(), keypoints, descriptors);
  EXPECT_GT(keypoints.size(), 0);
  EXPECT_EQ(descriptors.rows, keypoints.size());
  EXPECT_EQ(descriptors.cols, xfeat->descriptorSize());
}

TEST_F(XFeatFullTestFixture, DetectAndComputeColor) {
  ASSERT_FALSE(img_color.empty());
  std::vector<KeyPoint> keypoints;
  Mat descriptors;
  xfeat->detectAndCompute(img_color, noArray(), keypoints, descriptors);
  EXPECT_GT(keypoints.size(), 0);
  EXPECT_EQ(descriptors.rows, keypoints.size());
  EXPECT_EQ(descriptors.cols, xfeat->descriptorSize());
}

TEST_F(XFeatFullTestFixture, HeadNetvladInferenceWithXFeatOutput) {
  ASSERT_FALSE(img_gray.empty());
  std::vector<KeyPoint> keypoints;
  cv::Mat descriptors;
  cv::Mat M1, x_prep;
  xfeat->detectAndCompute(img_gray, noArray(), keypoints, descriptors, false, &M1, &x_prep);

  // print the size of M1 and x_prep
  std::cout << "M1 size: " << M1.size << ", type : " << M1.type() << std::endl;
  std::cout << "x_prep size: " << x_prep.size << ", type: " << x_prep.type() << std::endl;

  EXPECT_GT(keypoints.size(), 0);
  EXPECT_EQ(descriptors.rows, keypoints.size());
  EXPECT_EQ(descriptors.cols, xfeat->descriptorSize());

  std::cout << "M1 type: " << M1.type() << " (should be " << CV_32F << ")" << std::endl;
  std::cout << "x_prep type: " << x_prep.type() << " (should be " << CV_32F << ")" << std::endl;

  cv::Mat output = head_netvlad->run(M1, x_prep);
  ASSERT_EQ(output.dims, 4);
  EXPECT_EQ(output.size[1], 256);
  EXPECT_EQ(output.size[2], 30);
  EXPECT_EQ(output.size[3], 40);
  EXPECT_EQ(output.type(), CV_32F);
}

TEST_F(XFeatFullTestFixture, HeadNetvladToNetvladInferenceWithXFeatOutput) {
  ASSERT_FALSE(img_gray.empty());
  std::vector<KeyPoint> keypoints;
  cv::Mat descriptors;
  cv::Mat M1, x_prep;
  xfeat->detectAndCompute(img_gray, noArray(), keypoints, descriptors, false, &M1, &x_prep);

  // Ensure M1 and x_prep are CV_32F
  if (M1.type() != CV_32F) {
    M1.convertTo(M1, CV_32F);
  }
  if (x_prep.type() != CV_32F) {
    x_prep.convertTo(x_prep, CV_32F);
  }

  cv::Mat head_output = head_netvlad->run(M1, x_prep);
  ASSERT_EQ(head_output.dims, 4);
  EXPECT_EQ(head_output.size[1], 256);
  EXPECT_EQ(head_output.size[2], 30);
  EXPECT_EQ(head_output.size[3], 40);
  EXPECT_EQ(head_output.type(), CV_32F);

  // Feed head_output to netvlad
  auto netvlad_result = netvlad->infer(head_output);
  // netvlad_result: std::vector<std::vector<float>>
  ASSERT_FALSE(netvlad_result.empty());
  size_t batch_size = head_output.size[0];
  size_t output_dim = netvlad_result[0].size();
  EXPECT_EQ(netvlad_result.size(), batch_size);
  // Print output dim for debug
  std::cout << "NetVLAD output dim: " << output_dim << std::endl;
  // Optionally, check values are finite
  for (const auto& vec : netvlad_result) {
    for (float v : vec) {
      EXPECT_TRUE(std::isfinite(v));
    }
  }
}
