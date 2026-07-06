#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "xfeat-cpp/lighterglue_cv.h"
#include "xfeat-cpp/xfeat_cv.h"

using namespace xfeat;

// Helper to get test image path
std::string getRepoPath(const std::string& relative_path) {
  return (std::filesystem::path(XFEAT_CPP_SOURCE_DIR) / relative_path).string();
}

std::string getTestImagePath(const std::string& name) { return getRepoPath("image/" + name); }

class XFeatFullTestFixture : public ::testing::Test {
 protected:
  cv::Ptr<XFeatCV> xfeat;
  std::unique_ptr<xfeat::LighterGlueCV> lighterglue;
  cv::Mat img_gray;
  cv::Mat img_color;
  std::shared_ptr<Ort::Env> env;
  void SetUp() override {
    const char* run_model_tests = std::getenv("XFEAT_RUN_MODEL_TESTS");
    if (run_model_tests == nullptr || std::string(run_model_tests) != "1") {
      GTEST_SKIP() << "Set XFEAT_RUN_MODEL_TESTS=1 to run ONNX model integration tests";
    }

    const std::vector<std::string> required_models = {
        getRepoPath("onnx_model/xfeat_640x480.onnx"),
        getRepoPath("onnx_model/interpolator_bilinear_640x480.onnx"),
        getRepoPath("onnx_model/interpolator_bicubic_640x480.onnx"),
        getRepoPath("onnx_model/interpolator_nearest_640x480.onnx"),
        getRepoPath("onnx_model/lg_640x480_500.onnx"),
    };
    for (const auto& model_path : required_models) {
      if (!std::filesystem::exists(model_path)) {
        GTEST_SKIP() << "Missing model artifact: " << model_path;
      }
    }

    XFeatCV::Params params;
    params.xfeat_path = required_models[0];
    params.interp_bilinear_path = required_models[1];
    params.interp_bicubic_path = required_models[2];
    params.interp_nearest_path = required_models[3];
    params.use_gpu = true;
    params.max_features = 500;
    env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "xfeat-shared-env");
    xfeat = XFeatCV::create(*env, params);
    lighterglue = std::make_unique<xfeat::LighterGlueCV>(
        *env,
        xfeat::LighterGlueCV::Params{
            .model_path = required_models[4], .use_gpu = true, .min_score = -1, .n_kpts = 500});
    img_gray = imread(getTestImagePath("sample1.jpg"), IMREAD_GRAYSCALE);
    img_color = imread(getTestImagePath("sample2.jpg"), IMREAD_COLOR);
  }
};

// TEST_F(XFeatFullTestFixture, DetectAndComputeGray) {
//   ASSERT_FALSE(img_gray.empty());
//   std::vector<KeyPoint> keypoints;
//   Mat descriptors;
//   xfeat->detectAndCompute(img_gray, noArray(), keypoints, descriptors);
//   EXPECT_GT(keypoints.size(), 0);
//   EXPECT_EQ(descriptors.rows, keypoints.size());
//   EXPECT_EQ(descriptors.cols, xfeat->descriptorSize());
// }

// TEST_F(XFeatFullTestFixture, DetectAndComputeColor) {
//   ASSERT_FALSE(img_color.empty());
//   std::vector<KeyPoint> keypoints;
//   Mat descriptors;
//   xfeat->detectAndCompute(img_color, noArray(), keypoints, descriptors);
//   EXPECT_GT(keypoints.size(), 0);
//   EXPECT_EQ(descriptors.rows, keypoints.size());
//   EXPECT_EQ(descriptors.cols, xfeat->descriptorSize());
// }

TEST_F(XFeatFullTestFixture, XfeatWarmup) { xfeat->warmup(); }

TEST_F(XFeatFullTestFixture, LighterGlueWarmup) { lighterglue->warmup(); }
