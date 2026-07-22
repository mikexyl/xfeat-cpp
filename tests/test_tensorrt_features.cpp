#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

#include "xfeat-cpp/place_recognition/mixvpr_onnx.h"

#ifdef HAVE_TENSORRT
#include "xfeat-cpp/lighterglue_trt.h"
#include "xfeat-cpp/place_recognition/jist_trt.h"
#include "xfeat-cpp/place_recognition/mixvpr_trt.h"
#include "xfeat-cpp/xfeat_trt.h"
#endif

namespace {

std::string sourcePath(const std::string& relative) { return std::string(XFEAT_CPP_SOURCE_DIR) + "/" + relative; }

const char* requiredEnvironment(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::string(value).empty()) {
    return nullptr;
  }
  return value;
}

}  // namespace

TEST(TensorRTFeatures, XFeatSmokeRunsWhenEngineIsProvided) {
#ifndef HAVE_TENSORRT
  GTEST_SKIP() << "TensorRT support is not available";
#else
  const char* engine = requiredEnvironment("XFEAT_TRT_ENGINE");
  if (engine == nullptr) GTEST_SKIP() << "XFEAT_TRT_ENGINE is not set";
  const cv::Mat image = cv::imread(sourcePath("image/sample1.jpg"), cv::IMREAD_COLOR);
  ASSERT_FALSE(image.empty());

  xfeat::XFeatTRT model({.engine_path = engine});
  const xfeat::DetectionResult result = model.detect_and_compute(image, 500);
  EXPECT_GT(result.keypoints.rows, 0);
  EXPECT_LE(result.keypoints.rows, 500);
  EXPECT_EQ(result.keypoints.rows, result.scores.rows);
  EXPECT_EQ(result.keypoints.rows, result.descriptors.rows);
  EXPECT_EQ(result.descriptors.cols, 64);
#endif
}

TEST(TensorRTFeatures, LighterGlueSmokeRunsWhenEnginesAreProvided) {
#ifndef HAVE_TENSORRT
  GTEST_SKIP() << "TensorRT support is not available";
#else
  const char* xfeat_engine = requiredEnvironment("XFEAT_TRT_ENGINE");
  const char* lighterglue_engine = requiredEnvironment("LIGHTERGLUE_TRT_ENGINE");
  if (xfeat_engine == nullptr || lighterglue_engine == nullptr) {
    GTEST_SKIP() << "XFEAT_TRT_ENGINE and LIGHTERGLUE_TRT_ENGINE must both be set";
  }
  const cv::Mat image0 = cv::imread(sourcePath("image/sample1.jpg"), cv::IMREAD_COLOR);
  const cv::Mat image1 = cv::imread(sourcePath("image/sample2.jpg"), cv::IMREAD_COLOR);
  ASSERT_FALSE(image0.empty());
  ASSERT_FALSE(image1.empty());

  xfeat::XFeatTRT extractor({.engine_path = xfeat_engine});
  const xfeat::DetectionResult features0 = extractor.detect_and_compute(image0, 500);
  const xfeat::DetectionResult features1 = extractor.detect_and_compute(image1, 500);
  ASSERT_GT(features0.keypoints.rows, 0);
  ASSERT_GT(features1.keypoints.rows, 0);

  xfeat::LighterGlueTRT matcher(lighterglue_engine);
  const std::array<float, 2> size0 = {static_cast<float>(image0.cols), static_cast<float>(image0.rows)};
  const std::array<float, 2> size1 = {static_cast<float>(image1.cols), static_cast<float>(image1.rows)};
  std::vector<float> scores;
  const auto matches = matcher.match(features0, size0, features1, size1, -1.0f, &scores);
  EXPECT_EQ(matches.size(), static_cast<size_t>(features0.keypoints.rows));
  EXPECT_EQ(scores.size(), static_cast<size_t>(features0.keypoints.rows));
#endif
}

TEST(TensorRTFeatures, LighterGlueHandlesMinimumDynamicShape) {
#ifndef HAVE_TENSORRT
  GTEST_SKIP() << "TensorRT support is not available";
#else
  const char* engine = requiredEnvironment("LIGHTERGLUE_TRT_ENGINE");
  if (engine == nullptr) GTEST_SKIP() << "LIGHTERGLUE_TRT_ENGINE is not set";
  const std::vector<float> keypoint = {20.0f, 20.0f};
  std::vector<float> descriptor(64, 0.0f);
  descriptor.front() = 1.0f;
  const std::array<float, 2> image_size = {320.0f, 224.0f};

  xfeat::LighterGlueTRT matcher(engine);
  const auto [matches, scores] = matcher.match(keypoint, descriptor, image_size, keypoint, descriptor, image_size);
  EXPECT_EQ(matches.size(), scores.size());
#endif
}

TEST(TensorRTFeatures, JistSmokeRunsWhenEngineIsProvided) {
#ifndef HAVE_TENSORRT
  GTEST_SKIP() << "TensorRT support is not available";
#else
  const char* engine = requiredEnvironment("JIST_TRT_ENGINE");
  if (engine == nullptr) GTEST_SKIP() << "JIST_TRT_ENGINE is not set";
  const cv::Mat image = cv::imread(sourcePath("image/sample1.jpg"), cv::IMREAD_COLOR);
  ASSERT_FALSE(image.empty());

  xfeat::JistTRT::Params params;
  params.model_path = engine;
  xfeat::JistTRT model(params);
  const cv::Mat descriptor = model.infer(std::vector<cv::Mat>(model.get_seq_length(), image));
  EXPECT_EQ(descriptor.rows, 1);
  EXPECT_EQ(descriptor.cols, model.get_descriptor_dim());
  EXPECT_EQ(descriptor.type(), CV_32F);
  EXPECT_NEAR(cv::norm(descriptor, cv::NORM_L2), 1.0, 1e-4);
#endif
}

TEST(TensorRTFeatures, MixVPRSmokeRunsWhenEngineIsProvided) {
#ifndef HAVE_TENSORRT
  GTEST_SKIP() << "TensorRT support is not available";
#else
  const char* engine = requiredEnvironment("MIXVPR_TRT_ENGINE");
  if (engine == nullptr) GTEST_SKIP() << "MIXVPR_TRT_ENGINE is not set";
  const cv::Mat image = cv::imread(sourcePath("image/sample1.jpg"), cv::IMREAD_COLOR);
  ASSERT_FALSE(image.empty());

  xfeat::MixVPRTRT::Params params;
  params.model_path = engine;
  xfeat::MixVPRTRT model(params);
  const cv::Mat descriptor = model.infer(image);
  EXPECT_EQ(descriptor.rows, 1);
  EXPECT_EQ(descriptor.cols, 4096);
  EXPECT_EQ(descriptor.cols, model.get_descriptor_dim());
  EXPECT_EQ(descriptor.type(), CV_32F);
  EXPECT_NEAR(cv::norm(descriptor, cv::NORM_L2), 1.0, 1e-4);
#endif
}

TEST(TensorRTFeatures, MixVPRTensorRTAgreesWithONNXRuntime) {
#ifndef HAVE_TENSORRT
  GTEST_SKIP() << "TensorRT support is not available";
#else
  const char* engine = requiredEnvironment("MIXVPR_TRT_ENGINE");
  const char* onnx = requiredEnvironment("MIXVPR_ONNX_MODEL");
  if (engine == nullptr || onnx == nullptr) {
    GTEST_SKIP() << "MIXVPR_TRT_ENGINE and MIXVPR_ONNX_MODEL must both be set";
  }
  const cv::Mat image = cv::imread(sourcePath("image/sample1.jpg"), cv::IMREAD_COLOR);
  ASSERT_FALSE(image.empty());

  Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "mixvpr-backend-test");
  xfeat::MixVPRONNX::Params onnx_params;
  onnx_params.model_path = onnx;
  onnx_params.use_gpu = false;
  xfeat::MixVPRONNX onnx_model(environment, onnx_params);

  xfeat::MixVPRTRT::Params trt_params;
  trt_params.model_path = engine;
  xfeat::MixVPRTRT trt_model(trt_params);
  const cv::Mat expected = onnx_model.infer(image);
  const cv::Mat actual = trt_model.infer(image);
  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_GT(actual.dot(expected), 0.999f);
#endif
}
