#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <string>
#include <vector>

#include "xfeat-cpp/mono_depth/detail/depth_anything_v3_postprocess.h"

#ifdef HAVE_TENSORRT
#include "xfeat-cpp/mono_depth/depth_anything_v3_trt.h"
#endif

namespace {
namespace detail = xfeat::mono_depth_detail;

TEST(MonoDepthPostprocess, ComputesFocalScaleAtModelResolution) {
  xfeat::CameraIntrinsics intrinsics;
  intrinsics.fx = 600.0;
  intrinsics.fy = 600.0;

  const double scale = detail::computeFocalScale(intrinsics, cv::Size(1008, 560), cv::Size(504, 280));
  EXPECT_DOUBLE_EQ(scale, 1.0);
}

TEST(MonoDepthPostprocess, ClampsInvalidDepthValues) {
  cv::Mat depth = (cv::Mat_<float>(1, 4) << -1.0f,
                   std::numeric_limits<float>::quiet_NaN(),
                   std::numeric_limits<float>::infinity(),
                   2.0f);

  cv::Mat clamped = detail::clampInvalidDepth(depth);
  EXPECT_FLOAT_EQ(clamped.at<float>(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(clamped.at<float>(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(clamped.at<float>(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(clamped.at<float>(0, 3), 2.0f);
}

TEST(MonoDepthPostprocess, ThresholdsSkyWithStrictGreaterThan) {
  cv::Mat sky = (cv::Mat_<float>(1, 3) << 0.1f, 0.3f, 0.8f);
  cv::Mat mask = detail::skyMaskFromPrediction(sky, 0.3f);

  EXPECT_EQ(mask.at<uint8_t>(0, 0), 0);
  EXPECT_EQ(mask.at<uint8_t>(0, 1), 0);
  EXPECT_EQ(mask.at<uint8_t>(0, 2), 255);
}

TEST(MonoDepthPostprocess, FillsSkyWithPercentileCappedDepth) {
  cv::Mat depth(1, 102, CV_32FC1);
  for (int x = 0; x < 101; ++x) {
    depth.at<float>(0, x) = static_cast<float>(x + 1);
  }
  depth.at<float>(0, 101) = 0.0f;

  cv::Mat sky_mask = cv::Mat::zeros(depth.size(), CV_8UC1);
  sky_mask.at<uint8_t>(0, 101) = 255;

  const float fill = detail::fillSkyDepthWithPercentile(depth, sky_mask, 50.0f);
  EXPECT_FLOAT_EQ(fill, 50.0f);
  EXPECT_FLOAT_EQ(depth.at<float>(0, 101), 50.0f);
}

TEST(MonoDepthPostprocess, PostprocessResizesAndReturnsOriginalSkyMask) {
  cv::Mat raw_depth = (cv::Mat_<float>(2, 2) << 1.0f, 2.0f, 3.0f, 4.0f);
  cv::Mat sky = (cv::Mat_<float>(2, 2) << 1.0f, 0.1f, 0.1f, 0.1f);

  detail::DepthAnythingPostprocessOptions options;
  options.original_size = cv::Size(4, 4);
  options.sky_threshold = 0.3f;
  options.sky_depth_cap = 10.0f;

  auto output = detail::postprocessDepthAnything(raw_depth, sky, options);
  EXPECT_EQ(output.depth.size(), options.original_size);
  EXPECT_EQ(output.sky_mask.size(), options.original_size);
  EXPECT_EQ(output.model_sky_mask.at<uint8_t>(0, 0), 255);
  EXPECT_FALSE(output.raw_depth.empty());
}

TEST(DepthAnythingV3TRT, SmokeRunsOnlyWhenEngineIsProvided) {
  const char* engine = std::getenv("DA3_TRT_ENGINE");
  if (engine == nullptr || std::string(engine).empty()) {
    GTEST_SKIP() << "DA3_TRT_ENGINE is not set";
  }

#ifndef HAVE_TENSORRT
  GTEST_SKIP() << "TensorRT support is not available in this build";
#else
  std::string image_path;
  if (const char* env_image = std::getenv("DA3_TRT_IMAGE")) {
    image_path = env_image;
  } else {
    image_path = std::string(XFEAT_CPP_SOURCE_DIR) + "/image/sample1.jpg";
  }

  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    GTEST_SKIP() << "DA3 smoke image is unavailable: " << image_path;
  }

  xfeat::DepthAnythingV3TRT::Params params;
  params.engine_path = engine;
  xfeat::DepthAnythingV3TRT model(params);

  auto result = model.infer(image);
  EXPECT_EQ(result.depth.type(), CV_32FC1);
  EXPECT_EQ(result.depth.size(), image.size());
  EXPECT_FALSE(result.raw_depth.empty());
#endif
}

}  // namespace
