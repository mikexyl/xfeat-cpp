#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <string>
#include <vector>

#include "xfeat-cpp/mono_depth/detail/depth_anything_v3_postprocess.h"
#include "xfeat-cpp/mono_depth/detail/depth_anything_v3_pose.h"
#include "xfeat-cpp/mono_depth/detail/depth_anything_v3_tensor.h"

#ifdef HAVE_TENSORRT
#include "xfeat-cpp/mono_depth/depth_anything_v3_trt.h"
#endif

namespace {
namespace detail = xfeat::mono_depth_detail;

std::vector<float> sequenceValues(size_t count) {
  std::vector<float> values(count);
  std::iota(values.begin(), values.end(), 0.0f);
  return values;
}

void expectPlaneStartsAt(const cv::Mat& plane, int height, int width, float first_value) {
  ASSERT_EQ(plane.type(), CV_32FC1);
  ASSERT_EQ(plane.rows, height);
  ASSERT_EQ(plane.cols, width);
  EXPECT_FLOAT_EQ(plane.at<float>(0, 0), first_value);
  EXPECT_FLOAT_EQ(plane.at<float>(height - 1, width - 1), first_value + static_cast<float>(height * width - 1));
}

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

TEST(MonoDepthTensorShapes, ExtractsViewHeightWidthOutput) {
  const auto planes = detail::extractDepthAnythingTensorPlanes(sequenceValues(3 * 2 * 4), {3, 2, 4}, 3, "depth");

  ASSERT_EQ(planes.size(), 3u);
  expectPlaneStartsAt(planes[0], 2, 4, 0.0f);
  expectPlaneStartsAt(planes[1], 2, 4, 8.0f);
  expectPlaneStartsAt(planes[2], 2, 4, 16.0f);
}

TEST(MonoDepthTensorShapes, ExtractsViewChannelHeightWidthOutput) {
  const auto planes = detail::extractDepthAnythingTensorPlanes(sequenceValues(3 * 1 * 2 * 4), {3, 1, 2, 4}, 3, "depth");

  ASSERT_EQ(planes.size(), 3u);
  expectPlaneStartsAt(planes[0], 2, 4, 0.0f);
  expectPlaneStartsAt(planes[1], 2, 4, 8.0f);
  expectPlaneStartsAt(planes[2], 2, 4, 16.0f);
}

TEST(MonoDepthTensorShapes, ExtractsBatchViewChannelHeightWidthOutput) {
  const auto planes =
      detail::extractDepthAnythingTensorPlanes(sequenceValues(1 * 3 * 1 * 2 * 4), {1, 3, 1, 2, 4}, 3, "depth");

  ASSERT_EQ(planes.size(), 3u);
  expectPlaneStartsAt(planes[0], 2, 4, 0.0f);
  expectPlaneStartsAt(planes[1], 2, 4, 8.0f);
  expectPlaneStartsAt(planes[2], 2, 4, 16.0f);
}

TEST(MonoDepthTensorShapes, ExtractsBatchViewHeightWidthOutput) {
  const auto planes = detail::extractDepthAnythingTensorPlanes(sequenceValues(1 * 3 * 2 * 4), {1, 3, 2, 4}, 3, "depth");

  ASSERT_EQ(planes.size(), 3u);
  expectPlaneStartsAt(planes[0], 2, 4, 0.0f);
  expectPlaneStartsAt(planes[1], 2, 4, 8.0f);
  expectPlaneStartsAt(planes[2], 2, 4, 16.0f);
}

TEST(MonoDepthTensorShapes, ExtractsBatchViewHeightWidthChannelOutput) {
  const auto planes =
      detail::extractDepthAnythingTensorPlanes(sequenceValues(1 * 3 * 2 * 4 * 1), {1, 3, 2, 4, 1}, 3, "depth");

  ASSERT_EQ(planes.size(), 3u);
  expectPlaneStartsAt(planes[0], 2, 4, 0.0f);
  expectPlaneStartsAt(planes[1], 2, 4, 8.0f);
  expectPlaneStartsAt(planes[2], 2, 4, 16.0f);
}

TEST(MonoDepthPoseScale, UsesPredictedToInputBaselineRatio) {
  std::vector<cv::Matx44f> input{cv::Matx44f::eye(), cv::Matx44f::eye()};
  std::vector<cv::Matx44f> predicted{cv::Matx44f::eye(), cv::Matx44f::eye()};
  input[1](0, 3) = -10.0f;
  predicted[1](0, 3) = -20.0f;

  EXPECT_DOUBLE_EQ(detail::estimateInputToPredictedPoseScale(predicted, input), 2.0);
}

TEST(MonoDepthPoseScale, RejectsUnavailableDegenerateAndNonFiniteBaselines) {
  const std::vector<cv::Matx44f> one_pose{cv::Matx44f::eye()};
  const std::vector<cv::Matx44f> two_poses{cv::Matx44f::eye(), cv::Matx44f::eye()};
  EXPECT_THROW(detail::estimateInputToPredictedPoseScale(one_pose, two_poses), std::invalid_argument);
  EXPECT_THROW(detail::estimateInputToPredictedPoseScale(two_poses, two_poses), std::runtime_error);

  std::vector<cv::Matx44f> non_finite = two_poses;
  non_finite[1](0, 3) = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(detail::estimateInputToPredictedPoseScale(non_finite, two_poses), std::invalid_argument);
}

TEST(DepthAnythingV3TRT, TwoViewSmokeRunsOnlyWhenEngineIsProvided) {
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

  EXPECT_FALSE(model.has_camera_inputs());
  xfeat::CameraIntrinsics intrinsics;
  intrinsics.fx = static_cast<double>(std::max(image.cols, image.rows));
  intrinsics.fy = intrinsics.fx;
  intrinsics.cx = 0.5 * static_cast<double>(image.cols - 1);
  intrinsics.cy = 0.5 * static_cast<double>(image.rows - 1);
  intrinsics.width = image.cols;
  intrinsics.height = image.rows;

  const std::vector<cv::Mat> images{image, image};
  const std::vector<xfeat::CameraIntrinsics> batch_intrinsics{
      intrinsics, intrinsics};
  const auto results = model.infer_multi_view(images, batch_intrinsics);
  ASSERT_EQ(results.size(), 2u);
  for (std::size_t view = 0; view < results.size(); ++view) {
    const auto& result = results[view];
    EXPECT_EQ(result.depth.type(), CV_32FC1);
    EXPECT_EQ(result.depth.size(), image.size());
    EXPECT_EQ(result.confidence.type(), CV_32FC1);
    EXPECT_EQ(result.confidence.size(), image.size());
    EXPECT_FALSE(result.raw_depth.empty());
    EXPECT_FALSE(result.raw_confidence.empty());
    EXPECT_TRUE(result.predicted_world_to_camera.has_value());
    EXPECT_DOUBLE_EQ(result.metadata.pose_scale, 1.0);
    EXPECT_FALSE(result.metadata.pose_scaled);
    EXPECT_EQ(result.metadata.view_index, static_cast<int>(view));
    EXPECT_EQ(result.metadata.view_count, 2);
  }
  const auto camera_center = [](const cv::Matx44f& extrinsic) {
    const cv::Vec3f translation(
        extrinsic(0, 3), extrinsic(1, 3), extrinsic(2, 3));
    const cv::Matx33f rotation(extrinsic(0, 0),
                               extrinsic(0, 1),
                               extrinsic(0, 2),
                               extrinsic(1, 0),
                               extrinsic(1, 1),
                               extrinsic(1, 2),
                               extrinsic(2, 0),
                               extrinsic(2, 1),
                               extrinsic(2, 2));
    return -(rotation.t() * translation);
  };
  const cv::Vec3f da3_baseline =
      camera_center(*results[1].predicted_world_to_camera) -
      camera_center(*results[0].predicted_world_to_camera);
  EXPECT_TRUE(std::isfinite(cv::norm(da3_baseline)));
  EXPECT_GT(cv::norm(da3_baseline), 1e-6);
#endif
}

TEST(DepthAnythingV3TRT, PoseConditionedScaleRunsOnlyWhenEngineIsProvided) {
  const char* engine = std::getenv("DA3_POSE_TRT_ENGINE");
  if (engine == nullptr || std::string(engine).empty()) {
    GTEST_SKIP() << "DA3_POSE_TRT_ENGINE is not set";
  }

#ifndef HAVE_TENSORRT
  GTEST_SKIP() << "TensorRT support is not available in this build";
#else
  const std::string image_path = std::string(XFEAT_CPP_SOURCE_DIR) + "/image/sample1.jpg";
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    GTEST_SKIP() << "DA3 smoke image is unavailable: " << image_path;
  }

  xfeat::DepthAnythingV3TRT::Params params;
  params.engine_path = engine;
  xfeat::DepthAnythingV3TRT model(params);
  ASSERT_TRUE(model.has_pose_inputs());

  xfeat::CameraIntrinsics intrinsics;
  intrinsics.fx = static_cast<double>(std::max(image.cols, image.rows));
  intrinsics.fy = intrinsics.fx;
  intrinsics.cx = 0.5 * static_cast<double>(image.cols - 1);
  intrinsics.cy = 0.5 * static_cast<double>(image.rows - 1);
  intrinsics.width = image.cols;
  intrinsics.height = image.rows;
  const std::vector<cv::Mat> images{image, image};
  const std::vector<xfeat::CameraIntrinsics> batch_intrinsics{intrinsics, intrinsics};
  std::vector<cv::Matx44f> input_world_to_camera{cv::Matx44f::eye(), cv::Matx44f::eye()};
  input_world_to_camera[1](0, 3) = -10.0f;

  const auto results = model.infer_multi_view(images, batch_intrinsics, input_world_to_camera);
  ASSERT_EQ(results.size(), 2u);
  ASSERT_TRUE(results[0].predicted_world_to_camera.has_value());
  ASSERT_TRUE(results[1].predicted_world_to_camera.has_value());
  const double predicted_baseline =
      cv::norm(detail::cameraCenterFromWorldToCamera(*results[1].predicted_world_to_camera) -
               detail::cameraCenterFromWorldToCamera(*results[0].predicted_world_to_camera));
  ASSERT_TRUE(std::isfinite(predicted_baseline));
  ASSERT_GT(predicted_baseline, 1e-6);
  const double expected_scale = predicted_baseline / 10.0;
  for (const auto& result : results) {
    EXPECT_EQ(result.depth.type(), CV_32FC1);
    EXPECT_EQ(result.depth.size(), image.size());
    EXPECT_NEAR(result.metadata.pose_scale, expected_scale, expected_scale * 1e-5);
    EXPECT_EQ(result.metadata.pose_scaled, std::abs(expected_scale - 1.0) > 1e-6);
  }
#endif
}

}  // namespace
