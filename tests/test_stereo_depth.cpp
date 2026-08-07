#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "xfeat-cpp/stereo_depth/stereo_depth.h"
#ifdef HAVE_VPI
#include <cuda_runtime_api.h>

#include "../src/stereo_depth/stereo_depth_vpi_utils.h"
#include "xfeat-cpp/stereo_depth/stereo_depth_vpi.h"
#endif

namespace xfeat {
namespace {

TEST(OpenCVStereoDepth, ConvertsFixedPointDisparityToMetricDepth) {
  OpenCVStereoDepth::Params params;
  params.min_disparity = 10;
  OpenCVStereoDepth stereo(params);
  ASSERT_EQ(stereo.getDisparityScale(), 16);

  const cv::Mat disparity = (cv::Mat_<int16_t>(1, 4) << 9 * 16, 10 * 16, 20 * 16, 0);
  cv::Mat depth;
  stereo.disparityToDepth(disparity, depth, 100.0f, 0.1f);

  ASSERT_EQ(depth.type(), CV_32F);
  EXPECT_FLOAT_EQ(depth.at<float>(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(depth.at<float>(0, 1), 1.0f);
  EXPECT_FLOAT_EQ(depth.at<float>(0, 2), 0.5f);
  EXPECT_FLOAT_EQ(depth.at<float>(0, 3), 0.0f);
}

TEST(OpenCVStereoDepth, ComputesNativeSgbmFixedPointDisparity) {
  constexpr int kWidth = 160;
  constexpr int kHeight = 96;
  constexpr int kDisparity = 8;
  cv::Mat left(kHeight, kWidth, CV_8U);
  cv::RNG rng(7);
  rng.fill(left, cv::RNG::UNIFORM, 0, 256);

  cv::Mat right(kHeight, kWidth, CV_8U);
  rng.fill(right, cv::RNG::UNIFORM, 0, 256);
  left(cv::Rect(kDisparity, 0, kWidth - kDisparity, kHeight))
      .copyTo(right(cv::Rect(0, 0, kWidth - kDisparity, kHeight)));

  OpenCVStereoDepth::Params params;
  params.algorithm = OpenCVStereoDepth::Algorithm::SGBM;
  params.min_disparity = 0;
  params.num_disparities = 16;
  params.block_size = 3;
  params.P1 = 8 * params.block_size * params.block_size;
  params.P2 = 32 * params.block_size * params.block_size;
  params.uniqueness_ratio = 0;
  params.speckle_window_size = 0;
  params.mode = cv::StereoSGBM::MODE_SGBM;
  OpenCVStereoDepth stereo(params);

  cv::Mat disparity;
  stereo.compute(left, right, disparity);

  ASSERT_EQ(disparity.type(), CV_16S);
  ASSERT_EQ(disparity.size(), left.size());
  const int16_t raw_disparity = disparity.at<int16_t>(kHeight / 2, kWidth / 2);
  EXPECT_NEAR(static_cast<float>(raw_disparity) / static_cast<float>(stereo.getDisparityScale()),
              static_cast<float>(kDisparity),
              0.5f);
}

TEST(OpenCVStereoDepth, RestoresDisparityAfterWorkingResolutionDownscale) {
  constexpr int kWidth = 320;
  constexpr int kHeight = 192;
  constexpr int kDisparity = 20;
  cv::Mat left(kHeight, kWidth, CV_8U);
  cv::RNG rng(11);
  rng.fill(left, cv::RNG::UNIFORM, 0, 256);

  cv::Mat right(kHeight, kWidth, CV_8U);
  rng.fill(right, cv::RNG::UNIFORM, 0, 256);
  left(cv::Rect(kDisparity, 0, kWidth - kDisparity, kHeight))
      .copyTo(right(cv::Rect(0, 0, kWidth - kDisparity, kHeight)));

  OpenCVStereoDepth::Params params;
  params.algorithm = OpenCVStereoDepth::Algorithm::SGBM;
  params.min_disparity = 0;
  params.num_disparities = 32;
  params.block_size = 3;
  params.P1 = 8 * params.block_size * params.block_size;
  params.P2 = 32 * params.block_size * params.block_size;
  params.uniqueness_ratio = 0;
  params.speckle_window_size = 0;
  params.mode = cv::StereoSGBM::MODE_SGBM;
  // Exercise a non-integer horizontal restoration ratio (320 / 128 = 2.5),
  // as used by real camera/model-resolution combinations.
  params.target_size = cv::Size(128, kHeight / 2);
  OpenCVStereoDepth stereo(params);

  cv::Mat disparity;
  stereo.compute(left, right, disparity);

  ASSERT_EQ(disparity.type(), CV_16S);
  ASSERT_EQ(disparity.size(), left.size());
  const float physical_disparity = static_cast<float>(disparity.at<int16_t>(kHeight / 2, kWidth / 2)) /
                                   static_cast<float>(stereo.getDisparityScale());
  EXPECT_NEAR(physical_disparity, static_cast<float>(kDisparity), 1.0f);
  EXPECT_LT(disparity.at<int16_t>(kHeight / 2, 0), 0);
}

#ifdef HAVE_VPI
TEST(VPIStereoDepth, RejectsDisparityAtSearchFloor) {
  const cv::Mat disparity = (cv::Mat_<int16_t>(1, 3) << 10 * 32 - 1, 10 * 32, 10 * 32 + 1);
  const cv::Mat confidence = (cv::Mat_<uint16_t>(1, 3) << 60000, 60000, 60000);

  const cv::Mat valid = vpi_detail::makeValidityMask(disparity, confidence, 10, 55535, 32);

  EXPECT_EQ(valid.at<uint8_t>(0, 0), 0);
  EXPECT_EQ(valid.at<uint8_t>(0, 1), 0);
  EXPECT_NE(valid.at<uint8_t>(0, 2), 0);
}

TEST(VPIStereoDepth, ValidityAwareResizeDoesNotBlendInvalidSentinel) {
  const cv::Mat disparity = (cv::Mat_<int16_t>(1, 2) << 10 * 32, -1 * 32);
  const cv::Mat valid = (cv::Mat_<uint8_t>(1, 2) << 255, 0);

  cv::Mat restored;
  vpi_detail::restoreDisparityWithValidity(disparity, valid, cv::Size(4, 1), 2.0, -32, &restored);

  ASSERT_EQ(restored.type(), CV_16S);
  ASSERT_EQ(restored.size(), cv::Size(4, 1));
  EXPECT_EQ(restored.at<int16_t>(0, 0), 20 * 32);
  EXPECT_EQ(restored.at<int16_t>(0, 1), 20 * 32);
  EXPECT_EQ(restored.at<int16_t>(0, 2), -32);
  EXPECT_EQ(restored.at<int16_t>(0, 3), -32);
}

TEST(VPIStereoDepth, RejectsInvalidParametersWithoutInitializingCuda) {
  VPIStereoDepth::Params params;
  params.max_disparity = 0;
  EXPECT_THROW(VPIStereoDepth stereo(params), std::invalid_argument);

  params.max_disparity = 64;
  params.min_valid_disparity = 64;
  EXPECT_THROW(VPIStereoDepth stereo(params), std::invalid_argument);

  params.min_valid_disparity = 0;
  params.p1 = 50;
  params.p2 = 49;
  EXPECT_THROW(VPIStereoDepth stereo(params), std::invalid_argument);

  params.p1 = 3;
  params.p2 = 48;
  params.uniqueness = 1.1f;
  EXPECT_THROW(VPIStereoDepth stereo(params), std::invalid_argument);
}

TEST(VPIStereoDepth, ComputesQ10Point5AndRestoresWorkingResolution) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device is unavailable in this test context";
  }

  constexpr int kWidth = 320;
  constexpr int kHeight = 192;
  constexpr int kDisparity = 20;
  cv::Mat left(kHeight, kWidth, CV_8U);
  cv::RNG rng(23);
  rng.fill(left, cv::RNG::UNIFORM, 0, 256);

  cv::Mat right(kHeight, kWidth, CV_8U);
  rng.fill(right, cv::RNG::UNIFORM, 0, 256);
  left(cv::Rect(kDisparity, 0, kWidth - kDisparity, kHeight))
      .copyTo(right(cv::Rect(0, 0, kWidth - kDisparity, kHeight)));

  VPIStereoDepth::Params params;
  params.min_disparity = 0;
  params.max_disparity = 64;
  params.confidence_threshold = 0;
  params.uniqueness = -1.0f;
  params.include_diagonals = false;
  params.target_size = cv::Size(kWidth / 2, kHeight / 2);
  VPIStereoDepth stereo(params);

  cv::Mat disparity;
  stereo.compute(left, right, disparity);

  ASSERT_EQ(stereo.getDisparityScale(), 32);
  ASSERT_EQ(disparity.type(), CV_16S);
  ASSERT_EQ(disparity.size(), left.size());
  const float physical_disparity = static_cast<float>(disparity.at<int16_t>(kHeight / 2, kWidth / 2)) /
                                   static_cast<float>(stereo.getDisparityScale());
  EXPECT_NEAR(physical_disparity, static_cast<float>(kDisparity), 1.0f);

  // Reusing the object exercises the persistent VPI stream/payload while the
  // per-call OpenCV wrappers are recreated safely.
  cv::Mat second_disparity;
  stereo.compute(left, right, second_disparity);
  EXPECT_EQ(cv::countNonZero(disparity != second_disparity), 0);
}
#endif

}  // namespace
}  // namespace xfeat
