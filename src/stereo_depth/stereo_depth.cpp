#include "xfeat-cpp/stereo_depth/stereo_depth.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace xfeat {

// Base class implementation
void StereoDepth::computeDepth(const cv::Mat& left, const cv::Mat& right, 
                                cv::Mat& depth, float focal_length, float baseline) {
  cv::Mat disparity;
  compute(left, right, disparity);
  disparityToDepth(disparity, depth, focal_length, baseline);
}

void StereoDepth::disparityToDepth(const cv::Mat& disparity, cv::Mat& depth,
                                   float focal_length, float baseline) {
  if (disparity.empty()) {
    throw std::invalid_argument("Disparity map is empty");
  }

  const int scale = getDisparityScale();
  const float baseline_focal = baseline * focal_length;
  const float minimum_disparity =
      static_cast<float>(getMinDisparity());

  // Convert disparity to float if needed
  cv::Mat disp_float;
  if (disparity.type() == CV_16S) {
    disparity.convertTo(disp_float, CV_32F, 1.0 / scale);
  } else if (disparity.type() == CV_32F) {
    if (scale != 1) {
      disp_float = disparity / static_cast<float>(scale);
    } else {
      disp_float = disparity;
    }
  } else {
    throw std::invalid_argument("Unsupported disparity type");
  }

  // Allocate output depth map
  depth.create(disp_float.size(), CV_32F);

  // Convert disparity to depth: depth = (baseline * focal_length) / disparity
  // Handle invalid disparities (0 or negative)
  for (int y = 0; y < disp_float.rows; ++y) {
    const float* disp_ptr = disp_float.ptr<float>(y);
    float* depth_ptr = depth.ptr<float>(y);

    for (int x = 0; x < disp_float.cols; ++x) {
      float d = disp_ptr[x];
      if (d > 0.0f && d >= minimum_disparity) {
        depth_ptr[x] = baseline_focal / d;
      } else {
        depth_ptr[x] = 0.0f;  // Invalid depth
      }
    }
  }
}

// OpenCV implementation
OpenCVStereoDepth::OpenCVStereoDepth(const Params& params) : params_(params) {
  // Validate parameters
  if (params_.num_disparities <= 0 || params_.num_disparities % 16 != 0) {
    throw std::invalid_argument("num_disparities must be positive and divisible by 16");
  }
  if (params_.block_size < 3 || params_.block_size % 2 == 0) {
    throw std::invalid_argument("block_size must be odd and >= 3");
  }

  // Create the appropriate matcher
  if (params_.algorithm == Algorithm::BM) {
    auto bm = cv::StereoBM::create(params_.num_disparities, params_.block_size);
    bm->setMinDisparity(params_.min_disparity);
    bm->setPreFilterCap(params_.pre_filter_cap);
    bm->setUniquenessRatio(params_.uniqueness_ratio);
    bm->setSpeckleWindowSize(params_.speckle_window_size);
    bm->setSpeckleRange(params_.speckle_range);
    bm->setDisp12MaxDiff(params_.disp12_max_diff);
    matcher_ = bm;
  } else {  // SGBM
    auto sgbm = cv::StereoSGBM::create(
      params_.min_disparity,
      params_.num_disparities,
      params_.block_size,
      params_.P1,
      params_.P2,
      params_.disp12_max_diff,
      params_.pre_filter_cap,
      params_.uniqueness_ratio,
      params_.speckle_window_size,
      params_.speckle_range,
      params_.mode
    );
    matcher_ = sgbm;
  }
}

void OpenCVStereoDepth::compute(const cv::Mat& left, const cv::Mat& right, 
                                cv::Mat& disparity) {
  if (left.empty() || right.empty()) {
    throw std::invalid_argument("Input images are empty");
  }
  if (left.size() != right.size()) {
    throw std::invalid_argument("Left and right images must have the same size");
  }

  // Convert to grayscale if needed
  cv::Mat left_gray, right_gray;
  if (left.channels() == 3) {
    cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
  } else {
    left_gray = left;
  }
  if (right.channels() == 3) {
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
  } else {
    right_gray = right;
  }

  const cv::Size original_size = left_gray.size();
  const bool needs_resize = !params_.target_size.empty() &&
                            params_.target_size != original_size;
  if (needs_resize) {
    if (params_.target_size.width > original_size.width ||
        params_.target_size.height > original_size.height) {
      throw std::invalid_argument(
          "target_size must not enlarge the stereo input");
    }
    cv::resize(left_gray, left_gray, params_.target_size, 0.0, 0.0,
               cv::INTER_LINEAR);
    cv::resize(right_gray, right_gray, params_.target_size, 0.0, 0.0,
               cv::INTER_LINEAR);
  }

  cv::Mat working_disparity;
  matcher_->compute(left_gray, right_gray, working_disparity);
  if (working_disparity.type() != CV_16S) {
    throw std::runtime_error("OpenCV stereo matcher returned non-CV_16S data");
  }

  if (!needs_resize) {
    disparity = working_disparity;
    return;
  }

  // OpenCV stores disparity with four fractional bits. Preserve validity at
  // the working resolution before interpolation; otherwise the
  // (minDisparity - 1) sentinel can become a plausible positive disparity.
  const int invalid_working_disparity =
      (params_.min_disparity - 1) * getDisparityScale();
  const cv::Mat working_valid =
      working_disparity > invalid_working_disparity;

  cv::Mat resized_disparity;
  cv::resize(working_disparity, resized_disparity, original_size, 0.0, 0.0,
             cv::INTER_LINEAR);
  const double horizontal_scale =
      static_cast<double>(original_size.width) /
      static_cast<double>(params_.target_size.width);
  resized_disparity.convertTo(disparity, CV_16S, horizontal_scale);

  cv::Mat valid;
  cv::resize(working_valid, valid, original_size, 0.0, 0.0,
             cv::INTER_NEAREST);
  disparity.setTo(-getDisparityScale(), ~valid);
}

}  // namespace xfeat
