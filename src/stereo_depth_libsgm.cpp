#include "xfeat-cpp/stereo_depth_libsgm.h"
#include <stdexcept>
#include <opencv2/imgproc.hpp>

// Include LibSGM headers in implementation file
#ifdef BUILD_OPENCV_WRAPPER
#include <libsgm_wrapper.h>
#endif

namespace xfeat {

// Define Params constructor with defaults
LibSGMStereoDepth::Params::Params()
    : num_disparities(128),
      P1(10),
      P2(120),
      uniqueness_ratio(0.95f),
      subpixel(true),
      path_type(1),  // SCAN_8PATH
      min_disparity(0),
      lr_max_diff(1),
      census_type(1),  // SYMMETRIC_CENSUS_9x7
      use_gpu(true),
      target_size() {}  // Empty size = no resize

LibSGMStereoDepth::~LibSGMStereoDepth() = default;

LibSGMStereoDepth::LibSGMStereoDepth() : LibSGMStereoDepth(Params()) {}

LibSGMStereoDepth::LibSGMStereoDepth(const Params& params) : params_(params) {
#ifndef BUILD_OPENCV_WRAPPER
  throw std::runtime_error("LibSGM was built without OpenCV wrapper support");
#else
  // Create LibSGM wrapper
  wrapper_ = std::make_unique<sgm::LibSGMWrapper>(
    params_.num_disparities,
    params_.P1,
    params_.P2,
    params_.uniqueness_ratio,
    params_.subpixel,
    static_cast<sgm::PathType>(params_.path_type),
    params_.min_disparity,
    params_.lr_max_diff,
    static_cast<sgm::CensusType>(params_.census_type)
  );
#endif
}

void LibSGMStereoDepth::compute(const cv::Mat& left, const cv::Mat& right, 
                                cv::Mat& disparity) {
#ifndef BUILD_OPENCV_WRAPPER
  throw std::runtime_error("LibSGM was built without OpenCV wrapper support");
#else
  if (left.empty() || right.empty()) {
    throw std::invalid_argument("Input images are empty");
  }
  if (left.size() != right.size()) {
    throw std::invalid_argument("Left and right images must have the same size");
  }
  if (left.type() != right.type()) {
    throw std::invalid_argument("Left and right images must have the same type");
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

  // Convert to 8-bit if needed (LibSGM supports CV_8U, CV_16U, CV_32S)
  cv::Mat left_input, right_input;
  if (left_gray.type() != CV_8U && left_gray.type() != CV_16U && left_gray.type() != CV_32S) {
    left_gray.convertTo(left_input, CV_8U);
  } else {
    left_input = left_gray;
  }
  if (right_gray.type() != CV_8U && right_gray.type() != CV_16U && right_gray.type() != CV_32S) {
    right_gray.convertTo(right_input, CV_8U);
  } else {
    right_input = right_gray;
  }

  // Store original size for scaling back disparity
  cv::Size original_size = left_input.size();
  bool needs_resize = !params_.target_size.empty() && 
                      (params_.target_size.width != original_size.width || 
                       params_.target_size.height != original_size.height);
  
  // Resize inputs if target size is specified
  if (needs_resize) {
    cv::resize(left_input, left_input, params_.target_size, 0, 0, cv::INTER_LINEAR);
    cv::resize(right_input, right_input, params_.target_size, 0, 0, cv::INTER_LINEAR);
  }

  if (params_.use_gpu) {
    // Try GPU path first, fall back to CPU if CUDA not available in OpenCV
    try {
      // Check if OpenCV has CUDA support
      #ifdef HAVE_OPENCV_CUDAARITHM
      // Upload to GPU if not already there
      if (gpu_left_.size() != left_input.size() || gpu_left_.type() != left_input.type()) {
        gpu_left_.upload(left_input);
      } else {
        gpu_left_.upload(left_input);
      }
      
      if (gpu_right_.size() != right_input.size() || gpu_right_.type() != right_input.type()) {
        gpu_right_.upload(right_input);
      } else {
        gpu_right_.upload(right_input);
      }

      // Compute disparity on GPU
      wrapper_->execute(gpu_left_, gpu_right_, gpu_disparity_);

      // Download result
      gpu_disparity_.download(disparity);
      #else
      // OpenCV built without CUDA, use CPU path
      wrapper_->execute(left_input, right_input, disparity);
      #endif
    } catch (const cv::Exception& e) {
      // Fall back to CPU if GPU fails
      try {
        wrapper_->execute(left_input, right_input, disparity);
      } catch (const cv::Exception& cpu_e) {
        throw std::runtime_error(std::string("LibSGM execution failed: ") + cpu_e.what());
      }
    }
  } else {
    // CPU path
    try {
      wrapper_->execute(left_input, right_input, disparity);
    } catch (const cv::Exception& e) {
      throw std::runtime_error(std::string("LibSGM CPU execution failed: ") + e.what());
    }
  }

  // Scale disparity map back to original size if needed
  if (needs_resize) {
    // Calculate scaling factors
    float scale_x = static_cast<float>(original_size.width) / params_.target_size.width;
    float scale_y = static_cast<float>(original_size.height) / params_.target_size.height;
    
    // Resize disparity map back to original size
    cv::Mat disparity_resized;
    cv::resize(disparity, disparity_resized, original_size, 0, 0, cv::INTER_LINEAR);
    
    // Scale disparity values by the horizontal scaling factor
    // (disparity is proportional to horizontal image resolution)
    disparity_resized.convertTo(disparity, disparity.type(), scale_x);
  }
#endif
}

void LibSGMStereoDepth::warmup(const cv::Size& image_size) {
#ifdef BUILD_OPENCV_WRAPPER
  if (params_.use_gpu) {
    // Create dummy images for warmup
    cv::Mat dummy_left = cv::Mat::zeros(image_size, CV_8UC1);
    cv::Mat dummy_right = cv::Mat::zeros(image_size, CV_8UC1);
    cv::Mat dummy_disparity;

    // Run once to initialize GPU memory and CUDA kernels
    try {
      compute(dummy_left, dummy_right, dummy_disparity);
    } catch (const std::exception& e) {
      // Ignore errors during warmup
    }
  }
#endif
}

}  // namespace xfeat
