#pragma once

#include "xfeat-cpp/stereo_depth.h"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <memory>

// Forward declarations to avoid including LibSGM headers in the header file
namespace sgm {
class LibSGMWrapper;
enum class PathType;
enum class CensusType;
class StereoSGM;
}

namespace xfeat {

/**
 * @brief LibSGM-based stereo depth estimator using CUDA acceleration
 * 
 * This implementation uses the LibSGM library which provides a GPU-accelerated
 * Semi-Global Matching algorithm. It supports both CPU and GPU execution modes.
 */
class LibSGMStereoDepth : public StereoDepth {
 public:
  struct Params {
    int num_disparities;      // Maximum disparity minus minimum disparity
    int P1;                     // Penalty on disparity change by ±1 between neighbor pixels
    int P2;                    // Penalty on disparity change by >1 between neighbor pixels
    float uniqueness_ratio;  // Margin ratio for uniqueness check
    bool subpixel;            // Enable subpixel precision (4 fractional bits)
    int path_type;  // 4-path (0) or 8-path (1)
    int min_disparity;           // Minimum possible disparity value
    int lr_max_diff;             // Max diff for LR consistency check (-1 to disable)
    int census_type;  // Census type
    bool use_gpu;             // Use GPU acceleration if available
    cv::Size target_size;     // Resize input images to this size (empty = no resize)
    
    // Constructor with defaults
    Params();
  };

  explicit LibSGMStereoDepth(const Params& params);
  LibSGMStereoDepth();  // Default constructor
  ~LibSGMStereoDepth() override;  // Must be defined in .cpp where LibSGMWrapper is complete

  void compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity) override;

  int getDisparityScale() const override { 
    return params_.subpixel ? 16 : 1;  // StereoSGM::SUBPIXEL_SCALE = 16
  }
  
  int getMinDisparity() const override { return params_.min_disparity; }
  int getNumDisparities() const override { return params_.num_disparities; }
  int getBlockSize() const override { return 9; }  // Census window is 9x7
  bool requiresGrayscale() const override { return true; }

  void warmup(const cv::Size& image_size) override;

  // Get the underlying LibSGM wrapper for advanced configuration
  void* getWrapper() { return wrapper_.get(); }

 private:
  Params params_;
  std::unique_ptr<sgm::LibSGMWrapper> wrapper_;
  
  // GPU memory buffers (reused across compute calls)
  cv::cuda::GpuMat gpu_left_;
  cv::cuda::GpuMat gpu_right_;
  cv::cuda::GpuMat gpu_disparity_;
};

}  // namespace xfeat
