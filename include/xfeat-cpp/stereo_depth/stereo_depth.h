#pragma once

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

namespace xfeat {

/**
 * @brief Base class for stereo depth estimation algorithms
 *
 * This abstract class provides a common interface for various stereo depth
 * estimation methods including OpenCV's native algorithms and deep
 * learning-based methods like LightStereo.
 */
class StereoDepth {
 public:
  virtual ~StereoDepth() = default;

  /**
   * @brief Compute disparity map from stereo image pair
   *
   * @param left Left rectified image (grayscale or color depending on implementation)
   * @param right Right rectified image (grayscale or color depending on implementation)
   * @param disparity Output disparity map (CV_16S or CV_32F depending on implementation)
   *
   * @note The disparity values may be scaled by a factor (e.g., 16 for subpixel precision)
   *       Check getDisparityScale() to get the scaling factor
   */
  virtual void compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity) = 0;

  /**
   * @brief Compute depth map from stereo image pair
   *
   * @param left Left rectified image
   * @param right Right rectified image
   * @param depth Output depth map in meters (CV_32F)
   * @param focal_length Focal length in pixels
   * @param baseline Baseline distance in meters
   */
  virtual void computeDepth(const cv::Mat& left, const cv::Mat& right, cv::Mat& depth, 
                           float focal_length, float baseline);

  /**
   * @brief Convert disparity map to depth map
   *
   * @param disparity Input disparity map
   * @param depth Output depth map in meters (CV_32F)
   * @param focal_length Focal length in pixels
   * @param baseline Baseline distance in meters
   */
  virtual void disparityToDepth(const cv::Mat& disparity, cv::Mat& depth,
                               float focal_length, float baseline);

  /**
   * @brief Get the disparity scaling factor
   *
   * @return Scaling factor (e.g., 16 for subpixel precision, 1 for no scaling)
   */
  virtual int getDisparityScale() const = 0;

  /**
   * @brief Get minimum disparity value
   *
   * @return Minimum disparity
   */
  virtual int getMinDisparity() const = 0;

  /**
   * @brief Get number of disparity levels
   *
   * @return Number of disparities
   */
  virtual int getNumDisparities() const = 0;

  /**
   * @brief Get block/window size used for matching
   *
   * @return Block size (odd number, typically 3-21)
   */
  virtual int getBlockSize() const = 0;

  /**
   * @brief Warm up the algorithm (useful for GPU-based methods)
   *
   * Performs initialization and memory allocation to ensure optimal
   * performance for subsequent compute calls.
   */
  virtual void warmup(const cv::Size& image_size) {}

  /**
   * @brief Check if the algorithm requires grayscale input
   *
   * @return true if grayscale input is required, false otherwise
   */
  virtual bool requiresGrayscale() const { return true; }
};

/**
 * @brief Factory function to create OpenCV-based stereo depth estimators
 */
class OpenCVStereoDepth : public StereoDepth {
 public:
  enum class Algorithm {
    BM,      // Block Matching
    SGBM     // Semi-Global Block Matching
  };

  struct Params {
    Algorithm algorithm;
    int min_disparity;
    int num_disparities;  // Must be divisible by 16
    int block_size;          // Odd number, typically 3-21

    // SGBM-specific parameters
    int P1;                  // Penalty for small disparity changes (P1 = 8*channels*block_size^2)
    int P2;                 // Penalty for large disparity changes (P2 = 32*channels*block_size^2)
    int disp12_max_diff;     // Maximum allowed difference in left-right disparity check
    int pre_filter_cap;
    int uniqueness_ratio;   // Margin in percentage
    int speckle_window_size;
    int speckle_range;
    int mode;  // SGBM mode
    // Optional CPU working resolution. Disparity is returned at the input
    // resolution and expressed in input-image pixels.
    cv::Size target_size;

    // Constructor with defaults
    Params() 
      : algorithm(Algorithm::SGBM),
        min_disparity(0),
        num_disparities(128),
        block_size(5),
        P1(8),
        P2(32),
        disp12_max_diff(1),
        pre_filter_cap(63),
        uniqueness_ratio(10),
        speckle_window_size(100),
        speckle_range(32),
        mode(cv::StereoSGBM::MODE_SGBM_3WAY),
        target_size() {}
  };

  explicit OpenCVStereoDepth(const Params& params = Params());

  void compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity) override;

  int getDisparityScale() const override { return 16; }
  int getMinDisparity() const override { return params_.min_disparity; }
  int getNumDisparities() const override { return params_.num_disparities; }
  int getBlockSize() const override { return params_.block_size; }
  bool requiresGrayscale() const override { return true; }

  // Get the underlying OpenCV matcher for advanced configuration
  cv::Ptr<cv::StereoMatcher> getMatcher() { return matcher_; }

 private:
  Params params_;
  cv::Ptr<cv::StereoMatcher> matcher_;
};

}  // namespace xfeat
