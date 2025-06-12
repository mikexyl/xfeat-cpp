#pragma once

#include <opencv2/core/mat.hpp>

namespace xfeat {

struct DetectionResult {
  cv::Mat keypoints;
  cv::Mat scores;
  cv::Mat descriptors;
};

}  // namespace xfeat