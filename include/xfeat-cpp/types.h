#pragma once

#include <opencv2/core/mat.hpp>

namespace xfeat {

struct DetectionResult {
  cv::Mat keypoints;
  cv::Mat scores;
  cv::Mat descriptors;
};

enum class MatcherType { BF = 0, FLANN = 1, LIGHTERGLUE = 2 };

}  // namespace xfeat