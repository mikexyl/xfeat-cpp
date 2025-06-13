#pragma once

#include <opencv2/features2d.hpp>

#include "xfeat-cpp/xfeat_onnx.h"

namespace xfeat {
using namespace cv;

class XFeatCV : public cv::Feature2D {
 public:
  struct Params : XFeatONNX::Params {
    int max_features;  // Default maximum number of features to detect
  };

  XFeatCV(const Params& params = Params()) : Feature2D(), xfeat_onnx_(params), params_(params) {}

  // Factory method to create an instance of XFeatCV
  static cv::Ptr<XFeatCV> create(const Params& params) { return Ptr<XFeatCV>(new XFeatCV(params)); }

  /** Detects keypoints and computes the descriptors */
  void detectAndCompute(InputArray image,
                        InputArray mask,
                        CV_OUT std::vector<KeyPoint>& keypoints,
                        OutputArray descriptors,
                        bool useProvidedKeypoints = false) CV_OVERRIDE {
    // Ensure the input image is valid
    CV_Assert(image.type() == CV_8UC1 || image.type() == CV_8UC3);
    // Clear keypoints and descriptors
    keypoints.clear();
    // Call the XFeatONNX method to detect and compute keypoints and descriptors

    auto result = xfeat_onnx_.detect_and_compute(image.getMat(), params_.max_features, nullptr);
    for (int i = 0; i < result.keypoints.rows; i++) {
      KeyPoint kp;
      kp.pt = Point2f(result.keypoints.at<float>(i, 0), result.keypoints.at<float>(i, 1));
      keypoints.push_back(kp);
    }
    if (!result.descriptors.empty()) {
      descriptors.create(result.descriptors.rows, result.descriptors.cols, CV_32F);
      result.descriptors.copyTo(descriptors);
    } else {
      descriptors.release();
    }
  }

  CV_WRAP void detect(InputArray image,
                      CV_OUT std::vector<KeyPoint>& keypoints,
                      InputArray mask = noArray()) CV_OVERRIDE {
    // Ensure the input image is valid
    CV_Assert(image.type() == CV_8UC1 || image.type() == CV_8UC3);
    // Clear keypoints
    keypoints.clear();
    // Call the XFeatONNX method to detect keypoints
    auto result = xfeat_onnx_.detect_and_compute(image.getMat(), params_.max_features);
    for (int i = 0; i < result.keypoints.rows; i++) {
      KeyPoint kp;
      kp.pt = Point2f(result.keypoints.at<float>(i, 0), result.keypoints.at<float>(i, 1));
      keypoints.push_back(kp);
    }
    // If mask is provided, apply it (not implemented in XFeatONNX)
    if (!mask.empty()) {
      CV_Assert(mask.type() == CV_8UC1);
      cv::Mat mask_mat = mask.getMat();
      for (auto it = keypoints.begin(); it != keypoints.end();) {
        if (mask_mat.at<uchar>(static_cast<int>(it->pt.y), static_cast<int>(it->pt.x)) == 0) {
          it = keypoints.erase(it);  // Remove keypoint if mask is zero
        } else {
          ++it;
        }
      }
    }
  }

  int descriptorSize() const CV_OVERRIDE { return 64; }
  int descriptorType() const CV_OVERRIDE { return CV_32F; }
  bool empty() const CV_OVERRIDE { return false; }
  String getDefaultName() const CV_OVERRIDE { return "XFeat"; }

 private:
  Params params_;  // Parameters for XFeatCV

  XFeatONNX xfeat_onnx_;
};

};  // namespace xfeat