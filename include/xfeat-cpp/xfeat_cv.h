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

  XFeatCV(Ort::Env& env, const Params& params = Params()) : Feature2D(), env_(env), xfeat_onnx_(env, params), params_(params) {}

  // Factory method to create an instance of XFeatCV
  static cv::Ptr<XFeatCV> create(Ort::Env& env, const Params& params) { return Ptr<XFeatCV>(new XFeatCV(env, params)); }

  /** Detects keypoints and computes the descriptors */
  void detectAndCompute(InputArray image,
                        InputArray mask,
                        CV_OUT std::vector<KeyPoint>& keypoints,
                        OutputArray descriptors,
                        bool useProvidedKeypoints = false) CV_OVERRIDE {
    return detectAndCompute(image, mask, keypoints, descriptors, useProvidedKeypoints, nullptr, nullptr);
  }

  /** Detects keypoints and computes the descriptors */
  void detectAndCompute(InputArray image,
                        InputArray mask,
                        CV_OUT std::vector<KeyPoint>& keypoints,
                        OutputArray descriptors,
                        bool useProvidedKeypoints,
                        cv::Mat* M1,
                        cv::Mat* x_prep) {
    // not implemented
    CV_Assert(!useProvidedKeypoints);

    if (not descriptors.empty()) {
      CV_Error(Error::StsBadArg, "Output descriptors must be empty.");
    }
    // check descriptors are not fixed size
    CV_Assert(descriptors.empty() || descriptors.type() == CV_32F);

    // Ensure the input image is valid
    CV_Assert(image.type() == CV_8UC1 || image.type() == CV_8UC3);
    // Clear keypoints and descriptors
    keypoints.clear();
    // Call the XFeatONNX method to detect and compute keypoints and descriptors

    auto result = xfeat_onnx_.detect_and_compute(image.getMat(), params_.max_features, nullptr, M1, x_prep);
    for (int i = 0; i < result.keypoints.rows; i++) {
      KeyPoint kp;
      kp.pt = Point2f(result.keypoints.at<float>(i, 0), result.keypoints.at<float>(i, 1));
      keypoints.push_back(kp);
    }
    if (!result.descriptors.empty()) {
      // copy the descriptors to the output
      CV_Assert(result.descriptors.type() == CV_32F);
      CV_Assert(result.descriptors.rows == keypoints.size());
      CV_Assert(result.descriptors.cols == 64);  // Assuming 64-dimensional descriptors
      result.descriptors.copyTo(descriptors);
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
  Ort::Env& env_;
  Params params_;  // Parameters for XFeatCV

  XFeatONNX xfeat_onnx_;
};

};  // namespace xfeat