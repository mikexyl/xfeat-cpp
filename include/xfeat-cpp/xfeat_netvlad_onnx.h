#pragma once

#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <string>

#include "xfeat-cpp/netvlad_onnx.h"

namespace xfeat {

class HeadNetVLADONNX {
 public:
  HeadNetVLADONNX(Ort::Env& env, const std::string& model_path, bool use_gpu = true);

  // Run inference: input and input1 are cv::Mat (float32)
  // Returns output as cv::Mat (float32)
  cv::Mat run(const cv::Mat& M1, const cv::Mat& x_prep);

 private:
  Ort::Session session_;
  Ort::SessionOptions session_options_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
};

class XfeatNetVLADONNX {
 public:
  XfeatNetVLADONNX(Ort::Env& env,
                   const std::string& head_model_path,
                   const std::string& netvlad_model_path,
                   bool use_gpu = true)
      : head_(env, head_model_path, use_gpu), netvlad_(env, netvlad_model_path, use_gpu) {}

  // Example combined inference: runs head, then netvlad
  // M1 and x_prep are inputs for head, returns NetVLAD output
  std::vector<std::vector<float>> run(const cv::Mat& M1, const cv::Mat& x_prep) {
    cv::Mat head_output = head_.run(M1, x_prep);
    return netvlad_.infer(head_output);
  }

  // use the same name as dbow vocab
  cv::Mat transform(const cv::Mat& M1, const cv::Mat& x_prep) {
    cv::Mat head_output = head_.run(M1, x_prep);
    // NetVLADONNX::infer returns std::vector<std::vector<float>>, so we convert to cv::Mat
    auto netvlad_result = netvlad_.infer(head_output);
    if (netvlad_result.empty()) return cv::Mat();
    int rows = static_cast<int>(netvlad_result.size());
    int cols = static_cast<int>(netvlad_result[0].size());
    cv::Mat mat(rows, cols, CV_32F);
    for (int i = 0; i < rows; ++i) {
      std::memcpy(mat.ptr<float>(i), netvlad_result[i].data(), cols * sizeof(float));
    }
    return mat;
  }

  // Access to underlying models if needed
  HeadNetVLADONNX& head() { return head_; }
  NetVLADONNX& netvlad() { return netvlad_; }

 private:
  HeadNetVLADONNX head_;
  NetVLADONNX netvlad_;
};

}  // namespace xfeat
