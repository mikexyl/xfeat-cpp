#pragma once

#include <opencv2/core.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <string>
#include <memory>

namespace xfeat {

class HeadNetVLADONNX {
 public:
  HeadNetVLADONNX(const std::string& model_path);

  // Run inference: input and input1 are cv::Mat (float32)
  // Returns output as cv::Mat (float32)
  cv::Mat run(const cv::Mat& M1, const cv::Mat& x_prep);

 private:
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::SessionOptions session_options_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
};

} // namespace xfeat
