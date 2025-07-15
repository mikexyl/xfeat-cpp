#pragma once

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace xfeat {
class NetVLADONNX {
 public:
  NetVLADONNX(Ort::Env& env, const std::string& model_path);
  // input: [batch_size, 256, 30, 40], output: [batch_size, output_dim]
  std::vector<std::vector<float>> infer(const std::vector<float>& input, size_t batch_size);
  // Optional: OpenCV Mat interface
  std::vector<std::vector<float>> infer(const cv::Mat& input);

 private:
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  size_t input_size_ = 256 * 30 * 40;
};
}  // namespace xfeat
