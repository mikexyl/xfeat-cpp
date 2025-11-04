#pragma once

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace xfeat {
class NetVLADONNX {
 public:
  // height and width correspond to the spatial dimensions in the model input
  // Default values kept for backward compatibility (30 x 40)
  NetVLADONNX(Ort::Env& env, const std::string& model_path, bool use_gpu = true,
              size_t height = 30, size_t width = 40);
  // input: [batch_size, 256, height, width], output: [batch_size, output_dim]
  std::vector<std::vector<float>> infer(const std::vector<float>& input, size_t batch_size);
  // Optional: OpenCV Mat interface
  std::vector<std::vector<float>> infer(const cv::Mat& input);

 private:
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  // spatial dimensions expected by the model
  size_t height_ = 30;
  size_t width_ = 40;
  // total input size (channels * height * width)
  size_t input_size_ = 256 * height_ * width_;
};
}  // namespace xfeat
