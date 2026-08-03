#include "xfeat-cpp/stereo_depth/stereo_depth_fast_foundation_stereo.h"

#ifdef HAVE_TENSORRT

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

#include "ffs_gwc_plugin.hpp"
#include "xfeat-cpp/tensorrt/detail/trt_engine.h"

namespace xfeat {
namespace {

int checkedDimension(const std::vector<int64_t>& shape, size_t index, const std::string& label) {
  if (index >= shape.size() || shape[index] <= 0 || shape[index] > std::numeric_limits<int>::max()) {
    throw std::runtime_error("FastFoundationStereoDepth requires a fixed positive " + label + " dimension");
  }
  return static_cast<int>(shape[index]);
}

void requireShape(const std::vector<int64_t>& actual,
                  const std::vector<int64_t>& expected,
                  const std::string& tensor_name) {
  if (actual != expected) {
    throw std::runtime_error("FastFoundationStereoDepth tensor '" + tensor_name + "' has an unexpected shape");
  }
}

cv::Mat toRgb(const cv::Mat& image) {
  if (image.depth() != CV_8U) {
    throw std::invalid_argument("FastFoundationStereoDepth input must have 8-bit channels");
  }
  if (image.channels() == 1) {
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
    return rgb;
  }
  if (image.channels() == 3) {
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    return rgb;
  }
  throw std::invalid_argument("FastFoundationStereoDepth input must have one or three channels");
}

cv::Size scaledSize(const cv::Size& source, const cv::Size& target) {
  if (source == target) return target;
  const float scale =
      std::min(static_cast<float>(target.width) / source.width, static_cast<float>(target.height) / source.height);
  return {std::max(1, static_cast<int>(std::round(source.width * scale))),
          std::max(1, static_cast<int>(std::round(source.height * scale)))};
}

std::vector<float> prepareInput(const cv::Mat& image, const cv::Size& model_size, const cv::Size& scaled_size) {
  cv::Mat rgb = toRgb(image);
  if (rgb.size() != scaled_size) {
    cv::resize(rgb, rgb, scaled_size, 0.0, 0.0, cv::INTER_LINEAR);
  }
  if (rgb.size() != model_size) {
    cv::Mat padded;
    cv::copyMakeBorder(
        rgb, padded, 0, model_size.height - rgb.rows, 0, model_size.width - rgb.cols, cv::BORDER_REPLICATE);
    rgb = padded;
  }

  cv::Mat float_rgb;
  rgb.convertTo(float_rgb, CV_32F);
  std::vector<cv::Mat> channels;
  cv::split(float_rgb, channels);
  const size_t plane = static_cast<size_t>(model_size.height) * model_size.width;
  std::vector<float> tensor(3 * plane);
  for (int channel = 0; channel < 3; ++channel) {
    std::memcpy(
        tensor.data() + static_cast<size_t>(channel) * plane, channels[channel].ptr<float>(), plane * sizeof(float));
  }
  return tensor;
}

}  // namespace

FastFoundationStereoDepth::FastFoundationStereoDepth(const Params& params) : params_(params) {
  if (params_.engine_path.empty()) {
    throw std::invalid_argument("FastFoundationStereoDepth engine path cannot be empty");
  }
  if (params_.max_disparity <= 0) {
    throw std::invalid_argument("FastFoundationStereoDepth max_disparity must be positive");
  }
  if (params_.warmup_iterations < 0) {
    throw std::invalid_argument("FastFoundationStereoDepth warmup_iterations cannot be negative");
  }
  if (!ffs_depth::registerFFSGWCPlugin()) {
    throw std::runtime_error("FastFoundationStereoDepth failed to register the FFSGWCVolume plugin");
  }

  engine_ = std::make_unique<trt_detail::Engine>(params_.engine_path, params_.verbose);
  const auto input_names = engine_->input_names();
  const auto output_names = engine_->output_names();
  if (input_names.size() != 2 || output_names.size() != 1) {
    throw std::runtime_error("FastFoundationStereoDepth requires exactly two inputs and one output");
  }

  for (const auto& name : input_names) {
    if (name == "left") left_input_name_ = name;
    if (name == "right") right_input_name_ = name;
  }
  if (left_input_name_.empty() || right_input_name_.empty() || output_names.front() != "disp") {
    throw std::runtime_error(
        "FastFoundationStereoDepth engine must expose float32 tensors named left, right, and disp");
  }
  output_name_ = output_names.front();

  if (engine_->tensor_type(left_input_name_) != trt_detail::DataType::kFloat32 ||
      engine_->tensor_type(right_input_name_) != trt_detail::DataType::kFloat32 ||
      engine_->tensor_type(output_name_) != trt_detail::DataType::kFloat32) {
    throw std::runtime_error("FastFoundationStereoDepth requires float32 engine I/O tensors");
  }

  const auto left_shape = engine_->tensor_shape(left_input_name_);
  const auto right_shape = engine_->tensor_shape(right_input_name_);
  if (left_shape.size() != 4 || left_shape[0] != 1 || left_shape[1] != 3) {
    throw std::runtime_error("FastFoundationStereoDepth input must have shape [1, 3, height, width]");
  }
  requireShape(right_shape, left_shape, right_input_name_);
  const int height = checkedDimension(left_shape, 2, "input height");
  const int width = checkedDimension(left_shape, 3, "input width");
  input_size_ = cv::Size(width, height);
  requireShape(engine_->tensor_shape(output_name_), {1, 1, height, width}, output_name_);
}

FastFoundationStereoDepth::~FastFoundationStereoDepth() = default;

void FastFoundationStereoDepth::compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity) {
  if (left.empty() || right.empty()) {
    throw std::invalid_argument("FastFoundationStereoDepth input images are empty");
  }
  if (left.size() != right.size()) {
    throw std::invalid_argument("FastFoundationStereoDepth images must have the same size");
  }

  const cv::Size scaled_size = scaledSize(left.size(), input_size_);
  std::vector<float> left_tensor = prepareInput(left, input_size_, scaled_size);
  std::vector<float> right_tensor = prepareInput(right, input_size_, scaled_size);
  const std::vector<int64_t> input_shape = {1, 3, input_size_.height, input_size_.width};
  const auto outputs = engine_->run({
      trt_detail::InputTensor{left_input_name_, input_shape, left_tensor.data(), left_tensor.size() * sizeof(float)},
      trt_detail::InputTensor{right_input_name_, input_shape, right_tensor.data(), right_tensor.size() * sizeof(float)},
  });
  const auto& output = trt_detail::find_output(outputs, output_name_);
  const std::vector<int64_t> output_shape = {1, 1, input_size_.height, input_size_.width};
  if (output.type != trt_detail::DataType::kFloat32 || output.shape != output_shape) {
    throw std::runtime_error("FastFoundationStereoDepth returned an unexpected disparity tensor");
  }

  const std::vector<float> values = output.values<float>();
  cv::Mat model_disparity(input_size_, CV_32F);
  std::memcpy(model_disparity.ptr<float>(), values.data(), values.size() * sizeof(float));
  cv::patchNaNs(model_disparity, 0.0);
  cv::threshold(model_disparity, model_disparity, 0.0, 0.0, cv::THRESH_TOZERO);

  if (left.size() == input_size_) {
    disparity = model_disparity;
    return;
  }

  const cv::Mat cropped = model_disparity(cv::Rect(0, 0, scaled_size.width, scaled_size.height));
  cv::resize(cropped, disparity, left.size(), 0.0, 0.0, cv::INTER_NEAREST);
  disparity *= static_cast<float>(left.cols) / scaled_size.width;
}

void FastFoundationStereoDepth::warmup(const cv::Size& image_size) {
  if (image_size.width <= 0 || image_size.height <= 0) {
    throw std::invalid_argument("FastFoundationStereoDepth warmup image size must be positive");
  }
  cv::Mat dummy(image_size, CV_8UC3, cv::Scalar(128, 128, 128));
  cv::Mat output;
  for (int iteration = 0; iteration < params_.warmup_iterations; ++iteration) {
    compute(dummy, dummy, output);
  }
}

}  // namespace xfeat

#endif  // HAVE_TENSORRT
