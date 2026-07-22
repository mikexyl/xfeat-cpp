#include "xfeat-cpp/place_recognition/mixvpr_trt.h"

#ifdef HAVE_TENSORRT

#include <cstring>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "xfeat-cpp/tensorrt/detail/trt_engine.h"

namespace xfeat {
namespace {

int checkedDimension(const std::vector<int64_t>& shape, size_t index, const std::string& label) {
  if (index >= shape.size() || shape[index] <= 0 || shape[index] > std::numeric_limits<int>::max()) {
    throw std::runtime_error("MixVPRTRT requires a fixed positive " + label + " dimension");
  }
  return static_cast<int>(shape[index]);
}

}  // namespace

MixVPRTRT::MixVPRTRT(const Params& params)
    : engine_(std::make_unique<trt_detail::Engine>(params.model_path, params.verbose)),
      img_height_(params.img_height),
      img_width_(params.img_width),
      normalize_output_(params.normalize_output) {
  const auto input_names = engine_->input_names();
  const auto output_names = engine_->output_names();
  if (input_names.size() != 1 || output_names.size() != 1) {
    throw std::runtime_error("MixVPRTRT requires exactly one input and one output tensor");
  }
  input_name_ = input_names.front();
  output_name_ = output_names.front();
  if (engine_->tensor_type(input_name_) != trt_detail::DataType::kFloat32 ||
      engine_->tensor_type(output_name_) != trt_detail::DataType::kFloat32) {
    throw std::runtime_error("MixVPRTRT requires float32 engine I/O tensors");
  }

  const auto input_shape = engine_->tensor_shape(input_name_);
  const auto output_shape = engine_->tensor_shape(output_name_);
  if (input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] != 3) {
    throw std::runtime_error("MixVPRTRT input must have shape [1, 3, height, width]");
  }
  if (output_shape.size() != 2 || output_shape[0] != 1) {
    throw std::runtime_error("MixVPRTRT output must have shape [1, descriptor_dim]");
  }
  const int engine_height = checkedDimension(input_shape, 2, "height");
  const int engine_width = checkedDimension(input_shape, 3, "width");
  descriptor_dim_ = checkedDimension(output_shape, 1, "descriptor");
  if (engine_height != img_height_ || engine_width != img_width_) {
    throw std::runtime_error("MixVPRTRT engine image dimensions do not match configured dimensions");
  }
}

MixVPRTRT::~MixVPRTRT() = default;

std::vector<float> MixVPRTRT::prepare_input_tensor(const cv::Mat& image) const {
  if (image.empty()) throw std::invalid_argument("MixVPRTRT input image is empty");
  cv::Mat color;
  if (image.channels() == 1) {
    cv::cvtColor(image, color, cv::COLOR_GRAY2RGB);
  } else if (image.channels() == 3) {
    cv::cvtColor(image, color, cv::COLOR_BGR2RGB);
  } else {
    throw std::invalid_argument("MixVPRTRT input must have one or three channels");
  }
  cv::resize(color, color, cv::Size(img_width_, img_height_));
  color.convertTo(color, CV_32F, 1.0 / 255.0);

  static constexpr float kMean[] = {0.485f, 0.456f, 0.406f};
  static constexpr float kStdDev[] = {0.229f, 0.224f, 0.225f};
  std::vector<cv::Mat> channels;
  cv::split(color, channels);
  const size_t plane = static_cast<size_t>(img_height_) * img_width_;
  std::vector<float> tensor(3 * plane);
  for (int channel = 0; channel < 3; ++channel) {
    channels[channel] = (channels[channel] - kMean[channel]) / kStdDev[channel];
    std::memcpy(
        tensor.data() + static_cast<size_t>(channel) * plane, channels[channel].ptr<float>(), plane * sizeof(float));
  }
  return tensor;
}

void MixVPRTRT::normalize_descriptor(cv::Mat& descriptor) const {
  const double norm = cv::norm(descriptor, cv::NORM_L2);
  if (norm > 1e-8) descriptor /= norm;
}

cv::Mat MixVPRTRT::infer(const cv::Mat& image) {
  std::vector<float> input = prepare_input_tensor(image);
  const std::vector<int64_t> shape = {1, 3, img_height_, img_width_};
  const auto outputs =
      engine_->run({trt_detail::InputTensor{input_name_, shape, input.data(), input.size() * sizeof(float)}});
  const auto& output = trt_detail::find_output(outputs, output_name_);
  if (output.type != trt_detail::DataType::kFloat32 || output.shape != std::vector<int64_t>({1, descriptor_dim_})) {
    throw std::runtime_error("MixVPRTRT returned an unexpected output tensor");
  }
  const std::vector<float> values = output.values<float>();
  cv::Mat descriptor(1, descriptor_dim_, CV_32F);
  std::memcpy(descriptor.ptr<float>(), values.data(), values.size() * sizeof(float));
  if (normalize_output_) normalize_descriptor(descriptor);
  return descriptor;
}

cv::Mat MixVPRTRT::infer(const std::vector<cv::Mat>& images) {
  if (images.size() != 1) {
    throw std::invalid_argument("MixVPRTRT expected exactly one image, got " + std::to_string(images.size()));
  }
  return infer(images.front());
}

}  // namespace xfeat

#endif  // HAVE_TENSORRT
