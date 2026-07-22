#include "xfeat-cpp/place_recognition/jist_trt.h"

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
    throw std::runtime_error("JistTRT requires a fixed positive " + label + " dimension");
  }
  return static_cast<int>(shape[index]);
}

}  // namespace

JistTRT::JistTRT(const Params& params)
    : engine_(std::make_unique<trt_detail::Engine>(params.model_path, params.verbose)),
      img_height_(params.img_height),
      img_width_(params.img_width),
      normalize_output_(params.normalize_output) {
  const auto input_names = engine_->input_names();
  const auto output_names = engine_->output_names();
  if (input_names.size() != 1 || output_names.size() != 1) {
    throw std::runtime_error("JistTRT requires exactly one input and one output tensor");
  }
  input_name_ = input_names.front();
  output_name_ = output_names.front();
  if (engine_->tensor_type(input_name_) != trt_detail::DataType::kFloat32 ||
      engine_->tensor_type(output_name_) != trt_detail::DataType::kFloat32) {
    throw std::runtime_error("JistTRT requires float32 engine I/O tensors");
  }

  const auto input_shape = engine_->tensor_shape(input_name_);
  const auto output_shape = engine_->tensor_shape(output_name_);
  if (input_shape.size() != 5 || input_shape[0] != 1 || input_shape[2] != 3) {
    throw std::runtime_error("JistTRT input must have shape [1, sequence, 3, height, width]");
  }
  if (output_shape.size() != 2 || output_shape[0] != 1) {
    throw std::runtime_error("JistTRT output must have shape [1, descriptor_dim]");
  }

  seq_length_ = checkedDimension(input_shape, 1, "sequence");
  const int engine_height = checkedDimension(input_shape, 3, "height");
  const int engine_width = checkedDimension(input_shape, 4, "width");
  descriptor_dim_ = checkedDimension(output_shape, 1, "descriptor");
  if (engine_height != img_height_ || engine_width != img_width_) {
    throw std::runtime_error("JistTRT engine image dimensions do not match configured dimensions");
  }
}

JistTRT::~JistTRT() = default;

cv::Mat JistTRT::preprocess_image(const cv::Mat& image) const {
  if (image.empty()) {
    throw std::invalid_argument("JistTRT input image is empty");
  }
  cv::Mat color;
  if (image.channels() == 1) {
    cv::cvtColor(image, color, cv::COLOR_GRAY2RGB);
  } else if (image.channels() == 3) {
    cv::cvtColor(image, color, cv::COLOR_BGR2RGB);
  } else {
    throw std::invalid_argument("JistTRT input must have one or three channels");
  }
  cv::resize(color, color, cv::Size(img_width_, img_height_));
  color.convertTo(color, CV_32F, 1.0 / 255.0);
  return color;
}

std::vector<float> JistTRT::prepare_input_tensor(const std::vector<cv::Mat>& image_sequence) const {
  if (image_sequence.size() != static_cast<size_t>(seq_length_)) {
    throw std::invalid_argument("JistTRT expected " + std::to_string(seq_length_) + " images, got " +
                                std::to_string(image_sequence.size()));
  }
  const size_t plane = static_cast<size_t>(img_height_) * img_width_;
  std::vector<float> tensor(static_cast<size_t>(seq_length_) * 3 * plane);
  for (int sequence_index = 0; sequence_index < seq_length_; ++sequence_index) {
    const cv::Mat image = preprocess_image(image_sequence[sequence_index]);
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    for (int channel = 0; channel < 3; ++channel) {
      const size_t offset = (static_cast<size_t>(sequence_index) * 3 + channel) * plane;
      std::memcpy(tensor.data() + offset, channels[channel].ptr<float>(), plane * sizeof(float));
    }
  }
  return tensor;
}

void JistTRT::normalize_descriptor(cv::Mat& descriptor) const {
  const double norm = cv::norm(descriptor, cv::NORM_L2);
  if (norm > 1e-8) descriptor /= norm;
}

cv::Mat JistTRT::infer(const std::vector<cv::Mat>& image_sequence) {
  std::vector<float> input = prepare_input_tensor(image_sequence);
  const std::vector<int64_t> shape = {1, seq_length_, 3, img_height_, img_width_};
  const auto outputs =
      engine_->run({trt_detail::InputTensor{input_name_, shape, input.data(), input.size() * sizeof(float)}});
  const auto& output = trt_detail::find_output(outputs, output_name_);
  if (output.type != trt_detail::DataType::kFloat32 || output.shape != std::vector<int64_t>({1, descriptor_dim_})) {
    throw std::runtime_error("JistTRT returned an unexpected output tensor");
  }
  const std::vector<float> values = output.values<float>();
  cv::Mat descriptor(1, descriptor_dim_, CV_32F);
  std::memcpy(descriptor.ptr<float>(), values.data(), values.size() * sizeof(float));
  if (normalize_output_) normalize_descriptor(descriptor);
  return descriptor;
}

cv::Mat JistTRT::infer_batch(const std::vector<std::vector<cv::Mat>>& batch_sequences) {
  if (batch_sequences.empty()) {
    throw std::invalid_argument("JistTRT batch is empty");
  }
  cv::Mat descriptors;
  for (const auto& sequence : batch_sequences) {
    descriptors.push_back(infer(sequence));
  }
  return descriptors;
}

}  // namespace xfeat

#endif  // HAVE_TENSORRT
