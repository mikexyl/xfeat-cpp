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
  if (input_names.size() != 1 || output_names.empty() || output_names.size() > 2) {
    throw std::runtime_error("JistTRT requires one input and one or two output tensors");
  }
  input_name_ = input_names.front();
  if (engine_->tensor_type(input_name_) != trt_detail::DataType::kFloat32) {
    throw std::runtime_error("JistTRT requires float32 engine input");
  }

  const auto input_shape = engine_->tensor_shape(input_name_);
  if (input_shape.size() != 5 || input_shape[0] != 1 || input_shape[2] != 3) {
    throw std::runtime_error("JistTRT input must have shape [1, sequence, 3, height, width]");
  }

  seq_length_ = checkedDimension(input_shape, 1, "sequence");
  const int engine_height = checkedDimension(input_shape, 3, "height");
  const int engine_width = checkedDimension(input_shape, 4, "width");
  if (engine_height != img_height_ || engine_width != img_width_) {
    throw std::runtime_error("JistTRT engine image dimensions do not match configured dimensions");
  }

  for (const std::string& name : output_names) {
    if (engine_->tensor_type(name) != trt_detail::DataType::kFloat32) {
      throw std::runtime_error("JistTRT requires float32 engine outputs");
    }
    const auto shape = engine_->tensor_shape(name);
    if (shape.size() != 2) {
      throw std::runtime_error("JistTRT outputs must be rank-two descriptor tensors");
    }
    const int output_descriptor_dim = checkedDimension(shape, 1, "descriptor");
    if (shape[0] == 1 && output_name_.empty()) {
      output_name_ = name;
      if (descriptor_dim_ != 0 && descriptor_dim_ != output_descriptor_dim) {
        throw std::runtime_error("JistTRT sequence and frame descriptor dimensions do not match");
      }
      descriptor_dim_ = output_descriptor_dim;
    } else if (shape[0] == seq_length_ && frame_output_name_.empty()) {
      frame_output_name_ = name;
      if (descriptor_dim_ != 0 && descriptor_dim_ != output_descriptor_dim) {
        throw std::runtime_error("JistTRT sequence and frame descriptor dimensions do not match");
      }
      descriptor_dim_ = output_descriptor_dim;
    } else {
      throw std::runtime_error("JistTRT output shapes must be [1, D] and optionally [sequence, D]");
    }
  }
  if (output_name_.empty()) {
    throw std::runtime_error("JistTRT engine does not expose a [1, descriptor_dim] sequence output");
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

JistTRT::InferenceResult JistTRT::infer_all(const std::vector<cv::Mat>& image_sequence) {
  std::vector<float> input = prepare_input_tensor(image_sequence);
  const std::vector<int64_t> shape = {1, seq_length_, 3, img_height_, img_width_};
  const auto outputs =
      engine_->run({trt_detail::InputTensor{input_name_, shape, input.data(), input.size() * sizeof(float)}});

  InferenceResult result;
  const auto& output = trt_detail::find_output(outputs, output_name_);
  if (output.type != trt_detail::DataType::kFloat32 || output.shape != std::vector<int64_t>({1, descriptor_dim_})) {
    throw std::runtime_error("JistTRT returned an unexpected output tensor");
  }
  const std::vector<float> values = output.values<float>();
  result.sequence_descriptor = cv::Mat(1, descriptor_dim_, CV_32F);
  std::memcpy(result.sequence_descriptor.ptr<float>(), values.data(), values.size() * sizeof(float));
  if (normalize_output_) normalize_descriptor(result.sequence_descriptor);

  if (!frame_output_name_.empty()) {
    const auto& frame_output = trt_detail::find_output(outputs, frame_output_name_);
    if (frame_output.type != trt_detail::DataType::kFloat32 ||
        frame_output.shape != std::vector<int64_t>({seq_length_, descriptor_dim_})) {
      throw std::runtime_error("JistTRT returned an unexpected frame descriptor tensor");
    }
    const std::vector<float> frame_values = frame_output.values<float>();
    result.frame_descriptors = cv::Mat(seq_length_, descriptor_dim_, CV_32F);
    std::memcpy(result.frame_descriptors.ptr<float>(), frame_values.data(), frame_values.size() * sizeof(float));
    if (normalize_output_) {
      for (int frame = 0; frame < seq_length_; ++frame) {
        cv::Mat row = result.frame_descriptors.row(frame);
        normalize_descriptor(row);
      }
    }
  }
  return result;
}

cv::Mat JistTRT::infer(const std::vector<cv::Mat>& image_sequence) {
  return infer_all(image_sequence).sequence_descriptor;
}

JistTRT::InferenceResult JistTRT::infer_with_frame_descriptors(const std::vector<cv::Mat>& image_sequence) {
  if (!has_frame_descriptors()) {
    throw std::runtime_error("JistTRT engine does not expose per-frame descriptors");
  }
  return infer_all(image_sequence);
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
