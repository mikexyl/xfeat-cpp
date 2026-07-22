#include "xfeat-cpp/xfeat_trt.h"

#ifdef HAVE_TENSORRT

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "xfeat-cpp/helpers.h"
#include "xfeat-cpp/nms/anms/anms.h"
#include "xfeat-cpp/tensorrt/detail/trt_engine.h"

namespace xfeat {
namespace {

size_t indexCHW(int channel, int y, int x, int height, int width) {
  return (static_cast<size_t>(channel) * height + y) * width + x;
}

float readZeroPadded(const std::vector<float>& values, int channel, int y, int x, int channels, int height, int width) {
  if (channel < 0 || channel >= channels || x < 0 || x >= width || y < 0 || y >= height) return 0.0f;
  return values[indexCHW(channel, y, x, height, width)];
}

cv::Point2f sourceCoordinate(const cv::Point2f& point,
                             int input_width,
                             int input_height,
                             int source_width,
                             int source_height) {
  const float x = point.x * static_cast<float>(source_width) / static_cast<float>(input_width - 1) - 0.5f;
  const float y = point.y * static_cast<float>(source_height) / static_cast<float>(input_height - 1) - 0.5f;
  return {x, y};
}

float sampleNearest(const std::vector<float>& values,
                    int channel,
                    const cv::Point2f& point,
                    int channels,
                    int height,
                    int width,
                    int input_width,
                    int input_height) {
  const cv::Point2f source = sourceCoordinate(point, input_width, input_height, width, height);
  return readZeroPadded(values,
                        channel,
                        static_cast<int>(std::nearbyint(source.y)),
                        static_cast<int>(std::nearbyint(source.x)),
                        channels,
                        height,
                        width);
}

float sampleBilinear(const std::vector<float>& values,
                     int channel,
                     const cv::Point2f& point,
                     int channels,
                     int height,
                     int width,
                     int input_width,
                     int input_height) {
  const cv::Point2f source = sourceCoordinate(point, input_width, input_height, width, height);
  const int x0 = static_cast<int>(std::floor(source.x));
  const int y0 = static_cast<int>(std::floor(source.y));
  const float dx = source.x - x0;
  const float dy = source.y - y0;
  const float top = (1.0f - dx) * readZeroPadded(values, channel, y0, x0, channels, height, width) +
                    dx * readZeroPadded(values, channel, y0, x0 + 1, channels, height, width);
  const float bottom = (1.0f - dx) * readZeroPadded(values, channel, y0 + 1, x0, channels, height, width) +
                       dx * readZeroPadded(values, channel, y0 + 1, x0 + 1, channels, height, width);
  return (1.0f - dy) * top + dy * bottom;
}

float cubicWeight(float distance) {
  constexpr float alpha = -0.75f;
  const float x = std::abs(distance);
  if (x <= 1.0f) return (alpha + 2.0f) * x * x * x - (alpha + 3.0f) * x * x + 1.0f;
  if (x < 2.0f) return alpha * x * x * x - 5.0f * alpha * x * x + 8.0f * alpha * x - 4.0f * alpha;
  return 0.0f;
}

float sampleBicubic(const std::vector<float>& values,
                    int channel,
                    const cv::Point2f& point,
                    int channels,
                    int height,
                    int width,
                    int input_width,
                    int input_height) {
  const cv::Point2f source = sourceCoordinate(point, input_width, input_height, width, height);
  const int base_x = static_cast<int>(std::floor(source.x));
  const int base_y = static_cast<int>(std::floor(source.y));
  float result = 0.0f;
  for (int y_offset = -1; y_offset <= 2; ++y_offset) {
    const int y = base_y + y_offset;
    const float wy = cubicWeight(source.y - y);
    for (int x_offset = -1; x_offset <= 2; ++x_offset) {
      const int x = base_x + x_offset;
      const float wx = cubicWeight(source.x - x);
      result += wx * wy * readZeroPadded(values, channel, y, x, channels, height, width);
    }
  }
  return result;
}

cv::Mat keypointHeatmap(const std::vector<float>& logits, int height, int width) {
  constexpr int channels = 65;
  constexpr int grid = 8;
  cv::Mat heatmap(height * grid, width * grid, CV_32F);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float max_logit = -std::numeric_limits<float>::infinity();
      for (int channel = 0; channel < channels; ++channel) {
        max_logit = std::max(max_logit, logits[indexCHW(channel, y, x, height, width)]);
      }
      float denominator = 0.0f;
      float probabilities[64];
      for (int channel = 0; channel < channels; ++channel) {
        const float value = std::exp(logits[indexCHW(channel, y, x, height, width)] - max_logit);
        denominator += value;
        if (channel < 64) probabilities[channel] = value;
      }
      for (int grid_y = 0; grid_y < grid; ++grid_y) {
        for (int grid_x = 0; grid_x < grid; ++grid_x) {
          heatmap.at<float>(y * grid + grid_y, x * grid + grid_x) = probabilities[grid_y * grid + grid_x] / denominator;
        }
      }
    }
  }
  return heatmap;
}

cv::Mat nonMaximumSuppression(const cv::Mat& heatmap, float threshold = 0.05f, int kernel_size = 5) {
  cv::Mat maximum;
  cv::dilate(heatmap, maximum, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size)));
  const cv::Mat mask = (heatmap == maximum) & (heatmap > threshold);
  std::vector<cv::Point> points;
  cv::findNonZero(mask, points);
  cv::Mat keypoints(static_cast<int>(points.size()), 2, CV_32F);
  for (size_t i = 0; i < points.size(); ++i) {
    keypoints.at<float>(static_cast<int>(i), 0) = static_cast<float>(points[i].x);
    keypoints.at<float>(static_cast<int>(i), 1) = static_cast<float>(points[i].y);
  }
  return keypoints;
}

}  // namespace

XFeatTRT::XFeatTRT(const Params& params)
    : engine_(std::make_unique<trt_detail::Engine>(params.engine_path, params.verbose)),
      anms_(params.anms),
      nkpts_before_anms_(params.nkpts_before_anms),
      keypoint_detection_(params.keypoint_detection) {
  const auto inputs = engine_->input_names();
  if (inputs.size() != 1 || engine_->tensor_type(inputs.front()) != trt_detail::DataType::kFloat32) {
    throw std::runtime_error("XFeatTRT requires exactly one float32 input tensor");
  }
  input_name_ = inputs.front();
  const auto input_shape = engine_->tensor_shape(input_name_);
  if (input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] != 3 || input_shape[2] <= 0 ||
      input_shape[3] <= 0) {
    throw std::runtime_error("XFeatTRT input must have fixed shape [1, 3, height, width]");
  }
  input_height_ = static_cast<int>(input_shape[2]);
  input_width_ = static_cast<int>(input_shape[3]);

  for (const auto& name : engine_->output_names()) {
    if (engine_->tensor_type(name) != trt_detail::DataType::kFloat32) {
      throw std::runtime_error("XFeatTRT requires float32 output tensors");
    }
    const auto shape = engine_->tensor_shape(name);
    if (shape.size() == 4 && shape[0] == 1 && shape[1] == 64) feature_output_name_ = name;
    if (shape.size() == 4 && shape[0] == 1 && shape[1] == 65) keypoint_output_name_ = name;
  }
  if (feature_output_name_.empty() || keypoint_output_name_.empty()) {
    throw std::runtime_error("XFeatTRT engine does not expose 64-channel features and 65-channel keypoint logits");
  }
}

XFeatTRT::~XFeatTRT() = default;

std::vector<float> XFeatTRT::preprocess_image(const cv::Mat& image, float& resize_rate_w, float& resize_rate_h) const {
  if (image.empty()) throw std::invalid_argument("XFeatTRT input image is empty");
  cv::Mat color;
  if (image.channels() == 1) {
    cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
  } else if (image.channels() == 3) {
    color = image;
  } else {
    throw std::invalid_argument("XFeatTRT input must have one or three channels");
  }
  resize_rate_w = static_cast<float>(image.cols) / input_width_;
  resize_rate_h = static_cast<float>(image.rows) / input_height_;
  cv::resize(color, color, cv::Size(input_width_, input_height_));
  color.convertTo(color, CV_32F, 1.0 / 255.0);
  std::vector<cv::Mat> channels;
  cv::split(color, channels);
  const size_t plane = static_cast<size_t>(input_height_) * input_width_;
  std::vector<float> tensor(3 * plane);
  for (int channel = 0; channel < 3; ++channel) {
    std::memcpy(tensor.data() + channel * plane, channels[channel].ptr<float>(), plane * sizeof(float));
  }
  return tensor;
}

DetectionResult XFeatTRT::detect_and_compute(cv::Mat image,
                                             int top_k,
                                             cv::Mat* heatmap,
                                             cv::Mat* feature_map,
                                             cv::Mat* preprocessed,
                                             std::vector<cv::Vec2d>* uncertainty,
                                             const std::vector<cv::KeyPoint>& provided_keypoints,
                                             cv::Mat mask) {
  if (top_k <= 0) throw std::invalid_argument("XFeatTRT top_k must be positive");
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }

  float resize_rate_w = 1.0f;
  float resize_rate_h = 1.0f;
  std::vector<float> input = preprocess_image(image, resize_rate_w, resize_rate_h);
  if (preprocessed != nullptr) {
    const int sizes[] = {1, 3, input_height_, input_width_};
    *preprocessed = cv::Mat(4, sizes, CV_32F, input.data()).clone();
  }
  const auto outputs =
      engine_->run({{input_name_, {1, 3, input_height_, input_width_}, input.data(), input.size() * sizeof(float)}});
  const auto& feature_output = trt_detail::find_output(outputs, feature_output_name_);
  const auto& keypoint_output = trt_detail::find_output(outputs, keypoint_output_name_);
  if (feature_output.shape.size() != 4 || keypoint_output.shape.size() != 4) {
    throw std::runtime_error("XFeatTRT returned invalid output ranks");
  }
  const int channels = static_cast<int>(feature_output.shape[1]);
  const int feature_height = static_cast<int>(feature_output.shape[2]);
  const int feature_width = static_cast<int>(feature_output.shape[3]);
  if (channels != 64 || keypoint_output.shape[1] != 65 || keypoint_output.shape[2] != feature_height ||
      keypoint_output.shape[3] != feature_width) {
    throw std::runtime_error("XFeatTRT returned incompatible feature and keypoint outputs");
  }

  std::vector<float> features = feature_output.values<float>();
  const std::vector<float> logits = keypoint_output.values<float>();
  for (int y = 0; y < feature_height; ++y) {
    for (int x = 0; x < feature_width; ++x) {
      float norm_squared = 0.0f;
      for (int channel = 0; channel < channels; ++channel) {
        const float value = features[indexCHW(channel, y, x, feature_height, feature_width)];
        norm_squared += value * value;
      }
      const float norm = std::sqrt(norm_squared) + 1e-8f;
      for (int channel = 0; channel < channels; ++channel) {
        features[indexCHW(channel, y, x, feature_height, feature_width)] /= norm;
      }
    }
  }
  if (feature_map != nullptr) {
    const int sizes[] = {1, channels, feature_height, feature_width};
    *feature_map = cv::Mat(4, sizes, CV_32F, features.data()).clone();
  }

  cv::Mat keypoint_heatmap = keypointHeatmap(logits, feature_height, feature_width);
  if (heatmap != nullptr) *heatmap = keypoint_heatmap.clone();
  cv::Mat keypoint_matrix;
  if (keypoint_detection_ == 0) {
    if (provided_keypoints.empty()) {
      keypoint_matrix = nonMaximumSuppression(keypoint_heatmap);
    } else {
      keypoint_matrix = cv::Mat(static_cast<int>(provided_keypoints.size()), 2, CV_32F);
      for (size_t i = 0; i < provided_keypoints.size(); ++i) {
        keypoint_matrix.at<float>(static_cast<int>(i), 0) = provided_keypoints[i].pt.x / resize_rate_w;
        keypoint_matrix.at<float>(static_cast<int>(i), 1) = provided_keypoints[i].pt.y / resize_rate_h;
      }
    }
  } else {
    std::vector<cv::Point2f> new_keypoints;
    const int desired = std::max(0, nkpts_before_anms_ - static_cast<int>(provided_keypoints.size()));
    if (desired > 0) cv::goodFeaturesToTrack(gray, new_keypoints, desired, 0.001, 20, mask, 3, false, 0.04);
    keypoint_matrix = cv::Mat(static_cast<int>(provided_keypoints.size() + new_keypoints.size()), 2, CV_32F);
    size_t row = 0;
    for (const auto& keypoint : provided_keypoints) {
      keypoint_matrix.at<float>(static_cast<int>(row), 0) = keypoint.pt.x / resize_rate_w;
      keypoint_matrix.at<float>(static_cast<int>(row), 1) = keypoint.pt.y / resize_rate_h;
      ++row;
    }
    for (const auto& point : new_keypoints) {
      keypoint_matrix.at<float>(static_cast<int>(row), 0) = point.x / resize_rate_w;
      keypoint_matrix.at<float>(static_cast<int>(row), 1) = point.y / resize_rate_h;
      ++row;
    }
  }

  if (anms_ == 1 && keypoint_matrix.rows > top_k) {
    std::vector<cv::KeyPoint> candidates;
    for (int row = 0; row < keypoint_matrix.rows; ++row) {
      candidates.emplace_back(keypoint_matrix.at<float>(row, 0), keypoint_matrix.at<float>(row, 1), 1.0f);
    }
    const auto selected = anms::Ssc(candidates, top_k, 0.1f, input_width_, input_height_);
    keypoint_matrix = cv::Mat(static_cast<int>(selected.size()), 2, CV_32F);
    for (size_t i = 0; i < selected.size(); ++i) {
      keypoint_matrix.at<float>(static_cast<int>(i), 0) = selected[i].pt.x;
      keypoint_matrix.at<float>(static_cast<int>(i), 1) = selected[i].pt.y;
    }
  }

  if (keypoint_matrix.empty()) {
    return {cv::Mat(0, 2, CV_32F), cv::Mat(0, 1, CV_32F), cv::Mat(0, channels, CV_32F)};
  }
  const std::vector<float> heatmap_values(keypoint_heatmap.ptr<float>(),
                                          keypoint_heatmap.ptr<float>() + keypoint_heatmap.total());
  std::vector<float> candidate_scores(static_cast<size_t>(keypoint_matrix.rows));
  for (int row = 0; row < keypoint_matrix.rows; ++row) {
    const cv::Point2f point(keypoint_matrix.at<float>(row, 0), keypoint_matrix.at<float>(row, 1));
    candidate_scores[static_cast<size_t>(row)] =
        sampleNearest(
            heatmap_values, 0, point, 1, keypoint_heatmap.rows, keypoint_heatmap.cols, input_width_, input_height_) *
        sampleBilinear(
            heatmap_values, 0, point, 1, keypoint_heatmap.rows, keypoint_heatmap.cols, input_width_, input_height_);
    if (point.x == 0.0f && point.y == 0.0f) candidate_scores[static_cast<size_t>(row)] = -1.0f;
  }

  const size_t provided_count = std::min(provided_keypoints.size(), static_cast<size_t>(keypoint_matrix.rows));
  std::vector<int> order(static_cast<size_t>(keypoint_matrix.rows) - provided_count);
  std::iota(order.begin(), order.end(), static_cast<int>(provided_count));
  std::sort(order.begin(), order.end(), [&](int left, int right) {
    return candidate_scores[static_cast<size_t>(left)] > candidate_scores[static_cast<size_t>(right)];
  });

  std::vector<cv::Point2f> selected_points;
  std::vector<float> selected_scores;
  selected_points.reserve(std::min<size_t>(top_k, keypoint_matrix.rows));
  selected_scores.reserve(selected_points.capacity());
  for (size_t i = 0; i < provided_count; ++i) {
    selected_points.emplace_back(keypoint_matrix.at<float>(static_cast<int>(i), 0),
                                 keypoint_matrix.at<float>(static_cast<int>(i), 1));
    selected_scores.push_back(1.0f);
  }
  for (const int row : order) {
    if (selected_points.size() >= static_cast<size_t>(top_k)) break;
    selected_points.emplace_back(keypoint_matrix.at<float>(row, 0), keypoint_matrix.at<float>(row, 1));
    selected_scores.push_back(candidate_scores[static_cast<size_t>(row)]);
  }

  cv::Mat descriptors(static_cast<int>(selected_points.size()), channels, CV_32F);
  for (size_t row = 0; row < selected_points.size(); ++row) {
    float norm_squared = 0.0f;
    for (int channel = 0; channel < channels; ++channel) {
      const float value = sampleBicubic(features,
                                        channel,
                                        selected_points[row],
                                        channels,
                                        feature_height,
                                        feature_width,
                                        input_width_,
                                        input_height_);
      descriptors.at<float>(static_cast<int>(row), channel) = value;
      norm_squared += value * value;
    }
    const float norm = std::sqrt(norm_squared) + 1e-8f;
    descriptors.row(static_cast<int>(row)) /= norm;
  }

  std::vector<cv::KeyPoint> valid_keypoints;
  std::vector<float> valid_scores;
  cv::Mat valid_descriptors;
  for (size_t row = 0; row < selected_points.size(); ++row) {
    if (selected_scores[row] <= 0.0f && row >= provided_count) continue;
    cv::Point2f point(selected_points[row].x * resize_rate_w, selected_points[row].y * resize_rate_h);
    valid_keypoints.emplace_back(point, 1.0f);
    valid_scores.push_back(selected_scores[row]);
    valid_descriptors.push_back(descriptors.row(static_cast<int>(row)));
  }

  cv::Mat valid_keypoint_matrix(static_cast<int>(valid_keypoints.size()), 2, CV_32F);
  for (size_t row = 0; row < valid_keypoints.size(); ++row) {
    valid_keypoint_matrix.at<float>(static_cast<int>(row), 0) = valid_keypoints[row].pt.x;
    valid_keypoint_matrix.at<float>(static_cast<int>(row), 1) = valid_keypoints[row].pt.y;
  }
  cv::Mat valid_score_matrix(static_cast<int>(valid_scores.size()), 1, CV_32F);
  if (!valid_scores.empty()) {
    std::memcpy(valid_score_matrix.ptr<float>(), valid_scores.data(), valid_scores.size() * sizeof(float));
  }
  if (uncertainty != nullptr) {
    *uncertainty = computeUncertaintySobel(keypoint_heatmap, valid_keypoints);
    for (auto& value : *uncertainty) {
      value[0] *= resize_rate_w;
      value[1] *= resize_rate_h;
    }
  }
  return {valid_keypoint_matrix, valid_score_matrix, valid_descriptors};
}

}  // namespace xfeat

#endif  // HAVE_TENSORRT
