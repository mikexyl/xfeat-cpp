#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <opencv2/core.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace xfeat {
namespace mono_depth_detail {

inline std::string tensorShapeToString(const std::vector<int64_t>& dims) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << dims[i];
  }
  oss << "]";
  return oss.str();
}

inline size_t checkedTensorVolume(const std::vector<int64_t>& dims, const std::string& label) {
  if (dims.empty()) {
    throw std::runtime_error("Unsupported " + label + " output shape []");
  }

  size_t count = 1;
  for (int64_t dim : dims) {
    if (dim <= 0) {
      throw std::runtime_error("Tensor " + label +
                               " output dimensions are not fully specified: " + tensorShapeToString(dims));
    }
    const auto dim_size = static_cast<size_t>(dim);
    if (count > std::numeric_limits<size_t>::max() / dim_size) {
      throw std::runtime_error("Tensor " + label + " output shape is too large: " + tensorShapeToString(dims));
    }
    count *= dim_size;
  }
  return count;
}

inline std::vector<cv::Mat> extractDepthAnythingTensorPlanes(const std::vector<float>& values,
                                                             const std::vector<int64_t>& dims,
                                                             int requested_views,
                                                             const std::string& label) {
  if (requested_views <= 0) {
    throw std::runtime_error("Requested " + label + " view count must be positive");
  }
  if (checkedTensorVolume(dims, label) != values.size()) {
    throw std::runtime_error("TensorRT " + label + " output shape is inconsistent with its buffer size");
  }

  std::vector<cv::Mat> planes;
  planes.reserve(static_cast<size_t>(requested_views));

  auto copyPlane = [&](size_t offset, int64_t height, int64_t width) {
    if (height <= 0 || width <= 0 || height > std::numeric_limits<int>::max() ||
        width > std::numeric_limits<int>::max()) {
      throw std::runtime_error("Unsupported " + label + " output shape " + tensorShapeToString(dims));
    }
    const auto h = static_cast<int>(height);
    const auto w = static_cast<int>(width);
    const size_t plane_size = static_cast<size_t>(h) * static_cast<size_t>(w);
    if (offset + plane_size > values.size()) {
      throw std::runtime_error("TensorRT " + label + " output shape is inconsistent with its buffer size");
    }
    cv::Mat plane(h, w, CV_32FC1);
    std::memcpy(plane.data, values.data() + offset, plane_size * sizeof(float));
    planes.push_back(std::move(plane));
  };

  if (dims.size() == 5) {
    const int64_t batch = dims[0];
    const int64_t views = dims[1];
    if (batch != 1 || views != requested_views) {
      throw std::runtime_error("Unsupported " + label + " output shape " + tensorShapeToString(dims));
    }

    if (dims[4] == 1 && dims[2] > 1) {
      const int64_t height = dims[2];
      const int64_t width = dims[3];
      const size_t view_stride = static_cast<size_t>(height) * static_cast<size_t>(width);
      for (int view = 0; view < requested_views; ++view) {
        copyPlane(static_cast<size_t>(view) * view_stride, height, width);
      }
      return planes;
    }

    const int64_t channels = dims[2];
    const int64_t height = dims[3];
    const int64_t width = dims[4];
    if (channels < 1) {
      throw std::runtime_error("Unsupported " + label + " output shape " + tensorShapeToString(dims));
    }
    const size_t view_stride = static_cast<size_t>(channels) * static_cast<size_t>(height) * static_cast<size_t>(width);
    for (int view = 0; view < requested_views; ++view) {
      copyPlane(static_cast<size_t>(view) * view_stride, height, width);
    }
    return planes;
  }

  if (dims.size() == 4) {
    if (dims[0] == requested_views) {
      const int64_t channels = dims[1];
      const int64_t height = dims[2];
      const int64_t width = dims[3];
      if (channels < 1) {
        throw std::runtime_error("Unsupported " + label + " output shape " + tensorShapeToString(dims));
      }
      const size_t view_stride =
          static_cast<size_t>(channels) * static_cast<size_t>(height) * static_cast<size_t>(width);
      for (int view = 0; view < requested_views; ++view) {
        copyPlane(static_cast<size_t>(view) * view_stride, height, width);
      }
      return planes;
    }

    if (dims[0] == 1 && dims[1] == requested_views) {
      const int64_t height = dims[2];
      const int64_t width = dims[3];
      const size_t view_stride = static_cast<size_t>(height) * static_cast<size_t>(width);
      for (int view = 0; view < requested_views; ++view) {
        copyPlane(static_cast<size_t>(view) * view_stride, height, width);
      }
      return planes;
    }

    throw std::runtime_error("Unsupported " + label + " output shape " + tensorShapeToString(dims));
  }

  if (dims.size() == 3) {
    const int64_t views = dims[0];
    const int64_t height = dims[1];
    const int64_t width = dims[2];
    if (views != requested_views) {
      throw std::runtime_error("Unsupported " + label + " output shape " + tensorShapeToString(dims));
    }
    const size_t view_stride = static_cast<size_t>(height) * static_cast<size_t>(width);
    for (int view = 0; view < requested_views; ++view) {
      copyPlane(static_cast<size_t>(view) * view_stride, height, width);
    }
    return planes;
  }

  if (dims.size() == 2 && requested_views == 1) {
    copyPlane(0, dims[0], dims[1]);
    return planes;
  }

  throw std::runtime_error("Unsupported " + label + " output shape " + tensorShapeToString(dims));
}

}  // namespace mono_depth_detail
}  // namespace xfeat
