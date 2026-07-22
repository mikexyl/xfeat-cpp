#pragma once

#ifdef HAVE_TENSORRT

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace xfeat {
namespace trt_detail {

enum class DataType {
  kFloat32,
  kFloat16,
  kInt8,
  kInt32,
  kInt64,
  kUInt8,
  kBool,
  kBFloat16,
  kFloat8,
  kInt4,
  kUnknown,
};

struct InputTensor {
  std::string name;
  std::vector<int64_t> shape;
  const void* data = nullptr;
  size_t bytes = 0;
};

struct OutputTensor {
  std::string name;
  std::vector<int64_t> shape;
  DataType type = DataType::kUnknown;
  std::vector<uint8_t> bytes;

  size_t element_count() const;

  template <typename T>
  std::vector<T> values() const {
    if (bytes.size() % sizeof(T) != 0) {
      throw std::runtime_error("TensorRT output byte size is incompatible with requested value type");
    }
    std::vector<T> result(bytes.size() / sizeof(T));
    if (!bytes.empty()) {
      std::memcpy(result.data(), bytes.data(), bytes.size());
    }
    return result;
  }
};

class Engine {
 public:
  explicit Engine(const std::string& engine_path, bool verbose = false);
  ~Engine();

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) noexcept;
  Engine& operator=(Engine&&) noexcept;

  std::vector<OutputTensor> run(const std::vector<InputTensor>& inputs);

  std::vector<std::string> input_names() const;
  std::vector<std::string> output_names() const;
  std::vector<int64_t> tensor_shape(const std::string& name) const;
  DataType tensor_type(const std::string& name) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

const OutputTensor& find_output(const std::vector<OutputTensor>& outputs, const std::string& name);

}  // namespace trt_detail
}  // namespace xfeat

#endif  // HAVE_TENSORRT
