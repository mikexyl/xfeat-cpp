#include "xfeat-cpp/tensorrt/detail/trt_engine.h"

#ifdef HAVE_TENSORRT

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace xfeat {
namespace trt_detail {
namespace {

void checkCuda(cudaError_t status, const std::string& operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(operation + ": " + cudaGetErrorString(status));
  }
}

size_t checkedVolume(const std::vector<int64_t>& shape) {
  size_t volume = 1;
  for (const int64_t dimension : shape) {
    if (dimension < 0) {
      throw std::runtime_error("TensorRT tensor still has a dynamic dimension at execution time");
    }
    if (dimension == 0) {
      return 0;
    }
    const size_t value = static_cast<size_t>(dimension);
    if (volume > std::numeric_limits<size_t>::max() / value) {
      throw std::overflow_error("TensorRT tensor element count overflows size_t");
    }
    volume *= value;
  }
  return volume;
}

std::vector<int64_t> toShape(const nvinfer1::Dims& dims) {
  std::vector<int64_t> shape;
  shape.reserve(dims.nbDims);
  for (int i = 0; i < dims.nbDims; ++i) {
    shape.push_back(dims.d[i]);
  }
  return shape;
}

nvinfer1::Dims toDims(const std::vector<int64_t>& shape) {
  if (shape.size() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS)) {
    throw std::invalid_argument("TensorRT input rank exceeds the supported maximum");
  }
  nvinfer1::Dims dims{};
  dims.nbDims = static_cast<int>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0 || shape[i] > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument("TensorRT input dimensions must be non-negative int32 values");
    }
    dims.d[i] = static_cast<int32_t>(shape[i]);
  }
  return dims;
}

DataType toDataType(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return DataType::kFloat32;
    case nvinfer1::DataType::kHALF:
      return DataType::kFloat16;
    case nvinfer1::DataType::kINT8:
      return DataType::kInt8;
    case nvinfer1::DataType::kINT32:
      return DataType::kInt32;
    case nvinfer1::DataType::kINT64:
      return DataType::kInt64;
    case nvinfer1::DataType::kUINT8:
      return DataType::kUInt8;
    case nvinfer1::DataType::kBOOL:
      return DataType::kBool;
    case nvinfer1::DataType::kBF16:
      return DataType::kBFloat16;
    case nvinfer1::DataType::kFP8:
      return DataType::kFloat8;
    case nvinfer1::DataType::kINT4:
      return DataType::kInt4;
    default:
      return DataType::kUnknown;
  }
}

size_t elementSize(DataType type) {
  switch (type) {
    case DataType::kFloat32:
    case DataType::kInt32:
      return 4;
    case DataType::kFloat16:
    case DataType::kBFloat16:
      return 2;
    case DataType::kInt64:
      return 8;
    case DataType::kInt8:
    case DataType::kUInt8:
    case DataType::kBool:
    case DataType::kFloat8:
      return 1;
    case DataType::kInt4:
      throw std::runtime_error("Packed TensorRT INT4 I/O tensors are not supported");
    default:
      throw std::runtime_error("Unsupported TensorRT tensor data type");
  }
}

class Logger : public nvinfer1::ILogger {
 public:
  explicit Logger(bool verbose) : verbose_(verbose) {}

  void enableVerbose(bool verbose) { verbose_ = verbose_ || verbose; }

  void log(Severity severity, const char* message) noexcept override {
    const Severity threshold = verbose_ ? Severity::kINFO : Severity::kWARNING;
    if (severity <= threshold) {
      std::cerr << "[TensorRT] " << message << '\n';
    }
  }

 private:
  bool verbose_;
};

Logger& globalLogger(bool verbose) {
  static Logger logger(false);
  logger.enableVerbose(verbose);
  return logger;
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes) { allocate(bytes); }
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), bytes_(other.bytes_) {
    other.data_ = nullptr;
    other.bytes_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) cudaFree(data_);
      data_ = other.data_;
      bytes_ = other.bytes_;
      other.data_ = nullptr;
      other.bytes_ = 0;
    }
    return *this;
  }

  void allocate(size_t bytes) {
    if (bytes == 0) {
      throw std::invalid_argument("TensorRT device buffer cannot be zero bytes");
    }
    checkCuda(cudaMalloc(&data_, bytes), "Failed to allocate TensorRT device buffer");
    bytes_ = bytes;
  }

  void* data() const { return data_; }
  size_t bytes() const { return bytes_; }

 private:
  void* data_ = nullptr;
  size_t bytes_ = 0;
};

class OutputAllocator : public nvinfer1::IOutputAllocator {
 public:
  ~OutputAllocator() override {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  void* reallocateOutputAsync(const char*,
                              void* current_memory,
                              uint64_t size,
                              uint64_t,
                              cudaStream_t) noexcept override {
    if (data_ != nullptr && current_memory == data_ && capacity_ >= size) {
      return data_;
    }

    void* replacement = nullptr;
    if (cudaMalloc(&replacement, std::max<uint64_t>(size, 1)) != cudaSuccess) {
      return nullptr;
    }
    if (data_ != nullptr) {
      cudaFree(data_);
    }
    data_ = replacement;
    capacity_ = std::max<uint64_t>(size, 1);
    return data_;
  }

  void notifyShape(const char*, const nvinfer1::Dims& dims) noexcept override {
    dims_ = dims;
    shape_notified_ = true;
  }

  void* data() const { return data_; }
  const nvinfer1::Dims& dims() const { return dims_; }
  bool shape_notified() const { return shape_notified_; }

 private:
  void* data_ = nullptr;
  uint64_t capacity_ = 0;
  nvinfer1::Dims dims_{};
  bool shape_notified_ = false;
};

}  // namespace

size_t OutputTensor::element_count() const { return checkedVolume(shape); }

class Engine::Impl {
 public:
  Impl(const std::string& engine_path, bool verbose) : logger_(globalLogger(verbose)) {
    std::ifstream stream(engine_path, std::ios::binary | std::ios::ate);
    if (!stream) {
      throw std::invalid_argument("TensorRT engine is not readable: " + engine_path);
    }
    const std::streamsize size = stream.tellg();
    if (size <= 0) {
      throw std::invalid_argument("TensorRT engine is empty: " + engine_path);
    }
    stream.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<size_t>(size));
    if (!stream.read(bytes.data(), size)) {
      throw std::runtime_error("Failed to read TensorRT engine: " + engine_path);
    }

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (runtime_ == nullptr) throw std::runtime_error("Failed to create TensorRT runtime");
    engine_ = runtime_->deserializeCudaEngine(bytes.data(), bytes.size());
    if (engine_ == nullptr) throw std::runtime_error("Failed to deserialize TensorRT engine: " + engine_path);
    context_ = engine_->createExecutionContext();
    if (context_ == nullptr) throw std::runtime_error("Failed to create TensorRT execution context");
    checkCuda(cudaStreamCreate(&stream_), "Failed to create TensorRT CUDA stream");

    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char* name = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        input_names_.emplace_back(name);
      } else {
        output_names_.emplace_back(name);
      }
    }
  }

  ~Impl() {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    delete context_;
    delete engine_;
    delete runtime_;
  }

  std::vector<OutputTensor> run(const std::vector<InputTensor>& inputs) {
    if (inputs.size() != input_names_.size()) {
      throw std::invalid_argument("TensorRT input count mismatch: expected " + std::to_string(input_names_.size()) +
                                  ", got " + std::to_string(inputs.size()));
    }

    std::unordered_map<std::string, const InputTensor*> by_name;
    for (const auto& input : inputs) {
      if (!by_name.emplace(input.name, &input).second) {
        throw std::invalid_argument("Duplicate TensorRT input: " + input.name);
      }
    }

    std::vector<DeviceBuffer> input_buffers;
    input_buffers.reserve(inputs.size());
    for (const auto& name : input_names_) {
      const auto found = by_name.find(name);
      if (found == by_name.end()) {
        throw std::invalid_argument("Missing TensorRT input: " + name);
      }
      const InputTensor& input = *found->second;
      if (input.data == nullptr) {
        throw std::invalid_argument("TensorRT input data is null: " + name);
      }
      const size_t required_bytes = checkedVolume(input.shape) * elementSize(tensorType(name));
      if (input.bytes != required_bytes) {
        throw std::invalid_argument("TensorRT input byte size mismatch for " + name + ": expected " +
                                    std::to_string(required_bytes) + ", got " + std::to_string(input.bytes));
      }
      if (!context_->setInputShape(name.c_str(), toDims(input.shape))) {
        throw std::runtime_error("TensorRT rejected input shape for " + name);
      }
      input_buffers.emplace_back(input.bytes);
      checkCuda(cudaMemcpyAsync(input_buffers.back().data(), input.data, input.bytes, cudaMemcpyHostToDevice, stream_),
                "Failed to upload TensorRT input " + name);
      if (!context_->setInputTensorAddress(name.c_str(), input_buffers.back().data())) {
        throw std::runtime_error("Failed to bind TensorRT input " + name);
      }
    }

    std::vector<const char*> missing_names(static_cast<size_t>(engine_->getNbIOTensors()));
    const int missing = context_->inferShapes(static_cast<int>(missing_names.size()), missing_names.data());
    if (missing < 0) {
      throw std::runtime_error("TensorRT shape inference failed");
    }
    if (missing > 0) {
      std::ostringstream message;
      message << "TensorRT shape inference is missing " << missing << " input(s)";
      for (int i = 0; i < std::min<int>(missing, missing_names.size()); ++i) {
        message << (i == 0 ? ": " : ", ") << missing_names[i];
      }
      throw std::runtime_error(message.str());
    }

    std::vector<std::unique_ptr<OutputAllocator>> allocators;
    allocators.reserve(output_names_.size());
    for (const auto& name : output_names_) {
      allocators.push_back(std::make_unique<OutputAllocator>());
      if (!context_->setTensorAddress(name.c_str(), nullptr) ||
          !context_->setOutputAllocator(name.c_str(), allocators.back().get())) {
        throw std::runtime_error("Failed to bind TensorRT output " + name);
      }
    }

    if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error("TensorRT inference enqueue failed");
    }

    std::vector<OutputTensor> outputs;
    outputs.reserve(output_names_.size());
    for (size_t i = 0; i < output_names_.size(); ++i) {
      const std::string& name = output_names_[i];
      const nvinfer1::Dims dims =
          allocators[i]->shape_notified() ? allocators[i]->dims() : context_->getTensorShape(name.c_str());
      OutputTensor output;
      output.name = name;
      output.shape = toShape(dims);
      output.type = tensorType(name);
      const size_t output_bytes = checkedVolume(output.shape) * elementSize(output.type);
      output.bytes.resize(output_bytes);
      if (output_bytes > 0) {
        if (allocators[i]->data() == nullptr) {
          throw std::runtime_error("TensorRT did not allocate output " + name);
        }
        checkCuda(
            cudaMemcpyAsync(output.bytes.data(), allocators[i]->data(), output_bytes, cudaMemcpyDeviceToHost, stream_),
            "Failed to download TensorRT output " + name);
      }
      outputs.push_back(std::move(output));
    }
    checkCuda(cudaStreamSynchronize(stream_), "TensorRT CUDA stream synchronization failed");
    return outputs;
  }

  DataType tensorType(const std::string& name) const { return toDataType(engine_->getTensorDataType(name.c_str())); }

  Logger& logger_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  cudaStream_t stream_ = nullptr;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

Engine::Engine(const std::string& engine_path, bool verbose) : impl_(std::make_unique<Impl>(engine_path, verbose)) {}
Engine::~Engine() = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

std::vector<OutputTensor> Engine::run(const std::vector<InputTensor>& inputs) { return impl_->run(inputs); }
std::vector<std::string> Engine::input_names() const { return impl_->input_names_; }
std::vector<std::string> Engine::output_names() const { return impl_->output_names_; }

std::vector<int64_t> Engine::tensor_shape(const std::string& name) const {
  return toShape(impl_->engine_->getTensorShape(name.c_str()));
}

DataType Engine::tensor_type(const std::string& name) const { return impl_->tensorType(name); }

const OutputTensor& find_output(const std::vector<OutputTensor>& outputs, const std::string& name) {
  const auto found =
      std::find_if(outputs.begin(), outputs.end(), [&](const OutputTensor& output) { return output.name == name; });
  if (found == outputs.end()) {
    throw std::runtime_error("TensorRT output not found: " + name);
  }
  return *found;
}

}  // namespace trt_detail
}  // namespace xfeat

#endif  // HAVE_TENSORRT
