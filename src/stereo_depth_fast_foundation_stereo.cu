#include "xfeat-cpp/stereo_depth_fast_foundation_stereo.h"

#ifdef HAVE_TENSORRT

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace xfeat {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

class TrtLogger : public nvinfer1::ILogger {
 public:
  explicit TrtLogger(bool verbose) : verbose_(verbose) {}
  void log(Severity sev, const char* msg) noexcept override {
    if (verbose_ || sev <= Severity::kWARNING)
      std::cout << "[TRT] " << msg << "\n";
  }
  bool verbose_;
};

size_t getElementSize(nvinfer1::DataType dt) {
  switch (dt) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT8:  return 1;
    default: throw std::runtime_error("Unsupported TRT DataType");
  }
}

size_t getVolume(const nvinfer1::Dims& d) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= (size_t)d.d[i];
  return v;
}

// ---------------------------------------------------------------------------
// GWC volume CUDA kernels
//
// Builds the Group-Wise Correlation volume between left and right features
// entirely on GPU, matching Python's build_gwc_volume_triton exactly.
//
// ref, tar : [B, C, H, W]  NCHW  (float16 or float32 from the TRT engine)
// out      : [B, G, D, H, W]  float32
//
//   out[b,g,d,h,w] = dot( ref[b, g*K:(g+1)*K, h, w   ],
//                         tar[b, g*K:(g+1)*K, h, w-d ] )
//                   optionally L2-normalised per group.
// ---------------------------------------------------------------------------

template <typename InT, bool NORMALIZE>
__global__ void gwc_volume_kernel(const InT* __restrict__ ref,
                                  const InT* __restrict__ tar,
                                  float*     __restrict__ out,
                                  int B, int C, int H, int W, int G, int D) {
  const int K = C / G;
  const long long total = (long long)B * G * D * H * W;
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  int w = idx % W;    idx /= W;
  int h = idx % H;    idx /= H;
  int d = idx % D;    idx /= D;
  int g = idx % G;
  int b = (int)(idx / G);

  const int w_tar = w - d;
  float acc = 0.f, ref_sq = 0.f, tar_sq = 0.f;

  if (w_tar >= 0) {
    for (int k = 0; k < K; ++k) {
      int c = g * K + k;
      // NCHW index: [b,c,h,w] = (b*C + c)*H*W + h*W + w
      float rv = (float)ref[((long long)(b * C + c) * H + h) * W + w];
      float tv = (float)tar[((long long)(b * C + c) * H + h) * W + w_tar];
      acc += rv * tv;
      if (NORMALIZE) { ref_sq += rv * rv; tar_sq += tv * tv; }
    }
    if (NORMALIZE)
      acc /= (sqrtf(ref_sq) * sqrtf(tar_sq) + 1e-5f);
  }

  long long out_idx = ((long long)(b * G + g) * D + d) * H * W + (long long)h * W + w;
  out[out_idx] = acc;
}

// Cast float32 buffer to float16 when the post engine expects kHALF.
__global__ void cast_f32_to_f16(const float* __restrict__ in,
                                 __half*       __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------
class FastFoundationStereoDepth::Impl {
 public:
  explicit Impl(const Params& params) : params_(params), logger_(params.verbose) {
    loadEngine(params.feature_engine_path, feature_runtime_, feature_engine_, feature_ctx_);
    loadEngine(params.post_engine_path,    post_runtime_,    post_engine_,    post_ctx_);
    if (cudaStreamCreate(&stream_) != cudaSuccess)
      throw std::runtime_error("Failed to create CUDA stream");
    allocateBuffers();
  }

  ~Impl() {
    if (stream_) cudaStreamDestroy(stream_);
    if (gwc_f32_buf_) cudaFree(gwc_f32_buf_);
    for (auto& [name, ptr] : buf_) if (ptr) cudaFree(ptr);
    delete feature_ctx_;  delete feature_engine_;  delete feature_runtime_;
    delete post_ctx_;     delete post_engine_;     delete post_runtime_;
  }

  cv::Mat run(const cv::Mat& left_bgr, const cv::Mat& right_bgr) {
    uploadImage(left_bgr,  "left");
    uploadImage(right_bgr, "right");

    // 1. Feature runner: images → multi-scale features
    setAddresses(feature_engine_, feature_ctx_);
    if (!feature_ctx_->enqueueV3(stream_))
      throw std::runtime_error("Feature runner enqueue failed");

    // 2. GWC volume: features_left_04 × features_right_04 → gwc_volume  (on GPU)
    buildGwcVolume();

    // 3. Post runner: features + gwc_volume → disparity
    setAddresses(post_engine_, post_ctx_);
    if (!post_ctx_->enqueueV3(stream_))
      throw std::runtime_error("Post runner enqueue failed");

    return downloadDisparity();
  }

 private:
  Params   params_;
  TrtLogger logger_;

  nvinfer1::IRuntime*          feature_runtime_ = nullptr;
  nvinfer1::ICudaEngine*       feature_engine_  = nullptr;
  nvinfer1::IExecutionContext* feature_ctx_     = nullptr;

  nvinfer1::IRuntime*          post_runtime_ = nullptr;
  nvinfer1::ICudaEngine*       post_engine_  = nullptr;
  nvinfer1::IExecutionContext* post_ctx_     = nullptr;

  cudaStream_t stream_ = nullptr;

  std::unordered_map<std::string, void*>              buf_;
  std::unordered_map<std::string, nvinfer1::Dims>     dims_;
  std::unordered_map<std::string, nvinfer1::DataType> dtypes_;

  // Intermediate float32 GWC buffer (needed when post engine expects fp16)
  void*  gwc_f32_buf_    = nullptr;
  size_t gwc_f32_nelems_ = 0;

  int feat_B_ = 1, feat_C_ = 0, feat_H_ = 0, feat_W_ = 0;

  // ---------------------------------------------------------------------------
  void loadEngine(const std::string& path,
                  nvinfer1::IRuntime*& runtime,
                  nvinfer1::ICudaEngine*& engine,
                  nvinfer1::IExecutionContext*& ctx) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open engine: " + path);
    f.seekg(0, std::ios::end);
    std::vector<char> data(f.tellg());
    f.seekg(0);
    f.read(data.data(), data.size());
    runtime = nvinfer1::createInferRuntime(logger_);
    if (!runtime) throw std::runtime_error("createInferRuntime failed");
    engine = runtime->deserializeCudaEngine(data.data(), data.size());
    if (!engine) throw std::runtime_error("deserializeCudaEngine failed: " + path);
    ctx = engine->createExecutionContext();
    if (!ctx) throw std::runtime_error("createExecutionContext failed: " + path);
  }

  void allocateTensor(const std::string& name, nvinfer1::ICudaEngine* engine) {
    if (buf_.count(name)) return;  // shared between feature & post engines
    nvinfer1::Dims     d  = engine->getTensorShape(name.c_str());
    nvinfer1::DataType dt = engine->getTensorDataType(name.c_str());
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, getVolume(d) * getElementSize(dt)) != cudaSuccess)
      throw std::runtime_error("cudaMalloc failed for: " + name);
    buf_[name]   = ptr;
    dims_[name]  = d;
    dtypes_[name] = dt;
  }

  void allocateBuffers() {
    for (int i = 0; i < feature_engine_->getNbIOTensors(); ++i)
      allocateTensor(feature_engine_->getIOTensorName(i), feature_engine_);
    for (int i = 0; i < post_engine_->getNbIOTensors(); ++i)
      allocateTensor(post_engine_->getIOTensorName(i), post_engine_);

    auto it = dims_.find("features_left_04");
    if (it == dims_.end())
      throw std::runtime_error("features_left_04 not found in engines");
    feat_B_ = it->second.d[0]; feat_C_ = it->second.d[1];
    feat_H_ = it->second.d[2]; feat_W_ = it->second.d[3];

    int D = params_.max_disp / 4;
    gwc_f32_nelems_ = (size_t)feat_B_ * params_.cv_group * D * feat_H_ * feat_W_;
    if (cudaMalloc(&gwc_f32_buf_, gwc_f32_nelems_ * sizeof(float)) != cudaSuccess)
      throw std::runtime_error("cudaMalloc failed for gwc_f32");
  }

  void setAddresses(nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* ctx) {
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
      const char* name = engine->getIOTensorName(i);
      ctx->setTensorAddress(name, buf_.at(name));
    }
  }

  void uploadImage(const cv::Mat& bgr, const std::string& name) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (rgb.size() != params_.target_size)
      cv::resize(rgb, rgb, params_.target_size, 0, 0, cv::INTER_LINEAR);

    int H = rgb.rows, W = rgb.cols;
    cv::Mat f32;
    rgb.convertTo(f32, CV_32F);
    std::vector<cv::Mat> chans(3);
    cv::split(f32, chans);
    cv::Mat chw(3 * H, W, CV_32F);
    for (int c = 0; c < 3; ++c)
      chans[c].copyTo(chw(cv::Rect(0, c * H, W, H)));

    if (dtypes_.at(name) != nvinfer1::DataType::kFLOAT)
      throw std::runtime_error("Expected float32 engine input for " + name);
    cudaMemcpyAsync(buf_.at(name), chw.data,
                    (size_t)3 * H * W * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
  }

  void buildGwcVolume() {
    const void* ref_gpu = buf_.at("features_left_04");
    const void* tar_gpu = buf_.at("features_right_04");
    auto feat_dt = dtypes_.at("features_left_04");
    int D = params_.max_disp / 4;

    long long total = (long long)feat_B_ * params_.cv_group * D * feat_H_ * feat_W_;
    const int block = 256;
    int grid = (int)((total + block - 1) / block);

    // Launch GWC kernel (accumulates to float32 intermediate buffer)
    auto launch = [&](auto dummy_type, bool norm) {
      using T = decltype(dummy_type);
      if (norm)
        gwc_volume_kernel<T, true> <<<grid, block, 0, stream_>>>(
            reinterpret_cast<const T*>(ref_gpu),
            reinterpret_cast<const T*>(tar_gpu),
            reinterpret_cast<float*>(gwc_f32_buf_),
            feat_B_, feat_C_, feat_H_, feat_W_, params_.cv_group, D);
      else
        gwc_volume_kernel<T, false><<<grid, block, 0, stream_>>>(
            reinterpret_cast<const T*>(ref_gpu),
            reinterpret_cast<const T*>(tar_gpu),
            reinterpret_cast<float*>(gwc_f32_buf_),
            feat_B_, feat_C_, feat_H_, feat_W_, params_.cv_group, D);
    };

    if (feat_dt == nvinfer1::DataType::kHALF)
      launch(__half{}, params_.normalize_gwc);
    else if (feat_dt == nvinfer1::DataType::kFLOAT)
      launch(float{}, params_.normalize_gwc);
    else
      throw std::runtime_error("Unsupported feature dtype for GWC kernel");

    // Copy to post engine's gwc_volume buffer, casting to its expected dtype
    void* gwc_dst = buf_.at("gwc_volume");
    auto  gwc_dt  = dtypes_.at("gwc_volume");

    if (gwc_dt == nvinfer1::DataType::kFLOAT) {
      cudaMemcpyAsync(gwc_dst, gwc_f32_buf_,
                      gwc_f32_nelems_ * sizeof(float),
                      cudaMemcpyDeviceToDevice, stream_);
    } else if (gwc_dt == nvinfer1::DataType::kHALF) {
      int n = (int)gwc_f32_nelems_;
      cast_f32_to_f16<<<(n + block - 1) / block, block, 0, stream_>>>(
          reinterpret_cast<const float*>(gwc_f32_buf_),
          reinterpret_cast<__half*>(gwc_dst), n);
    } else {
      throw std::runtime_error("Unsupported gwc_volume dtype in post engine");
    }
  }

  cv::Mat downloadDisparity() {
    const auto& d = dims_.at("disp");
    int H = d.d[d.nbDims - 2];
    int W = d.d[d.nbDims - 1];
    auto dt = dtypes_.at("disp");
    size_t nelems = (size_t)H * W;

    cv::Mat disp(H, W, CV_32FC1);
    if (dt == nvinfer1::DataType::kFLOAT) {
      cudaMemcpyAsync(disp.data, buf_.at("disp"),
                      nelems * sizeof(float), cudaMemcpyDeviceToHost, stream_);
      cudaStreamSynchronize(stream_);
    } else if (dt == nvinfer1::DataType::kHALF) {
      std::vector<uint16_t> tmp(nelems);
      cudaMemcpy(tmp.data(), buf_.at("disp"),
                 nelems * sizeof(uint16_t), cudaMemcpyDeviceToHost);
      float* dst = reinterpret_cast<float*>(disp.data);
      for (size_t i = 0; i < nelems; ++i)
        dst[i] = __half2float(reinterpret_cast<const __half*>(tmp.data())[i]);
    } else {
      throw std::runtime_error("Unsupported disp dtype");
    }

    cv::threshold(disp, disp, 0.0, 0.0, cv::THRESH_TOZERO);
    return disp;
  }
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
FastFoundationStereoDepth::FastFoundationStereoDepth(const Params& params)
    : params_(params) {
  if (params_.feature_engine_path.empty() || params_.post_engine_path.empty())
    throw std::invalid_argument("Engine paths cannot be empty");
  impl_ = std::make_unique<Impl>(params_);
}

FastFoundationStereoDepth::~FastFoundationStereoDepth() = default;

void FastFoundationStereoDepth::compute(const cv::Mat& left,
                                        const cv::Mat& right,
                                        cv::Mat& disparity) {
  if (left.empty() || right.empty())
    throw std::invalid_argument("Input images are empty");
  if (left.size() != right.size())
    throw std::invalid_argument("Left and right images must have the same size");

  cv::Size orig = left.size();
  cv::Mat disp_model = impl_->run(left, right);

  if (disp_model.size() != orig) {
    float sx = static_cast<float>(orig.width) / params_.target_size.width;
    cv::resize(disp_model, disp_model, orig, 0, 0, cv::INTER_LINEAR);
    disp_model *= sx;
  }
  disparity = disp_model;
}

void FastFoundationStereoDepth::warmup(const cv::Size& image_size) {
  cv::Mat dummy(image_size, CV_8UC3, cv::Scalar(128, 128, 128));
  cv::Mat out;
  for (int i = 0; i < params_.warmup_iterations; ++i) {
    try { compute(dummy, dummy, out); } catch (...) {}
  }
}

}  // namespace xfeat

#endif  // HAVE_TENSORRT
