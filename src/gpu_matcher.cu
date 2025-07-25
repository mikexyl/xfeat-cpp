// Gpu Inner Product + Motion‑aware Matching (row‑major, CUDA + cuBLAS)
// ------------------------------------------------------------------
//  - Computes inner‑product matrix on GPU (cuBLAS)
//  - Adds anisotropic Mahalanobis penalty aligned with predicted flow
//  - Finds best match per descriptor entirely on GPU
//
//  author: ChatGPT demo (v2 – finished main())
// ------------------------------------------------------------------
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <cusolverDn.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

#include "xfeat-cpp/gpu_matcher.h"

namespace xfeat {

struct Homography {
  float h[9];
};

// ------------------------- macros -------------------------
#define CUDA_CHECK(expr)                                                                                    \
  do {                                                                                                      \
    cudaError_t _err = (expr);                                                                              \
    if (_err != cudaSuccess) {                                                                              \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, cudaGetErrorString(_err)); \
      std::exit(EXIT_FAILURE);                                                                              \
    }                                                                                                       \
  } while (0)

#define CUBLAS_CHECK(expr)                           \
  do {                                               \
    cublasStatus_t _st = (expr);                     \
    if (_st != CUBLAS_STATUS_SUCCESS) {              \
      fprintf(stderr,                                \
              "cuBLAS error %s at %s:%d: %s (%d)\n", \
              #expr,                                 \
              __FILE__,                              \
              __LINE__,                              \
              cublasGetStatusString(_st),            \
              static_cast<int>(_st));                \
      std::exit(EXIT_FAILURE);                       \
    }                                                \
  } while (0)

#define CUSOLVER_CHECK(expr)                                                                                        \
  do {                                                                                                              \
    cusolverStatus_t _st = (expr);                                                                                  \
    if (_st != CUSOLVER_STATUS_SUCCESS) {                                                                           \
      fprintf(stderr, "cuSOLVER error %s at %s:%d: status=%d\n", #expr, __FILE__, __LINE__, static_cast<int>(_st)); \
      std::exit(EXIT_FAILURE);                                                                                      \
    }                                                                                                               \
  } while (0)

#define CURAND_CHECK(expr)                           \
  do {                                               \
    curandStatus_t _st = (expr);                     \
    if (_st != CURAND_STATUS_SUCCESS) {              \
      fprintf(stderr,                                \
              "cuRAND error %s at %s:%d: %s (%d)\n", \
              #expr,                                 \
              __FILE__,                              \
              __LINE__,                              \
              curandGetStatusString(_st),            \
              static_cast<int>(_st));                \
      std::exit(EXIT_FAILURE);                       \
    }                                                \
  } while (0)

// -------------------------------- fuse: descriptor – λ·mahalanobis
__global__ void fusePenaltyKernel(const float* __restrict__ d_in,
                                  float* __restrict__ d_out,
                                  const float2* __restrict__ d_pred,
                                  const float2* __restrict__ d_kp2,
                                  const float4* __restrict__ d_param,
                                  float lambda,
                                  int N1,
                                  int N2) {
  int i = blockIdx.y;                             // row (query)
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // column (db)
  if (i >= N1 || j >= N2) return;

  float2 pred = d_pred[i];
  float2 kp = d_kp2[j];
  float dx = kp.x - pred.x;
  float dy = kp.y - pred.y;

  float4 p = d_param[i];
  float u = p.x * dx + p.y * dy;
  float v = -p.y * dx + p.x * dy;
  float maha = p.z * u * u + p.w * v * v;

  int idx = i * N2 + j;                    // row major
  d_out[idx] = d_in[idx] - lambda * maha;  // higher is better
}

__global__ void argmaxKernelRows(const float* scores,  // column‑major (N1 × N2)
                                 int* bestIdx,         // N1
                                 float* bestScore,     // N1
                                 int N1,
                                 int N2) {
  extern __shared__ unsigned char sm[];
  float* s_val = (float*)sm;
  int* s_idx = (int*)(s_val + blockDim.x);

  const int i = blockIdx.x;  // one row per block
  if (i >= N1) return;

  float best_v = -FLT_MAX;
  int best_j = -1;

  // ---- scan this row (strided in column‑major) --------------------------
  for (int j = threadIdx.x; j < N2; j += blockDim.x) {
    float v = scores[i + j * N1];  // <<< column‑major access
    if (v > best_v) {
      best_v = v;
      best_j = j;
    }
  }

  s_val[threadIdx.x] = best_v;
  s_idx[threadIdx.x] = best_j;
  __syncthreads();

  // parallel reduction inside the block
  for (int stride = blockDim.x >> 1; stride; stride >>= 1) {
    if (threadIdx.x < stride && s_val[threadIdx.x + stride] > s_val[threadIdx.x]) {
      s_val[threadIdx.x] = s_val[threadIdx.x + stride];
      s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    bestScore[i] = s_val[0];
    bestIdx[i] = s_idx[0];
  }
}

void CuMatcher::init(int maxN1, int maxN2, int D) {
  max_N1 = maxN1;
  max_N2 = maxN2;
  max_D = D;

  const size_t bytesScores = static_cast<size_t>(max_N1) * max_N2 * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_scores, bytesScores));
  CUDA_CHECK(cudaMalloc(&d_pred, max_N1 * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&d_kp2, max_N2 * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&d_param, max_N1 * sizeof(float4)));

  // --- row‑wise arg‑max (already used by computeMatches) ------------------
  CUDA_CHECK(cudaMalloc(&d_bestIdxRow, max_N1 * sizeof(int)));      // NEW
  CUDA_CHECK(cudaMalloc(&d_bestScoreRow, max_N1 * sizeof(float)));  // NEW

  // --- column‑wise arg‑max (needed by match_mkpts) ------------------------
  CUDA_CHECK(cudaMalloc(&d_bestIdxCol, max_N2 * sizeof(int)));      // NEW
  CUDA_CHECK(cudaMalloc(&d_bestScoreCol, max_N2 * sizeof(float)));  // NEW

  // --- descriptor matrices ------------------------------------------------
  CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(max_N1) * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, static_cast<size_t>(max_N2) * D * sizeof(float)));

  CUBLAS_CHECK(cublasCreate(&handle));
}

void CuMatcher::destroy() {
  auto freeIf = [&](void* p) {
    if (p) cudaFree(p);
  };
  freeIf(d_scores);
  freeIf(d_pred);
  freeIf(d_kp2);
  freeIf(d_param);
  freeIf(d_bestIdx);
  freeIf(d_bestScore);
  freeIf(d_A);
  freeIf(d_B);
  if (handle) cublasDestroy(handle);
}
__global__ void argmaxKernelCols(const float* scores,  // column‑major N1×N2
                                 int* bestIdx,         // N2
                                 float* bestScore,     // N2
                                 int N1,
                                 int N2) {
  extern __shared__ unsigned char sm[];
  float* s_val = (float*)sm;
  int* s_idx = (int*)(s_val + blockDim.x);

  const int j = blockIdx.x;  // one column per block
  if (j >= N2) return;

  // stride over rows i = threadIdx.x + k*blockDim.x
  int best_i = -1;
  float best_v = -FLT_MAX;
  for (int i = threadIdx.x; i < N1; i += blockDim.x) {
    const float v = scores[i + j * N1];  // column‑major access
    if (v > best_v) {
      best_v = v;
      best_i = i;
    }
  }

  s_val[threadIdx.x] = best_v;
  s_idx[threadIdx.x] = best_i;
  __syncthreads();

  // parallel reduction to find global max inside block
  for (int stride = blockDim.x >> 1; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      if (s_val[threadIdx.x + stride] > s_val[threadIdx.x]) {
        s_val[threadIdx.x] = s_val[threadIdx.x + stride];
        s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    bestScore[j] = s_val[0];
    bestIdx[j] = s_idx[0];
  }
}

std::vector<std::vector<int>> CuMatcher::match_mkpts(const cv::Mat& desc1,
                                                     const cv::Mat& desc2,
                                                     float min_cossim)  // ≥0 ⇒ apply threshold
{
  const int N1 = desc1.rows, N2 = desc2.rows, D = desc1.cols;
  CV_Assert(desc1.type() == CV_32F && desc2.type() == CV_32F && D == desc2.cols);

  // -----------------------------------------------------------------------
  // 1. Upload descriptors
  // -----------------------------------------------------------------------
  CUDA_CHECK(cudaMemcpy(d_A, desc1.ptr<float>(), static_cast<size_t>(N1) * D * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, desc2.ptr<float>(), static_cast<size_t>(N2) * D * sizeof(float), cudaMemcpyHostToDevice));

  // -----------------------------------------------------------------------
  // 2. Cosine‑similarity matrix  (N1 × N2)   scores = Aᵀ·B
  // -----------------------------------------------------------------------
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasSgemm(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, N1, N2, D, &alpha, d_A, D, d_B, D, &beta, d_scores, N1));  // column‑major view

  // -----------------------------------------------------------------------
  // 3. Best‑of‑ROW   (for each i∈[0,N1) find j* and score r_i)
  // -----------------------------------------------------------------------
  const int threads = 256;
  const size_t smem = threads * (sizeof(float) + sizeof(int));

  // --- row‑wise arg‑max (unchanged) -----------------------------------------
  argmaxKernelRows<<<N1, threads, smem>>>(d_scores,
                                          d_bestIdxRow,    // row2col
                                          d_bestScoreRow,  // rowBest
                                          N1,
                                          N2);

  // --- column‑wise arg‑max (needed for colBest only) ------------------------
  argmaxKernelCols<<<N2, threads, smem>>>(d_scores,
                                          d_bestIdxCol,    // col2row  (unused)
                                          d_bestScoreCol,  // colBest
                                          N1,
                                          N2);

  // -------------------------------------------------------------------------
  // copy results back
  // -------------------------------------------------------------------------
  std::vector<int> row2col(N1);
  std::vector<float> rowBest(N1);
  std::vector<float> colBest(N2);

  CUDA_CHECK(cudaMemcpy(row2col.data(), d_bestIdxRow, N1 * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(rowBest.data(), d_bestScoreRow, N1 * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(colBest.data(), d_bestScoreCol, N2 * sizeof(float), cudaMemcpyDeviceToHost));

  // -------------------------------------------------------------------------
  // accept if BOTH rowBest[i] and colBest[j*] clear the threshold
  // -------------------------------------------------------------------------
  std::vector<std::vector<int>> idx(desc1.rows, std::vector<int>());
  for (int i = 0; i < N1; ++i) {
    const int j = row2col[i];
    if (j >= 0 && j < N2 && (min_cossim <= 0.f || (rowBest[i] > min_cossim && colBest[j] > min_cossim))) {
      idx[i].push_back(j);
    }
  }
  return idx;
}

// ---------------------------------------------------------------------------
// neighbourMaskKernel.cu
//   – column‑major similarity matrix  (N1 × N2)
//   – zeroes out scores (sets to -FLT_MAX) when pair is outside search radius
// ---------------------------------------------------------------------------
__global__ void neighbourMaskKernel(float* scores,      // in/out
                                    const float2* kp1,  // N1
                                    const float2* kp2,  // N2
                                    float radius2,      // r²
                                    int N1,
                                    int N2) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;  // column (desc2)
  int i = blockIdx.y;                             // row    (desc1)
  if (i >= N1 || j >= N2) return;

  float2 p1 = kp1[i];
  float2 p2 = kp2[j];
  float dx = p1.x - p2.x;
  float dy = p1.y - p2.y;
  if (dx * dx + dy * dy > radius2) scores[i + j * N1] = -FLT_MAX;  // mask‑out
}

// ---------------------------------------------------------------------------
// CuMatcher::match_mkpts_local
//   – local (radius‑limited) GPU brute‑force matcher
// ---------------------------------------------------------------------------
std::tuple<std::vector<int>, std::vector<int>> CuMatcher::match_mkpts_local(const cv::Mat& desc1,
                                                                            const cv::Mat& desc2,
                                                                            const std::vector<cv::Point2f>& kp1,
                                                                            const std::vector<cv::Point2f>& kp2,
                                                                            float search_radius,  // pixels
                                                                            float min_cossim)  // threshold (≤0 ⇒ off)
{
  const int N1 = desc1.rows, N2 = desc2.rows, D = desc1.cols;
  CV_Assert(desc1.type() == CV_32F && desc2.type() == CV_32F && D == desc2.cols && kp1.size() == (size_t)N1 &&
            kp2.size() == (size_t)N2);

  // --- 1. upload descriptors --------------------------------------------
  CUDA_CHECK(cudaMemcpy(d_A, desc1.ptr<float>(), static_cast<size_t>(N1) * D * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, desc2.ptr<float>(), static_cast<size_t>(N2) * D * sizeof(float), cudaMemcpyHostToDevice));

  // --- 2. upload key‑point coordinates -----------------------------------
  std::vector<float2> h_kp1(N1), h_kp2(N2);
  for (int i = 0; i < N1; ++i) h_kp1[i] = make_float2(kp1[i].x, kp1[i].y);
  for (int j = 0; j < N2; ++j) h_kp2[j] = make_float2(kp2[j].x, kp2[j].y);

  CUDA_CHECK(cudaMemcpy(d_pred, h_kp1.data(), N1 * sizeof(float2),
                        cudaMemcpyHostToDevice));  // reuse d_pred
  CUDA_CHECK(cudaMemcpy(d_kp2, h_kp2.data(), N2 * sizeof(float2), cudaMemcpyHostToDevice));

  // --- 3. SGEMM: cosine‑similarity matrix  (column‑major N1×N2) ----------
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N1, N2, D, &alpha, d_A, D, d_B, D, &beta, d_scores, N1));

  // --- 4. radius mask ----------------------------------------------------
  dim3 blk(256);
  dim3 grd((N2 + blk.x - 1) / blk.x, N1);  // (columns, rows)
  const float radius2 = search_radius * search_radius;
  neighbourMaskKernel<<<grd, blk>>>(d_scores,
                                    d_pred,  // kp1
                                    d_kp2,   // kp2
                                    radius2,
                                    N1,
                                    N2);
  CUDA_CHECK(cudaGetLastError());

  // --- 5. per‑row arg‑max (column‑major aware) ---------------------------
  const int threads = 256;
  const size_t smem = threads * (sizeof(float) + sizeof(int));
  argmaxKernelRows<<<N1, threads, smem>>>(d_scores,
                                          d_bestIdxRow,    // j*
                                          d_bestScoreRow,  // score*
                                          N1,
                                          N2);
  CUDA_CHECK(cudaGetLastError());

  // --- 6. copy back + accept pairs ---------------------------------------
  std::vector<int> row2col(N1);
  std::vector<float> rowBest(N1);
  CUDA_CHECK(cudaMemcpy(row2col.data(), d_bestIdxRow, N1 * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(rowBest.data(), d_bestScoreRow, N1 * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<int> idx0, idx1;
  idx0.reserve(N1);
  idx1.reserve(N1);

  for (int i = 0; i < N1; ++i) {
    int j = row2col[i];
    float score = rowBest[i];

    if (j >= 0 && j < N2 && score > -FLT_MAX / 2 &&  // not masked out
        (min_cossim <= 0.f || score > min_cossim)) {
      idx0.push_back(i);
      idx1.push_back(j);
    }
  }
  return {std::move(idx0), std::move(idx1)};
}

struct NonNegToMask {
  __host__ __device__ unsigned char operator()(int j) const { return (j >= 0) ? 1u : 0u; }
};

struct IsNonZeroMask {
  __host__ __device__ bool operator()(unsigned char m) const { return m != 0u; }
};

struct MarkInlier {
  float thr2;
  __host__ __device__ explicit MarkInlier(float t) : thr2(t) {}
  __host__ __device__ unsigned char operator()(float e2) const { return (e2 <= thr2) ? 1u : 0u; }
};

__device__ __forceinline__ void warp_pt(const Homography& H, float x, float y, float& u, float& v) {
  float X = H.h[0] * x + H.h[1] * y + H.h[2];
  float Y = H.h[3] * x + H.h[4] * y + H.h[5];
  float W = H.h[6] * x + H.h[7] * y + H.h[8];
  float invW = 1.f / W;
  u = X * invW;
  v = Y * invW;
}

__device__ __forceinline__ void warp_point(const Homography& H, float x, float y, float& u, float& v) {
  float X = H.h[0] * x + H.h[1] * y + H.h[2];
  float Y = H.h[3] * x + H.h[4] * y + H.h[5];
  float W = H.h[6] * x + H.h[7] * y + H.h[8];
  float invW = 1.f / W;
  u = X * invW;
  v = Y * invW;
}

__device__ __forceinline__ void jacobianAt(const Homography& H, float x, float y, float J[4]) {
  // u = (ax + by + c) / (gx + hy + 1)
  // v = (dx + ey + f) / (gx + hy + 1)
  const float a = H.h[0], b = H.h[1], c = H.h[2];
  const float d = H.h[3], e = H.h[4], f = H.h[5];
  const float g = H.h[6], h = H.h[7], i = H.h[8];

  const float den = g * x + h * y + i;
  const float den2 = den * den;

  const float num_u = a * x + b * y + c;
  const float num_v = d * x + e * y + f;

  const float du_dx = (a * den - num_u * g) / den2;
  const float du_dy = (b * den - num_u * h) / den2;
  const float dv_dx = (d * den - num_v * g) / den2;
  const float dv_dy = (e * den - num_v * h) / den2;

  J[0] = du_dx;
  J[1] = du_dy;
  J[2] = dv_dx;
  J[3] = dv_dy;
}

__device__ __forceinline__ float det2(const float J[4]) { return J[0] * J[3] - J[1] * J[2]; }

// Return sigma_max / sigma_min of J using J^T J eigenvalues (2x2 closed form)
__device__ __forceinline__ float anisotropy(const float J[4]) {
  // JTJ = [a b; b c]
  float a = J[0] * J[0] + J[2] * J[2];
  float b = J[0] * J[1] + J[2] * J[3];
  float c = J[1] * J[1] + J[3] * J[3];

  float tr = a + c;
  float det = a * c - b * b;
  det = fmaxf(det, 1e-20f);
  float disc = fmaxf(tr * tr - 4.f * det, 0.f);
  float lmax = 0.5f * (tr + sqrtf(disc));
  float lmin = 0.5f * (tr - sqrtf(disc));
  lmin = fmaxf(lmin, 1e-20f);
  return sqrtf(lmax / lmin);
}

__device__ __forceinline__ float det3x3(const Homography& H) {
  const float* h = H.h;
  return h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) + h[2] * (h[3] * h[7] - h[4] * h[6]);
}

__device__ __forceinline__ void adjugate3x3(const Homography& H, float A[9]) {
  const float* h = H.h;
  A[0] = (h[4] * h[8] - h[5] * h[7]);
  A[1] = -(h[1] * h[8] - h[2] * h[7]);
  A[2] = (h[1] * h[5] - h[2] * h[4]);
  A[3] = -(h[3] * h[8] - h[5] * h[6]);
  A[4] = (h[0] * h[8] - h[2] * h[6]);
  A[5] = -(h[0] * h[5] - h[2] * h[3]);
  A[6] = (h[3] * h[7] - h[4] * h[6]);
  A[7] = -(h[0] * h[7] - h[1] * h[6]);
  A[8] = (h[0] * h[4] - h[1] * h[3]);
}

__device__ __forceinline__ float frob_norm3x3(const float* m) {
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 9; ++k) s += m[k] * m[k];
  return sqrtf(s);
}

__device__ __forceinline__ float frob_diff_normed33(const Homography& H, const Homography& Hp) {
  float s = (fabsf(H.h[8]) > 1e-12f) ? H.h[8] : 1.f;
  float sp = (fabsf(Hp.h[8]) > 1e-12f) ? Hp.h[8] : 1.f;
  float diff2 = 0.f;
#pragma unroll
  for (int k = 0; k < 9; ++k) {
    float a = H.h[k] / s;
    float b = Hp.h[k] / sp;
    float d = a - b;
    diff2 += d * d;
  }
  return sqrtf(diff2);
}

__device__ __forceinline__ float mean_reproj_gap(const Homography& H,
                                                 const Homography& Hp,
                                                 float imgW,
                                                 float imgH,
                                                 int S /*samples/side*/) {
  float acc = 0.f;
  int cnt = 0;
  for (int sy = 0; sy < S; ++sy)
    for (int sx = 0; sx < S; ++sx) {
      float x = (imgW - 1) * (sx + 0.5f) / S;
      float y = (imgH - 1) * (sy + 0.5f) / S;
      float u1, v1, u2, v2;
      warp_pt(H, x, y, u1, v1);
      warp_pt(Hp, x, y, u2, v2);
      float dx = u1 - u2, dy = v1 - v2;
      acc += sqrtf(dx * dx + dy * dy);
      ++cnt;
    }
  return acc / max(cnt, 1);
}

static inline float2 toFloat2(const cv::Point2f& p) { return float2{p.x, p.y}; }

struct IsNonZero {
  __host__ __device__ bool operator()(char x) const { return x != 0; }
};

// Build tentative (i,j) list by mutual score + threshold
__global__ void buildTentativePairs(const int* __restrict__ bestIdxRow,
                                    const float* __restrict__ bestScoreRow,
                                    const float* __restrict__ bestScoreCol,
                                    int N1,
                                    int N2,
                                    float min_cossim,
                                    int* pair_i,
                                    int* pair_j,
                                    int* M_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N1) return;

  int j = bestIdxRow[i];
  if (j >= 0 && j < N2) {
    if (min_cossim <= 0.f || (bestScoreRow[i] > min_cossim && bestScoreCol[j] > min_cossim)) {
      int idx = atomicAdd(M_out, 1);
      pair_i[idx] = i;
      pair_j[idx] = j;
    }
  }
}

// ---------------------------------------------------------------
// cuRAND: init RNG states
// ---------------------------------------------------------------
__global__ void initCurand(curandStatePhilox4_32_10_t* states, unsigned long long seed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, id, 0, &states[id]);
}

// Sample 4 unique indices per hypothesis
__global__ void sampleMinimalSets(curandStatePhilox4_32_10_t* states, int* samples, int iters, int M) {
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  if (hid >= iters) return;

  auto st = states[hid];
  int sel[4];
  while (true) {
    for (int k = 0; k < 4; ++k) sel[k] = curand(&st) % M;
    bool ok = true;
#pragma unroll
    for (int a = 0; a < 4; ++a)
      for (int b = a + 1; b < 4; ++b)
        if (sel[a] == sel[b]) ok = false;
    if (ok) break;
  }
  for (int k = 0; k < 4; ++k) samples[hid * 4 + k] = sel[k];
  states[hid] = st;
}

// Build 8x8 linear systems Ah=b for each hypothesis (h22=1 eliminated)
__global__ void buildLinearSystems8x8(const float2* __restrict__ p1,
                                      const float2* __restrict__ p2,
                                      const int* __restrict__ pair_i,
                                      const int* __restrict__ pair_j,
                                      const int* __restrict__ samples,
                                      float* __restrict__ A,  // [iters][64]
                                      float* __restrict__ b,  // [iters][8]
                                      int iters) {
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  if (hid >= iters) return;

  float x[4], y[4], X[4], Y[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    int m = samples[hid * 4 + k];
    float2 a = p1[m];
    float2 b2 = p2[m];
    x[k] = a.x;
    y[k] = a.y;
    X[k] = b2.x;
    Y[k] = b2.y;
  }

  float* A_h = A + hid * 64;
  float* b_h = b + hid * 8;

  for (int k = 0; k < 4; ++k) {
    int r0 = 2 * k, r1 = r0 + 1;
    float row0[8] = {x[k], y[k], 1, 0, 0, 0, -x[k] * X[k], -y[k] * X[k]};
    float row1[8] = {0, 0, 0, x[k], y[k], 1, -x[k] * Y[k], -y[k] * Y[k]};
    for (int c = 0; c < 8; ++c) {
      A_h[c * 8 + r0] = row0[c];  // column-major 8x8
      A_h[c * 8 + r1] = row1[c];
    }
    b_h[r0] = X[k];
    b_h[r1] = Y[k];
  }
}

// ---------------------------------------------------------------
// Kernel: pack solution h(8) -> 3x3 Homography with h22=1
// ---------------------------------------------------------------
__global__ void packHomographies(const float* __restrict__ h_solutions, Homography* H, int iters) {
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  if (hid >= iters) return;

  const float* h = h_solutions + hid * 8;
  Homography Ho;
  Ho.h[0] = h[0];
  Ho.h[1] = h[1];
  Ho.h[2] = h[2];
  Ho.h[3] = h[3];
  Ho.h[4] = h[4];
  Ho.h[5] = h[5];
  Ho.h[6] = h[6];
  Ho.h[7] = h[7];
  Ho.h[8] = 1.f;
  H[hid] = Ho;
}

// ---------------------------------------------------------------
// Kernel: score each H against all tentative matches
// ---------------------------------------------------------------
__global__ void scoreHypotheses(const Homography* __restrict__ Hs,
                                const unsigned char* __restrict__ valid,
                                const float2* __restrict__ p1,
                                const float2* __restrict__ p2,
                                int M,
                                int iters,
                                float thr2,
                                int* __restrict__ counts) {
  int hid = blockIdx.x;
  if (hid >= iters || !valid[hid]) {
    if (hid < iters && threadIdx.x == 0) counts[hid] = 0;
    return;
  }
  Homography H = Hs[hid];

  extern __shared__ int s_count[];
  if (threadIdx.x == 0) s_count[0] = 0;
  __syncthreads();

  for (int m = threadIdx.x; m < M; m += blockDim.x) {
    float u, v;
    warp_pt(H, p1[m].x, p1[m].y, u, v);
    float dx = u - p2[m].x, dy = v - p2[m].y;
    if (dx * dx + dy * dy <= thr2) atomicAdd(&s_count[0], 1);
  }
  __syncthreads();
  if (threadIdx.x == 0) counts[hid] = s_count[0];
}

using RNGState = curandStatePhilox4_32_10_t;

__global__ void computeSymmetricErrKernel(const Homography H,
                                          const Homography Hinv,
                                          const float2* __restrict__ p1,
                                          const float2* __restrict__ p2,
                                          float* __restrict__ errs,  // [M]
                                          int M) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;

  float u, v, x, y;
  // forward p1 -> p2
  warp_pt(H, p1[i].x, p1[i].y, u, v);
  float dx1 = u - p2[i].x, dy1 = v - p2[i].y;

  // backward p2 -> p1
  warp_pt(Hinv, p2[i].x, p2[i].y, x, y);
  float dx2 = x - p1[i].x, dy2 = y - p1[i].y;

  errs[i] = dx1 * dx1 + dy1 * dy1 + dx2 * dx2 + dy2 * dy2;  // symmetric error
}

__global__ void predictKernel(Homography H,
                              const float2* __restrict__ pts1,  // [N1]
                              float2* __restrict__ yhat,        // [N1]
                              int N1) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N1) return;
  float x = pts1[i].x, y = pts1[i].y;
  float X = H.h[0] * x + H.h[1] * y + H.h[2];
  float Y = H.h[3] * x + H.h[4] * y + H.h[5];
  float W = H.h[6] * x + H.h[7] * y + H.h[8];
  float invW = 1.f / W;
  yhat[i] = make_float2(X * invW, Y * invW);
}

__global__
void validateHomographiesKernel(const Homography* __restrict__ Hs,
                                int iters,
                                float imgW,
                                float imgH,
                                float max_anisotropy,
                                float max_scale,
                                float min_den,
                                float cond_frob_max,
                                Homography Hpred,
                                float max_frob_gap,
                                float max_reproj_gap,
                                int samples_per_side,
                                unsigned char* __restrict__ valid) {
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  if (hid >= iters) return;

  const Homography H = Hs[hid];

  // ----- (0) Global det / condition estimate (cheap) ---------------------
  float detH = det3x3(H);
  if (fabsf(detH) < 1e-8f) {
    valid[hid] = 0;
    return;
  }

  // --- NEW: distance to prior (fast) ------------------------------------
  float frob_gap = frob_diff_normed33(H, Hpred);
  if (!isfinite(frob_gap) || frob_gap > max_frob_gap) {
    valid[hid] = 0;
    return;
  }

  // --- NEW: distance to prior (geometric) --------------------------------
  float reproj_gap = mean_reproj_gap(H, Hpred, imgW, imgH, 3);
  if (!isfinite(reproj_gap) || reproj_gap > max_reproj_gap) {
    valid[hid] = 0;
    return;
  }

  // cond_est ~= ||H||_F * ||H^{-1}||_F
  float adj[9];
  adjugate3x3(H, adj);
  float invH[9];
  float inv_scale = 1.f / detH;
#pragma unroll
  for (int k = 0; k < 9; ++k) invH[k] = adj[k] * inv_scale;

  float cond_est = frob_norm3x3(H.h) * frob_norm3x3(invH);
  if (cond_est > cond_frob_max || !isfinite(cond_est)) {
    valid[hid] = 0;
    return;
  }

  // ----- (1) Quick triangle orientation at center ------------------------
  {
    float cx = imgW * 0.5f, cy = imgH * 0.5f;
    float u0, v0, u1, v1, u2, v2;
    warp_pt(H, cx, cy, u0, v0);
    warp_pt(H, cx + 1, cy, u1, v1);
    warp_pt(H, cx, cy + 1, u2, v2);
    float cross_z = (u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0);
    if (!isfinite(cross_z) || cross_z <= 0.f) {
      valid[hid] = 0;
      return;
    }
  }

  // ----- (2) Sample grid: denominator, det(J), anisotropy, scale ---------
  const int S = samples_per_side;
  float minDen = 1e30f;
  for (int sy = 0; sy < S; ++sy) {
    for (int sx = 0; sx < S; ++sx) {
      float x = (imgW - 1) * (sx + 0.5f) / S;
      float y = (imgH - 1) * (sy + 0.5f) / S;

      // denominator
      float den = H.h[6] * x + H.h[7] * y + H.h[8];
      minDen = fminf(minDen, fabsf(den));
      if (!isfinite(den)) {
        valid[hid] = 0;
        return;
      }

      // Jacobian tests
      float J[4];
      jacobianAt(H, x, y, J);

      // orientation preserve
      float detJ = det2(J);
      if (detJ <= 0.f || !isfinite(detJ)) {
        valid[hid] = 0;
        return;
      }

      // anisotropy
      float a = anisotropy(J);
      if (!isfinite(a) || a > max_anisotropy) {
        valid[hid] = 0;
        return;
      }

      // absolute scale clamp (sigma_max)
      // reuse JTJ lambda_max from anisotropy() derivation
      float xx = J[0] * J[0] + J[2] * J[2];
      float xy = J[0] * J[1] + J[2] * J[3];
      float yy = J[1] * J[1] + J[3] * J[3];
      float tr = xx + yy;
      float det = xx * yy - xy * xy;
      det = fmaxf(det, 1e-20f);
      float disc = fmaxf(tr * tr - 4.f * det, 0.f);
      float lmax = 0.5f * (tr + sqrtf(disc));
      float sigma_max = sqrtf(lmax);
      if (!isfinite(sigma_max) || sigma_max > max_scale) {
        valid[hid] = 0;
        return;
      }
    }
  }

  if (minDen < min_den) {
    valid[hid] = 0;
    return;
  }

  valid[hid] = 1;
}

// Put next to your Homography definition (host-side helper)
inline Homography toHomography(const cv::Mat& Hcv, bool renormalize = true) {
  CV_Assert(Hcv.rows == 3 && Hcv.cols == 3);
  Homography H{};

  if (Hcv.type() == CV_32F) {
    const float* m = Hcv.ptr<float>(0);
    for (int k = 0; k < 9; ++k) H.h[k] = m[k];
  } else if (Hcv.type() == CV_64F) {
    const double* m = Hcv.ptr<double>(0);
    for (int k = 0; k < 9; ++k) H.h[k] = static_cast<float>(m[k]);
  } else {
    CV_Error(cv::Error::StsUnsupportedFormat, "Homography must be CV_32F or CV_64F");
  }

  if (renormalize) {
    const float s = (std::abs(H.h[8]) > 1e-12f) ? H.h[8] : 1.f;
    for (int k = 0; k < 9; ++k) H.h[k] /= s;
    H.h[8] = 1.f;
  }
  return H;
}

// scores: [N1 x N2] (column-major), pts2: [N2], y_hat: [N1]
__global__ void localRematchKernel(const float* __restrict__ scores,
                                   const float2* __restrict__ pts2,
                                   const float2* __restrict__ y_hat,
                                   int N1,
                                   int N2,
                                   float radius2,
                                   float lambda,
                                   int* __restrict__ new_row2col,
                                   float* __restrict__ new_rowScore) {
  int i = blockIdx.x;
  if (i >= N1) return;

  const float2 yh = y_hat[i];

  float best = -FLT_MAX;
  int bestj = -1;

  for (int j = threadIdx.x; j < N2; j += blockDim.x) {
    float dx = pts2[j].x - yh.x;
    float dy = pts2[j].y - yh.y;
    float d2 = dx * dx + dy * dy;
    if (d2 <= radius2) {
      float s = scores[i + j * N1] - lambda * d2;
      if (s > best) {
        best = s;
        bestj = j;
      }
    }
  }

  // block reduce
  extern __shared__ unsigned char sm[];
  float* s_val = (float*)sm;
  int* s_idx = (int*)(s_val + blockDim.x);
  s_val[threadIdx.x] = best;
  s_idx[threadIdx.x] = bestj;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (threadIdx.x < off) {
      if (s_val[threadIdx.x + off] > s_val[threadIdx.x]) {
        s_val[threadIdx.x] = s_val[threadIdx.x + off];
        s_idx[threadIdx.x] = s_idx[threadIdx.x + off];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    new_row2col[i] = s_idx[0];
    new_rowScore[i] = s_val[0];
  }
}

CuMatcher::GpuMatchResult CuMatcher::match_mkpts_gpuRansac(const cv::Mat& desc1,
                                                           const cv::Mat& desc2,
                                                           const std::vector<cv::Point2f>& kpts1,
                                                           const std::vector<cv::Point2f>& kpts2,
                                                           cv::Size img_size,
                                                           cv::Mat predicted_H,
                                                           float min_cossim,
                                                           float ransac_thr_px,
                                                           int ransac_iters,
                                                           int min_inliers,
                                                           float local_radius_px,  // e.g. 8.f
                                                           float geo_lambda)       // e.g. 1e-3f
{
  CuMatcher::GpuMatchResult res;
  int img_width = img_size.width, img_height = img_size.height;

  const int N1 = desc1.rows, N2 = desc2.rows, D = desc1.cols;
  CV_Assert(desc1.type() == CV_32F && desc2.type() == CV_32F && D == desc2.cols);

  // -------------------- 0) Push descriptors & keypoints --------------------
  CUDA_CHECK(cudaMemcpy(d_A, desc1.ptr<float>(), size_t(N1) * D * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, desc2.ptr<float>(), size_t(N2) * D * sizeof(float), cudaMemcpyHostToDevice));

  thrust::device_vector<float2> d_pts1(N1), d_pts2(N2);
  {
    std::vector<float2> h1(N1), h2(N2);
    for (int i = 0; i < N1; ++i) h1[i] = toFloat2(kpts1[i]);
    for (int j = 0; j < N2; ++j) h2[j] = toFloat2(kpts2[j]);
    CUDA_CHECK(
        cudaMemcpy(thrust::raw_pointer_cast(d_pts1.data()), h1.data(), N1 * sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(thrust::raw_pointer_cast(d_pts2.data()), h2.data(), N2 * sizeof(float2), cudaMemcpyHostToDevice));
  }

  // -------------------- 1) Cosine sim matrix: scores = A^T B ---------------
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasSgemm(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, N1, N2, D, &alpha, d_A, D, d_B, D, &beta, d_scores, N1));  // column-major N1 x
                                                                                                   // N2

  // -------------------- 2) Argmax rows & cols ------------------------------
  {
    const int threads = 256;
    const size_t smem = threads * (sizeof(float) + sizeof(int));
    argmaxKernelRows<<<N1, threads, smem>>>(d_scores, d_bestIdxRow, d_bestScoreRow, N1, N2);
    CUDA_CHECK(cudaPeekAtLastError());
    argmaxKernelCols<<<N2, threads, smem>>>(d_scores, d_bestIdxCol, d_bestScoreCol, N1, N2);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // -------------------- 3) Tentative list (GPU) ----------------------------
  thrust::device_vector<int> d_pair_i(N1), d_pair_j(N1);
  thrust::device_vector<int> d_M(1, 0);
  {
    const int block = 256;
    const int grid = (N1 + block - 1) / block;
    buildTentativePairs<<<grid, block>>>(d_bestIdxRow,
                                         d_bestScoreRow,
                                         d_bestScoreCol,
                                         N1,
                                         N2,
                                         min_cossim,
                                         thrust::raw_pointer_cast(d_pair_i.data()),
                                         thrust::raw_pointer_cast(d_pair_j.data()),
                                         thrust::raw_pointer_cast(d_M.data()));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  int M = 0;
  CUDA_CHECK(cudaMemcpy(&M, thrust::raw_pointer_cast(d_M.data()), sizeof(int), cudaMemcpyDeviceToHost));
  if (M < min_inliers) {
    res.H = cv::Mat::eye(3, 3, CV_32F);
    return res;
  }

  // Build M correspondence buffers
  thrust::device_vector<float2> d_p1(M), d_p2(M);
  {
    std::vector<int> hi(M), hj(M);
    CUDA_CHECK(
        cudaMemcpy(hi.data(), thrust::raw_pointer_cast(d_pair_i.data()), M * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(hj.data(), thrust::raw_pointer_cast(d_pair_j.data()), M * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<float2> h1(M), h2(M);
    for (int m = 0; m < M; ++m) {
      h1[m] = toFloat2(kpts1[hi[m]]);
      h2[m] = toFloat2(kpts2[hj[m]]);
    }
    CUDA_CHECK(
        cudaMemcpy(thrust::raw_pointer_cast(d_p1.data()), h1.data(), M * sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(thrust::raw_pointer_cast(d_p2.data()), h2.data(), M * sizeof(float2), cudaMemcpyHostToDevice));
  }

  // -------------------- 4) GPU RANSAC on tentative set ---------------------
  Homography H_best;
  {
    // 4.1 sample
    thrust::device_vector<int> d_samples(ransac_iters * 4);
    thrust::device_vector<curandStatePhilox4_32_10_t> d_states(ransac_iters);
    initCurand<<<(ransac_iters + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_states.data()), 123456ULL);
    sampleMinimalSets<<<(ransac_iters + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(d_states.data()), thrust::raw_pointer_cast(d_samples.data()), ransac_iters, M);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4.2 build systems
    thrust::device_vector<float> d_A8(ransac_iters * 64);
    thrust::device_vector<float> d_b8(ransac_iters * 8);
    buildLinearSystems8x8<<<(ransac_iters + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_p1.data()),
                                                               thrust::raw_pointer_cast(d_p2.data()),
                                                               thrust::raw_pointer_cast(d_pair_i.data()),
                                                               thrust::raw_pointer_cast(d_pair_j.data()),
                                                               thrust::raw_pointer_cast(d_samples.data()),
                                                               thrust::raw_pointer_cast(d_A8.data()),
                                                               thrust::raw_pointer_cast(d_b8.data()),
                                                               ransac_iters);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4.3 batched LU / solve (cuBLAS)
    float **d_A_ptrs = nullptr, **d_b_ptrs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A_ptrs, ransac_iters * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_b_ptrs, ransac_iters * sizeof(float*)));
    {
      std::vector<float*> hA(ransac_iters), hB(ransac_iters);
      for (int i = 0; i < ransac_iters; ++i) {
        hA[i] = thrust::raw_pointer_cast(d_A8.data()) + i * 64;
        hB[i] = thrust::raw_pointer_cast(d_b8.data()) + i * 8;
      }
      CUDA_CHECK(cudaMemcpy(d_A_ptrs, hA.data(), ransac_iters * sizeof(float*), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_b_ptrs, hB.data(), ransac_iters * sizeof(float*), cudaMemcpyHostToDevice));
    }
    thrust::device_vector<int> d_info(ransac_iters);
    thrust::device_vector<int> d_piv(ransac_iters * 8);

    CUBLAS_CHECK(cublasSgetrfBatched(handle,
                                     8,
                                     d_A_ptrs,
                                     8,
                                     thrust::raw_pointer_cast(d_piv.data()),
                                     thrust::raw_pointer_cast(d_info.data()),
                                     ransac_iters));
    std::vector<int> h_info_rf(ransac_iters);
    CUDA_CHECK(cudaMemcpy(
        h_info_rf.data(), thrust::raw_pointer_cast(d_info.data()), ransac_iters * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> h_info_rs(ransac_iters, 0);
    CUBLAS_CHECK(cublasSgetrsBatched(handle,
                                     CUBLAS_OP_N,
                                     8,
                                     1,
                                     (const float**)d_A_ptrs,
                                     8,
                                     thrust::raw_pointer_cast(d_piv.data()),
                                     d_b_ptrs,
                                     8,
                                     h_info_rs.data(),
                                     ransac_iters));

    CUDA_CHECK(cudaFree(d_A_ptrs));
    CUDA_CHECK(cudaFree(d_b_ptrs));

    // 4.4 pack + score
    thrust::device_vector<Homography> d_Hs(ransac_iters);
    packHomographies<<<(ransac_iters + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(d_b8.data()), thrust::raw_pointer_cast(d_Hs.data()), ransac_iters);

    thrust::device_vector<unsigned char> d_valid(ransac_iters, 0);

    // call the validation kernel
    {
      const int threads = 256;
      const int blocks = (ransac_iters + threads - 1) / threads;

      float max_aniso = 4.0f;      // tune
      float max_scale = 100.0f;    // tune
      float min_den = 1e-4f;       // tune (avoid division blowups)
      float cond_frob_max = 1e6f;  // tune
      int samples_side = 3;

      Homography Hpred_dev = toHomography(predicted_H);  // pack to struct

      validateHomographiesKernel<<<blocks, threads>>>(thrust::raw_pointer_cast(d_Hs.data()),
                                                      ransac_iters,
                                                      (float)img_width,
                                                      (float)img_height,
                                                      max_aniso,
                                                      max_scale,
                                                      min_den,
                                                      cond_frob_max,
                                                      Hpred_dev,
                                                      /*max_frob_gap=*/0.15f,   // tune
                                                      /*max_reproj_gap=*/3.0f,  // px, tune
                                                      /*samples_per_side=*/3,
                                                      thrust::raw_pointer_cast(d_valid.data()));

      CUDA_CHECK(cudaDeviceSynchronize());
    }

    thrust::device_vector<int> d_counts(ransac_iters, 0);
    scoreHypotheses<<<ransac_iters, 256, sizeof(int)>>>(thrust::raw_pointer_cast(d_Hs.data()),
                                                        thrust::raw_pointer_cast(d_valid.data()),
                                                        thrust::raw_pointer_cast(d_p1.data()),
                                                        thrust::raw_pointer_cast(d_p2.data()),
                                                        M,
                                                        ransac_iters,
                                                        ransac_thr_px * ransac_thr_px,
                                                        thrust::raw_pointer_cast(d_counts.data()));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto it_best = thrust::max_element(d_counts.begin(), d_counts.end());
    int bestIdx = it_best - d_counts.begin();
    CUDA_CHECK(cudaMemcpy(
        &H_best, thrust::raw_pointer_cast(d_Hs.data()) + bestIdx, sizeof(Homography), cudaMemcpyDeviceToHost));
  }

  // -------------------- 5) Local re‑matching around H*p1 -------------------
  thrust::device_vector<float2> d_yhat(N1);
  predictKernel<<<(N1 + 255) / 256, 256>>>(
      H_best, thrust::raw_pointer_cast(d_pts1.data()), thrust::raw_pointer_cast(d_yhat.data()), N1);
  CUDA_CHECK(cudaDeviceSynchronize());

  thrust::device_vector<int> d_row2col_new(N1);
  thrust::device_vector<float> d_rowScore_new(N1);
  {
    const int threads = 256;
    const size_t smem = threads * (sizeof(float) + sizeof(int));
    localRematchKernel<<<N1, threads, smem>>>(d_scores,
                                              thrust::raw_pointer_cast(d_pts2.data()),
                                              thrust::raw_pointer_cast(d_yhat.data()),
                                              N1,
                                              N2,
                                              local_radius_px * local_radius_px,
                                              geo_lambda,
                                              thrust::raw_pointer_cast(d_row2col_new.data()),
                                              thrust::raw_pointer_cast(d_rowScore_new.data()));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Build new (i,j) pool
  thrust::device_vector<unsigned char> d_accept(N1);
  thrust::transform(d_row2col_new.begin(), d_row2col_new.end(), d_accept.begin(), NonNegToMask());

  int M2 = thrust::count_if(d_accept.begin(), d_accept.end(), IsNonZeroMask());

  thrust::device_vector<int> d_pair_i2(M2), d_pair_j2(M2);
  {
    thrust::device_vector<int> d_all_i(N1);
    thrust::sequence(d_all_i.begin(), d_all_i.end(), 0);
    auto end_it = thrust::copy_if(d_all_i.begin(), d_all_i.end(), d_accept.begin(), d_pair_i2.begin(), IsNonZero());
    d_pair_i2.resize(end_it - d_pair_i2.begin());
    thrust::gather(d_pair_i2.begin(), d_pair_i2.end(), d_row2col_new.begin(), d_pair_j2.begin());
  }

  // build correspondence buffers for the new pool
  thrust::device_vector<float2> d_p1_new(M2), d_p2_new(M2);
  thrust::gather(d_pair_i2.begin(), d_pair_i2.end(), d_pts1.begin(), d_p1_new.begin());
  thrust::gather(d_pair_j2.begin(), d_pair_j2.end(), d_pts2.begin(), d_p2_new.begin());

  // -------------------- 6) Inlier retrieval on the new pool ----------------
  Homography H_inv;
  {
    cv::Mat Hcv = (cv::Mat_<float>(3, 3) << H_best.h[0],
                   H_best.h[1],
                   H_best.h[2],
                   H_best.h[3],
                   H_best.h[4],
                   H_best.h[5],
                   H_best.h[6],
                   H_best.h[7],
                   H_best.h[8]);
    cv::Mat Hcvinv = Hcv.inv();
    for (int r = 0, k = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c, ++k) H_inv.h[k] = Hcvinv.at<float>(r, c);
  }

  thrust::device_vector<float> d_errs(M2);
  {
    const int threads = 256;
    const int blocks = (M2 + threads - 1) / threads;
    computeSymmetricErrKernel<<<blocks, threads>>>(H_best,
                                                   H_inv,
                                                   thrust::raw_pointer_cast(d_p1_new.data()),
                                                   thrust::raw_pointer_cast(d_p2_new.data()),
                                                   thrust::raw_pointer_cast(d_errs.data()),
                                                   M2);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  thrust::device_vector<unsigned char> d_mask(M2);
  const float thr2 = 2.f * ransac_thr_px * ransac_thr_px;
  thrust::transform(d_errs.begin(), d_errs.end(), d_mask.begin(), MarkInlier(thr2));

  int K = thrust::count_if(d_mask.begin(), d_mask.end(), IsNonZeroMask());

  thrust::device_vector<int> d_inlier_ids(K);
  {
    thrust::device_vector<int> d_all_ids(M2);
    thrust::sequence(d_all_ids.begin(), d_all_ids.end(), 0);
    auto end_it =
        thrust::copy_if(d_all_ids.begin(), d_all_ids.end(), d_mask.begin(), d_inlier_ids.begin(), IsNonZeroMask());
    d_inlier_ids.resize(end_it - d_inlier_ids.begin());
  }

  thrust::device_vector<int> d_inlier_i(K), d_inlier_j(K);
  thrust::gather(d_inlier_ids.begin(), d_inlier_ids.end(), d_pair_i2.begin(), d_inlier_i.begin());
  thrust::gather(d_inlier_ids.begin(), d_inlier_ids.end(), d_pair_j2.begin(), d_inlier_j.begin());

  std::vector<int> hi(K), hj(K);
  thrust::copy(d_inlier_i.begin(), d_inlier_i.end(), hi.begin());
  thrust::copy(d_inlier_j.begin(), d_inlier_j.end(), hj.begin());

  // -------------------- 7) (Optional) LS refine on K inliers ----------------
  // TODO: add an 8x8 normal-equation solve here to re-estimate H_best if you want.
  // (Kept out to keep this function compact.)

  // -------------------- 8) Return ------------------------------------------
  res.matches.reserve(K);
  for (int n = 0; n < K; ++n) res.matches.emplace_back(hi[n], hj[n]);
  res.H = (cv::Mat_<float>(3, 3) << H_best.h[0],
           H_best.h[1],
           H_best.h[2],
           H_best.h[3],
           H_best.h[4],
           H_best.h[5],
           H_best.h[6],
           H_best.h[7],
           H_best.h[8]);
  return res;
}

}  // namespace xfeat