// Gpu Inner Product + Motion‑aware Matching (row‑major, CUDA + cuBLAS)
// ------------------------------------------------------------------
//  - Computes inner‑product matrix on GPU (cuBLAS)
//  - Adds anisotropic Mahalanobis penalty aligned with predicted flow
//  - Finds best match per descriptor entirely on GPU
//
//  author: ChatGPT demo (v2 – finished main())
// ------------------------------------------------------------------
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

#include "xfeat-cpp/gpu_matcher.h"

// ------------------------------------------------ CUDA helpers
#define CUDA_CHECK(expr)                                                                                    \
  do {                                                                                                      \
    cudaError_t _err = (expr);                                                                              \
    if (_err != cudaSuccess) {                                                                              \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err) << " (" << _err << ") at " << __FILE__ << ':' \
                << __LINE__ << std::endl;                                                                   \
      std::exit(EXIT_FAILURE);                                                                              \
    }                                                                                                       \
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

std::tuple<std::vector<int>, std::vector<int>> CuMatcher::match_mkpts(const cv::Mat& desc1,
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

  // We'll reuse the same kernel twice: once across rows, once across columns.
  // First pass – per‑row arg‑max
  argmaxKernelRows<<<N1, threads, smem>>>(d_scores,  // M = N1 rows
                                          d_bestIdxRow,
                                          d_bestScoreRow,
                                          N1,
                                          N2);
  CUDA_CHECK(cudaGetLastError());

  // -----------------------------------------------------------------------
  // 4. Best‑of‑COLUMN  (for each j∈[0,N2) find i* and score c_j)
  //    The matrix is column‑major, so “columns” aren’t contiguous.
  //    We launch with grid = N2; inside kernel, stride over rows.
  // -----------------------------------------------------------------------
  argmaxKernelCols<<<N2, threads, smem>>>(d_scores,  // read same matrix
                                          d_bestIdxCol,
                                          d_bestScoreCol,
                                          N1,
                                          N2);  // dims swapped inside
  CUDA_CHECK(cudaGetLastError());

  // -----------------------------------------------------------------------
  // 5. Copy arg‑max results back
  // -----------------------------------------------------------------------
  std::vector<int> row2col(N1);
  std::vector<float> rowBest(N1);
  std::vector<int> col2row(N2);
  std::vector<float> colBest(N2);

  CUDA_CHECK(cudaMemcpy(row2col.data(), d_bestIdxRow, N1 * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(rowBest.data(), d_bestScoreRow, N1 * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaMemcpy(col2row.data(), d_bestIdxCol, N2 * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(colBest.data(), d_bestScoreCol, N2 * sizeof(float), cudaMemcpyDeviceToHost));

  // -----------------------------------------------------------------------
  // 6. Mutual‑nearest check  +  min‑cossim threshold
  // -----------------------------------------------------------------------
  std::vector<int> idx0, idx1;
  idx0.reserve(std::min(N1, N2));
  idx1.reserve(std::min(N1, N2));

  for (int i = 0; i < N1; ++i) {
    const int j = row2col[i];
    if (j < 0 || j >= N2) continue;  // safety

    if (col2row[j] == i)  // mutual
    {
      if (min_cossim <= 0.f || rowBest[i] > min_cossim) {
        idx0.push_back(i);
        idx1.push_back(j);
      }
    }
  }

  return {std::move(idx0), std::move(idx1)};
}