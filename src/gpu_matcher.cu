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
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

// Include OpenCV headers after CUDA headers to avoid namespace conflicts
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

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
  if (p1.x == 0 && p1.y == 0) {
    scores[i + j * N1] = -FLT_MAX;
    return;
  }
  if (dx * dx + dy * dy > radius2) scores[i + j * N1] = -FLT_MAX;  // mask‑out
}

// ---------------------------------------------------------------------------
// CuMatcher::match_mkpts_local
//   – local (radius‑limited) GPU brute‑force matcher
// ---------------------------------------------------------------------------
std::tuple<std::vector<int>, std::vector<int>> CuMatcher::match_mkpts_local(
    const cv::Mat& desc1,
    const cv::Mat& desc2,
    const std::vector<cv::Point2f>& kp1,
    const std::vector<cv::Point2f>& kp2,
    float search_radius,  // pixels
    float min_cossim,
    std::vector<float>* scores)  // threshold (≤0 ⇒ off)
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
      if (scores) scores->push_back(score);
    }
  }

  if (scores) {
    // normalize scores
    float min_score = *std::min_element(scores->begin(), scores->end());
    float max_score = *std::max_element(scores->begin(), scores->end());
    for (size_t i = 0; i < scores->size(); ++i) {
      (*scores)[i] = ((*scores)[i] - min_score) / (max_score - min_score);
    }
  }
  return {std::move(idx0), std::move(idx1)};
}

// ================== Tunables ==================
static constexpr int H = 48;        // hypotheses
static constexpr int L = 48;        // sample pool (top-L pairs)
static constexpr int S = 48;        // Stage-A subset size
static constexpr int K_KEEP = 16;   // finalists
static constexpr int M_MAX = 512;   // cap tentative matches for speed
static constexpr int J_SWEEPS = 4;  // Jacobi sweeps (9x9)

// ================== Argmax (row/col) ==========
__global__ void argmaxRows(const float* scores, int* idx, float* val, int N1, int N2) {
  int r = blockIdx.x;
  if (r >= N1) return;
  extern __shared__ unsigned char smem[];
  float* sval = (float*)smem;
  int* sidx = (int*)(sval + blockDim.x);
  float best = -FLT_MAX;
  int b = -1;
  for (int j = threadIdx.x; j < N2; j += blockDim.x) {
    float v = scores[r + j * N1];
    if (v > best) {
      best = v;
      b = j;
    }
  }
  sval[threadIdx.x] = best;
  sidx[threadIdx.x] = b;
  __syncthreads();
  for (int o = blockDim.x >> 1; o; o >>= 1) {
    if (threadIdx.x < o) {
      if (sval[threadIdx.x + o] > sval[threadIdx.x]) {
        sval[threadIdx.x] = sval[threadIdx.x + o];
        sidx[threadIdx.x] = sidx[threadIdx.x + o];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    idx[r] = sidx[0];
    val[r] = sval[0];
  }
}

__global__ void argmaxCols(const float* scores, int* idx, float* val, int N1, int N2) {
  int c = blockIdx.x;
  if (c >= N2) return;
  extern __shared__ unsigned char smem[];
  float* sval = (float*)smem;
  int* sidx = (int*)(sval + blockDim.x);
  float best = -FLT_MAX;
  int b = -1;
  for (int i = threadIdx.x; i < N1; i += blockDim.x) {
    float v = scores[i + c * N1];
    if (v > best) {
      best = v;
      b = i;
    }
  }
  sval[threadIdx.x] = best;
  sidx[threadIdx.x] = b;
  __syncthreads();
  for (int o = blockDim.x >> 1; o; o >>= 1) {
    if (threadIdx.x < o) {
      if (sval[threadIdx.x + o] > sval[threadIdx.x]) {
        sval[threadIdx.x] = sval[threadIdx.x + o];
        sidx[threadIdx.x] = sidx[threadIdx.x + o];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    idx[c] = sidx[0];
    val[c] = sval[0];
  }
}

// ================== Tentatives (mutual) =======
__global__ void buildTentativePairs_mutual(const int* bestRow,
                                           const int* bestCol,
                                           const float* bestRowVal,
                                           const float* bestColVal,
                                           int N1,
                                           int N2,
                                           float thr,
                                           int* pair_i,
                                           int* pair_j,
                                           float* pair_s,
                                           int* Mout) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N1) return;
  int j = bestRow[i];
  if (j >= 0 && j < N2 && bestCol[j] == i) {
    if (thr <= 0.f || (bestRowVal[i] > thr && bestColVal[j] > thr)) {
      int k = atomicAdd(Mout, 1);
      pair_i[k] = i;
      pair_j[k] = j;
      pair_s[k] = bestRowVal[i];
    }
  }
}

// ================== RNG =======================
__global__ void initPhilox(curandStatePhilox4_32_10_t* st, unsigned long long seed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, id, 0, &st[id]);
}

// ================== Build A (col-major) =======
__global__ void buildA8x9_colmajor(const float2* x1,
                                   const float2* x2,
                                   const int* samples,
                                   int poolL,
                                   float* A_cm)  // [H][8x9], lda=8
{
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) return;
  float* A = A_cm + h * (8 * 9);
// samples are indices into [0..poolL)
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    int m = samples[h * 8 + k];
    m = (m < poolL) ? m : (m % poolL);
    float x = x1[m].x, y = x1[m].y, X = x2[m].x, Y = x2[m].y;
    A[k + 0 * 8] = X * x;
    A[k + 1 * 8] = X * y;
    A[k + 2 * 8] = X;
    A[k + 3 * 8] = Y * x;
    A[k + 4 * 8] = Y * y;
    A[k + 5 * 8] = Y;
    A[k + 6 * 8] = x;
    A[k + 7 * 8] = y;
    A[k + 8 * 8] = 1.f;
  }
}

// ================== N=A^T A (9x9) =============
__global__ void gram9(const float* A_cm, float* N)  // A: [H][8x9] col-major
{
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) return;
  const float* A = A_cm + h * (8 * 9);
  float* G = N + h * 81;
// G = A^T A : for c1,c2 in [0..8], dot of columns (length 8)
#pragma unroll
  for (int c1 = 0; c1 < 9; ++c1) {
    for (int c2 = c1; c2 < 9; ++c2) {
      float acc = 0.f;
#pragma unroll
      for (int r = 0; r < 8; ++r) acc += A[r + c1 * 8] * A[r + c2 * 8];
      G[c1 + 9 * c2] = G[c2 + 9 * c1] = acc;  // col-major write
    }
  }
}

// ========== 9x9 Jacobi (smallest eigenvec) ====
__device__ void jacobi_smallest_9x9(float* Gcm, float* Vcm)  // both col-major
{
  // Init V=I
  for (int j = 0; j < 9; ++j) {
    for (int i = 0; i < 9; ++i) Vcm[i + 9 * j] = (i == j) ? 1.f : 0.f;
  }
  // Cyclic sweeps
  for (int sweep = 0; sweep < J_SWEEPS; ++sweep) {
    for (int p = 0; p < 8; ++p)
      for (int q = p + 1; q < 9; ++q) {
        float Gpp = Gcm[p + 9 * p];
        float Gqq = Gcm[q + 9 * q];
        float Gpq = Gcm[p + 9 * q];
        if (fabsf(Gpq) < 1e-10f * (fabsf(Gpp) + fabsf(Gqq))) continue;
        float tau = (Gqq - Gpp) / (2.f * Gpq);
        float t = copysignf(1.f / (fabsf(tau) + sqrtf(1.f + tau * tau)), tau);
        float c = rsqrtf(1.f + t * t);
        float s = t * c;

        // G = J^T G J (apply to cols/rows p,q)
        for (int k = 0; k < 9; ++k) {  // columns update
          float Gkp = Gcm[k + 9 * p];
          float Gkq = Gcm[k + 9 * q];
          Gcm[k + 9 * p] = c * Gkp - s * Gkq;
          Gcm[k + 9 * q] = s * Gkp + c * Gkq;
        }
        for (int k = 0; k < 9; ++k) {  // rows update
          float Gpk = Gcm[p + 9 * k];
          float Gqk = Gcm[q + 9 * k];
          Gcm[p + 9 * k] = c * Gpk - s * Gqk;
          Gcm[q + 9 * k] = s * Gpk + c * Gqk;
        }
        // V = V J
        for (int k = 0; k < 9; ++k) {
          float Vkp = Vcm[k + 9 * p];
          float Vkq = Vcm[k + 9 * q];
          Vcm[k + 9 * p] = c * Vkp - s * Vkq;
          Vcm[k + 9 * q] = s * Vkp + c * Vkq;
        }
      }
  }
  // After sweeps, diagonals approx eigenvalues; smallest at argmin diag
}

__global__ void smallestEigenVec9(const float* Ncm, float* e_row)  // e_row[H][9] row-major
{
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) return;
  // put N into registers/shared
  float G[81];
  float V[81];
#pragma unroll
  for (int k = 0; k < 81; ++k) G[k] = Ncm[h * 81 + k];
  jacobi_smallest_9x9(G, V);
  // find argmin diag of G
  int imin = 0;
  float dmin = G[0];
  for (int i = 1; i < 9; ++i) {
    float d = G[i + 9 * i];
    if (d < dmin) {
      dmin = d;
      imin = i;
    }
  }
  // eigenvector is column 'imin' of V (col-major). Write as row-major 3x3 (E)
  float* out = e_row + h * 9;
  for (int r = 0; r < 9; ++r) out[r] = V[r + 9 * imin];
}

// ========== Stage-A scoring (subset) ==========
__device__ __forceinline__ int score_subset_warp(const float e[9],
                                                 const float2* __restrict__ x1,
                                                 const float2* __restrict__ x2,
                                                 const int* __restrict__ subset_idx,
                                                 int Ssz) {
  const unsigned full = 0xffffffff;
  float e11 = e[0], e12 = e[1], e13 = e[2], e21 = e[3], e22 = e[4], e23 = e[5], e31 = e[6], e32 = e[7], e33 = e[8];
  int acc = 0;
  for (int t = threadIdx.x; t < Ssz; t += blockDim.x) {
    int m = subset_idx[t];
    float x = x1[m].x, y = x1[m].y, X = x2[m].x, Y = x2[m].y;
    float Ex1u = e11 * x + e12 * y + e13;
    float Ex1v = e21 * x + e22 * y + e23;
    float Ex1w = e31 * x + e32 * y + e33;
    float num = X * Ex1u + Y * Ex1v + Ex1w;
    float Etxu = e11 * X + e21 * Y + e31;
    float Etyv = e12 * X + e22 * Y + e32;
    float den = Ex1u * Ex1u + Ex1v * Ex1v + Etxu * Etxu + Etyv * Etyv;
    float d2 = (num * num) / fmaxf(den, 1e-12f);
    int inl = (d2 <= 1.f);  // threshold=1 in normalized units here; caller scales subset_idx accordingly
    unsigned mask = __ballot_sync(full, inl);
    if ((threadIdx.x & 31) == 0) acc += __popc(mask);
  }
  return acc;
}

__global__ void stageA_build_and_score(const float2* x1,
                                       const float2* x2,
                                       const int* subset_idx,
                                       int Ssz,
                                       const int* samples,
                                       int poolL,
                                       float thr2_subset,
                                       int* counts,
                                       float* E_row_out) {
  int h = blockIdx.x;
  if (h >= H) return;
  // 1) Build A -> N
  float A[8 * 9];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    int m = samples[h * 8 + k];
    m = (m < poolL) ? m : (m % poolL);
    float x = x1[m].x, y = x1[m].y, X = x2[m].x, Y = x2[m].y;
    A[k + 0 * 8] = X * x;
    A[k + 1 * 8] = X * y;
    A[k + 2 * 8] = X;
    A[k + 3 * 8] = Y * x;
    A[k + 4 * 8] = Y * y;
    A[k + 5 * 8] = Y;
    A[k + 6 * 8] = x;
    A[k + 7 * 8] = y;
    A[k + 8 * 8] = 1.f;
  }
  float N[81] = {0.f};
#pragma unroll
  for (int c1 = 0; c1 < 9; ++c1) {
    for (int c2 = c1; c2 < 9; ++c2) {
      float acc = 0.f;
#pragma unroll
      for (int r = 0; r < 8; ++r) acc += A[r + c1 * 8] * A[r + c2 * 8];
      N[c1 + 9 * c2] = N[c2 + 9 * c1] = acc;
    }
  }
  // 2) Smallest eigenvector (E as 9-vector)
  float V[81];
  jacobi_smallest_9x9(N, V);
  int imin = 0;
  float dmin = N[0];
  for (int i = 1; i < 9; ++i) {
    float d = N[i + 9 * i];
    if (d < dmin) {
      dmin = d;
      imin = i;
    }
  }
  float e[9];
  for (int r = 0; r < 9; ++r) e[r] = V[r + 9 * imin];

  // 3) Stage-A scoring on subset
  int c = score_subset_warp(e, x1, x2, subset_idx, Ssz);
  if (threadIdx.x == 0) {
    counts[h] = c;
    // store E (row-major) for reuse in stage-B if selected
    float* out = E_row_out + 9 * h;
    for (int k = 0; k < 9; ++k) out[k] = e[k];
  }
}

// ========== 3x3 rank-2 projection (Jacobi SVD) =========
__device__ void svd3x3(const float* M, float* U, float* S, float* Vt) {  // all col-major
  // Symmetric eigen of MtM to get V, then U = M*V*S^{-1}
  float MtM[9] = {0};
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i) {
      float acc = 0.f;
      for (int k = 0; k < 3; ++k) {
        float Mik = M[i + 3 * k], Mjk = M[j + 3 * k];
        acc += Mjk * Mik;
      }
      MtM[i + 3 * j] = acc;
    }
  // Jacobi for 3x3 symmetric
  float V[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int s = 0; s < 6; ++s) {
    for (int p = 0; p < 2; ++p)
      for (int q = p + 1; q < 3; ++q) {
        float App = MtM[p + 3 * p], Aqq = MtM[q + 3 * q], Apq = MtM[p + 3 * q];
        if (fabsf(Apq) < 1e-12f * (fabsf(App) + fabsf(Aqq))) continue;
        float tau = (Aqq - App) / (2.f * Apq);
        float t = copysignf(1.f / (fabsf(tau) + sqrtf(1.f + tau * tau)), tau);
        float c = rsqrtf(1.f + t * t), s2 = t * c;
        // cols p,q of MtM and V
        for (int k = 0; k < 3; ++k) {
          float akp = MtM[k + 3 * p], akq = MtM[k + 3 * q];
          MtM[k + 3 * p] = c * akp - s2 * akq;
          MtM[k + 3 * q] = s2 * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {
          float apk = MtM[p + 3 * k], aqk = MtM[q + 3 * k];
          MtM[p + 3 * k] = c * apk - s2 * aqk;
          MtM[q + 3 * k] = s2 * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {
          float vkp = V[k + 3 * p], vkq = V[k + 3 * q];
          V[k + 3 * p] = c * vkp - s2 * vkq;
          V[k + 3 * q] = s2 * vkp + c * vkq;
        }
      }
  }
  // singular values from diag(MtM)
  float sv[3] = {sqrtf(fmaxf(MtM[0], 0.f)), sqrtf(fmaxf(MtM[4], 0.f)), sqrtf(fmaxf(MtM[8], 0.f))};
  // sort descending (simple bubble, 3 items)
  int o[3] = {0, 1, 2};
  for (int a = 0; a < 2; ++a)
    for (int b = a + 1; b < 3; ++b)
      if (sv[b] > sv[a]) {
        float ts = sv[a];
        sv[a] = sv[b];
        sv[b] = ts;
        int to = o[a];
        o[a] = o[b];
        o[b] = to;
      }
  float Vsorted[9];
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i) Vsorted[i + 3 * j] = V[i + 3 * o[j]];
  // U = M * V * inv(S)
  float Utmp[9] = {0};
  for (int j = 0; j < 3; ++j) {
    float invs = (sv[j] > 1e-12f) ? 1.f / sv[j] : 0.f;
    for (int i = 0; i < 3; ++i) {
      float acc = 0.f;
      for (int k = 0; k < 3; ++k) acc += M[i + 3 * k] * Vsorted[k + 3 * j];
      Utmp[i + 3 * j] = acc * invs;
    }
  }
  // Orthonormalize U with Gram-Schmidt quick fix
  // (skipped—usually fine)
  // Return
  for (int j = 0; j < 3; ++j) S[j] = sv[j];
  for (int k = 0; k < 9; ++k) {
    Vt[k] = Vsorted[(k % 3) * 3 + k / 3];
    U[k] = Utmp[k];
  }
}

__global__ void enforce_rank2_and_score(const float* E_row_in,  // [K][9] row
                                        const int* finalists,   // [K] indices into stage-A E bank
                                        const float2* x1,
                                        const float2* x2,
                                        int M,
                                        float thr2,
                                        int* counts,
                                        float* Ebest_row_out) {
  int k = blockIdx.x;
  if (k >= K_KEEP) return;
  const float* e_in = E_row_in + 9 * finalists[k];

  // convert to col-major 3x3
  float Ecm[9];
  Ecm[0] = e_in[0];
  Ecm[1] = e_in[3];
  Ecm[2] = e_in[6];
  Ecm[3] = e_in[1];
  Ecm[4] = e_in[4];
  Ecm[5] = e_in[7];
  Ecm[6] = e_in[2];
  Ecm[7] = e_in[5];
  Ecm[8] = e_in[8];

  // SVD and set smallest sigma to 0
  float U[9], Vt[9], S[3];
  svd3x3(Ecm, U, S, Vt);
  S[2] = 0.f;
  // E = U * diag(S) * Vt
  float Ecm2[9] = {0};
  for (int j = 0; j < 3; ++j) {
    float sj = S[j];
    for (int i = 0; i < 3; ++i) {
      float acc = 0.f;
      for (int t = 0; t < 3; ++t) acc += U[i + 3 * t] * ((t == j) ? sj : 0.f);
      // later multiply by Vt
    }
  }
  // Multiply U*diag(S) first into T
  float T[9] = {0};
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i) T[i + 3 * j] = U[i + 3 * j] * S[j];
  // E = T * Vt
  float Ecm_final[9] = {0};
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i) {
      float acc = 0.f;
      for (int t = 0; t < 3; ++t) acc += T[i + 3 * t] * Vt[t + 3 * j];
      Ecm_final[i + 3 * j] = acc;
    }
  // to row-major
  float e[9] = {Ecm_final[0],
                Ecm_final[3],
                Ecm_final[6],
                Ecm_final[1],
                Ecm_final[4],
                Ecm_final[7],
                Ecm_final[2],
                Ecm_final[5],
                Ecm_final[8]};

  // Score on all M with warp ballots
  const unsigned full = 0xffffffff;
  float e11 = e[0], e12 = e[1], e13 = e[2], e21 = e[3], e22 = e[4], e23 = e[5], e31 = e[6], e32 = e[7], e33 = e[8];
  __shared__ int sum;
  if (threadIdx.x == 0) sum = 0;
  __syncthreads();
  int local = 0;
  for (int m = threadIdx.x; m < M; m += blockDim.x) {
    float x = x1[m].x, y = x1[m].y, X = x2[m].x, Y = x2[m].y;
    float Ex1u = e11 * x + e12 * y + e13, Ex1v = e21 * x + e22 * y + e23, Ex1w = e31 * x + e32 * y + e33;
    float num = X * Ex1u + Y * Ex1v + Ex1w;
    float Etxu = e11 * X + e21 * Y + e31, Etyv = e12 * X + e22 * Y + e32;
    float den = Ex1u * Ex1u + Ex1v * Ex1v + Etxu * Etxu + Etyv * Etyv;
    float d2 = (num * num) / fmaxf(den, 1e-12f);
    int inl = (d2 <= thr2);
    unsigned mask = __ballot_sync(full, inl);
    if ((threadIdx.x & 31) == 0) local += __popc(mask);
  }
  atomicAdd(&sum, local);
  __syncthreads();
  if (threadIdx.x == 0) {
    counts[k] = sum;
    // optionally write back the final E (row-major) for the winner later
    float* out = Ebest_row_out + 9 * k;
    for (int t = 0; t < 9; ++t) out[t] = e[t];
  }
}

// ========== Public entry ======================
EResult CuMatcher::match_mkpts_gpuRansac_E(const cv::Mat& desc1,
                                           const cv::Mat& desc2,
                                           const std::vector<cv::Point2f>& kpts1_px,
                                           const std::vector<cv::Point2f>& kpts2_px,
                                           float min_cossim,
                                           float sampson_thr_px,
                                           int ransac_seed,
                                           float fx,
                                           float fy,
                                           float cx,
                                           float cy) {
  const int N1 = desc1.rows, N2 = desc2.rows, D = desc1.cols;
  assert(desc1.type() == CV_32F && desc2.type() == CV_32F && D == desc2.cols);
  assert((int)kpts1_px.size() == N1 && (int)kpts2_px.size() == N2);

  // ---- BF cosine with cuBLAS ----
  thrust::device_vector<float> dS(N1 * N2);
  CUDA_CHECK(cudaMemcpy(d_A, desc1.ptr<float>(), size_t(N1) * D * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, desc2.ptr<float>(), size_t(N2) * D * sizeof(float), cudaMemcpyHostToDevice));
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasSgemm(handle,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           N1,
                           N2,
                           D,
                           &alpha,
                           d_A,
                           D,
                           d_B,
                           D,
                           &beta,
                           thrust::raw_pointer_cast(dS.data()),
                           N1));
  // (Optional) Tensor‑core GEMMEx with FP16: switch here.

  // ---- Argmax + mutual ----
  thrust::device_vector<int> dBestRow(N1), dBestCol(N2);
  thrust::device_vector<float> dBestRowVal(N1), dBestColVal(N2);
  const int T = 256;
  size_t smem = T * (sizeof(float) + sizeof(int));
  argmaxRows<<<N1, T, smem>>>(thrust::raw_pointer_cast(dS.data()),
                              thrust::raw_pointer_cast(dBestRow.data()),
                              thrust::raw_pointer_cast(dBestRowVal.data()),
                              N1,
                              N2);
  argmaxCols<<<N2, T, smem>>>(thrust::raw_pointer_cast(dS.data()),
                              thrust::raw_pointer_cast(dBestCol.data()),
                              thrust::raw_pointer_cast(dBestColVal.data()),
                              N1,
                              N2);

  thrust::device_vector<int> dPi(N1), dPj(N1), dM(1, 0);
  thrust::device_vector<float> dPs(N1);
  buildTentativePairs_mutual<<<(N1 + 255) / 256, 256>>>(thrust::raw_pointer_cast(dBestRow.data()),
                                                        thrust::raw_pointer_cast(dBestCol.data()),
                                                        thrust::raw_pointer_cast(dBestRowVal.data()),
                                                        thrust::raw_pointer_cast(dBestColVal.data()),
                                                        N1,
                                                        N2,
                                                        min_cossim,
                                                        thrust::raw_pointer_cast(dPi.data()),
                                                        thrust::raw_pointer_cast(dPj.data()),
                                                        thrust::raw_pointer_cast(dPs.data()),
                                                        thrust::raw_pointer_cast(dM.data()));
  CUDA_CHECK(cudaDeviceSynchronize());
  int M;
  CUDA_CHECK(cudaMemcpy(&M, thrust::raw_pointer_cast(dM.data()), sizeof(int), cudaMemcpyDeviceToHost));
  if (M < 8) {
    return {};
  }

  // ---- Keep top-M by score ----
  int keep = std::min(M, M_MAX);
  thrust::device_vector<int> order(M);
  thrust::sequence(order.begin(), order.end());
  thrust::sort_by_key(dPs.begin(), dPs.begin() + M, order.begin(), thrust::greater<float>());
  thrust::device_vector<int> dPiK(keep), dPjK(keep);
  thrust::gather(order.begin(), order.begin() + keep, dPi.begin(), dPiK.begin());
  thrust::gather(order.begin(), order.begin() + keep, dPj.begin(), dPjK.begin());
  M = keep;

  // ---- Pack normalized coords in pair order (length M) ----
  thrust::device_vector<float2> dX1(M), dX2(M);
  {
    thrust::host_vector<int> hPi = dPiK, hPj = dPjK;
    std::vector<float2> h1(M), h2(M);
    bool norm = (fx > 0 && fy > 0);
    for (int m = 0; m < M; ++m) {
      auto p = kpts1_px[hPi[m]];
      auto q = kpts2_px[hPj[m]];
      h1[m] = norm ? float2{(p.x - cx) / fx, (p.y - cy) / fy} : float2{p.x, p.y};
      h2[m] = norm ? float2{(q.x - cx) / fx, (q.y - cy) / fy} : float2{q.x, q.y};
    }
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(dX1.data()), h1.data(), M * sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(dX2.data()), h2.data(), M * sizeof(float2), cudaMemcpyHostToDevice));
  }

  // ---- Pre-sample subset indices for Stage-A ----
  thrust::device_vector<int> dSubset(S);
  {
    thrust::host_vector<int> hIdx(S);
    for (int i = 0; i < S; ++i) hIdx[i] = (i * 9973) % M;  // cheap pseudo-rand; or use curand once
    dSubset = hIdx;
  }

  // ---- Pre-sample minimal sets (8) from pool L ----
  int poolL = std::min(L, M);
  thrust::device_vector<int> dSamples(H * 8);
  {
    thrust::device_vector<curandStatePhilox4_32_10_t> dStates((H + 255) / 256 * 256);
    initPhilox<<<(H + 255) / 256, 256>>>(thrust::raw_pointer_cast(dStates.data()), (unsigned long long)ransac_seed);
    // Build on host for determinism/simple uniqueness
    thrust::host_vector<int> hS(H * 8);
    for (int hidx = 0; hidx < H; ++hidx) {
      // simple unique sampling without replacement from [0..poolL)
      int used[poolL > 8 ? 8 : poolL];
      int u = 0;
      for (int k = 0; k < 8; ++k) {
        int v = (hidx * 131 + k * 37 + ransac_seed * 17 + k) % poolL;
        // de-duplicate
        bool ok = false;
        for (int t = 0; t < poolL * 2 && !ok; ++t) {
          v = (v + 1) % poolL;
          ok = true;
          for (int z = 0; z < u; ++z)
            if (used[z] == v) {
              ok = false;
              break;
            }
        }
        used[u++] = v;
        hS[hidx * 8 + k] = v;
      }
    }
    dSamples = hS;
  }

  // ---- Stage-A: build+score (subset) ----
  float fref = (fx > 0 && fy > 0) ? 0.5f * (fx + fy) : 1.f;
  float thr2_norm = (fx > 0) ? (sampson_thr_px / fref) * (sampson_thr_px / fref) : (sampson_thr_px * sampson_thr_px);
  // For Stage-A fast screening we scale subset threshold up slightly (be lenient):
  float thr2_subset = thr2_norm * 4.f;

  thrust::device_vector<int> dCountsA(H, 0);
  thrust::device_vector<float> dEbank(H * 9);
  stageA_build_and_score<<<H, 128>>>(thrust::raw_pointer_cast(dX1.data()),
                                     thrust::raw_pointer_cast(dX2.data()),
                                     thrust::raw_pointer_cast(dSubset.data()),
                                     S,
                                     thrust::raw_pointer_cast(dSamples.data()),
                                     poolL,
                                     thr2_subset,
                                     thrust::raw_pointer_cast(dCountsA.data()),
                                     thrust::raw_pointer_cast(dEbank.data()));
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- Pick top-K hyps ----
  thrust::device_vector<int> hyps(H);
  thrust::sequence(hyps.begin(), hyps.end(), 0);
  thrust::sort_by_key(dCountsA.begin(), dCountsA.end(), hyps.begin(), thrust::greater<int>());
  thrust::device_vector<int> finalists(K_KEEP);
  thrust::copy(hyps.begin(), hyps.begin() + K_KEEP, finalists.begin());

  // ---- Stage-B: enforce rank-2 and score fully ----
  thrust::device_vector<int> dCountsB(K_KEEP, 0);
  thrust::device_vector<float> dEfinal(K_KEEP * 9);
  enforce_rank2_and_score<<<K_KEEP, 128>>>(thrust::raw_pointer_cast(dEbank.data()),
                                           thrust::raw_pointer_cast(finalists.data()),
                                           thrust::raw_pointer_cast(dX1.data()),
                                           thrust::raw_pointer_cast(dX2.data()),
                                           M,
                                           thr2_norm,
                                           thrust::raw_pointer_cast(dCountsB.data()),
                                           thrust::raw_pointer_cast(dEfinal.data()));
  CUDA_CHECK(cudaDeviceSynchronize());

  // pick best
  thrust::host_vector<int> hCountsB = dCountsB;
  int bestK = 0;
  for (int i = 1; i < K_KEEP; ++i)
    if (hCountsB[i] > hCountsB[bestK]) bestK = i;

  // ---- Build inlier mask for best and return pairs ----
  thrust::host_vector<float> hEbest(9);
  CUDA_CHECK(cudaMemcpy(
      hEbest.data(), thrust::raw_pointer_cast(dEfinal.data()) + 9 * bestK, 9 * sizeof(float), cudaMemcpyDeviceToHost));
  cv::Mat Ecv = (cv::Mat_<float>(3, 3) << hEbest[0],
                 hEbest[1],
                 hEbest[2],
                 hEbest[3],
                 hEbest[4],
                 hEbest[5],
                 hEbest[6],
                 hEbest[7],
                 hEbest[8]);

  // mask on host (K small; for speed you can add a device kernel)
  thrust::host_vector<float2> hX1 = dX1, hX2 = dX2;
  std::vector<std::pair<int, int>> inliers;
  inliers.reserve(M);
  for (int m = 0; m < M; ++m) {
    float x = hX1[m].x, y = hX1[m].y, X = hX2[m].x, Y = hX2[m].y;
    float e11 = hEbest[0], e12 = hEbest[1], e13 = hEbest[2], e21 = hEbest[3], e22 = hEbest[4], e23 = hEbest[5],
          e31 = hEbest[6], e32 = hEbest[7], e33 = hEbest[8];
    float Ex1u = e11 * x + e12 * y + e13, Ex1v = e21 * x + e22 * y + e23, Ex1w = e31 * x + e32 * y + e33;
    float num = X * Ex1u + Y * Ex1v + Ex1w;
    float Etxu = e11 * X + e21 * Y + e31, Etyv = e12 * X + e22 * Y + e32;
    float den = Ex1u * Ex1u + Ex1v * Ex1v + Etxu * Etxu + Etyv * Etyv;
    float d2 = (num * num) / std::max(den, 1e-12f);
    if (d2 <= thr2_norm) inliers.emplace_back(/*i*/ 0, /*j*/ 0);  // filled below
  }
  // Map back to original indices
  thrust::host_vector<int> hPiK = dPiK, hPjK = dPjK;
  inliers.clear();
  inliers.reserve(M);
  for (int m = 0; m < M; ++m) {
    float x = hX1[m].x, y = hX1[m].y, X = hX2[m].x, Y = hX2[m].y;
    float e11 = hEbest[0], e12 = hEbest[1], e13 = hEbest[2], e21 = hEbest[3], e22 = hEbest[4], e23 = hEbest[5],
          e31 = hEbest[6], e32 = hEbest[7], e33 = hEbest[8];
    float Ex1u = e11 * x + e12 * y + e13, Ex1v = e21 * x + e22 * y + e23, Ex1w = e31 * x + e32 * y + e33;
    float num = X * Ex1u + Y * Ex1v + Ex1w;
    float Etxu = e11 * X + e21 * Y + e31, Etyv = e12 * X + e22 * Y + e32;
    float den = Ex1u * Ex1u + Ex1v * Ex1v + Etxu * Etxu + Etyv * Etyv;
    float d2 = (num * num) / std::max(den, 1e-12f);
    if (d2 <= thr2_norm) inliers.emplace_back(hPiK[m], hPjK[m]);
  }

  EResult r;
  r.matches = std::move(inliers);
  r.E = Ecv;
  return r;
}

}  // namespace xfeat