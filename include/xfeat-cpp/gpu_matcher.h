#pragma once

#include <cublas_v2.h>

#include <iostream>
#include <opencv2/core.hpp>

struct RansacParams {
  int maxIter = 10000;
  float reprojThresh = 3.0f;  // px
  int minInliers = 50;
};

// ------------------------------------------------ Matcher class (same as
// before)
class CuMatcher {
 public:
  CuMatcher() = default;
  ~CuMatcher() { destroy(); }

  void init(int maxN1, int maxN2, int D);

  void destroy();

  void computeMatches(const cv::Mat& desc1,
                      const cv::Mat& desc2,
                      std::vector<int>& bestIdx,
                      std::vector<float>& bestScore);

  std::tuple<std::vector<int>, std::vector<int>> match_mkpts(const cv::Mat& desc1,
                                                             const cv::Mat& desc2,
                                                             float min_cossim);

 private:
  float *d_scores = nullptr, *d_A = nullptr, *d_B = nullptr;
  float2 *d_pred = nullptr, *d_kp2 = nullptr;
  float4* d_param = nullptr;
  int* d_bestIdx = nullptr;
  float* d_bestScore = nullptr;
  int* d_bestIdxRow = nullptr;      // N1
  float* d_bestScoreRow = nullptr;  // N1
  int* d_bestIdxCol = nullptr;      // N2
  float* d_bestScoreCol = nullptr;  // N2

  int max_N1{}, max_N2{}, max_D{};
  cublasHandle_t handle = nullptr;
  static void CUBLAS_CHECK(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "cuBLAS error " << s << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
};
