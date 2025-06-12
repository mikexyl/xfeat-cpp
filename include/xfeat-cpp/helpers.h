#pragma once

#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

namespace xfeat {
/**
 * @brief Computes per-keypoint localization uncertainty from a heatmap.
 *
 * @param heatmap   Single-channel floating-point heatmap (CV_32F or CV_64F).
 * @param keypoints Vector of integer keypoint locations (cv::Point).
 * @param radius    Half‐size of the square patch around each keypoint (default=2 → 5×5 patch).
 * @param ridge     Small ridge λ to add to Hessian for numeric stability (default=1e-3).
 * @return std::vector<cv::Vec2d>
 *         For each keypoint: (sigma_x, sigma_y).
 */
inline std::vector<cv::Vec2d> computeKeypointsStd(const cv::Mat& heatmap,
                                                  const std::vector<cv::KeyPoint>& keypoints,
                                                  int radius = 7,
                                                  double ridgeFactor = 1e-3,
                                                  double eps = 1e-12) {
  CV_Assert(heatmap.type() == CV_32F || heatmap.type() == CV_64F);
  const int rows = heatmap.rows;
  const int cols = heatmap.cols;
  std::vector<cv::Vec2d> uncertainties;
  uncertainties.reserve(keypoints.size());

  // Helper: safe pixel read as double, with border replication
  auto atD = [&](int y, int x) -> double {
    x = cv::borderInterpolate(x, cols, cv::BORDER_REPLICATE);
    y = cv::borderInterpolate(y, rows, cv::BORDER_REPLICATE);
    if (heatmap.type() == CV_32F)
      return static_cast<double>(heatmap.at<float>(y, x));
    else
      return heatmap.at<double>(y, x);
  };

  for (const auto& kp : keypoints) {
    int u = kp.pt.x, v = kp.pt.y;

    // If too close to border, assign a large fallback uncertainty
    if (u < radius || u >= cols - radius || v < radius || v >= rows - radius) {
      double large = 1e3;
      uncertainties.emplace_back(large, large);
      continue;
    }

    // Sample central value and clamp
    double Hc = std::max(atD(v, u), eps);
    double Lc = std::log(Hc);

    // Second derivatives via central finite differences on log(H)
    double L_uplus = std::log(std::max(atD(v, u + 1), eps));
    double L_uminus = std::log(std::max(atD(v, u - 1), eps));
    double fxx = L_uplus - 2 * Lc + L_uminus;

    double L_vplus = std::log(std::max(atD(v + 1, u), eps));
    double L_vminus = std::log(std::max(atD(v - 1, u), eps));
    double fyy = L_vplus - 2 * Lc + L_vminus;

    double L_pp = std::log(std::max(atD(v + 1, u + 1), eps));
    double L_pm = std::log(std::max(atD(v - 1, u + 1), eps));
    double L_mp = std::log(std::max(atD(v + 1, u - 1), eps));
    double L_mm = std::log(std::max(atD(v - 1, u - 1), eps));
    double fxy = (L_pp - L_pm - L_mp + L_mm) * 0.25;

    // Build the Hessian of -log H
    cv::Mat Hmat = (cv::Mat_<double>(2, 2) << -fxx, -fxy, -fxy, -fyy);

    // Adaptive ridge regularization based on trace
    double trace = Hmat.at<double>(0, 0) + Hmat.at<double>(1, 1);
    double ridge = std::abs(trace) * ridgeFactor + 1e-6;
    Hmat += ridge * cv::Mat::eye(2, 2, CV_64F);

    // Check determinant to detect near-singularity
    double det = Hmat.at<double>(0, 0) * Hmat.at<double>(1, 1) - Hmat.at<double>(0, 1) * Hmat.at<double>(1, 0);
    if (std::abs(det) < 1e-8) {
      // Fallback to isotropic gradient-magnitude heuristic
      double gx = (atD(v, u + 1) - atD(v, u - 1)) * 0.5;
      double gy = (atD(v + 1, u) - atD(v - 1, u)) * 0.5;
      double gmag = std::sqrt(gx * gx + gy * gy);
      double iso = (gmag > 1e-6 ? 1.0 / gmag : 1e3);
      uncertainties.emplace_back(iso, iso);
      continue;
    }

    // Invert with SVD for stability
    cv::Mat cov;
    cv::invert(Hmat, cov, cv::DECOMP_SVD);

    // Extract stddevs, clamp negatives or NaNs
    double var_u = cov.at<double>(0, 0), var_v = cov.at<double>(1, 1);
    double su = (var_u > 0 && std::isfinite(var_u)) ? std::sqrt(var_u) : 1e3;
    double sv = (var_v > 0 && std::isfinite(var_v)) ? std::sqrt(var_v) : 1e3;

    uncertainties.emplace_back(su, sv);
  }

  return uncertainties;
}

inline std::vector<cv::Vec2d> computeKeypointUncertaintiesSoftmax(const cv::Mat& heatmap,
                                                                  const std::vector<cv::KeyPoint>& keypoints,
                                                                  int radius = 5,
                                                                  double eps = 1e-12) {
  CV_Assert(heatmap.type() == CV_32F || heatmap.type() == CV_64F);
  int rows = heatmap.rows, cols = heatmap.cols;
  int patchSize = 2 * radius + 1;

  auto atD = [&](int y, int x) {
    // border-replicate
    x = cv::borderInterpolate(x, cols, cv::BORDER_REPLICATE);
    y = cv::borderInterpolate(y, rows, cv::BORDER_REPLICATE);
    return (heatmap.type() == CV_32F) ? static_cast<double>(heatmap.at<float>(y, x)) : heatmap.at<double>(y, x);
  };

  std::vector<cv::Vec2d> uncertainties;
  uncertainties.reserve(keypoints.size());

  for (auto& kp : keypoints) {
    int u = kp.pt.x, v = kp.pt.y;

    // 1) Find max in patch
    double m = -1e300;
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        m = std::max(m, atD(v + dy, u + dx));
      }
    }

    // 2) Compute softmax weights and accumulate Z, weighted sums
    double Z = 0, sumX = 0, sumY = 0;
    std::vector<std::vector<double>> w(patchSize, std::vector<double>(patchSize));
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        double val = atD(v + dy, u + dx);
        double ex = std::exp(val - m);
        int iy = dy + radius, ix = dx + radius;
        w[iy][ix] = ex;
        Z += ex;
        sumX += (u + dx) * ex;
        sumY += (v + dy) * ex;
      }
    }
    if (Z < eps) {
      // degenerate patch
      uncertainties.emplace_back(1e3, 1e3);
      continue;
    }
    double muX = sumX / Z;
    double muY = sumY / Z;

    // 3) Compute variances
    double varX = 0, varY = 0;
    for (int iy = 0; iy < patchSize; ++iy) {
      for (int ix = 0; ix < patchSize; ++ix) {
        double p = w[iy][ix] / Z;
        double dx = (u + ix - radius) - muX;
        double dy = (v + iy - radius) - muY;
        varX += dx * dx * p;
        varY += dy * dy * p;
      }
    }

    uncertainties.emplace_back(std::sqrt(varX), std::sqrt(varY));
  }

  return uncertainties;
}

/**
 * @brief Compute per-keypoint uncertainty via Sobel‐based Hessian.
 *
 * 1) Gaussian blur (5×5) + log(eps+H)
 * 2) Sobel 2nd derivatives: dxx, dyy, dxy
 * 3) At each keypoint, H = -[dxx dxy; dxy dyy] + ridge*I
 * 4) Invert H → cov → σx, σy
 */
inline std::vector<cv::Vec2d> computeUncertaintySobel(const cv::Mat& heatmap,
                                                      const std::vector<cv::KeyPoint>& keypoints,
                                                      double ridge = 1e-2,
                                                      double eps = 1e-12,
                                                      cv::Mat* debug = nullptr) {
  CV_Assert(heatmap.type() == CV_32F || heatmap.type() == CV_64F);

  static constexpr double kMaxStd = 8 * 1.5;

  // 0) Max-pool via dilation with 8×8 structuring element
  cv::Mat pooled;
  cv::dilate(heatmap, pooled, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8)));

  // 0b) Downsample to 1/8 size
  cv::Mat small;
  cv::resize(pooled, small, cv::Size(heatmap.cols / 8, heatmap.rows / 8), 0, 0, cv::INTER_NEAREST);

  // 1) Stronger blur + log
  cv::Mat Hf;
  cv::GaussianBlur(small, Hf, cv::Size(5, 5), 1.2, 1.2, cv::BORDER_REPLICATE);
  Hf.convertTo(Hf, CV_64F);
  Hf += eps;
  cv::log(Hf, Hf);

  if (debug) {
    *debug = Hf.clone();
    cv::normalize(*debug, *debug, 0, 255, cv::NORM_MINMAX, CV_8U);
  }

  // 2) Compute derivatives: second-order for Hessian, first-order for fallback
  cv::Mat dxx, dyy, dxy, gradX, gradY;
  // second derivatives (wider kernel)
  cv::Sobel(Hf, dxx, CV_64F, 2, 0, 5, 1.0, 0.0, cv::BORDER_REPLICATE);
  cv::Sobel(Hf, dyy, CV_64F, 0, 2, 5, 1.0, 0.0, cv::BORDER_REPLICATE);
  cv::Sobel(Hf, dxy, CV_64F, 1, 1, 5, 0.5, 0.0, cv::BORDER_REPLICATE);
  // first derivatives for isotropic fallback
  cv::Sobel(Hf, gradX, CV_64F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
  cv::Sobel(Hf, gradY, CV_64F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);

  std::vector<cv::Vec2d> sigmas;
  sigmas.reserve(keypoints.size());

  for (const auto& kp : keypoints) {
    // map original keypoint coords to downsampled grid
    int u = cvRound(kp.pt.x / 8.0);
    int v = cvRound(kp.pt.y / 8.0);

    // read second-derivatives
    double fxx = dxx.at<double>(v, u);
    double fyy = dyy.at<double>(v, u);
    double fxy = dxy.at<double>(v, u);

    // build Hessian of -log H
    cv::Mat Hmat = (cv::Mat_<double>(2, 2) << -fxx, -fxy, -fxy, -fyy);
    // adaptive ridge based on trace
    double tr = std::abs(Hmat.at<double>(0, 0) + Hmat.at<double>(1, 1));
    double ridgeAdaptive = std::max(ridge, tr * 0.1);
    Hmat += ridgeAdaptive * cv::Mat::eye(2, 2, CV_64F);

    // check conditioning via determinant
    double det = Hmat.at<double>(0, 0) * Hmat.at<double>(1, 1) - Hmat.at<double>(0, 1) * Hmat.at<double>(1, 0);
    if (std::abs(det) < 1e-6) {
      // fallback to isotropic gradient-magnitude heuristic
      double gx = gradX.at<double>(v, u);
      double gy = gradY.at<double>(v, u);
      double gmag = std::sqrt(gx * gx + gy * gy);
      double iso = (gmag > eps ? 1.0 / gmag : kMaxStd);
      sigmas.emplace_back(iso, iso);
      continue;
    }

    // invert Hessian
    cv::Mat cov;
    cv::invert(Hmat, cov, cv::DECOMP_SVD);

    // extract variances and clamp
    double vxx_cov = cov.at<double>(0, 0);
    double vyy_cov = cov.at<double>(1, 1);
    double sx = (vxx_cov > 0 && std::isfinite(vxx_cov)) ? std::sqrt(vxx_cov) : 1;
    double sy = (vyy_cov > 0 && std::isfinite(vyy_cov)) ? std::sqrt(vyy_cov) : 1;

    sigmas.emplace_back(sx * kMaxStd, sy * kMaxStd);
  }

  return sigmas;
}

}  // namespace xfeat