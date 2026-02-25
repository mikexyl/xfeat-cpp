#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "xfeat-cpp/place_recognition/patchnetvlad_matcher.h"
#include "xfeat-cpp/place_recognition/patchnetvlad_onnx.h"

using namespace xfeat;

int main(int argc, char* argv[]) {
  std::filesystem::path model_dir = (argc > 1) ? argv[1] : "onnx_model";
  std::filesystem::path img_dir = (argc > 2) ? argv[2] : "examples";

  std::string model_path = (model_dir / "patchnetvlad_trt.onnx").string();
  std::string db_img_path = (img_dir / "tokyo_db.png").string();
  std::string query_img_path = (img_dir / "tokyo_query.jpg").string();

  // Load images
  cv::Mat db_img = cv::imread(db_img_path, cv::IMREAD_COLOR);
  cv::Mat query_img = cv::imread(query_img_path, cv::IMREAD_COLOR);

  if (db_img.empty() || query_img.empty()) {
    std::cerr << "Failed to load images from " << img_dir << std::endl;
    std::cerr << "  DB:    " << db_img_path << std::endl;
    std::cerr << "  Query: " << query_img_path << std::endl;
    return 1;
  }

  std::cout << "DB image:    " << db_img.cols << "x" << db_img.rows << std::endl;
  std::cout << "Query image: " << query_img.cols << "x" << query_img.rows << std::endl;

  // Initialise ONNX Runtime environment
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "patchnetvlad-example");

  // Build model
  PatchNetVLADONNX::Params params;
  params.model_path = model_path;
  params.use_gpu = true;

  std::cout << "\nLoading model from " << model_path << " ..." << std::endl;
  PatchNetVLADONNX model(env, params);
  PatchNetVLADMatcher matcher;

  // Warmup pass to ensure CUDA kernels are compiled
  std::cout << "\nWarming up..." << std::endl;
  model.extract(db_img);

  // ── DB image ────────────────────────────────────────────────────────────────
  std::cout << "\n[DB image]" << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto db_feat = model.extract(db_img);
  auto t1 = std::chrono::high_resolution_clock::now();
  double db_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << "  global_desc: " << db_feat.global_desc.size() << " (CV_32F)" << std::endl;
  std::cout << "  local_descs: " << db_feat.local_descs.size() << " scales";
  for (size_t i = 0; i < db_feat.local_descs.size(); ++i)
    std::cout << "  [" << db_feat.local_descs[i].rows << "x" << db_feat.local_descs[i].cols << "]";
  std::cout << std::endl;
  std::cout << "  extraction time: " << db_ms << " ms" << std::endl;

  // ── Query image ─────────────────────────────────────────────────────────────
  std::cout << "\n[Query image]" << std::endl;
  auto t2 = std::chrono::high_resolution_clock::now();
  auto query_feat = model.extract(query_img);
  auto t3 = std::chrono::high_resolution_clock::now();
  double query_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

  std::cout << "  global_desc: " << query_feat.global_desc.size() << " (CV_32F)" << std::endl;
  std::cout << "  extraction time: " << query_ms << " ms" << std::endl;

  // ── Global similarity ───────────────────────────────────────────────────────
  // Both descriptors are L2-normalised → dot product == cosine similarity
  float global_sim = static_cast<float>(db_feat.global_desc.dot(query_feat.global_desc));
  std::cout << "\nGlobal cosine similarity (query vs DB): " << global_sim << std::endl;

  // ── Local patch RANSAC re-ranking ───────────────────────────────────────────
  std::cout << "\nRunning local patch RANSAC re-ranking..." << std::endl;
  auto t4 = std::chrono::high_resolution_clock::now();
  auto result = matcher.match(query_feat, db_feat);
  auto t5 = std::chrono::high_resolution_clock::now();
  double match_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

  std::cout << "  re-ranking time: " << match_ms << " ms" << std::endl;
  std::cout << "\nMatch result:" << std::endl;
  std::cout << "  Weighted inlier score: " << result.score << std::endl;
  std::cout << "  Global similarity:     " << result.global_sim << std::endl;
  for (size_t i = 0; i < result.inlier_counts.size(); ++i)
    std::cout << "  Scale " << i << " RANSAC inliers: " << result.inlier_counts[i] << std::endl;

  return 0;
}
