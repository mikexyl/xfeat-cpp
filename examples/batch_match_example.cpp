// filepath: examples/batch_match_example.cpp
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

#include "xfeat-cpp/faiss_database.h"
#include "xfeat-cpp/netvlad_onnx.h"
#include "xfeat-cpp/xfeat_netvlad_onnx.h"
#include "xfeat-cpp/xfeat_onnx.h"
// Include LighterGlue headers as needed
// #include "xfeat-cpp/lighterglue_onnx.h"

using namespace xfeat;

int main(int argc, char* argv[]) {
  std::filesystem::path image_folder((argc > 1) ? argv[1] : "image");
  const std::string image_resolution = "640x480";
  constexpr int max_kpts = 500;
  const int num_images = 100;

  std::filesystem::path xfeat_model_folder = (argc > 2) ? argv[2] : "onnx_model";
  std::filesystem::path xfeat_model_path = xfeat_model_folder / ("xfeat_" + image_resolution + ".onnx");
  std::filesystem::path interp_bilinear_path =
      xfeat_model_folder / ("interpolator_bilinear_" + image_resolution + ".onnx");
  std::filesystem::path interp_bicubic_path =
      xfeat_model_folder / ("interpolator_bicubic_" + image_resolution + ".onnx");
  std::filesystem::path interp_nearest_path =
      xfeat_model_folder / ("interpolator_nearest_" + image_resolution + ".onnx");
  std::filesystem::path lighterglue_model_path =
      xfeat_model_folder / ("lg_" + image_resolution + "_" + std::to_string(max_kpts) + ".onnx");

  std::filesystem::path netvlad_model_path = xfeat_model_folder / ("netvlad.onnx");

  // List all images in the folder
  std::vector<std::filesystem::path> image_paths;
  for (const auto& entry : std::filesystem::directory_iterator(image_folder)) {
    if (entry.is_regular_file()) {
      auto ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
        image_paths.push_back(entry.path());
      }
    }
  }
  // Randomly select 20 images
  std::random_device rd;
  std::mt19937 g(rd());
  if (image_paths.size() > num_images) {
    std::shuffle(image_paths.begin(), image_paths.end(), g);
    image_paths.resize(num_images);
  }

  // Create a single ONNX Runtime environment
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "xfeat-shared-env");

  // Load models
  auto lighterglue_model = std::make_unique<LighterGlueOnnx>(env, lighterglue_model_path.string(), true);
  XFeatONNX xfeat_onnx(env,
                       XFeatONNX::Params{
                           .xfeat_path = xfeat_model_path.string(),
                           .interp_bilinear_path = interp_bilinear_path.string(),
                           .interp_bicubic_path = interp_bicubic_path.string(),
                           .interp_nearest_path = interp_nearest_path.string(),
                           .use_gpu = true,
                           .matcher_type = MatcherType::LIGHTERGLUE,
                       },
                       std::move(lighterglue_model));

  std::cout << netvlad_model_path.string() << std::endl;
  xfeat::NetVLADONNX netvlad_onnx(env, netvlad_model_path.string());
  xfeat::HeadNetVLADONNX head_netvlad_onnx(env, (xfeat_model_folder / ("xfeat_nv.onnx")).string());

  auto extract_netvlad_desc = [&](const cv::Mat& img) {
    cv::Mat M1, x_prep;
    auto start = std::chrono::high_resolution_clock::now();
    xfeat_onnx.detect_and_compute(img, max_kpts, nullptr, &M1, &x_prep);
    auto t1 = std::chrono::high_resolution_clock::now();
    cv::Mat dense_desc = head_netvlad_onnx.run(M1, x_prep);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto result = netvlad_onnx.infer(dense_desc);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "detect_and_compute: " << std::chrono::duration<double, std::milli>(t1 - start).count() << " ms\n";
    std::cout << "head_netvlad_onnx.run: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
    std::cout << "netvlad_onnx.infer: " << std::chrono::duration<double, std::milli>(end - t2).count() << " ms\n";
    return result;
  };

  // Pick query image separately from the folder (not from the samples)
  std::vector<std::filesystem::path> all_image_paths;
  for (const auto& entry : std::filesystem::directory_iterator(image_folder)) {
    if (entry.is_regular_file()) {
      auto ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
        all_image_paths.push_back(entry.path());
      }
    }
  }
  if (all_image_paths.empty()) {
    std::cerr << "No images found in folder!" << std::endl;
    return 1;
  }
  std::uniform_int_distribution<size_t> dist_query(0, all_image_paths.size() - 1);
  size_t query_idx = dist_query(g);
  auto query_path = all_image_paths[query_idx];
  cv::Mat query_img = cv::imread(query_path, cv::IMREAD_GRAYSCALE);
  if (query_img.empty()) {
    std::cerr << "Error loading query image: " << query_path << std::endl;
    return 1;
  }
  auto vlad_query = extract_netvlad_desc(query_img);

  // --- Get the image 10 frames before the query image (if possible) ---
  float vlad_threshold = 0.0f;
  if (query_idx >= 10) {
    auto before10_path = all_image_paths[query_idx - 10];
    cv::Mat before10_img = cv::imread(before10_path, cv::IMREAD_GRAYSCALE);
    if (!before10_img.empty()) {
      auto vlad_before10 = extract_netvlad_desc(before10_img);
      // faiss::fvec_L2sqr(a, b, d)
      vlad_threshold = faiss::fvec_L2sqr(vlad_query[0].data(), vlad_before10[0].data(), vlad_query[0].size());
      std::cout << "VLAD distance to image 10 frames before: " << vlad_threshold << std::endl;
    }
  }

  // Randomly select 20 sample images (excluding the query image)
  std::vector<std::filesystem::path> sample_paths = all_image_paths;
  sample_paths.erase(sample_paths.begin() + query_idx);
  std::shuffle(sample_paths.begin(), sample_paths.end(), g);
  sample_paths.resize(num_images);

  // Print the name of the picked query image
  std::cout << "Query image: " << query_path.filename() << std::endl;

  // Print the names of the sampled images
  std::cout << "Sample images:" << std::endl;
  for (const auto& p : sample_paths) {
    std::cout << "  " << p.filename() << std::endl;
  }
  // Match query image with all samples using NetVLAD + Faiss GPU
  const float min_score = 0.3f;

  // Prepare VLAD descriptors for Faiss
  std::vector<std::vector<float>> vlad_targets(sample_paths.size());
  int vlad_dim = -1;

  for (size_t j = 0; j < sample_paths.size(); ++j) {
    cv::Mat target_img = cv::imread(sample_paths[j], cv::IMREAD_GRAYSCALE);
    if (target_img.empty()) {
      std::cerr << "Error loading image: " << sample_paths[j] << std::endl;
      continue;
    }
    auto vlad_target = extract_netvlad_desc(target_img);
    if (vlad_dim < 0) vlad_dim = vlad_target[0].size();
    vlad_targets[j] = vlad_target[0];  // Assuming single descriptor per image
  }

  std::vector<float> vlad_query_vec = vlad_query[0];

  // === FAISS index loading and transfer to GPU ===
  std::filesystem::path faiss_index_path = xfeat_model_folder / "faiss_ivfflat.index.bin";
  xfeat::FaissDatabase faiss_db(faiss_index_path.string());

  // === FLATTEN and ADD VLAD TARGETS TO INDEX ===
  int n = static_cast<int>(vlad_targets.size());
  std::vector<float> flat_targets(n * vlad_dim);
  for (int i = 0; i < n; ++i) {
    std::copy(vlad_targets[i].begin(), vlad_targets[i].end(), flat_targets.begin() + i * vlad_dim);
  }
  cv::Mat flat_targets_mat(n, vlad_dim, CV_32F, flat_targets.data());
  faiss_db.add(flat_targets_mat);

  // === SEARCH TOP-K ===
  int k = 30;
  std::vector<faiss::idx_t> faiss_labels(k);
  std::vector<float> faiss_distances(k);
  cv::Mat vlad_query_mat(1, vlad_dim, CV_32F, vlad_query_vec.data());

  // Time the search function
  auto search_start = std::chrono::high_resolution_clock::now();
  faiss_db.search(vlad_query_mat, k, faiss_labels, faiss_distances);
  auto search_end = std::chrono::high_resolution_clock::now();
  std::cout << "Faiss search time: " << std::chrono::duration<double, std::milli>(search_end - search_start).count()
            << " ms" << std::endl;

  // print matches and their distances
  for (size_t i = 0; i < faiss_labels.size(); ++i) {
    if (vlad_threshold > 0.0f && faiss_distances[i] > vlad_threshold) continue;
    auto label = faiss_labels[i];
    std::cout << "Match label: " << label << std::endl;
  }
  std::cout << "Faiss distances: ";
  for (size_t i = 0; i < faiss_distances.size(); ++i) {
    if (vlad_threshold > 0.0f && faiss_distances[i] > vlad_threshold) continue;
    std::cout << faiss_distances[i] << " ";
  }
  std::cout << std::endl;

  // === LighterGlue matching for each matched image ===
  // Extract DetectionResult for query image
  auto query_det = xfeat_onnx.detect_and_compute(query_img, max_kpts);
  std::array<float, 2> query_img_size = {static_cast<float>(query_img.cols), static_cast<float>(query_img.rows)};

  std::vector<int> lighterglue_match_counts(k, 0);
  for (size_t match_idx = 0; match_idx < faiss_labels.size(); ++match_idx) {
    faiss::idx_t idx = faiss_labels[match_idx];
    if (idx < 0 || idx >= static_cast<faiss::idx_t>(sample_paths.size())) continue;
    cv::Mat match_img_gray = cv::imread(sample_paths[idx], cv::IMREAD_GRAYSCALE);
    if (match_img_gray.empty()) continue;
    auto match_det = xfeat_onnx.detect_and_compute(match_img_gray, max_kpts);
    std::array<float, 2> match_img_size = {static_cast<float>(match_img_gray.cols),
                                           static_cast<float>(match_img_gray.rows)};
    auto [idx0, idx1, _, __] = xfeat_onnx.match(query_det, match_det, query_img);
    lighterglue_match_counts[match_idx] = idx0.rows;
    std::cout << "LighterGlue matches for Faiss match " << match_idx << " (label " << idx << "): " << idx0.size()
              << std::endl;
  }

  // Visualize the query and sampled images in one big image grid
  int grid_cols = 5;
  int num_matched = 0;
  for (size_t match_idx = 0; match_idx < faiss_labels.size(); ++match_idx) {
    if (faiss_labels[match_idx] < 0 || faiss_labels[match_idx] >= static_cast<faiss::idx_t>(sample_paths.size())) continue;
    if (vlad_threshold > 0.0f && faiss_distances[match_idx] > vlad_threshold) continue;
    ++num_matched;
  }
  int grid_offset = 2; // rows for query and before10
  int grid_rows = ((num_matched + grid_cols - 1) / grid_cols) + grid_offset;
  int thumb_w = 160, thumb_h = 88;
  cv::Mat grid_img = cv::Mat::zeros(grid_rows * thumb_h, grid_cols * thumb_w, CV_8UC3);

  // Place query image and before10 image at the top left
  cv::Mat query_img_color = cv::imread(query_path, cv::IMREAD_COLOR);
  cv::Mat before10_img_color;
  if (query_idx >= 10) {
    before10_img_color = cv::imread(all_image_paths[query_idx - 10], cv::IMREAD_COLOR);
  }
  if (!query_img_color.empty()) {
    cv::Mat query_thumb;
    cv::resize(query_img_color, query_thumb, cv::Size(thumb_w, thumb_h));
    query_thumb.copyTo(grid_img(cv::Rect(0, 0, thumb_w, thumb_h)));
    cv::rectangle(grid_img, cv::Rect(0, 0, thumb_w, thumb_h), cv::Scalar(0, 0, 255), 2);
    cv::putText(grid_img, "Query", cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2);
  }
  if (!before10_img_color.empty()) {
    cv::Mat before10_thumb;
    cv::resize(before10_img_color, before10_thumb, cv::Size(thumb_w, thumb_h));
    before10_thumb.copyTo(grid_img(cv::Rect(thumb_w, 0, thumb_w, thumb_h)));
    cv::rectangle(grid_img, cv::Rect(thumb_w, 0, thumb_w, thumb_h), cv::Scalar(255, 0, 0), 2);
    cv::putText(grid_img, "Before-10", cv::Point(thumb_w + 5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0), 2);
    // Print the before10 distance on the image
    char dist_text[64];
    snprintf(dist_text, sizeof(dist_text), "Dist: %.2f", vlad_threshold);
    cv::putText(grid_img, dist_text, cv::Point(thumb_w + 5, thumb_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0), 2);
  }

  // Only show the matched images (top-k matches)
  int match_count = 0;
  for (size_t match_idx = 0; match_idx < faiss_labels.size(); ++match_idx) {
    faiss::idx_t idx = faiss_labels[match_idx];
    if (idx < 0 || idx >= static_cast<faiss::idx_t>(sample_paths.size())) continue;
    if (vlad_threshold > 0.0f && faiss_distances[match_idx] > vlad_threshold) continue;
    cv::Mat img = cv::imread(sample_paths[idx], cv::IMREAD_COLOR);
    if (img.empty()) continue;
    cv::Mat thumb;
    cv::resize(img, thumb, cv::Size(thumb_w, thumb_h));
    std::string text = "LG: " + std::to_string(lighterglue_match_counts[match_idx]);
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.7;
    int thickness = 2;
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, font, font_scale, thickness, &baseline);
    cv::Point text_org(5, thumb_h - 10);
    cv::putText(thumb, text, text_org, font, font_scale, cv::Scalar(0, 255, 255), thickness, cv::LINE_AA);
    // Print the distance on the image
    char match_dist_text[64];
    snprintf(match_dist_text, sizeof(match_dist_text), "Dist: %.2f", faiss_distances[match_idx]);
    cv::putText(thumb, match_dist_text, cv::Point(5, 30), font, 0.7, cv::Scalar(255,255,0), 2);
    int row = (match_count / grid_cols) + grid_offset;
    int col = match_count % grid_cols;
    if ((row + 1) * thumb_h <= grid_img.rows && (col + 1) * thumb_w <= grid_img.cols) {
      thumb.copyTo(grid_img(cv::Rect(col * thumb_w, row * thumb_h, thumb_w, thumb_h)));
      cv::Scalar rect_color = (vlad_threshold > 0.0f && faiss_distances[match_idx] <= vlad_threshold) ? cv::Scalar(0,0,255) : cv::Scalar(0,255,0);
      cv::rectangle(grid_img, cv::Rect(col * thumb_w, row * thumb_h, thumb_w, thumb_h), rect_color, 2);
    }
    ++match_count;
  }
  cv::imwrite("query_and_matches_grid.png", grid_img);  // Save the grid image to file
  cv::imshow("Query + Sampled Images Grid", grid_img);
  cv::waitKey(0);

  return 0;
}
