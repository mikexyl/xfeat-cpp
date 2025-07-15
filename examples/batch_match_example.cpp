// filepath: examples/batch_match_example.cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

#include "xfeat-cpp/helpers.h"
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
  const int num_images = 20;

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
  if (image_paths.size() < num_images) {
    std::cerr << "Not enough images in folder!" << std::endl;
    return 1;
  }

  // Randomly select 20 images
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(image_paths.begin(), image_paths.end(), g);
  image_paths.resize(num_images);

  // Load models
  auto lighterglue_model = std::make_unique<LighterGlueOnnx>(lighterglue_model_path.string(), true);
  XFeatONNX xfeat_onnx(
      XFeatONNX::Params{
          .xfeat_path = xfeat_model_path.string(),
          .interp_bilinear_path = interp_bilinear_path.string(),
          .interp_bicubic_path = interp_bicubic_path.string(),
          .interp_nearest_path = interp_nearest_path.string(),
          .use_gpu = true,
          .matcher_type = MatcherType::LIGHTERGLUE,
      },
      std::move(lighterglue_model));

  xfeat::NetVLADONNX netvlad_onnx(netvlad_model_path.string());
  xfeat::HeadNetVLADONNX head_netvlad_onnx(xfeat_model_folder / ("xfeat_nv.onnx"));

  auto extract_netvlad_desc = [&](const cv::Mat& img) {
    cv::Mat M1, x_prep;
    xfeat_onnx.detect_and_compute(img, max_kpts, nullptr, &M1, &x_prep);
    cv::Mat dense_desc = head_netvlad_onnx.run(M1, x_prep);
    return netvlad_onnx.infer(dense_desc);
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

  // Match query image with all samples using NetVLAD + brute-force
  std::vector<float> vlad_scores(sample_paths.size(), 0.0f);
  int best_idx = -1;
  float best_score = std::numeric_limits<float>::max();
  const float min_score = 0.3f;
  for (size_t j = 0; j < sample_paths.size(); ++j) {
    cv::Mat target_img = cv::imread(sample_paths[j], cv::IMREAD_GRAYSCALE);
    if (target_img.empty()) {
      std::cerr << "Error loading image: " << sample_paths[j] << std::endl;
      continue;
    }
    auto vlad_target = extract_netvlad_desc(target_img);
    float vlad_score = cv::norm(cv::Mat(vlad_query[0]), cv::Mat(vlad_target[0]), cv::NORM_L2);
    vlad_scores[j] = vlad_score;
    if (vlad_score < best_score) {
      best_score = vlad_score;
      best_idx = j;
    }
    if (vlad_score >= min_score) {
      std::cout << "MATCH: NetVLAD L2 distance (query <-> " << sample_paths[j].filename() << "): " << vlad_score << std::endl;
    } else {
      std::cout << "NO MATCH: NetVLAD L2 distance (query <-> " << sample_paths[j].filename() << "): " << vlad_score << std::endl;
    }
  }

  // Visualize the query and sampled images in one big image grid
  int grid_cols = 5;
  int grid_rows = (num_images + grid_cols - 1) / grid_cols;
  int thumb_w = 160, thumb_h = 88;
  int total_rows = grid_rows + 1;
  cv::Mat grid_img = cv::Mat::zeros(total_rows * thumb_h, grid_cols * thumb_w, CV_8UC3);

  // Place query image at the top left
  cv::Mat query_img_color = cv::imread(query_path, cv::IMREAD_COLOR);
  if (!query_img_color.empty()) {
    cv::Mat query_thumb;
    cv::resize(query_img_color, query_thumb, cv::Size(thumb_w, thumb_h));
    query_thumb.copyTo(grid_img(cv::Rect(0, 0, thumb_w, thumb_h)));
    cv::rectangle(grid_img, cv::Rect(0, 0, thumb_w, thumb_h), cv::Scalar(0, 0, 255), 2);
  }

  // Place sampled images below query image
  for (size_t idx = 0; idx < sample_paths.size(); ++idx) {
    cv::Mat img = cv::imread(sample_paths[idx], cv::IMREAD_COLOR);
    if (img.empty()) continue;
    cv::Mat thumb;
    cv::resize(img, thumb, cv::Size(thumb_w, thumb_h));
    int row = (idx / grid_cols) + 1;
    int col = idx % grid_cols;
    thumb.copyTo(grid_img(cv::Rect(col * thumb_w, row * thumb_h, thumb_w, thumb_h)));
    if (idx == best_idx) {
      cv::rectangle(grid_img, cv::Rect(col * thumb_w, row * thumb_h, thumb_w, thumb_h), cv::Scalar(0, 255, 0), 3);
    }
  }
  cv::imshow("Query + Sampled Images Grid", grid_img);
  cv::waitKey(0);

  return 0;
}
