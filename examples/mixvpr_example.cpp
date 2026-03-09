#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

#include "xfeat-cpp/place_recognition/mixvpr_onnx.h"

namespace fs = std::filesystem;
using namespace xfeat;

// ── Load sorted image paths from a directory ────────────────────────────────
static std::vector<fs::path> load_image_paths(const std::string& dir) {
  std::vector<fs::path> paths;
  for (const auto& entry : fs::directory_iterator(dir)) {
    const auto& p = entry.path();
    if (p.extension() == ".png" || p.extension() == ".jpg") {
      paths.push_back(p);
    }
  }
  std::sort(paths.begin(), paths.end());  // filenames are timestamps → chronological order
  return paths;
}

// ── Similarity plot ──────────────────────────────────────────────────────────
static cv::Mat draw_similarity_plot(const std::vector<float>& scores,
                                    int best_idx,
                                    int plot_w = 1200,
                                    int plot_h = 300) {
  cv::Mat plot(plot_h, plot_w, CV_8UC3, cv::Scalar(30, 30, 30));
  if (scores.empty()) return plot;

  const int pad_l = 60, pad_r = 20, pad_t = 20, pad_b = 40;
  int inner_w = plot_w - pad_l - pad_r;
  int inner_h = plot_h - pad_t - pad_b;

  float min_s = *std::min_element(scores.begin(), scores.end());
  float max_s = *std::max_element(scores.begin(), scores.end());
  float range = (max_s - min_s) < 1e-6f ? 1.0f : (max_s - min_s);
  int n = static_cast<int>(scores.size());

  auto score_to_y = [&](float s) -> int {
    float t = (s - min_s) / range;
    return pad_t + inner_h - static_cast<int>(t * inner_h);
  };
  auto idx_to_x = [&](int i) -> int {
    return pad_l + static_cast<int>(i * (inner_w - 1.0f) / std::max(n - 1, 1));
  };

  // Horizontal grid lines + Y-axis labels
  for (int g = 0; g <= 4; ++g) {
    int y = pad_t + g * inner_h / 4;
    cv::line(plot, {pad_l, y}, {pad_l + inner_w, y}, cv::Scalar(60, 60, 60), 1);
    float val = max_s - g * range / 4.0f;
    cv::putText(plot, cv::format("%.2f", val), {2, y + 4}, cv::FONT_HERSHEY_SIMPLEX, 0.35,
                cv::Scalar(160, 160, 160), 1);
  }

  // X-axis label
  cv::putText(plot, "DB image index (downsampled)", {pad_l + inner_w / 2 - 80, plot_h - 5},
              cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(160, 160, 160), 1);

  // Best-match vertical marker
  if (best_idx >= 0 && best_idx < n) {
    int bx = idx_to_x(best_idx);
    cv::line(plot, {bx, pad_t}, {bx, pad_t + inner_h}, cv::Scalar(0, 0, 220), 2);
    cv::putText(plot, cv::format("best=%d", best_idx), {std::max(pad_l, bx - 30), pad_t + 14},
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(80, 80, 255), 1);
  }

  // Score line
  for (int i = 0; i + 1 < n; ++i) {
    cv::line(plot, {idx_to_x(i), score_to_y(scores[i])}, {idx_to_x(i + 1), score_to_y(scores[i + 1])},
             cv::Scalar(0, 200, 100), 2);
  }

  // Best-match dot
  if (best_idx >= 0 && best_idx < n)
    cv::circle(plot, {idx_to_x(best_idx), score_to_y(scores[best_idx])}, 5, cv::Scalar(0, 0, 255), -1);

  cv::rectangle(plot, {pad_l, pad_t}, {pad_l + inner_w, pad_t + inner_h}, cv::Scalar(100, 100, 100), 1);
  return plot;
}

int main(int argc, char* argv[]) {
  std::string model_path =
      (argc > 4) ? argv[4] : "/workspaces/src/xfeat-cpp/onnx_model/mixvpr_resnet50_4096d.onnx";
  std::string query_dir = (argc > 2) ? argv[1]
                                     : "/datasets_extra/graco/aerial-05-40m_images/camera_left_image_raw";
  std::string db_dir = (argc > 3) ? argv[2]
                                  : "/datasets_extra/graco/aerial-07-25m_images/camera_left_image_raw";
  int downsample = (argc > 3) ? std::stoi(argv[3]) : 5;  // keep every Nth DB image

  std::cout << "Model:        " << model_path << "\n";
  std::cout << "Query dir:    " << query_dir << "\n";
  std::cout << "DB dir:       " << db_dir << "\n";
  std::cout << "Downsample:   every " << downsample << "-th DB image\n\n";

  // ── Load image paths ────────────────────────────────────────────────────────
  auto query_paths = load_image_paths(query_dir);
  auto db_paths_all = load_image_paths(db_dir);

  if (query_paths.empty()) { std::cerr << "No images in query dir: " << query_dir << "\n"; return 1; }
  if (db_paths_all.empty()) { std::cerr << "No images in DB dir: " << db_dir << "\n"; return 1; }

  std::cout << "Query images: " << query_paths.size() << "\n";
  std::cout << "DB images:    " << db_paths_all.size() << " → " << (db_paths_all.size() / downsample + 1)
            << " after downsampling\n\n";

  // ── Downsample DB paths (aerial-07) ─────────────────────────────────────────
  std::vector<fs::path> db_paths;
  for (size_t i = 0; i < db_paths_all.size(); i += downsample) db_paths.push_back(db_paths_all[i]);

  // ── Self-test DB paths (aerial-05 downsampled) ───────────────────────────────
  std::vector<fs::path> self_db_paths;
  for (size_t i = 0; i < query_paths.size(); i += downsample) self_db_paths.push_back(query_paths[i]);

  // ── Init MixVPR ──────────────────────────────────────────────────────────────
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mixvpr-example");
  MixVPRONNX::Params params;
  params.model_path = model_path;
  params.use_gpu = true;

  std::cout << "Loading MixVPR model...\n";
  MixVPRONNX model(env, params);

  // Warmup with the first query image
  model.infer(cv::imread(query_paths[0].string(), cv::IMREAD_COLOR));

  // Helper to extract descriptors for a set of paths
  auto extract_descs = [&](const std::vector<fs::path>& paths, const std::string& tag) {
    std::cout << "Extracting " << paths.size() << " descriptors [" << tag << "]...\n";
    std::vector<cv::Mat> descs;
    descs.reserve(paths.size());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < paths.size(); ++i) {
      cv::Mat frame = cv::imread(paths[i].string(), cv::IMREAD_COLOR);
      if (frame.empty()) { std::cerr << "  Warning: failed to read " << paths[i] << "\n"; continue; }
      descs.push_back(model.infer(frame));
      if ((i + 1) % 50 == 0) std::cout << "  " << (i + 1) << " / " << paths.size() << "\n";
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "  Done: " << ms << " ms total  (" << ms / descs.size() << " ms/frame)\n\n";
    return descs;
  };

  // ── Precompute both DBs ───────────────────────────────────────────────────────
  auto db_descs = extract_descs(db_paths, "aerial-07");
  auto self_db_descs = extract_descs(self_db_paths, "aerial-05 self");

  // ── Helpers ───────────────────────────────────────────────────────────────────
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> qdist(0, static_cast<int>(query_paths.size()) - 1);

  const int thumb_h = 320, thumb_w = 480;
  const int plot_w = std::max(thumb_w * 2, 1200);

  auto make_thumb = [&](const cv::Mat& img, const std::string& label) {
    cv::Mat thumb;
    cv::resize(img, thumb, cv::Size(thumb_w, thumb_h));
    cv::putText(thumb, label, {8, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 3);
    cv::putText(thumb, label, {8, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    return thumb;
  };

  bool self_mode = false;

  // Run one query against the active DB and return the rendered canvas
  auto run_query = [&]() -> cv::Mat {
    const auto& active_db_paths = self_mode ? self_db_paths : db_paths;
    const auto& active_db_descs = self_mode ? self_db_descs : db_descs;
    const std::string db_name = self_mode ? "aerial-05 (self)" : "aerial-07";

    int query_idx = qdist(rng);
    cv::Mat query_frame = cv::imread(query_paths[query_idx].string(), cv::IMREAD_COLOR);
    if (query_frame.empty()) {
      std::cerr << "Failed to read query image: " << query_paths[query_idx] << "\n";
      return {};
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    cv::Mat query_desc = model.infer(query_frame);
    double q_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

    std::vector<float> similarities(active_db_descs.size());
    for (size_t i = 0; i < active_db_descs.size(); ++i)
      similarities[i] = static_cast<float>(query_desc.dot(active_db_descs[i]));

    int best_idx =
        static_cast<int>(std::max_element(similarities.begin(), similarities.end()) - similarities.begin());

    std::cout << "[" << (self_mode ? "SELF" : "CROSS") << "] "
              << "Query [" << query_idx << "] " << query_paths[query_idx].filename().string()
              << "  (" << q_ms << " ms)"
              << "  →  best DB[" << best_idx << "] " << active_db_paths[best_idx].filename().string()
              << "  score=" << similarities[best_idx] << "\n";

    cv::Mat best_frame = cv::imread(active_db_paths[best_idx].string(), cv::IMREAD_COLOR);

    cv::Mat thumb_query = make_thumb(query_frame, "Query aerial-05 [" + std::to_string(query_idx) + "]");
    cv::Mat thumb_best = make_thumb(
        best_frame, "Best " + db_name + " [" + std::to_string(best_idx * downsample) + "] " +
                        cv::format("score=%.3f", similarities[best_idx]));

    cv::Mat top_row;
    cv::hconcat(thumb_query, thumb_best, top_row);
    if (top_row.cols < plot_w) {
      cv::Mat pad(top_row.rows, plot_w - top_row.cols, CV_8UC3, cv::Scalar(30, 30, 30));
      cv::hconcat(top_row, pad, top_row);
    }

    cv::Mat plot = draw_similarity_plot(similarities, best_idx, plot_w, 300);

    // Title bar — highlight self-test mode in orange
    cv::Mat title_bar(32, plot_w, CV_8UC3,
                      self_mode ? cv::Scalar(0, 40, 80) : cv::Scalar(20, 20, 20));
    std::string mode_tag = self_mode ? " [SELF-TEST: a5 vs a5]" : " [CROSS: a5 vs a7]";
    cv::putText(title_bar,
                "MixVPR | query: aerial-05 left | DB: " + db_name + " (every " +
                    std::to_string(downsample) + "-th)" + mode_tag +
                    "  [R/Spc] new query  [S] toggle mode  [Q/Esc] quit",
                {8, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1);

    cv::Mat canvas;
    cv::vconcat(title_bar, top_row, canvas);
    cv::vconcat(canvas, plot, canvas);
    return canvas;
  };

  // ── Interactive loop ──────────────────────────────────────────────────────────
  std::cout << "Press [R]/[Space] new query | [S] toggle self-test mode | [Q]/[Esc] quit\n\n";
  cv::namedWindow("MixVPR Place Recognition", cv::WINDOW_NORMAL);

  cv::Mat canvas = run_query();
  if (!canvas.empty()) cv::imshow("MixVPR Place Recognition", canvas);

  while (true) {
    int key = cv::waitKey(0) & 0xFF;
    if (key == 'q' || key == 'Q' || key == 27 /* Esc */) break;
    if (key == 'r' || key == 'R' || key == ' ') {
      canvas = run_query();
      if (!canvas.empty()) cv::imshow("MixVPR Place Recognition", canvas);
    }
    if (key == 's' || key == 'S') {
      self_mode = !self_mode;
      std::cout << "Switched to " << (self_mode ? "SELF-TEST (a5 vs a5)" : "CROSS (a5 vs a7)") << " mode\n";
      canvas = run_query();
      if (!canvas.empty()) cv::imshow("MixVPR Place Recognition", canvas);
    }
  }

  if (!canvas.empty()) {
    cv::imwrite("mixvpr_result.png", canvas);
    std::cout << "Saved mixvpr_result.png\n";
  }

  return 0;
}
