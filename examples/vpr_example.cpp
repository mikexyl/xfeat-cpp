#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

#include "xfeat-cpp/lighterglue_cv.h"
#include "xfeat-cpp/place_recognition/jist_onnx.h"
#include "xfeat-cpp/place_recognition/mixvpr_onnx.h"
#include "xfeat-cpp/place_recognition/place_recognizer.h"
#include "xfeat-cpp/xfeat_onnx.h"

namespace fs = std::filesystem;
using namespace xfeat;

// ── Layout constants ─────────────────────────────────────────────────────────
static constexpr int kThumbH = 320, kThumbW = 480;
static constexpr int kTitleH = 32, kPlotH = 300;
static constexpr int kPadL = 60, kPadR = 20, kPadT = 20;

static void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " <model-type> <model-path> <query-dir> <db-dir> [downsample] [xfeat-dir]\n"
            << "  model-type : mixvpr | jist\n"
            << "  downsample : keep every N-th DB image (default 5)\n"
            << "  xfeat-dir  : folder with XFeat/LighterGlue ONNX models (default onnx_model)\n";
}

// ── Load sorted image paths from a directory ────────────────────────────────
static std::vector<fs::path> load_image_paths(const std::string& dir) {
  std::vector<fs::path> paths;
  for (const auto& entry : fs::directory_iterator(dir)) {
    const auto& p = entry.path();
    if (p.extension() == ".png" || p.extension() == ".jpg") paths.push_back(p);
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

// ── Similarity plot ──────────────────────────────────────────────────────────
// best_idx  → red   vertical line + filled circle
// sel_idx   → yellow vertical line + filled circle (shown only when ≠ best_idx)
static cv::Mat draw_similarity_plot(const std::vector<float>& scores, int best_idx, int sel_idx,
                                    int plot_w = 1200, int plot_h = kPlotH) {
  cv::Mat plot(plot_h, plot_w, CV_8UC3, cv::Scalar(30, 30, 30));
  if (scores.empty()) return plot;

  const int pad_b = 40;
  int inner_w = plot_w - kPadL - kPadR;
  int inner_h = plot_h - kPadT - pad_b;
  int n = static_cast<int>(scores.size());

  float min_s = *std::min_element(scores.begin(), scores.end());
  float max_s = *std::max_element(scores.begin(), scores.end());
  float range = (max_s - min_s) < 1e-6f ? 1.0f : (max_s - min_s);

  auto score_to_y = [&](float s) {
    return kPadT + inner_h - static_cast<int>((s - min_s) / range * inner_h);
  };
  auto idx_to_x = [&](int i) {
    return kPadL + static_cast<int>(i * (inner_w - 1.0f) / std::max(n - 1, 1));
  };

  // Grid + Y labels
  for (int g = 0; g <= 4; ++g) {
    int y = kPadT + g * inner_h / 4;
    cv::line(plot, {kPadL, y}, {kPadL + inner_w, y}, cv::Scalar(60, 60, 60), 1);
    float val = max_s - g * range / 4.0f;
    cv::putText(plot, cv::format("%.2f", val), {2, y + 4}, cv::FONT_HERSHEY_SIMPLEX, 0.35,
                cv::Scalar(160, 160, 160), 1);
  }
  cv::putText(plot, "DB descriptor index  [click to inspect]",
              {kPadL + inner_w / 2 - 100, plot_h - 5}, cv::FONT_HERSHEY_SIMPLEX, 0.4,
              cv::Scalar(160, 160, 160), 1);

  // Score polyline
  for (int i = 0; i + 1 < n; ++i)
    cv::line(plot, {idx_to_x(i), score_to_y(scores[i])}, {idx_to_x(i + 1), score_to_y(scores[i + 1])},
             cv::Scalar(0, 200, 100), 2);

  // Best marker (red)
  if (best_idx >= 0 && best_idx < n) {
    int bx = idx_to_x(best_idx);
    cv::line(plot, {bx, kPadT}, {bx, kPadT + inner_h}, cv::Scalar(0, 0, 220), 2);
    cv::circle(plot, {bx, score_to_y(scores[best_idx])}, 6, cv::Scalar(0, 0, 255), -1);
    cv::putText(plot, cv::format("best=%d", best_idx), {std::max(kPadL, bx - 30), kPadT + 14},
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(80, 80, 255), 1);
  }

  // Selected marker (yellow), only when different from best
  if (sel_idx >= 0 && sel_idx < n && sel_idx != best_idx) {
    int sx = idx_to_x(sel_idx);
    cv::line(plot, {sx, kPadT}, {sx, kPadT + inner_h}, cv::Scalar(0, 200, 255), 2);
    cv::circle(plot, {sx, score_to_y(scores[sel_idx])}, 6, cv::Scalar(0, 215, 255), -1);
    cv::putText(plot, cv::format("sel=%d", sel_idx), {std::max(kPadL, sx - 24), kPadT + 28},
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(0, 215, 255), 1);
  }

  cv::rectangle(plot, {kPadL, kPadT}, {kPadL + inner_w, kPadT + inner_h}, cv::Scalar(100, 100, 100), 1);
  return plot;
}

// ── State shared between render and mouse callback ───────────────────────────
struct DisplayState {
  // current query
  cv::Mat query_frame;
  int query_idx = -1;
  double query_ms = 0.0;

  // current DB view
  const std::vector<fs::path>* active_paths = nullptr;
  std::string db_name;
  bool self_mode = false;

  // computed per-query
  std::vector<float> similarities;
  int best_idx = -1;
  int selected_idx = -1;  // -1 → show best; set by mouse click

  // display config
  int plot_w = 1200;
  int downsample = 1;
  std::string win_title;
  std::string model_type;
  int seq_length = 1;
  int descriptor_dim = 0;
};

// ── Render the full canvas from current DisplayState ─────────────────────────
static cv::Mat render_canvas(const DisplayState& s) {
  int sel = (s.selected_idx >= 0) ? s.selected_idx : s.best_idx;
  if (sel < 0 || s.active_paths == nullptr) return {};

  auto make_thumb = [&](const cv::Mat& img, const std::string& label, cv::Scalar border) {
    cv::Mat thumb;
    cv::resize(img, thumb, cv::Size(kThumbW, kThumbH));
    cv::rectangle(thumb, {0, 0}, {kThumbW - 1, kThumbH - 1}, border, 3);
    cv::putText(thumb, label, {8, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 3);
    cv::putText(thumb, label, {8, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
    return thumb;
  };

  const auto& ap = *s.active_paths;
  cv::Mat sel_frame = cv::imread(ap[sel].string(), cv::IMREAD_COLOR);
  if (sel_frame.empty()) sel_frame = cv::Mat::zeros(kThumbH, kThumbW, CV_8UC3);

  bool is_selected = (sel != s.best_idx);
  cv::Scalar sel_border = is_selected ? cv::Scalar(0, 215, 255) : cv::Scalar(0, 0, 255);
  std::string sel_label = (is_selected ? "Sel " : "Best ") + s.db_name + " [" +
                          std::to_string(sel) + "] " +
                          cv::format("score=%.3f", s.similarities[sel]);

  cv::Mat thumb_query = make_thumb(s.query_frame, "Query [" + std::to_string(s.query_idx) + "]",
                                   cv::Scalar(0, 200, 100));
  cv::Mat thumb_sel = make_thumb(sel_frame, sel_label, sel_border);

  cv::Mat top_row;
  cv::hconcat(thumb_query, thumb_sel, top_row);
  if (top_row.cols < s.plot_w) {
    cv::Mat pad(top_row.rows, s.plot_w - top_row.cols, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::hconcat(top_row, pad, top_row);
  }

  cv::Mat plot = draw_similarity_plot(s.similarities, s.best_idx, sel, s.plot_w);

  cv::Mat title_bar(kTitleH, s.plot_w, CV_8UC3,
                    s.self_mode ? cv::Scalar(0, 40, 80) : cv::Scalar(20, 20, 20));
  std::string mode_tag = s.self_mode ? " [SELF-TEST]" : " [CROSS]";
  cv::putText(title_bar,
              s.model_type + " | seq=" + std::to_string(s.seq_length) +
                  " dim=" + std::to_string(s.descriptor_dim) + " | DB: " + s.db_name + " (every " +
                  std::to_string(s.downsample) + "-th)" + mode_tag +
                  "  [R/Spc] new query  [S] toggle mode  [M] match  [Q/Esc] quit",
              {8, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(200, 200, 200), 1);

  cv::Mat canvas;
  cv::vconcat(title_bar, top_row, canvas);
  cv::vconcat(canvas, plot, canvas);
  return canvas;
}

// ── Mouse callback ────────────────────────────────────────────────────────────
// Maps a left-click inside the similarity plot to a DB index, then re-renders.
// Note: coordinate space matches the canvas pixel space (window at native res).
static void on_mouse_cb(int event, int x, int y, int /*flags*/, void* userdata) {
  if (event != cv::EVENT_LBUTTONDOWN) return;
  auto* s = static_cast<DisplayState*>(userdata);
  if (s->similarities.empty()) return;

  // Check the click falls within the plot area
  int plot_y0 = kTitleH + kThumbH;
  int plot_y1 = plot_y0 + kPlotH;
  if (y < plot_y0 || y >= plot_y1) return;

  // Map x → DB index
  int inner_w = s->plot_w - kPadL - kPadR;
  int n = static_cast<int>(s->similarities.size());
  int idx = static_cast<int>((x - kPadL) * (n - 1.0f) / inner_w + 0.5f);
  idx = std::clamp(idx, 0, n - 1);

  s->selected_idx = idx;

  const auto& ap = *s->active_paths;
  std::cout << "[CLICK] selected DB[" << idx << "] " << ap[idx].filename().string()
            << "  score=" << s->similarities[idx]
            << (idx == s->best_idx ? "  ← best" : "") << "\n";

  cv::Mat canvas = render_canvas(*s);
  if (!canvas.empty()) cv::imshow(s->win_title, canvas);
}

// ── Build descriptor for one query index ─────────────────────────────────────
static cv::Mat query_descriptor(PlaceRecognizer& model, const std::vector<fs::path>& paths, int idx) {
  int seq = model.get_seq_length();
  int n = static_cast<int>(paths.size());
  std::vector<cv::Mat> imgs;
  imgs.reserve(seq);
  for (int k = 0; k < seq; ++k) {
    int j = std::clamp(idx - seq / 2 + k, 0, n - 1);
    cv::Mat img = cv::imread(paths[j].string(), cv::IMREAD_COLOR);
    if (img.empty()) img = cv::Mat::zeros(1, 1, CV_8UC3);
    imgs.push_back(img);
  }
  return model.infer(imgs);
}

// ── Extract descriptors using non-overlapping windows ───────────────────────
static std::pair<std::vector<fs::path>, std::vector<cv::Mat>> extract_descs(
    PlaceRecognizer& model, const std::vector<fs::path>& paths, const std::string& tag) {
  int seq = model.get_seq_length();
  int n = static_cast<int>(paths.size());
  std::cout << "Extracting descriptors from " << n << " images [" << tag << "]"
            << " (seq=" << seq << ", windows=" << (n / seq) << ")...\n";

  std::vector<fs::path> out_paths;
  std::vector<cv::Mat> out_descs;
  auto t0 = std::chrono::high_resolution_clock::now();

  for (int start = 0; start + seq <= n; start += seq) {
    std::vector<cv::Mat> window;
    window.reserve(seq);
    for (int k = 0; k < seq; ++k) {
      cv::Mat frame = cv::imread(paths[start + k].string(), cv::IMREAD_COLOR);
      if (frame.empty()) frame = cv::Mat::zeros(4, 4, CV_8UC3);
      window.push_back(frame);
    }
    out_paths.push_back(paths[start + seq / 2]);
    out_descs.push_back(model.infer(window));

    int done = start / seq + 1, total = n / seq;
    if (done % 50 == 0) std::cout << "  " << done << " / " << total << "\n";
  }

  double ms =
      std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
  std::cout << "  Done: " << out_descs.size() << " descriptors, " << ms << " ms total  ("
            << (out_descs.empty() ? 0.0 : ms / out_descs.size()) << " ms/desc)\n\n";
  return {out_paths, out_descs};
}

// ── Convert DetectionResult keypoints (N×2 CV_32F) to vector<KeyPoint> ──────
static std::vector<cv::KeyPoint> to_keypoints(const cv::Mat& kpts_mat) {
  std::vector<cv::KeyPoint> kpts;
  kpts.reserve(kpts_mat.rows);
  for (int i = 0; i < kpts_mat.rows; ++i)
    kpts.emplace_back(kpts_mat.at<float>(i, 0), kpts_mat.at<float>(i, 1), 1.f);
  return kpts;
}

int main(int argc, char* argv[]) {
  if (argc < 5) { print_usage(argv[0]); return 1; }

  std::string model_type = argv[1];
  std::string model_path = argv[2];
  std::string query_dir = argv[3];
  std::string db_dir = argv[4];
  int downsample = (argc > 5) ? std::stoi(argv[5]) : 5;
  std::string xfeat_dir = (argc > 6) ? argv[6] : "onnx_model";

  if (model_type != "mixvpr" && model_type != "jist") {
    std::cerr << "Unknown model-type '" << model_type << "'. Choose mixvpr or jist.\n";
    return 1;
  }

  std::cout << "Model type:   " << model_type << "\n"
            << "Model path:   " << model_path << "\n"
            << "Query dir:    " << query_dir << "\n"
            << "DB dir:       " << db_dir << "\n"
            << "Downsample:   every " << downsample << "-th DB image\n\n";

  auto query_paths = load_image_paths(query_dir);
  auto db_paths_all = load_image_paths(db_dir);
  if (query_paths.empty()) { std::cerr << "No images in query dir: " << query_dir << "\n"; return 1; }
  if (db_paths_all.empty()) { std::cerr << "No images in DB dir: " << db_dir << "\n"; return 1; }

  std::cout << "Query images: " << query_paths.size() << "\n"
            << "DB images:    " << db_paths_all.size() << " → "
            << (db_paths_all.size() / downsample + 1) << " after downsampling\n\n";

  std::vector<fs::path> db_paths_ds, self_db_paths;
  for (size_t i = 0; i < db_paths_all.size(); i += downsample) db_paths_ds.push_back(db_paths_all[i]);
  for (size_t i = 0; i < query_paths.size(); i += downsample) self_db_paths.push_back(query_paths[i]);

  // ── Create model ──────────────────────────────────────────────────────────
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vpr-example");
  std::unique_ptr<PlaceRecognizer> model;
  std::cout << "Loading " << model_type << " model...\n";
  if (model_type == "mixvpr") {
    MixVPRONNX::Params p; p.model_path = model_path; p.use_gpu = true;
    model = std::make_unique<MixVPRONNX>(env, p);
  } else {
    JistONNX::Params p; p.model_path = model_path; p.use_gpu = true;
    model = std::make_unique<JistONNX>(env, p);
  }
  std::cout << "  seq_length=" << model->get_seq_length()
            << "  descriptor_dim=" << model->get_descriptor_dim() << "\n\n";

  // Warmup
  {
    int seq = model->get_seq_length();
    cv::Mat first = cv::imread(query_paths[0].string(), cv::IMREAD_COLOR);
    model->infer(std::vector<cv::Mat>(seq, first));
  }

  // ── Init XFeat + LighterGlue ─────────────────────────────────────────────
  const std::string res = "640x480";
  XFeatONNX::Params xfp;
  xfp.xfeat_path            = xfeat_dir + "/xfeat_" + res + ".onnx";
  xfp.interp_bilinear_path  = xfeat_dir + "/interpolator_bilinear_" + res + ".onnx";
  xfp.interp_bicubic_path   = xfeat_dir + "/interpolator_bicubic_" + res + ".onnx";
  xfp.interp_nearest_path   = xfeat_dir + "/interpolator_nearest_" + res + ".onnx";
  xfp.use_gpu = true;
  xfp.nkpts   = 1024;
  std::cout << "Loading XFeat + LighterGlue models from " << xfeat_dir << "...\n";
  auto xfeat = std::make_unique<XFeatONNX>(env, xfp);

  LighterGlueCV::Params lgp;
  lgp.model_path  = xfeat_dir + "/lg_" + res + "_dyn.onnx";
  lgp.use_gpu     = true;
  lgp.n_kpts      = xfp.nkpts;
  auto lighter_glue = std::make_unique<LighterGlueCV>(env, lgp);
  lighter_glue->warmup();
  std::cout << "  XFeat + LighterGlue ready.\n\n";

  // ── Extract DB descriptors ────────────────────────────────────────────────
  auto [db_paths, db_descs] = extract_descs(*model, db_paths_ds, "DB");
  auto [self_paths, self_descs] = extract_descs(*model, self_db_paths, "self-DB");
  if (db_descs.empty() || self_descs.empty()) {
    std::cerr << "Failed to extract any descriptors.\n"; return 1;
  }

  // ── Set up display state and mouse callback ───────────────────────────────
  const int plot_w = std::max(kThumbW * 2, 1200);
  DisplayState state;
  state.plot_w = plot_w;
  state.win_title = "VPR Example — " + model_type;
  state.model_type = model_type;
  state.seq_length = model->get_seq_length();
  state.descriptor_dim = model->get_descriptor_dim();
  state.downsample = downsample;

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> qdist(0, static_cast<int>(query_paths.size()) - 1);

  // Compute a new random query and update state
  auto run_query = [&]() {
    const auto& active_paths = state.self_mode ? self_paths : db_paths;
    const auto& active_descs = state.self_mode ? self_descs : db_descs;

    int qi = qdist(rng);
    cv::Mat qframe = cv::imread(query_paths[qi].string(), cv::IMREAD_COLOR);
    if (qframe.empty()) { std::cerr << "Failed to read " << query_paths[qi] << "\n"; return; }

    auto t0 = std::chrono::high_resolution_clock::now();
    cv::Mat qdesc = query_descriptor(*model, query_paths, qi);
    double qms = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t0).count();

    std::vector<float> sims(active_descs.size());
    for (size_t i = 0; i < active_descs.size(); ++i)
      sims[i] = static_cast<float>(qdesc.dot(active_descs[i]));
    int best = static_cast<int>(std::max_element(sims.begin(), sims.end()) - sims.begin());

    std::cout << "[" << (state.self_mode ? "SELF" : "CROSS") << "] "
              << "Query[" << qi << "] " << query_paths[qi].filename().string()
              << "  (" << qms << " ms)"
              << "  →  best DB[" << best << "] " << active_paths[best].filename().string()
              << "  score=" << sims[best] << "\n";

    // Update shared state atomically for the mouse callback
    state.query_frame = qframe;
    state.query_idx = qi;
    state.query_ms = qms;
    state.active_paths = &active_paths;
    state.db_name = state.self_mode ? "self-DB" : "DB";
    state.similarities = std::move(sims);
    state.best_idx = best;
    state.selected_idx = -1;  // reset selection on new query
  };

  // Run XFeat detect+LighterGlue match on query vs currently selected/best DB frame
  static const std::string kMatchWin = "XFeat + LighterGlue";
  auto run_matching = [&]() {
    int sel = (state.selected_idx >= 0) ? state.selected_idx : state.best_idx;
    if (sel < 0 || state.active_paths == nullptr || state.query_frame.empty()) {
      std::cerr << "[MATCH] No active query — press [R] first.\n";
      return;
    }
    const auto& ap = *state.active_paths;
    cv::Mat db_frame = cv::imread(ap[sel].string(), cv::IMREAD_COLOR);
    if (db_frame.empty()) { std::cerr << "[MATCH] Failed to read DB frame.\n"; return; }

    cv::Mat q_gray, db_gray;
    cv::cvtColor(state.query_frame, q_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(db_frame, db_gray, cv::COLOR_BGR2GRAY);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto det_q  = xfeat->detect_and_compute(q_gray, xfp.nkpts);
    auto det_db = xfeat->detect_and_compute(db_gray, xfp.nkpts);

    std::vector<cv::DMatch> matches;
    cv::Size q_size(state.query_frame.cols, state.query_frame.rows);
    cv::Size db_size(db_frame.cols, db_frame.rows);
    lighter_glue->match(det_q, q_size, det_db, db_size, matches);
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();

    std::cout << "[MATCH] " << matches.size() << " matches  (" << ms << " ms)"
              << "  DB[" << sel << "] " << ap[sel].filename().string()
              << "  score=" << state.similarities[sel] << "\n";

    auto q_kpts  = to_keypoints(det_q.keypoints);
    auto db_kpts = to_keypoints(det_db.keypoints);

    cv::Mat match_img;
    cv::drawMatches(state.query_frame, q_kpts, db_frame, db_kpts, matches, match_img,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), {},
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Overlay stats
    std::string label = "XFeat+LighterGlue | " + std::to_string(matches.size()) +
                        " matches | " + cv::format("%.1f ms", ms) +
                        " | DB[" + std::to_string(sel) + "] score=" +
                        cv::format("%.3f", state.similarities[sel]);
    cv::putText(match_img, label, {8, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 0), 3);
    cv::putText(match_img, label, {8, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 1);

    cv::namedWindow(kMatchWin, cv::WINDOW_NORMAL);
    cv::imshow(kMatchWin, match_img);
  };

  std::cout << "Press [R]/[Space] new query | [S] toggle self-test | [M] match | [Q]/[Esc] quit\n"
            << "Click on the plot to inspect any DB match.\n\n";

  cv::namedWindow(state.win_title, cv::WINDOW_NORMAL);
  cv::setMouseCallback(state.win_title, on_mouse_cb, &state);

  run_query();
  cv::Mat canvas = render_canvas(state);
  if (!canvas.empty()) cv::imshow(state.win_title, canvas);

  while (true) {
    int key = cv::waitKey(0) & 0xFF;
    if (key == 'q' || key == 'Q' || key == 27) break;
    if (key == 'r' || key == 'R' || key == ' ') {
      run_query();
      canvas = render_canvas(state);
      if (!canvas.empty()) cv::imshow(state.win_title, canvas);
    }
    if (key == 's' || key == 'S') {
      state.self_mode = !state.self_mode;
      std::cout << "Switched to " << (state.self_mode ? "SELF-TEST" : "CROSS") << " mode\n";
      run_query();
      canvas = render_canvas(state);
      if (!canvas.empty()) cv::imshow(state.win_title, canvas);
    }
    if (key == 'm' || key == 'M') run_matching();
  }

  if (!canvas.empty()) {
    std::string out_file = model_type + "_vpr_result.png";
    cv::imwrite(out_file, canvas);
    std::cout << "Saved " << out_file << "\n";
  }
  return 0;
}
