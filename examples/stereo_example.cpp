/**
 * stereo_example.cpp
 *
 * Unified stereo depth demo: runs every available model on a stereo pair and
 * shows a tiled comparison window.
 *
 * Usage:
 *   stereo_example <left> <right> [options]
 *
 * Options:
 *   --onnx PATH             OnnxStereoDepth model (.onnx)
 *   --acvnet PATH           FastACVNet+ model (.onnx, 288x512 input)
 *   --lightstereo PATH      LightStereoDepth engine (.engine)
 *   --ffs-engine PATH       FastFoundationStereo single TensorRT engine
 *   --ffs-maxdisp N         export-time max disparity (default 192)
 *   --focal F               focal length in pixels (default 721.5)
 *   --baseline B            stereo baseline in metres (default 0.54)
 */

#include "xfeat-cpp/stereo_depth/stereo_depth.h"
#include "xfeat-cpp/stereo_depth/stereo_depth_onnx.h"
#ifdef HAVE_TENSORRT
#include "xfeat-cpp/stereo_depth/stereo_depth_lightstereo.h"
#include "xfeat-cpp/stereo_depth/stereo_depth_fast_foundation_stereo.h"
#endif

#include <boost/program_options.hpp>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <string>
#include <filesystem>
#include <vector>

namespace po = boost::program_options;

namespace {

struct StereoResult {
  cv::Mat display;
  cv::Mat depth;
  std::string label;
};

constexpr float kMinDepthMeters = 0.1f;
constexpr float kMaxDepthMeters = 80.0f;
constexpr int kCloudStride = 4;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static cv::Mat coloriseDisparity(const cv::Mat& disp) {
  cv::Mat f;
  if (disp.type() == CV_16S)
    disp.convertTo(f, CV_32F, 1.0 / 16.0);
  else
    disp.copyTo(f);

  // Clip negatives and normalise
  cv::threshold(f, f, 0.0, 0.0, cv::THRESH_TOZERO);
  double mn, mx;
  cv::minMaxLoc(f, &mn, &mx);
  cv::Mat u8;
  if (mx > mn)
    f.convertTo(u8, CV_8U, 255.0 / mx);
  else
    f.convertTo(u8, CV_8U);
  cv::Mat col;
  cv::applyColorMap(u8, col, cv::COLORMAP_TURBO);
  return col;
}

static cv::Mat labelledTile(const cv::Mat& img, const std::string& label,
                            const cv::Size& tile_size) {
  cv::Mat resized;
  cv::resize(img, resized, tile_size);

  const int banner_h = 28;
  cv::Mat tile(tile_size.height + banner_h, tile_size.width, CV_8UC3,
               cv::Scalar(30, 30, 30));
  resized.copyTo(tile(cv::Rect(0, banner_h, tile_size.width, tile_size.height)));
  cv::putText(tile, label, cv::Point(6, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
  return tile;
}

static cv::Mat buildGrid(const std::vector<StereoResult>& items,
                         const cv::Size& tile_size, int cols) {
  int rows = (static_cast<int>(items.size()) + cols - 1) / cols;
  int tile_h = tile_size.height + 28;
  cv::Mat grid(rows * tile_h, cols * tile_size.width, CV_8UC3,
               cv::Scalar(20, 20, 20));

  for (int i = 0; i < static_cast<int>(items.size()); ++i) {
    cv::Mat src = items[i].display;
    if (src.channels() == 1) src = coloriseDisparity(src);

    cv::Mat tile = labelledTile(src, items[i].label, tile_size);
    int r = i / cols;
    int c = i % cols;
    tile.copyTo(
        grid(cv::Rect(c * tile_size.width, r * tile_h, tile.cols, tile.rows)));
  }
  return grid;
}

static pcl::PointCloud<pcl::PointXYZRGB>::Ptr depthToCloud(const cv::Mat& left_bgr,
                                                           const cv::Mat& depth,
                                                           float focal) {
  auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  cloud->reserve((depth.rows / kCloudStride) * (depth.cols / kCloudStride));

  const float cx = static_cast<float>(depth.cols) * 0.5f;
  const float cy = static_cast<float>(depth.rows) * 0.5f;

  for (int y = 0; y < depth.rows; y += kCloudStride) {
    const float* depth_ptr = depth.ptr<float>(y);
    const cv::Vec3b* color_ptr = left_bgr.ptr<cv::Vec3b>(y);
    for (int x = 0; x < depth.cols; x += kCloudStride) {
      const float z = depth_ptr[x];
      if (!std::isfinite(z) || z < kMinDepthMeters || z > kMaxDepthMeters) {
        continue;
      }

      pcl::PointXYZRGB pt;
      pt.z = z;
      pt.x = (static_cast<float>(x) - cx) * z / focal;
      pt.y = (static_cast<float>(y) - cy) * z / focal;
      pt.b = color_ptr[x][0];
      pt.g = color_ptr[x][1];
      pt.r = color_ptr[x][2];
      cloud->push_back(pt);
    }
  }

  cloud->width = cloud->size();
  cloud->height = 1;
  cloud->is_dense = false;
  return cloud;
}

static void savePointClouds(const cv::Mat& left_bgr,
                            const std::vector<StereoResult>& results,
                            float focal,
                            const std::string& output_dir) {
  std::filesystem::create_directories(output_dir);

  for (const auto& result : results) {
    if (result.depth.empty()) {
      continue;
    }

    auto cloud = depthToCloud(left_bgr, result.depth, focal);
    if (cloud->empty()) {
      std::fprintf(stderr, "Skipping %s point cloud: no valid depth samples\n",
                   result.label.c_str());
      continue;
    }

    std::string stem = result.label;
    for (char& ch : stem) {
      if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-')) {
        ch = '_';
      }
    }
    const std::filesystem::path ply_path = std::filesystem::path(output_dir) / (stem + ".ply");
    pcl::io::savePLYFileBinary(ply_path.string(), *cloud);
    std::cout << "Saved point cloud: " << ply_path << std::endl;
  }
}

static void appendResult(std::vector<StereoResult>& out,
                         const cv::Mat& disparity,
                         const cv::Mat& depth,
                         const std::string& label) {
  out.push_back({coloriseDisparity(disparity), depth, label});
}

// ---------------------------------------------------------------------------
// Per-model runners
// ---------------------------------------------------------------------------

static void runOpenCVSGBM(const cv::Mat& left, const cv::Mat& right,
                          float focal, float baseline,
                          std::vector<StereoResult>& out) {
  xfeat::OpenCVStereoDepth::Params p;
  p.algorithm = xfeat::OpenCVStereoDepth::Algorithm::SGBM;
  p.num_disparities = 128;
  p.block_size = 11;
  p.P1 = 8 * p.block_size * p.block_size;
  p.P2 = 32 * p.block_size * p.block_size;
  xfeat::OpenCVStereoDepth stereo(p);

  cv::Mat disp;
  auto t0 = std::chrono::steady_clock::now();
  stereo.compute(left, right, disp);
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();

  cv::Mat depth;
  stereo.disparityToDepth(disp, depth, focal, baseline);
  std::printf("  OpenCV SGBM: %.1f ms\n", ms);
  appendResult(out, disp, depth, "OpenCV SGBM  " + std::to_string(static_cast<int>(ms)) + " ms");
}

static void runOnnx(const cv::Mat& left, const cv::Mat& right,
                    const std::string& model_path, const std::string& label,
                    cv::Size input_size, float focal, float baseline,
                    std::vector<StereoResult>& out) {
  xfeat::OnnxStereoDepth::Params p;
  p.model_path = model_path;
  p.input_size = input_size;
  p.use_cuda = true;
  p.warmup_iterations = 3;
  xfeat::OnnxStereoDepth stereo(p);
  stereo.warmup(left.size());

  cv::Mat disp;
  auto t0 = std::chrono::steady_clock::now();
  stereo.compute(left, right, disp);
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();

  cv::Mat depth;
  stereo.disparityToDepth(disp, depth, focal, baseline);
  std::printf("  %s: %.1f ms\n", label.c_str(), ms);
  appendResult(out, disp, depth,
               label + "  " + std::to_string(static_cast<int>(ms)) + " ms");
}

#ifdef HAVE_TENSORRT
static void runLightStereo(const cv::Mat& left, const cv::Mat& right,
                           const std::string& engine_path,
                           float focal, float baseline,
                           std::vector<StereoResult>& out) {
  xfeat::LightStereoDepth::Params p;
  p.engine_path = engine_path;
  p.target_size = cv::Size(512, 288);
  p.warmup_iterations = 5;
  xfeat::LightStereoDepth stereo(p);
  stereo.warmup(left.size());

  cv::Mat disp;
  auto t0 = std::chrono::steady_clock::now();
  stereo.compute(left, right, disp);
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();

  cv::Mat depth;
  stereo.disparityToDepth(disp, depth, focal, baseline);
  std::printf("  LightStereo: %.1f ms\n", ms);
  appendResult(out, disp, depth,
               "LightStereo  " + std::to_string(static_cast<int>(ms)) + " ms");
}

static void runFFS(const cv::Mat& left, const cv::Mat& right,
                   const std::string& engine_path, int max_disp,
                   float focal, float baseline,
                   std::vector<StereoResult>& out) {
  xfeat::FastFoundationStereoDepth::Params p;
  p.engine_path = engine_path;
  p.max_disparity = max_disp;
  p.warmup_iterations = 3;
  xfeat::FastFoundationStereoDepth stereo(p);
  stereo.warmup(left.size());

  cv::Mat disp;
  auto t0 = std::chrono::steady_clock::now();
  stereo.compute(left, right, disp);
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();

  cv::Mat depth;
  stereo.disparityToDepth(disp, depth, focal, baseline);
  std::printf("  FastFoundationStereo: %.1f ms\n", ms);
  appendResult(out, disp, depth,
               "FastFoundation " + std::to_string(static_cast<int>(ms)) + " ms");
}
#endif

}  // namespace

int main(int argc, char** argv) {
  std::string left_path, right_path;
  std::string onnx_model, acvnet_model, lightstereo_engine, ffs_engine;
  float focal = 721.5f, baseline = 0.54f;
  int ffs_maxdisp = 192;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help,h",                                     "Show help")
    ("left",    po::value(&left_path)->required(),  "Left image path")
    ("right",   po::value(&right_path)->required(), "Right image path")
    ("onnx",    po::value(&onnx_model),             "OnnxStereoDepth model (.onnx)")
    ("acvnet",  po::value(&acvnet_model),           "FastACVNet+ model (.onnx, 288x512 input)")
    ("lightstereo", po::value(&lightstereo_engine), "LightStereoDepth engine (.engine)")
    ("ffs-engine",  po::value(&ffs_engine),         "FastFoundationStereo single engine")
    ("ffs-maxdisp", po::value(&ffs_maxdisp),        "FFS export-time max disparity (default 192)")
    ("focal",       po::value(&focal),              "Focal length px (default 721.5)")
    ("baseline",    po::value(&baseline),           "Baseline m (default 0.54)");
  // clang-format on

  po::positional_options_description pos;
  pos.add("left", 1).add("right", 1);

  po::variables_map vm;
  try {
    po::store(
        po::command_line_parser(argc, argv).options(desc).positional(pos).run(),
        vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << "Argument error: " << e.what() << "\n" << desc << "\n";
    return 1;
  }

  cv::Mat left = cv::imread(left_path, cv::IMREAD_COLOR);
  cv::Mat right = cv::imread(right_path, cv::IMREAD_COLOR);
  if (left.empty() || right.empty()) {
    std::cerr << "Failed to load images\n";
    return 1;
  }
  std::printf("Images: %dx%d   focal=%.1f px  baseline=%.3f m\n",
              left.cols, left.rows, focal, baseline);

  std::vector<StereoResult> results;
  results.push_back({left, cv::Mat(), "Left"});
  results.push_back({right, cv::Mat(), "Right"});

  std::puts("\n--- OpenCV SGBM ---");
  try {
    runOpenCVSGBM(left, right, focal, baseline, results);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "  SKIP: %s\n", e.what());
  }

  if (!onnx_model.empty()) {
    std::puts("\n--- ONNX Stereo ---");
    try {
      runOnnx(left, right, onnx_model, "ONNX stereo", cv::Size(512, 288), focal,
              baseline, results);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "  SKIP: %s\n", e.what());
    }
  }

  if (!acvnet_model.empty()) {
    std::puts("\n--- FastACVNet+ ---");
    try {
      runOnnx(left, right, acvnet_model, "FastACVNet+", cv::Size(512, 288),
              focal, baseline, results);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "  SKIP: %s\n", e.what());
    }
  }

#ifdef HAVE_TENSORRT
  if (!lightstereo_engine.empty()) {
    std::puts("\n--- LightStereo ---");
    try {
      runLightStereo(left, right, lightstereo_engine, focal, baseline, results);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "  SKIP: %s\n", e.what());
    }
  }

  if (!ffs_engine.empty()) {
    std::puts("\n--- FastFoundationStereo ---");
    try {
      runFFS(left, right, ffs_engine, ffs_maxdisp, focal, baseline, results);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "  SKIP: %s\n", e.what());
    }
  }
#endif

  const std::string cloud_dir = "build/pointclouds";
  savePointClouds(left, results, focal, cloud_dir);
  std::cout << "Point clouds written under " << cloud_dir << std::endl;
  return 0;
}
