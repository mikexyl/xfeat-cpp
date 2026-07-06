#include <iostream>

#ifdef HAVE_TENSORRT

#include <algorithm>
#include <boost/program_options.hpp>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xfeat-cpp/mono_depth/depth_anything_v3_trt.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

cv::Mat colorizeDepth(const cv::Mat& depth) {
  cv::Mat mask = depth > 0.0f;
  double min_depth = 0.0;
  double max_depth = 0.0;
  cv::minMaxLoc(depth, &min_depth, &max_depth, nullptr, nullptr, mask);

  cv::Mat normalized = cv::Mat::zeros(depth.size(), CV_8UC1);
  if (max_depth > min_depth) {
    depth.convertTo(normalized, CV_8UC1, 255.0 / (max_depth - min_depth), -min_depth * 255.0 / (max_depth - min_depth));
    normalized.setTo(0, mask == 0);
  }

  cv::Mat color;
  cv::applyColorMap(normalized, color, cv::COLORMAP_TURBO);
  color.setTo(cv::Scalar(0, 0, 0), mask == 0);
  return color;
}

std::string toLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool isImagePath(const fs::path& path) {
  const std::string ext = toLower(path.extension().string());
  return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

std::vector<fs::path> listImages(const fs::path& sequence_dir) {
  if (!fs::is_directory(sequence_dir)) {
    throw std::runtime_error("Sequence directory does not exist: " + sequence_dir.string());
  }

  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(sequence_dir)) {
    if (entry.is_regular_file() && isImagePath(entry.path())) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
  return images;
}

std::string outputPrefix(int view_index, const fs::path& image_path) {
  std::ostringstream oss;
  oss << "view_" << std::setw(2) << std::setfill('0') << view_index << "_" << image_path.stem().string();
  return oss.str();
}

std::optional<xfeat::CameraIntrinsics> makeIntrinsics(const cv::Size& size,
                                                      bool enabled,
                                                      double fx,
                                                      double fy,
                                                      double cx,
                                                      double cy) {
  if (!enabled) {
    return std::nullopt;
  }

  xfeat::CameraIntrinsics intrinsics;
  intrinsics.fx = fx;
  intrinsics.fy = fy;
  intrinsics.cx = cx > 0.0 ? cx : 0.5 * static_cast<double>(size.width - 1);
  intrinsics.cy = cy > 0.0 ? cy : 0.5 * static_cast<double>(size.height - 1);
  intrinsics.width = size.width;
  intrinsics.height = size.height;
  return intrinsics;
}

std::vector<fs::path> chooseSubsequence(const std::vector<fs::path>& images,
                                        int view_count,
                                        int interval,
                                        unsigned int seed,
                                        size_t* start_index) {
  if (view_count <= 0) {
    throw std::runtime_error("--views must be positive");
  }
  if (interval <= 0) {
    throw std::runtime_error("--interval must be positive");
  }
  const size_t required_span = static_cast<size_t>(interval) * static_cast<size_t>(view_count - 1);
  if (images.size() <= required_span) {
    throw std::runtime_error("Sequence has " + std::to_string(images.size()) + " images, but " +
                             std::to_string(required_span + 1) + " are needed for the requested subsequence");
  }

  const size_t max_start = images.size() - required_span - 1;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<size_t> distribution(0, max_start);
  *start_index = distribution(rng);

  std::vector<fs::path> selected;
  selected.reserve(static_cast<size_t>(view_count));
  for (int view = 0; view < view_count; ++view) {
    selected.push_back(images[*start_index + static_cast<size_t>(view * interval)]);
  }
  return selected;
}

void writeSelectedPaths(const fs::path& out_dir,
                        const std::vector<fs::path>& selected,
                        size_t start_index,
                        int interval,
                        unsigned int seed) {
  std::ofstream out(out_dir / "selected_views.txt");
  if (!out) {
    throw std::runtime_error("Failed to write selected view list");
  }

  out << "seed " << seed << "\n";
  out << "start_index " << start_index << "\n";
  out << "interval " << interval << "\n";
  for (size_t i = 0; i < selected.size(); ++i) {
    out << i << " " << selected[i].string() << "\n";
  }
}

void writeResult(const fs::path& out_dir,
                 int view_index,
                 const fs::path& image_path,
                 const cv::Mat& image,
                 const xfeat::MonoDepthResult& result,
                 std::vector<cv::Mat>* summary_tiles) {
  const std::string prefix = outputPrefix(view_index, image_path);
  const cv::Mat depth_vis = colorizeDepth(result.depth);

  cv::FileStorage storage((out_dir / (prefix + "_depth.yml")).string(), cv::FileStorage::WRITE);
  storage << "depth" << result.depth;
  storage << "raw_depth" << result.raw_depth;
  storage << "focal_scale" << result.metadata.focal_scale;
  storage << "sky_fill_value" << result.metadata.sky_fill_value;
  storage.release();

  cv::imwrite((out_dir / (prefix + "_image.png")).string(), image);
  cv::imwrite((out_dir / (prefix + "_depth_vis.png")).string(), depth_vis);
  if (!result.sky_mask.empty()) {
    cv::imwrite((out_dir / (prefix + "_sky_mask.png")).string(), result.sky_mask);
  }

  cv::Mat image_tile;
  cv::Mat depth_tile;
  cv::resize(image, image_tile, cv::Size(400, 275), 0.0, 0.0, cv::INTER_AREA);
  cv::resize(depth_vis, depth_tile, cv::Size(400, 275), 0.0, 0.0, cv::INTER_AREA);
  cv::putText(image_tile,
              "view " + std::to_string(view_index),
              cv::Point(12, 28),
              cv::FONT_HERSHEY_SIMPLEX,
              0.8,
              cv::Scalar(255, 255, 255),
              2,
              cv::LINE_AA);
  cv::putText(depth_tile,
              "depth " + std::to_string(view_index),
              cv::Point(12, 28),
              cv::FONT_HERSHEY_SIMPLEX,
              0.8,
              cv::Scalar(255, 255, 255),
              2,
              cv::LINE_AA);
  summary_tiles->push_back(std::move(image_tile));
  summary_tiles->push_back(std::move(depth_tile));
}

void writeSummary(const fs::path& out_dir, const std::vector<cv::Mat>& tiles) {
  if (tiles.empty()) {
    return;
  }

  std::vector<cv::Mat> rows;
  rows.reserve((tiles.size() + 1) / 2);
  for (size_t i = 0; i < tiles.size(); i += 2) {
    if (i + 1 >= tiles.size()) {
      rows.push_back(tiles[i]);
      continue;
    }

    cv::Mat row;
    cv::hconcat(std::vector<cv::Mat>{tiles[i], tiles[i + 1]}, row);
    rows.push_back(std::move(row));
  }

  cv::Mat summary;
  cv::vconcat(rows, summary);
  cv::imwrite((out_dir / "multiview_summary.png").string(), summary);
}

}  // namespace

int main(int argc, char** argv) {
  std::string engine_path;
  std::string sequence_dir = "/data/graco/ground-03_images/camera_left_image_raw";
  std::string out_dir = "output/mono_depth_graco_ground03_multiview";
  int views = 5;
  int interval = 10;
  unsigned int seed = std::random_device{}();
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  bool verbose = false;

  po::options_description desc("Random strided sequence example for grouped DA3 TensorRT inference");
  desc.add_options()("help,h", "Show this help message")(
      "engine,e", po::value<std::string>(&engine_path)->required(), "Path to a grouped-view DA3 TensorRT .engine file")(
      "sequence-dir", po::value<std::string>(&sequence_dir)->default_value(sequence_dir), "Sorted image sequence")(
      "out-dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "Directory for result files")(
      "views", po::value<int>(&views)->default_value(views), "Number of views in the grouped inference")(
      "interval", po::value<int>(&interval)->default_value(interval), "Frame interval between selected views")(
      "seed", po::value<unsigned int>(&seed)->default_value(seed), "Random seed for the start index")(
      "fx", po::value<double>(&fx), "Camera fx in pixels")("fy", po::value<double>(&fy), "Camera fy in pixels")(
      "cx", po::value<double>(&cx), "Camera cx in pixels (defaults to image center when fx/fy are set)")(
      "cy", po::value<double>(&cy), "Camera cy in pixels (defaults to image center when fx/fy are set)")(
      "verbose,v", po::bool_switch(&verbose), "Enable TensorRT metadata logging");

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << "Argument error: " << e.what() << "\n\n" << desc << std::endl;
    return 2;
  }

  const bool has_intrinsics = vm.count("fx") > 0 || vm.count("fy") > 0 || vm.count("cx") > 0 || vm.count("cy") > 0;
  if (has_intrinsics && (fx <= 0.0 || fy <= 0.0)) {
    std::cerr << "Both --fx and --fy must be positive when intrinsics are provided" << std::endl;
    return 2;
  }

  try {
    const auto all_images = listImages(sequence_dir);
    size_t start_index = 0;
    const auto selected = chooseSubsequence(all_images, views, interval, seed, &start_index);
    fs::create_directories(out_dir);
    writeSelectedPaths(out_dir, selected, start_index, interval, seed);

    std::vector<cv::Mat> images;
    images.reserve(selected.size());
    for (const auto& path : selected) {
      cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
      if (image.empty()) {
        throw std::runtime_error("Failed to read image: " + path.string());
      }
      images.push_back(std::move(image));
    }

    std::vector<xfeat::CameraIntrinsics> intrinsics;
    if (has_intrinsics) {
      intrinsics.reserve(images.size());
      for (const auto& image : images) {
        intrinsics.push_back(*makeIntrinsics(image.size(), true, fx, fy, cx, cy));
      }
    }

    xfeat::DepthAnythingV3TRT::Params params;
    params.engine_path = engine_path;
    params.verbose = verbose;
    xfeat::DepthAnythingV3TRT model(params);
    const auto results = model.infer_multi_view(images, intrinsics);

    std::vector<cv::Mat> summary_tiles;
    summary_tiles.reserve(results.size() * 2);
    for (size_t i = 0; i < results.size(); ++i) {
      writeResult(out_dir, static_cast<int>(i), selected[i], images[i], results[i], &summary_tiles);
      std::cout << "view " << i << ": " << selected[i] << ", depth " << results[i].depth.cols << "x"
                << results[i].depth.rows << ", raw " << results[i].raw_depth.cols << "x" << results[i].raw_depth.rows
                << ", focal_scale=" << results[i].metadata.focal_scale;
      if (!results[i].sky_mask.empty()) {
        std::cout << ", sky_fill=" << results[i].metadata.sky_fill_value;
      }
      std::cout << std::endl;
    }

    writeSummary(out_dir, summary_tiles);
    std::cout << "selected_start_index=" << start_index << ", interval=" << interval << ", seed=" << seed << std::endl;
    std::cout << "Results written to " << out_dir << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Multi-view depth example failed: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

#else

int main() {
  std::cerr << "mono_depth_multiview_sequence_example requires TensorRT support." << std::endl;
  return 1;
}

#endif  // HAVE_TENSORRT
