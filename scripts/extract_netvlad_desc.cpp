#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <random>
#include <thread>
#include <vector>

#include "xfeat-cpp/netvlad_onnx.h"
#include "xfeat-cpp/xfeat_netvlad_onnx.h"
#include "xfeat-cpp/xfeat_onnx.h"

using namespace xfeat;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <image_folder> <output_desc_file> [model_folder]" << std::endl;
    return 1;
  }
  std::filesystem::path image_folder(argv[1]);
  std::string image_resolution = "640x480";
  int max_kpts = 500;
  std::filesystem::path xfeat_model_folder = (argc > 3) ? argv[3] : "onnx_model";
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

  std::string output_desc_file = argv[2];

  // List all images in the folder and subfolders
  std::vector<std::filesystem::path> image_paths;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(image_folder)) {
    if (entry.is_regular_file()) {
      auto ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
        image_paths.push_back(entry.path());
      }
    }
  }

  // randomly select 4000 images
  if (image_paths.size() > 20000) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(image_paths.begin(), image_paths.end(), g);
    image_paths.resize(20000);
  }

  if (image_paths.empty()) {
    std::cerr << "No images found in folder!" << std::endl;
    return 1;
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "xfeat-shared-env");
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
  xfeat::NetVLADONNX netvlad_onnx(env, netvlad_model_path.string());
  xfeat::HeadNetVLADONNX head_netvlad_onnx(env, (xfeat_model_folder / ("xfeat_nv.onnx")).string());

  std::ofstream ofs(output_desc_file, std::ios::binary);
  int nb_total = 0;
  int vlad_dim = -1;
  std::vector<float> buffer;
  const int save_interval = 1000;
  bool header_written = false;
  std::mutex mtx;
  int num_threads = 1;
  size_t images_per_thread = (image_paths.size() + num_threads - 1) / num_threads;

  // Write placeholder header before threads start
  int nb_placeholder = 0;
  ofs.write(reinterpret_cast<const char*>(&nb_placeholder), sizeof(int));
  ofs.write(reinterpret_cast<const char*>(&vlad_dim), sizeof(int));
  header_written = true;

  auto worker = [&](size_t start_idx, size_t end_idx) {
    std::vector<float> local_buffer;
    int local_nb_total = 0;
    int local_vlad_dim = -1;
    for (size_t idx = start_idx; idx < end_idx; ++idx) {
      const auto& img_path = image_paths[idx];
      {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "[Thread " << std::this_thread::get_id() << "] Processing image " << (idx + 1) << "/"
                  << image_paths.size() << ": " << img_path << std::endl;
      }
      cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
      if (img.empty()) {
        std::lock_guard<std::mutex> lock(mtx);
        std::cerr << "Error loading image: " << img_path << std::endl;
        continue;
      }
      cv::Mat M1, x_prep;
      xfeat_onnx.detect_and_compute(img, max_kpts, nullptr, &M1, &x_prep);
      cv::Mat dense_desc = head_netvlad_onnx.run(M1, x_prep);
      auto vlad = netvlad_onnx.infer(dense_desc);
      if (local_vlad_dim < 0) local_vlad_dim = vlad[0].size();
      local_buffer.insert(local_buffer.end(), vlad[0].begin(), vlad[0].end());
      ++local_nb_total;
      if (local_nb_total % save_interval == 0) {
        std::lock_guard<std::mutex> lock(mtx);
        if (vlad_dim < 0 && local_vlad_dim > 0) vlad_dim = local_vlad_dim;
        ofs.write(reinterpret_cast<const char*>(local_buffer.data()), local_buffer.size() * sizeof(float));
        nb_total += local_nb_total;
        local_buffer.clear();
        local_nb_total = 0;
        std::cout << "Saved " << nb_total << " descriptors so far..." << std::endl;
      }
    }
    // Write any remaining buffer
    if (!local_buffer.empty()) {
      std::lock_guard<std::mutex> lock(mtx);
      if (vlad_dim < 0 && local_vlad_dim > 0) vlad_dim = local_vlad_dim;
      ofs.write(reinterpret_cast<const char*>(local_buffer.data()), local_buffer.size() * sizeof(float));
      nb_total += local_nb_total;
      std::cout << "Saved " << nb_total << " descriptors so far..." << std::endl;
    }
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    size_t start_idx = t * images_per_thread;
    size_t end_idx = std::min(image_paths.size(), (t + 1) * images_per_thread);
    if (start_idx >= end_idx) break;
    threads.emplace_back(worker, start_idx, end_idx);
  }
  for (auto& th : threads) th.join();

  ofs.seekp(0, std::ios::beg);
  ofs.write(reinterpret_cast<const char*>(&nb_total), sizeof(int));
  ofs.seekp(sizeof(int), std::ios::beg);
  ofs.write(reinterpret_cast<const char*>(&vlad_dim), sizeof(int));
  ofs.close();
  std::cout << "Saved total " << nb_total << " descriptors of dim " << vlad_dim << " to " << output_desc_file
            << std::endl;
  return 0;
}
