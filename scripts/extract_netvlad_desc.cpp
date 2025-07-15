#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
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
  std::filesystem::path interp_bilinear_path = xfeat_model_folder / ("interpolator_bilinear_" + image_resolution + ".onnx");
  std::filesystem::path interp_bicubic_path = xfeat_model_folder / ("interpolator_bicubic_" + image_resolution + ".onnx");
  std::filesystem::path interp_nearest_path = xfeat_model_folder / ("interpolator_nearest_" + image_resolution + ".onnx");
  std::filesystem::path lighterglue_model_path = xfeat_model_folder / ("lg_" + image_resolution + "_" + std::to_string(max_kpts) + ".onnx");
  std::filesystem::path netvlad_model_path = xfeat_model_folder / ("netvlad.onnx");

  std::string output_desc_file = argv[2];

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

  for (size_t idx = 0; idx < image_paths.size(); ++idx) {
    const auto& img_path = image_paths[idx];
    cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      std::cerr << "Error loading image: " << img_path << std::endl;
      continue;
    }
    cv::Mat M1, x_prep;
    xfeat_onnx.detect_and_compute(img, max_kpts, nullptr, &M1, &x_prep);
    cv::Mat dense_desc = head_netvlad_onnx.run(M1, x_prep);
    auto vlad = netvlad_onnx.infer(dense_desc);
    if (vlad_dim < 0) vlad_dim = vlad[0].size();
    buffer.insert(buffer.end(), vlad[0].begin(), vlad[0].end());
    ++nb_total;

    if (nb_total % save_interval == 0 || idx == image_paths.size() - 1) {
      if (!header_written) {
        // Write placeholder header
        int nb_placeholder = 0;
        ofs.write(reinterpret_cast<const char*>(&nb_placeholder), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&vlad_dim), sizeof(int));
        header_written = true;
      }
      ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(float));
      buffer.clear();
      std::cout << "Saved " << nb_total << " descriptors so far..." << std::endl;
    }
  }
  ofs.seekp(0, std::ios::beg);
  ofs.write(reinterpret_cast<const char*>(&nb_total), sizeof(int));
  ofs.seekp(sizeof(int), std::ios::beg);
  ofs.write(reinterpret_cast<const char*>(&vlad_dim), sizeof(int));
  ofs.close();
  std::cout << "Saved total " << nb_total << " descriptors of dim " << vlad_dim << " to " << output_desc_file << std::endl;
  return 0;
}
