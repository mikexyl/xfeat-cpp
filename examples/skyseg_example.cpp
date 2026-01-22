/**
 * @file skyseg_example.cpp
 * @brief Example usage of SkySegTRT class for sky segmentation
 *
 * This example demonstrates how to use the TensorRT-based sky segmentation
 * to process images and generate visualization outputs.
 */

#include <chrono>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef HAVE_TENSORRT
#include "xfeat-cpp/segmentation/skyseg_trt.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <engine_path> <image_path> [--show] [--benchmark]" << std::endl;
    std::cerr << "Example: " << argv[0] << " onnx_model/skyseg.engine image/sample1.jpg --show" << std::endl;
    return 1;
  }

  std::string engine_path = argv[1];
  std::string image_path = argv[2];
  bool show_results = false;
  bool benchmark = false;

  // Parse optional arguments
  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--show") {
      show_results = true;
    } else if (arg == "--benchmark") {
      benchmark = true;
    }
  }

  try {
    // Configure parameters
    xfeat::SkySegTRT::Params params;
    params.engine_path = engine_path;
    params.input_size = cv::Size(320, 320);
    params.verbose = true;
    params.warmup_iterations = 3;
    params.threshold = 127.0f;

    std::cout << "Loading TensorRT engine from: " << engine_path << std::endl;

    // Create sky segmentation instance
    xfeat::SkySegTRT sky_seg(params);

    // Load input image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
      std::cerr << "Error: Could not read image from " << image_path << std::endl;
      return 1;
    }

    std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;

    // Warmup
    std::cout << "\nWarming up..." << std::endl;
    sky_seg.warmup(image.size());

    if (benchmark) {
      // Benchmark mode
      const int iterations = 100;
      std::cout << "\nRunning benchmark with " << iterations << " iterations..." << std::endl;

      std::vector<double> times;
      cv::Mat mask;

      for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        sky_seg.segment(image, mask);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        times.push_back(duration.count());

        if ((i + 1) % 10 == 0) {
          std::cout << "  Progress: " << (i + 1) << "/" << iterations << std::endl;
        }
      }

      // Calculate statistics
      double mean_time = 0.0;
      for (double t : times) mean_time += t;
      mean_time /= times.size();

      double min_time = *std::min_element(times.begin(), times.end());
      double max_time = *std::max_element(times.begin(), times.end());

      std::cout << "\n=== Benchmark Results ===" << std::endl;
      std::cout << "Mean inference time: " << mean_time << " ms" << std::endl;
      std::cout << "Min inference time: " << min_time << " ms" << std::endl;
      std::cout << "Max inference time: " << max_time << " ms" << std::endl;
      std::cout << "Mean FPS: " << (1000.0 / mean_time) << std::endl;
      std::cout << "Max FPS: " << (1000.0 / min_time) << std::endl;

    } else {
      // Single inference mode
      std::cout << "\nRunning inference..." << std::endl;

      cv::Mat mask, overlay;

      auto start = std::chrono::high_resolution_clock::now();
      sky_seg.segmentWithVisualization(image, mask, overlay, 0.5f, cv::Scalar(255, 100, 0));
      auto end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double, std::milli> duration = end - start;

      std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
      std::cout << "FPS: " << (1000.0 / duration.count()) << std::endl;

      // Calculate sky percentage
      float sky_percentage = xfeat::SkySegTRT::calculateSkyPercentage(mask);
      std::cout << "Sky percentage: " << sky_percentage << "%" << std::endl;

      // Save results
      std::string output_dir = "output";
      std::string base_name = image_path.substr(image_path.find_last_of("/\\") + 1);
      base_name = base_name.substr(0, base_name.find_last_of("."));

      cv::imwrite(output_dir + "/" + base_name + "_mask.jpg", mask);
      cv::imwrite(output_dir + "/" + base_name + "_overlay.jpg", overlay);

      // Create side-by-side comparison
      cv::Mat mask_colored;
      cv::cvtColor(mask, mask_colored, cv::COLOR_GRAY2BGR);
      cv::Mat comparison;
      cv::hconcat(std::vector<cv::Mat>{image, mask_colored, overlay}, comparison);
      cv::imwrite(output_dir + "/" + base_name + "_comparison.jpg", comparison);

      std::cout << "\nResults saved to " << output_dir << "/" << std::endl;
      std::cout << "  - " << base_name << "_mask.jpg" << std::endl;
      std::cout << "  - " << base_name << "_overlay.jpg" << std::endl;
      std::cout << "  - " << base_name << "_comparison.jpg" << std::endl;

      // Display results
      if (show_results) {
        cv::namedWindow("Sky Segmentation Results", cv::WINDOW_NORMAL);
        cv::imshow("Sky Segmentation Results", comparison);
        std::cout << "\nPress any key to exit..." << std::endl;
        cv::waitKey(0);
      }
    }

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\nDone!" << std::endl;
  return 0;
}

#else

int main() {
  std::cerr << "Error: This example requires TensorRT support." << std::endl;
  std::cerr << "Please rebuild with -DHAVE_TENSORRT=ON" << std::endl;
  return 1;
}

#endif  // HAVE_TENSORRT
