#include "xfeat-cpp/stereo_depth.h"
#include "xfeat-cpp/stereo_depth_libsgm.h"
#ifdef HAVE_TENSORRT
#include "xfeat-cpp/stereo_depth_lightstereo.h"
#endif
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>

using namespace xfeat;

void visualizeDisparity(const cv::Mat& disparity, const std::string& window_name, int scale = 16) {
    // Normalize disparity for visualization
    cv::Mat disp_vis;
    double min_val, max_val;
    
    if (disparity.type() == CV_16S) {
        disparity.convertTo(disp_vis, CV_8U, 255.0 / (scale * 64.0));
    } else {
        cv::minMaxLoc(disparity, &min_val, &max_val);
        disparity.convertTo(disp_vis, CV_8U, 255.0 / max_val);
    }
    
    cv::Mat disp_color;
    cv::applyColorMap(disp_vis, disp_color, cv::COLORMAP_JET);
    cv::imshow(window_name, disp_color);
}

void exampleOpenCV(const cv::Mat& left, const cv::Mat& right) {
    std::cout << "\n=== OpenCV SGBM Example ===" << std::endl;
    
    // Create OpenCV SGBM stereo depth estimator
    OpenCVStereoDepth::Params params;
    params.algorithm = OpenCVStereoDepth::Algorithm::SGBM;
    params.num_disparities = 128;
    params.block_size = 5;
    params.min_disparity = 0;
    params.P1 = 8 * 1 * params.block_size * params.block_size;
    params.P2 = 32 * 1 * params.block_size * params.block_size;
    
    auto stereo = std::make_unique<OpenCVStereoDepth>(params);
    
    // Compute disparity
    cv::Mat disparity;
    auto start = cv::getTickCount();
    stereo->compute(left, right, disparity);
    auto end = cv::getTickCount();
    
    std::cout << "OpenCV SGBM computation time: " 
              << (end - start) / cv::getTickFrequency() * 1000.0 << " ms" << std::endl;
    std::cout << "Disparity scale: " << stereo->getDisparityScale() << std::endl;
    
    // Compute depth (example with typical stereo camera parameters)
    float focal_length = 721.5377f;  // pixels
    float baseline = 0.54f;           // meters
    
    cv::Mat depth;
    stereo->computeDepth(left, right, depth, focal_length, baseline);
    
    // Visualize
    visualizeDisparity(disparity, "OpenCV SGBM Disparity", stereo->getDisparityScale());
}

void exampleLibSGM(const cv::Mat& left, const cv::Mat& right) {
    std::cout << "\n=== LibSGM Example ===" << std::endl;
    
    try {
        // Create LibSGM stereo depth estimator
        LibSGMStereoDepth::Params params;
        params.num_disparities = 128;
        params.P1 = 10;
        params.P2 = 120;
        params.uniqueness_ratio = 0.95f;
        params.subpixel = true;
        params.path_type = 1;  // SCAN_8PATH
        params.use_gpu = true;  // Will automatically fall back to CPU if OpenCV lacks CUDA
        
        auto stereo = std::make_unique<LibSGMStereoDepth>(params);
        
        // Warmup (works for both CPU and GPU)
        std::cout << "Warming up..." << std::endl;
        stereo->warmup(left.size());
        
        // Compute disparity
        cv::Mat disparity;
        auto start = cv::getTickCount();
        stereo->compute(left, right, disparity);
        auto end = cv::getTickCount();
        
        std::cout << "LibSGM computation time: " 
                  << (end - start) / cv::getTickFrequency() * 1000.0 << " ms" << std::endl;
        std::cout << "Disparity scale: " << stereo->getDisparityScale() << std::endl;
        
        // Visualize
        visualizeDisparity(disparity, "LibSGM Disparity", stereo->getDisparityScale());
        
    } catch (const std::exception& e) {
        std::cerr << "LibSGM error: " << e.what() << std::endl;
        std::cerr << "This may be because LibSGM was built without OpenCV wrapper support" << std::endl;
    }
}

#ifdef HAVE_TENSORRT
void exampleLightStereo(const cv::Mat& left, const cv::Mat& right, 
                       const std::string& engine_path) {
    std::cout << "\n=== LightStereo Example ===" << std::endl;
    
    try {
        // Create LightStereo depth estimator
        LightStereoDepth::Params params;
        params.engine_path = engine_path;
        params.target_size = cv::Size(1248, 384);
        params.mean = {0.485f, 0.456f, 0.406f};
        params.std = {0.229f, 0.224f, 0.225f};
        params.warmup_iterations = 10;
        params.verbose = true;
        
        auto stereo = std::make_unique<LightStereoDepth>(params);
        
        // Warmup
        std::cout << "Warming up model..." << std::endl;
        stereo->warmup(left.size());
        
        // Compute disparity
        cv::Mat disparity;
        auto start = cv::getTickCount();
        stereo->compute(left, right, disparity);
        auto end = cv::getTickCount();
        
        std::cout << "LightStereo computation time: " 
                  << (end - start) / cv::getTickFrequency() * 1000.0 << " ms" << std::endl;
        std::cout << "Disparity scale: " << stereo->getDisparityScale() << std::endl;
        
        // Visualize (LightStereo already provides color disparity)
        cv::imshow("LightStereo Disparity", stereo->getColorDisparity());
        cv::imshow("LightStereo Raw Disparity", stereo->getRawDisparity());
        
    } catch (const std::exception& e) {
        std::cerr << "LightStereo error: " << e.what() << std::endl;
        std::cerr << "Make sure you have a valid TensorRT engine file" << std::endl;
    }
}
#endif  // HAVE_TENSORRT

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <left_image> <right_image> [lightstereo_engine]" 
                  << std::endl;
        return 1;
    }
    
    // Load stereo images
    cv::Mat left = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat right = cv::imread(argv[2], cv::IMREAD_COLOR);
    
    if (left.empty() || right.empty()) {
        std::cerr << "Failed to load images!" << std::endl;
        return 1;
    }
    
    std::cout << "Image size: " << left.size() << std::endl;
    
    // Show input images
    cv::imshow("Left Image", left);
    cv::imshow("Right Image", right);
    
    // Run OpenCV example
    exampleOpenCV(left, right);
    
    // Run LibSGM example (if available)
    exampleLibSGM(left, right);
    
#ifdef HAVE_TENSORRT
    // Run LightStereo example (if engine path provided)
    if (argc >= 4) {
        exampleLightStereo(left, right, argv[3]);
    }
#else
    if (argc >= 4) {
        std::cout << "\nLightStereo example skipped - TensorRT not available at build time" << std::endl;
    }
#endif
    
    std::cout << "\nPress any key to exit..." << std::endl;
    cv::waitKey(0);
    
    return 0;
}
