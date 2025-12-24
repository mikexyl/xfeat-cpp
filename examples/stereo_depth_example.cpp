#include "xfeat-cpp/stereo_depth.h"
#include "xfeat-cpp/stereo_depth_libsgm.h"
#ifdef HAVE_TENSORRT
#include "xfeat-cpp/stereo_depth_lightstereo.h"
#endif
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

// PCL headers
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

using namespace xfeat;

// Convert disparity map to point cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr disparityToPointCloud(
    const cv::Mat& disparity,
    const cv::Mat& left_image,
    float focal_length,
    float baseline,
    int disparity_scale = 16,
    float max_distance = 100.0f) {
  
  auto cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  
  int width = disparity.cols;
  int height = disparity.rows;
  
  float principal_point_x = width / 2.0f;
  float principal_point_y = height / 2.0f;
  
  int filtered_count = 0;
  
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float disp;
      
      // Handle different disparity data types
      if (disparity.type() == CV_16S) {
        disp = disparity.at<int16_t>(y, x) / static_cast<float>(disparity_scale);
      } else if (disparity.type() == CV_32F) {
        disp = disparity.at<float>(y, x);
      } else {
        disp = disparity.at<uint8_t>(y, x) / static_cast<float>(disparity_scale);
      }
      
      // Skip invalid disparities
      if (disp <= 0) {
        continue;
      }
      
      // Calculate 3D point from disparity
      float depth = (focal_length * baseline) / disp;
      float point_x = (x - principal_point_x) * depth / focal_length;
      float point_y = (y - principal_point_y) * depth / focal_length;
      float point_z = depth;
      
      // Skip points outside bounding box (distance > max_distance)
      float distance = std::sqrt(point_x * point_x + point_y * point_y + point_z * point_z);
      if (distance > max_distance) {
        filtered_count++;
        continue;
      }
      
      pcl::PointXYZRGB point;
      point.x = point_x;
      point.y = point_y;
      point.z = point_z;
      
      // Get color from left image
      if (!left_image.empty() && left_image.type() == CV_8UC3) {
        cv::Vec3b color = left_image.at<cv::Vec3b>(y, x);
        point.b = color[0];
        point.g = color[1];
        point.r = color[2];
      } else {
        // Default color based on depth
        uint8_t depth_color = cv::saturate_cast<uint8_t>(255.0f * std::min(depth / 50.0f, 1.0f));
        point.r = depth_color;
        point.g = 128;
        point.b = 255 - depth_color;
      }
      
      cloud->points.push_back(point);
    }
  }
  
  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;
  
  std::cout << "Generated point cloud with " << cloud->points.size() << " points" << std::endl;
  if (filtered_count > 0) {
    std::cout << "Filtered " << filtered_count << " points beyond " << max_distance << "m" << std::endl;
  }
  
  return cloud;
}

// Downsample point cloud using voxel grid
pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsamplePointCloud(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    float voxel_size = 0.1f) {
  
  auto filtered = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  
  pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
  voxel_grid.setInputCloud(cloud);
  voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
  voxel_grid.filter(*filtered);
  
  std::cout << "Downsampled point cloud from " << cloud->points.size() << " to " << filtered->points.size() 
            << " points (voxel size: " << voxel_size << "m)" << std::endl;
  
  return filtered;
}

// Visualize point cloud using PCL
void visualizePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const std::string& window_name, float voxel_size = 0.1f) {
  // Downsample the point cloud before visualization
  auto downsampled_cloud = downsamplePointCloud(cloud, voxel_size);
  
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(window_name));
  
  // Set background color
  viewer->setBackgroundColor(0, 0, 0);
  
  // Add the downsampled point cloud
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(downsampled_cloud);
  viewer->addPointCloud<pcl::PointXYZRGB>(downsampled_cloud, rgb, "cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
  
  // Add coordinate frame
  viewer->addCoordinateSystem(1.0);
  
  // Set camera to look at the point cloud
  viewer->initCameraParameters();
  viewer->setCameraPosition(0, -2, -5, 0, -1, 0, 0, 0, 1);
  
  std::cout << "\nPoint cloud visualization window: " << window_name << std::endl;
  std::cout << "Controls:" << std::endl;
  std::cout << "  - Rotate: Left click + drag" << std::endl;
  std::cout << "  - Zoom: Scroll wheel or right click + drag" << std::endl;
  std::cout << "  - Pan: Middle click + drag" << std::endl;
  std::cout << "  - Press 'Q' or close window to exit" << std::endl;
  
  // Spin the viewer until 'q' is pressed
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
  }
}

// Save point cloud to PCD file
void savePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const std::string& filename, float voxel_size = 0.1f) {
  // Downsample before saving
  auto downsampled_cloud = downsamplePointCloud(cloud, voxel_size);
  pcl::io::savePCDFileASCII(filename, *downsampled_cloud);
  std::cout << "Saved downsampled point cloud to: " << filename << std::endl;
}

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
  params.block_size = 25;
  params.min_disparity = 0;
  params.P1 = 8 * 1 * params.block_size * params.block_size;
  params.P2 = 32 * 1 * params.block_size * params.block_size;

  auto stereo = std::make_unique<OpenCVStereoDepth>(params);

  // Compute disparity
  cv::Mat disparity;
  auto start = cv::getTickCount();
  stereo->compute(left, right, disparity);
  auto end = cv::getTickCount();

  std::cout << "OpenCV SGBM computation time: " << (end - start) / cv::getTickFrequency() * 1000.0 << " ms"
            << std::endl;
  std::cout << "Disparity scale: " << stereo->getDisparityScale() << std::endl;

  // Compute depth (example with typical stereo camera parameters)
  float focal_length = 721.5377f;  // pixels
  float baseline = 0.54f;          // meters

  cv::Mat depth;
  stereo->computeDepth(left, right, depth, focal_length, baseline);

  // Visualize disparity
  visualizeDisparity(disparity, "OpenCV SGBM Disparity", stereo->getDisparityScale());
  
  // Convert to point cloud and visualize
  std::cout << "\nConverting disparity to point cloud..." << std::endl;
  auto point_cloud = disparityToPointCloud(disparity, left, focal_length, baseline, stereo->getDisparityScale());
  
  // Save point cloud
  savePointCloud(point_cloud, "/tmp/opencv_sgbm_cloud.pcd");
  
  // Visualize point cloud
  visualizePointCloud(point_cloud, "OpenCV SGBM Point Cloud");
}

void exampleLibSGM(const cv::Mat& left, const cv::Mat& right) {
  std::cout << "\n=== LibSGM Example ===" << std::endl;

  try {
    // Create LibSGM stereo depth estimator
    LibSGMStereoDepth::Params params;
    params.num_disparities = 128;
    params.P1 = 10;
    params.P2 = 120;
    params.uniqueness_ratio = 0.9f;
    params.subpixel = true;
    params.path_type = 1;   // SCAN_8PATH
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

    std::cout << "LibSGM computation time: " << (end - start) / cv::getTickFrequency() * 1000.0 << " ms" << std::endl;
    std::cout << "Disparity scale: " << stereo->getDisparityScale() << std::endl;

    // Visualize disparity
    visualizeDisparity(disparity, "LibSGM Disparity", stereo->getDisparityScale());
    
    // Convert to point cloud and visualize
    std::cout << "\nConverting disparity to point cloud..." << std::endl;
    float focal_length = 721.5377f;
    float baseline = 0.54f;
    auto point_cloud = disparityToPointCloud(disparity, left, focal_length, baseline, stereo->getDisparityScale());
    
    // Save point cloud
    savePointCloud(point_cloud, "/tmp/libsgm_cloud.pcd");
    
    // Visualize point cloud
    visualizePointCloud(point_cloud, "LibSGM Point Cloud");

  } catch (const std::exception& e) {
    std::cerr << "LibSGM error: " << e.what() << std::endl;
    std::cerr << "This may be because LibSGM was built without OpenCV wrapper support" << std::endl;
  }
}

#ifdef HAVE_TENSORRT
void exampleLightStereo(const cv::Mat& left, const cv::Mat& right, const std::string& engine_path) {
  std::cout << "\n=== LightStereo Example ===" << std::endl;

  try {
    // Create LightStereo depth estimator
    LightStereoDepth::Params params;
    params.engine_path = engine_path;
    // params.target_size = cv::Size(1248, 384);
    // params.target_size = cv::Size(640, 480);
    params.target_size = cv::Size(512, 288);
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

    std::cout << "LightStereo disparity size: " << disparity.size() << std::endl;

    std::cout << "LightStereo computation time: " << (end - start) / cv::getTickFrequency() * 1000.0 << " ms"
              << std::endl;
    std::cout << "Disparity scale: " << stereo->getDisparityScale() << std::endl;

    // Visualize (LightStereo already provides color disparity)
    cv::imshow("LightStereo Disparity", stereo->getColorDisparity());
    cv::imshow("LightStereo Raw Disparity", stereo->getRawDisparity());
    
    // Convert to point cloud and visualize
    std::cout << "\nConverting disparity to point cloud..." << std::endl;
    float focal_length = 721.5377f;
    float baseline = 0.54f;
    auto point_cloud = disparityToPointCloud(disparity, left, focal_length, baseline, stereo->getDisparityScale());
    
    // Save point cloud
    savePointCloud(point_cloud, "/tmp/lightstereo_cloud.pcd");
    
    // Visualize point cloud
    visualizePointCloud(point_cloud, "LightStereo Point Cloud");

  } catch (const std::exception& e) {
    std::cerr << "LightStereo error: " << e.what() << std::endl;
    std::cerr << "Make sure you have a valid TensorRT engine file" << std::endl;
  }
}
#endif  // HAVE_TENSORRT

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <left_image> <right_image> [lightstereo_engine]" << std::endl;
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
