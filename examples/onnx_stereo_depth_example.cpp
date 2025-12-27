/**
 * @file onnx_stereo_depth_example.cpp
 * @brief Example demonstrating ONNX-based stereo depth estimation with point cloud visualization
 * 
 * This example shows how to use the OnnxStereoDepth class to compute
 * disparity maps using ONNX Runtime with models like FastACVNet,
 * and convert the results to a point cloud for visualization.
 */

#include <xfeat-cpp/stereo_depth_onnx.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <chrono>
#include <thread>

// Convert disparity map to point cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr disparityToPointCloud(
    const cv::Mat& disparity,
    const cv::Mat& color_image,
    float focal_length,
    float baseline,
    float min_depth = 0.1f,
    float max_depth = 50.0f) {
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  
  const int width = disparity.cols;
  const int height = disparity.rows;
  const float cx = width / 2.0f;
  const float cy = height / 2.0f;
  
  cloud->width = width;
  cloud->height = height;
  cloud->is_dense = false;
  cloud->points.resize(width * height);
  
  int valid_points = 0;
  float min_depth_found = std::numeric_limits<float>::max();
  float max_depth_found = 0.0f;
  
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      float disp = disparity.at<float>(v, u);
      
      pcl::PointXYZRGB& point = cloud->points[v * width + u];
      
      // Convert disparity to depth
      if (disp > 0.1f) {  // Minimum disparity threshold
        float depth = (focal_length * baseline) / disp;
        
        // Filter invalid depths
        if (depth >= min_depth && depth <= max_depth) {
          // Project to 3D
          point.x = (u - cx) * depth / focal_length;
          point.y = (v - cy) * depth / focal_length;
          point.z = depth;
          
          // Add color
          cv::Vec3b color = color_image.at<cv::Vec3b>(v, u);
          point.r = color[2];
          point.g = color[1];
          point.b = color[0];
          
          valid_points++;
          min_depth_found = std::min(min_depth_found, depth);
          max_depth_found = std::max(max_depth_found, depth);
        } else {
          point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN();
        }
      } else {
        point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN();
      }
    }
  }
  
  std::cout << "Point cloud created with " << valid_points << " valid points" << std::endl;
  std::cout << "Depth range: [" << min_depth_found << ", " << max_depth_found << "] meters" << std::endl;
  return cloud;
}

// Downsample point cloud using voxel grid filter
pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsamplePointCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    float leaf_size = 0.01f) {
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
  
  pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
  voxel_filter.setInputCloud(cloud);
  voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
  voxel_filter.filter(*cloud_filtered);
  
  std::cout << "Downsampled from " << cloud->size() << " to " 
            << cloud_filtered->size() << " points" << std::endl;
  
  return cloud_filtered;
}

void printUsage(const char* program_name) {
  std::cout << "Usage: " << program_name << " <model_path> <left_image> <right_image> [output_prefix]" << std::endl;
  std::cout << "\nArguments:" << std::endl;
  std::cout << "  model_path      Path to ONNX model file (e.g., fast_acvnet.onnx)" << std::endl;
  std::cout << "  left_image      Path to left stereo image" << std::endl;
  std::cout << "  right_image     Path to right stereo image" << std::endl;
  std::cout << "  output_prefix   (Optional) Prefix for output files (default: 'output')" << std::endl;
  std::cout << "\nExample:" << std::endl;
  std::cout << "  " << program_name << " models/fast_acvnet.onnx left.png right.png result" << std::endl;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    printUsage(argv[0]);
    return 1;
  }

  std::string model_path = argv[1];
  std::string left_image_path = argv[2];
  std::string right_image_path = argv[3];
  std::string output_prefix = (argc > 4) ? argv[4] : "output";

  // Load stereo images
  cv::Mat left_img = cv::imread(left_image_path);
  cv::Mat right_img = cv::imread(right_image_path);

  if (left_img.empty() || right_img.empty()) {
    std::cerr << "Error: Could not load images" << std::endl;
    return 1;
  }

  std::cout << "Loaded images: " << left_img.size() << std::endl;

  // Configure ONNX stereo depth estimator
  xfeat::OnnxStereoDepth::Params params;
  params.model_path = model_path;
  params.input_size = cv::Size(512, 288);  // Must match model input size [width, height]
  params.use_cuda = true;
  params.verbose = true;
  params.warmup_iterations = 3;
  params.max_disparity = 192.0f;
  
  // Set camera parameters for depth computation and point cloud generation
  // These are approximate values - adjust based on your actual camera calibration
  // For typical stereo rigs: baseline ~5-15cm, focal_length ~image_width
  params.focal_length = left_img.cols * 0.8f;  // Approximate focal length in pixels
  params.baseline = 0.1f;                      // Approximate baseline in meters (10cm)

  try {
    // Create stereo depth estimator
    std::cout << "\nInitializing ONNX stereo depth estimator..." << std::endl;
    xfeat::OnnxStereoDepth stereo_depth(params);

    // Warm up the model
    std::cout << "Warming up..." << std::endl;
    stereo_depth.warmup(left_img.size());

    // Compute disparity
    std::cout << "\nComputing disparity map..." << std::endl;
    cv::Mat disparity;
    
    auto start = std::chrono::high_resolution_clock::now();
    stereo_depth.compute(left_img, right_img, disparity);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

    // Get statistics
    double min_disp, max_disp;
    cv::minMaxLoc(disparity, &min_disp, &max_disp);
    std::cout << "Disparity range: [" << min_disp << ", " << max_disp << "]" << std::endl;
    std::cout << "Camera parameters: focal_length=" << params.focal_length 
              << "px, baseline=" << params.baseline << "m" << std::endl;

    // Save raw disparity map
    std::string disparity_path = output_prefix + "_disparity.exr";
    cv::imwrite(disparity_path, disparity);
    std::cout << "Saved raw disparity to: " << disparity_path << std::endl;

    // Get and save color visualization
    cv::Mat color_disparity = stereo_depth.getColorDisparity();
    std::string color_path = output_prefix + "_color_disparity.png";
    cv::imwrite(color_path, color_disparity);
    std::cout << "Saved color disparity to: " << color_path << std::endl;

    // Create side-by-side comparison
    cv::Mat combined;
    cv::hconcat(left_img, color_disparity, combined);
    std::string combined_path = output_prefix + "_combined.png";
    cv::imwrite(combined_path, combined);
    std::cout << "Saved combined visualization to: " << combined_path << std::endl;

    // Visualize stereo pair with rectification lines
    std::cout << "\nCreating stereo pair visualization with rectification check..." << std::endl;
    cv::Mat stereo_viz;
    cv::hconcat(left_img, right_img, stereo_viz);
    
    // Draw horizontal lines every 50 pixels to check rectification
    for (int y = 0; y < stereo_viz.rows; y += 50) {
      cv::line(stereo_viz, cv::Point(0, y), cv::Point(stereo_viz.cols - 1, y), 
               cv::Scalar(0, 255, 0), 1);
    }
    
    // Add vertical separator line between images
    cv::line(stereo_viz, cv::Point(left_img.cols, 0), 
             cv::Point(left_img.cols, stereo_viz.rows - 1), 
             cv::Scalar(0, 0, 255), 2);
    
    std::string stereo_viz_path = output_prefix + "_stereo_rectification.png";
    cv::imwrite(stereo_viz_path, stereo_viz);
    std::cout << "Saved stereo rectification check to: " << stereo_viz_path << std::endl;
    
    // Display stereo pair
    cv::namedWindow("Stereo Rectification Check", cv::WINDOW_NORMAL);
    cv::imshow("Stereo Rectification Check", stereo_viz);
    std::cout << "Displaying stereo pair (press any key to continue)..." << std::endl;
    cv::waitKey(0);
    cv::destroyWindow("Stereo Rectification Check");

    // Compute depth map and create point cloud
    std::cout << "\nComputing depth map and generating point cloud..." << std::endl;
    cv::Mat depth;
    stereo_depth.computeDepthDirect(left_img, right_img, depth);
    
    std::string depth_path = output_prefix + "_depth.exr";
    cv::imwrite(depth_path, depth);
    std::cout << "Saved depth map to: " << depth_path << std::endl;
    
    // Visualize depth (clip to reasonable range)
    cv::Mat depth_vis;
    double max_depth_vis = 50.0;  // Maximum depth in meters for visualization
    depth.convertTo(depth_vis, CV_8U, 255.0 / max_depth_vis);
    cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_JET);
    
    std::string depth_color_path = output_prefix + "_depth_color.png";
    cv::imwrite(depth_color_path, depth_vis);
    std::cout << "Saved color depth to: " << depth_color_path << std::endl;
    
    // Convert to point cloud
    std::cout << "\nConverting to point cloud..." << std::endl;
    auto cloud = disparityToPointCloud(
        disparity, 
        left_img, 
        params.focal_length, 
        params.baseline,
        0.1f,   // min depth
        50.0f   // max depth
    );
    
    // Downsample point cloud
    std::cout << "Downsampling point cloud..." << std::endl;
    auto cloud_downsampled = downsamplePointCloud(cloud, 0.05f);  // 5cm voxel size
    
    // Save point cloud
    std::string pcd_path = output_prefix + "_cloud.pcd";
    pcl::io::savePCDFileBinary(pcd_path, *cloud_downsampled);
    std::cout << "Saved point cloud to: " << pcd_path << std::endl;
    
    // Visualize point cloud
    std::cout << "\nVisualizing point cloud (close window to continue)..." << std::endl;
    
    // Remove NaN points for better visualization
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (const auto& point : cloud_downsampled->points) {
      if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
        cloud_filtered->points.push_back(point);
      }
    }
    cloud_filtered->width = cloud_filtered->points.size();
    cloud_filtered->height = 1;
    cloud_filtered->is_dense = true;
    
    std::cout << "Filtered cloud has " << cloud_filtered->size() << " valid points" << std::endl;
    
    if (cloud_filtered->empty()) {
      std::cerr << "Warning: Point cloud is empty after filtering!" << std::endl;
      std::cerr << "Check that disparity values and camera parameters are correct." << std::endl;
    } else {
      pcl::visualization::CloudViewer viewer("Point Cloud Viewer");
      viewer.showCloud(cloud_filtered);
      
      while (!viewer.wasStopped()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }

    // Display results
    std::cout << "\nDisplaying results (press any key to close)..." << std::endl;
    cv::namedWindow("Combined Result", cv::WINDOW_NORMAL);
    cv::imshow("Combined Result", combined);
    cv::waitKey(0);

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\nDone!" << std::endl;
  return 0;
}
