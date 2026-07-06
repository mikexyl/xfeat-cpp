# Stereo Depth Quick Reference

## Quick Start

```cpp
#include "xfeat-cpp/stereo_depth/stereo_depth.h"

// Create stereo depth estimator
OpenCVStereoDepth::Params params;
auto stereo = std::make_unique<OpenCVStereoDepth>(params);

// Compute disparity
cv::Mat disparity;
stereo->compute(left_image, right_image, disparity);

// Or compute depth directly
cv::Mat depth;
stereo->computeDepth(left, right, depth, 721.5f, 0.54f);
```

## Choose Your Implementation

| Use Case | Implementation | Code |
|----------|---------------|------|
| Simple, no GPU | OpenCV BM | `OpenCVStereoDepth(Algorithm::BM)` |
| Best quality, CPU | OpenCV SGBM | `OpenCVStereoDepth(Algorithm::SGBM)` |
| Real-time, GPU | LibSGM | `LibSGMStereoDepth()` |
| State-of-art | LightStereo | `LightStereoDepth(engine_path)` |

## Key Parameters

### OpenCV
```cpp
params.num_disparities = 128;  // Max disparity (÷16)
params.block_size = 5;         // Window size (odd)
params.P1 = 200;               // Small penalty
params.P2 = 800;               // Large penalty
```

### LibSGM  
```cpp
params.num_disparities = 128;
params.P1 = 10;
params.P2 = 120;
params.subpixel = true;
params.use_gpu = true;
```

### LightStereo
```cpp
params.engine_path = "model.trt";
params.target_size = cv::Size(1248, 384);
```

## Common Operations

```cpp
// Get disparity info
int scale = stereo->getDisparityScale();  // 1 or 16
int min_d = stereo->getMinDisparity();
int num_d = stereo->getNumDisparities();

// Convert disparity to depth
stereo->disparityToDepth(disparity, depth, focal, baseline);

// Warmup GPU (optional)
stereo->warmup(cv::Size(1280, 720));
```

## Input Requirements

| Implementation | Type | Channels | Notes |
|---------------|------|----------|-------|
| OpenCV | Any | 1 or 3 | Auto converts to gray |
| LibSGM | CV_8U/16U/32S | 1 or 3 | Auto converts to gray |
| LightStereo | CV_8U | 3 (RGB) | Requires color |

## Output Formats

| Implementation | Disparity Type | Scale | Invalid Value |
|---------------|---------------|-------|---------------|
| OpenCV | CV_16S | 16 | < 0 |
| LibSGM | CV_16S | 16 | Special value |
| LightStereo | CV_32F | 1 | 0 |

## Performance Tips

1. **For OpenCV**: Use SGBM for better quality, BM for speed
2. **For LibSGM**: Always warmup GPU first, use 4-path for speed
3. **For LightStereo**: Warmup with 10+ iterations, batch processing
4. **General**: Smaller num_disparities = faster, larger = more range

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Slow performance | Use GPU implementation or reduce parameters |
| Poor quality | Increase P1/P2, use SGBM, or try LightStereo |
| Out of memory | Reduce image size or num_disparities |
| LibSGM not found | Build with `BUILD_OPENCV_WRAPPER=ON` |
| TensorRT error | Check engine file path and CUDA version |

## Example: Complete Pipeline

```cpp
#include "xfeat-cpp/stereo_depth/stereo_depth.h"
#include <opencv2/opencv.hpp>

int main() {
    // Load images
    cv::Mat left = cv::imread("left.png");
    cv::Mat right = cv::imread("right.png");
    
    // Create estimator
    OpenCVStereoDepth::Params params;
    params.num_disparities = 128;
    params.block_size = 5;
    auto stereo = std::make_unique<OpenCVStereoDepth>(params);
    
    // Compute disparity
    cv::Mat disparity;
    stereo->compute(left, right, disparity);
    
    // Visualize
    cv::Mat disp_vis;
    disparity.convertTo(disp_vis, CV_8U, 255.0/(16*64));
    cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_JET);
    cv::imshow("Disparity", disp_vis);
    
    // Compute depth
    float focal = 721.5f;      // pixels
    float baseline = 0.54f;    // meters
    cv::Mat depth;
    stereo->computeDepth(left, right, depth, focal, baseline);
    
    cv::waitKey(0);
    return 0;
}
```

## API Cheatsheet

```cpp
// Construction
OpenCVStereoDepth stereo(params);
LibSGMStereoDepth stereo(params);
LightStereoDepth stereo(params);

// Core methods
stereo.compute(left, right, disparity);
stereo.computeDepth(left, right, depth, focal, baseline);
stereo.disparityToDepth(disparity, depth, focal, baseline);

// Properties
stereo.getDisparityScale();
stereo.getMinDisparity();
stereo.getNumDisparities();
stereo.getBlockSize();
stereo.requiresGrayscale();

// GPU optimization
stereo.warmup(image_size);

// LightStereo specific
lightstereo.getRawDisparity();
lightstereo.getColorDisparity();
```
