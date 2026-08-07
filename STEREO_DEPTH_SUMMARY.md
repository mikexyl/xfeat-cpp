# Stereo Depth Implementation Summary

## Overview

This project provides a stereo depth framework with a base class and four implementations.

### Files Created

**Header Files:**
1. `/workspaces/src/xfeat-cpp/include/xfeat-cpp/stereo_depth/stereo_depth.h` - Base class and OpenCV implementation
2. `/workspaces/src/xfeat-cpp/include/xfeat-cpp/stereo_depth/stereo_depth_lightstereo.h` - LightStereo deep learning implementation
3. `/workspaces/src/xfeat-cpp/include/xfeat-cpp/stereo_depth/stereo_depth_onnx.h` - Generic ONNX Runtime implementation
4. `/workspaces/src/xfeat-cpp/include/xfeat-cpp/stereo_depth/stereo_depth_fast_foundation_stereo.h` - Fast-FoundationStereo single-engine implementation

**Implementation Files:**
1. `/workspaces/src/xfeat-cpp/src/stereo_depth/stereo_depth.cpp` - Base class and OpenCV implementation
2. `/workspaces/src/xfeat-cpp/src/stereo_depth/stereo_depth_lightstereo.cpp` - LightStereo implementation
3. `/workspaces/src/xfeat-cpp/src/stereo_depth/stereo_depth_onnx.cpp` - Generic ONNX Runtime implementation
4. `/workspaces/src/xfeat-cpp/src/stereo_depth/stereo_depth_fast_foundation_stereo.cpp` - Fast-FoundationStereo shared TensorRT engine wrapper

**Example and Documentation:**
1. `/workspaces/src/xfeat-cpp/examples/stereo_depth_example.cpp` - Complete usage example
2. `/workspaces/src/xfeat-cpp/docs/STEREO_DEPTH.md` - Comprehensive documentation

## Class Hierarchy

```
StereoDepth (abstract base class)
├── OpenCVStereoDepth (CPU-based Block Matching and SGBM)
├── LightStereoDepth (Deep learning with TensorRT)
├── OnnxStereoDepth (Learned stereo with ONNX Runtime)
└── FastFoundationStereoDepth (Official single TensorRT engine and GWC plugin)
```

## Key Features

### Base Class (StereoDepth)
- `compute()` - Compute disparity map
- `computeDepth()` - Compute depth map with calibration parameters
- `disparityToDepth()` - Convert disparity to depth
- Property getters for algorithm parameters
- Optional `warmup()` for GPU-based implementations

### OpenCVStereoDepth
- Supports Block Matching (BM) and Semi-Global Block Matching (SGBM)
- CPU-based, no external dependencies beyond OpenCV
- Fully configurable parameters (P1, P2, uniqueness ratio, etc.)
- 16x subpixel precision

### LightStereoDepth
- State-of-the-art deep learning model
- TensorRT inference for real-time performance
- Works with RGB images (no rectification needed)
- Built-in preprocessing and visualization
- Smooth, dense disparity maps

## Design Decisions

1. **Abstract Base Class**: Provides a common interface for all implementations, making it easy to swap algorithms

2. **PIMPL Pattern**: LightStereo uses PIMPL (Pointer to Implementation) to hide TensorRT dependencies

3. **Params Structs**: Each implementation has a Params struct for configuration with sensible defaults

4. **Error Handling**: All methods throw std::runtime_error or std::invalid_argument for errors

5. **OpenCV Integration**: All classes use cv::Mat for input/output compatibility

6. **GPU Optimization**: Learned backends support GPU acceleration with warmup methods

## Usage Example

```cpp
// OpenCV SGBM
OpenCVStereoDepth::Params opencv_params;
opencv_params.num_disparities = 128;
auto opencv_stereo = std::make_unique<OpenCVStereoDepth>(opencv_params);

// LightStereo (Deep Learning)
LightStereoDepth::Params lightstereo_params;
lightstereo_params.engine_path = "model.trt";
auto lightstereo_stereo = std::make_unique<LightStereoDepth>(lightstereo_params);

// Use any implementation with the same interface
cv::Mat disparity;
stereo->compute(left, right, disparity);

// Or compute depth directly
cv::Mat depth;
stereo->computeDepth(left, right, depth, focal_length, baseline);
```

## Performance Comparison

Based on typical stereo images (1242×375):

| Method | Platform | Time | Quality |
|--------|----------|------|---------|
| OpenCV BM | CPU | ~50 ms | Good |
| OpenCV SGBM | CPU | ~200 ms | Very Good |
| LightStereo | GPU | ~10 ms | Excellent |

## Build Requirements

### Base + OpenCV Implementation:
- OpenCV 4.x with calib3d module
- C++14 or later

### LightStereo Implementation:
- CUDA Toolkit
- TensorRT 8.x or later
- Pre-trained model converted to TensorRT engine

## Next Steps

To use these classes in your project:

1. **Add to CMakeLists.txt**:
```cmake
# Add stereo depth library
add_library(stereo_depth
    src/stereo_depth/stereo_depth.cpp
    src/stereo_depth/stereo_depth_lightstereo.cpp
)

target_link_libraries(stereo_depth
    PUBLIC
        ${OpenCV_LIBS}
    PRIVATE
        nvinfer      # if using LightStereo
        cudart
)

# Build example
add_executable(stereo_depth_example examples/stereo_depth_example.cpp)
target_link_libraries(stereo_depth_example stereo_depth)
```

2. **Build and run**:
```bash
mkdir build && cd build
cmake ..
make
./stereo_depth_example left.png right.png
```

3. **Integrate into your application**:
```cpp
#include "xfeat-cpp/stereo_depth/stereo_depth.h"

// Choose implementation based on your needs
auto stereo = std::make_unique<OpenCVStereoDepth>();
cv::Mat disparity;
stereo->compute(left, right, disparity);
```

## Notes

- All implementations handle grayscale conversion automatically
- Disparity values may be scaled (check `getDisparityScale()`)
- Invalid disparities are marked as 0 or negative values
- GPU implementations benefit from warmup to initialize CUDA kernels
- LightStereo produces the highest quality but requires a trained model
