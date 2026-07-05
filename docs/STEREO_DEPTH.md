# Stereo Depth Estimation Classes

This directory contains a comprehensive stereo depth estimation framework with multiple implementations.

## Overview

The framework provides a base `StereoDepth` class with three concrete implementations:

1. **OpenCVStereoDepth** - OpenCV's built-in algorithms (BM and SGBM)
2. **LibSGMStereoDepth** - GPU-accelerated Semi-Global Matching
3. **LightStereoDepth** - Deep learning-based depth estimation using TensorRT

## Class Hierarchy

```
StereoDepth (abstract base class)
├── OpenCVStereoDepth
├── LibSGMStereoDepth
└── LightStereoDepth
```

## Base Class API

The `StereoDepth` base class provides a common interface:

### Core Methods

```cpp
// Compute disparity map from stereo pair
virtual void compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity) = 0;

// Compute depth map from stereo pair
virtual void computeDepth(const cv::Mat& left, const cv::Mat& right, cv::Mat& depth, 
                         float focal_length, float baseline);

// Convert disparity to depth
virtual void disparityToDepth(const cv::Mat& disparity, cv::Mat& depth,
                             float focal_length, float baseline);
```

### Property Methods

```cpp
virtual int getDisparityScale() const = 0;     // Scaling factor (e.g., 16 for subpixel)
virtual int getMinDisparity() const = 0;        // Minimum disparity value
virtual int getNumDisparities() const = 0;      // Number of disparity levels
virtual int getBlockSize() const = 0;           // Block/window size
virtual bool requiresGrayscale() const;         // Whether grayscale input is required
virtual void warmup(const cv::Size& image_size); // GPU warmup (optional)
```

## Implementations

### 1. OpenCVStereoDepth

Uses OpenCV's built-in stereo matching algorithms.

**Features:**
- Block Matching (BM) - Fast, suitable for simple scenes
- Semi-Global Block Matching (SGBM) - More accurate, better for complex scenes
- CPU-based processing
- No external dependencies beyond OpenCV

**Example:**
```cpp
#include "xfeat-cpp/stereo_depth/stereo_depth.h"

OpenCVStereoDepth::Params params;
params.algorithm = OpenCVStereoDepth::Algorithm::SGBM;
params.num_disparities = 128;  // Must be divisible by 16
params.block_size = 5;          // Odd number, typically 3-21
params.P1 = 8 * 1 * params.block_size * params.block_size;
params.P2 = 32 * 1 * params.block_size * params.block_size;

auto stereo = std::make_unique<OpenCVStereoDepth>(params);

cv::Mat disparity;
stereo->compute(left, right, disparity);
```

**Parameters:**
- `algorithm`: BM or SGBM
- `num_disparities`: Maximum disparity range (must be divisible by 16)
- `block_size`: Window size for matching (odd number)
- `P1`, `P2`: Smoothness penalties for SGBM
- `uniqueness_ratio`: Margin for uniqueness check
- `speckle_window_size`, `speckle_range`: Noise filtering

**Output:**
- Disparity type: `CV_16S`
- Disparity scale: 16 (subpixel precision)
- Invalid disparities: negative values

### 2. LibSGMStereoDepth

GPU-accelerated Semi-Global Matching using the LibSGM library.

**Features:**
- CUDA GPU acceleration
- 4-path or 8-path optimization
- Subpixel precision
- Census transform
- Left-Right consistency check
- Significantly faster than CPU implementations

**Example:**
```cpp
#include "xfeat-cpp/stereo_depth/stereo_depth_libsgm.h"

LibSGMStereoDepth::Params params;
params.num_disparities = 128;
params.P1 = 10;
params.P2 = 120;
params.uniqueness_ratio = 0.95f;
params.subpixel = true;
params.path_type = sgm::PathType::SCAN_8PATH;
params.use_gpu = true;

auto stereo = std::make_unique<LibSGMStereoDepth>(params);

// Warmup GPU for optimal performance
stereo->warmup(image_size);

cv::Mat disparity;
stereo->compute(left, right, disparity);
```

**Parameters:**
- `num_disparities`: Maximum disparity range
- `P1`: Penalty for small disparity changes (±1)
- `P2`: Penalty for large disparity changes (>1)
- `uniqueness_ratio`: Uniqueness check threshold
- `subpixel`: Enable subpixel precision (4 fractional bits)
- `path_type`: SCAN_4PATH or SCAN_8PATH
- `lr_max_diff`: Max difference for LR consistency (-1 to disable)
- `census_type`: Census transform type
- `use_gpu`: Enable GPU acceleration

**Output:**
- Disparity type: `CV_16S`
- Disparity scale: 16 (if subpixel enabled), 1 otherwise
- Invalid disparities: special value from `getInvalidDisparity()`

**Requirements:**
- CUDA-capable GPU
- LibSGM compiled with `BUILD_OPENCV_WRAPPER=ON`

### 3. LightStereoDepth

Deep learning-based stereo depth estimation using TensorRT.

**Features:**
- State-of-the-art accuracy
- Real-time performance on GPU
- Direct RGB input (no rectification needed)
- Produces smooth, dense disparity maps
- Built-in visualization outputs

**Example:**
```cpp
#include "xfeat-cpp/stereo_depth/stereo_depth_lightstereo.h"

LightStereoDepth::Params params;
params.engine_path = "lightstereo.trt";
params.target_size = cv::Size(1248, 384);
params.mean = {0.485f, 0.456f, 0.406f};
params.std = {0.229f, 0.224f, 0.225f};
params.warmup_iterations = 10;

auto stereo = std::make_unique<LightStereoDepth>(params);

// Warmup for optimal performance
stereo->warmup(image_size);

cv::Mat disparity;
stereo->compute(left, right, disparity);

// Get color visualization
cv::Mat color_disp = stereo->getColorDisparity();
```

**Parameters:**
- `engine_path`: Path to TensorRT engine file (.trt)
- `target_size`: Target input size for the model
- `mean`, `std`: ImageNet normalization parameters
- `warmup_iterations`: Number of warmup runs
- `verbose`: Enable verbose logging
- `max_disparity`: Maximum disparity value for the model

**Output:**
- Disparity type: `CV_32F`
- Disparity scale: 1 (already in pixels)
- Additional outputs: raw disparity, color-mapped visualization

**Requirements:**
- CUDA-capable GPU
- TensorRT installed
- Pre-trained LightStereo model converted to TensorRT engine

## Depth Computation

All implementations support depth computation from disparity:

```cpp
// Direct depth computation
float focal_length = 721.5377f;  // pixels
float baseline = 0.54f;           // meters

cv::Mat depth;
stereo->computeDepth(left, right, depth, focal_length, baseline);

// Or convert existing disparity to depth
cv::Mat disparity;
stereo->compute(left, right, disparity);
stereo->disparityToDepth(disparity, depth, focal_length, baseline);
```

The depth is computed as:
```
depth = (baseline × focal_length) / disparity
```

Output depth is in meters (`CV_32F`). Invalid disparities produce depth = 0.

## Usage Examples

See `examples/stereo_depth_example.cpp` for complete usage examples.

### Building

```bash
# Build the library
cd build
cmake ..
make

# Run example
./examples/stereo_depth_example left.png right.png [lightstereo.trt]
```

## Performance Comparison

Typical performance on KITTI stereo images (1242×375):

| Method | Platform | Time | Quality |
|--------|----------|------|---------|
| OpenCV BM | CPU | ~50 ms | Good |
| OpenCV SGBM | CPU | ~200 ms | Very Good |
| LibSGM | GPU (RTX 3090) | ~5 ms | Very Good |
| LightStereo | GPU (RTX 3090) | ~10 ms | Excellent |

## Parameter Tuning

### For Indoor Scenes:
- Use smaller `num_disparities` (64-96)
- Smaller `block_size` (3-5)
- Higher `P1`, `P2` for smoother results

### For Outdoor Scenes:
- Larger `num_disparities` (128-192)
- Larger `block_size` (5-7)
- Lower `P1`, `P2` for more detail

### For Real-time Applications:
- Use LibSGM with GPU
- Use 4-path instead of 8-path
- Disable subpixel precision if not needed

## Error Handling

All implementations throw `std::runtime_error` or `std::invalid_argument` for errors:

```cpp
try {
    stereo->compute(left, right, disparity);
} catch (const std::invalid_argument& e) {
    std::cerr << "Invalid input: " << e.what() << std::endl;
} catch (const std::runtime_error& e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
}
```

## References

1. Hirschmuller, H. (2008). Stereo Processing by Semiglobal Matching and Mutual Information
2. OpenCV Stereo Matching: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
3. LibSGM: https://github.com/fixstars/libSGM
4. LightStereo: [Paper/Repository Link]

## License

See individual library licenses:
- OpenCV: Apache 2.0
- LibSGM: Apache 2.0
- LightStereo: [License]
