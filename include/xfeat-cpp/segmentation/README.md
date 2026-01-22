# Sky Segmentation TensorRT C++ Interface

This document provides information about the C++ interface for the sky segmentation TensorRT engine.

## Files

- **Header**: `include/xfeat-cpp/segmentation/skyseg_trt.h`
- **Implementation**: `src/segmentation/skyseg_trt.cpp`
- **Example**: `examples/skyseg_example.cpp`

## Class: `xfeat::SkySegTRT`

A C++ interface for running sky segmentation using TensorRT for high-performance inference on NVIDIA GPUs.

### Features

- **TensorRT Acceleration**: Optimized inference using TensorRT engine
- **Preprocessing**: Automatic image resizing, normalization (ImageNet), and format conversion (HWC → CHW)
- **Postprocessing**: Normalization and thresholding of segmentation masks
- **Visualization**: Built-in overlay generation with customizable colors and transparency
- **Benchmarking**: Warmup iterations and performance measurement support

### Parameters

```cpp
struct Params {
  std::string engine_path;              // Path to TensorRT engine file
  cv::Size input_size = cv::Size(320, 320);  // Model input size
  std::vector<float> mean = {0.485f, 0.456f, 0.406f};  // ImageNet mean (RGB)
  std::vector<float> std = {0.229f, 0.224f, 0.225f};   // ImageNet std (RGB)
  bool verbose = false;                 // Enable verbose output
  int warmup_iterations = 3;            // Number of warmup iterations
  float threshold = 127.0f;             // Threshold for binary mask (0-255)
};
```

### Basic Usage

```cpp
#include "xfeat-cpp/segmentation/skyseg_trt.h"

// Configure parameters
xfeat::SkySegTRT::Params params;
params.engine_path = "onnx_model/skyseg.engine";
params.input_size = cv::Size(320, 320);
params.verbose = true;

// Create instance
xfeat::SkySegTRT sky_seg(params);

// Load image
cv::Mat image = cv::imread("image.jpg");

// Run segmentation
cv::Mat mask;
sky_seg.segment(image, mask);

// Calculate sky percentage
float sky_pct = xfeat::SkySegTRT::calculateSkyPercentage(mask);
std::cout << "Sky percentage: " << sky_pct << "%" << std::endl;
```

### Segmentation with Visualization

```cpp
cv::Mat mask, overlay;
sky_seg.segmentWithVisualization(
    image, 
    mask, 
    overlay,
    0.5f,                          // alpha (transparency)
    cv::Scalar(255, 100, 0)        // color (BGR) - blue for sky
);

cv::imwrite("overlay.jpg", overlay);
```

### Warmup

```cpp
// Warmup the engine for optimal performance
sky_seg.warmup(cv::Size(640, 480));
```

## Building

The sky segmentation interface requires TensorRT support. Build with:

```bash
cmake -DHAVE_TENSORRT=ON ..
make
```

## Example Program

The example program demonstrates both single inference and benchmark modes:

```bash
# Single inference with visualization
./skyseg_example onnx_model/skyseg.engine image/sample1.jpg --show

# Benchmark mode
./skyseg_example onnx_model/skyseg.engine image/sample1.jpg --benchmark
```

### Example Output

```
Loading TensorRT engine from: onnx_model/skyseg.engine
TensorRT Sky Segmentation engine loaded successfully
Input shape: [1, 3, 320, 320]
Output shape: [1, 1, 320, 320]
Image loaded: 640x480

Warming up...
Warmup completed with 3 iterations

Running inference...
Inference time: 2.45 ms
FPS: 408.16
Sky percentage: 42.35%

Results saved to output/
  - sample1_mask.jpg
  - sample1_overlay.jpg
  - sample1_comparison.jpg
```

## API Reference

### Methods

#### `segment(const cv::Mat& image, cv::Mat& mask)`
Segment sky regions in the input image.

**Parameters:**
- `image`: Input image (BGR format, CV_8UC3)
- `mask`: Output segmentation mask (CV_8UC1, 0-255)

#### `segmentWithVisualization(...)`
Segment sky and create visualization overlay.

**Parameters:**
- `image`: Input image (BGR format)
- `mask`: Output segmentation mask
- `overlay`: Output visualization with sky overlay
- `alpha`: Transparency for overlay (0.0-1.0), default 0.5
- `color`: Sky color for overlay (BGR), default blue (255, 100, 0)

#### `warmup(const cv::Size& image_size)`
Warmup the inference engine with dummy images.

#### `getRawMask() const`
Get the raw segmentation output before thresholding.

**Returns:** CV_8UC1 mask (0-255)

#### `static calculateSkyPercentage(const cv::Mat& mask)`
Calculate the percentage of sky pixels in a mask.

**Returns:** Percentage (0-100)

## Performance

Typical performance on NVIDIA GPU (depends on hardware):
- **Inference time**: 2-5 ms
- **FPS**: 200-500
- **Speedup vs ONNX Runtime CPU**: ~50-100x

## Notes

- The engine file must be generated using the same TensorRT version as the runtime
- Input images are automatically resized to the model's input size (default 320x320)
- Output masks are resized back to the original image dimensions
- ImageNet normalization is applied: `(x/255 - mean) / std`
- The model expects RGB input (BGR is automatically converted)
