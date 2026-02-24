# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**xfeat-cpp** is a C++17 computer vision library providing GPU-accelerated feature detection, matching, place recognition, and stereo depth estimation. It wraps deep learning models via ONNX Runtime and TensorRT.

## Build System

### Prerequisites
- ONNX Runtime GPU (1.22.0) at `../onnxruntime-linux-x64-gpu-1.22.0`
- CUDA, OpenCV, FAISS, TBB, OpenMP
- TensorRT (optional — enables LightStereo deep stereo)
- GTest (for tests), Boost + PCL (for examples)

### Build Commands
```bash
# Using presets (default: Ninja, Debug, compile_commands.json generated)
cmake --preset default
cmake --build --preset default

# Run tests
ctest --preset default

# Run individual tests
./build/tests/test_xfeat_cv
./build/tests/test_faiss
```

The `CMakePresets.json` sets `ONNXRUNTIME_ROOTDIR` and exports compile commands automatically.

### CUDA MPS (multi-process GPU)
If running multiple GPU processes simultaneously:
```bash
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
export CUDA_VISIBLE_DEVICES=0
sudo nvidia-cuda-mps-control -d
```
CUDA architecture is hardcoded to SM 89 (RTX 4090) in `CMakeLists.txt`.

### LibSGM Submodule
CMakeLists.txt automatically patches `thirdparty/libsgm/CMakeLists.txt` for export compatibility. Run `git submodule update --init --recursive` before first build.

## Architecture

### Library Target
`xfeat-cpp` is a static library. All modules share a common type system defined in [include/xfeat-cpp/types.h](include/xfeat-cpp/types.h):
```cpp
struct DetectionResult { cv::Mat keypoints; cv::Mat scores; cv::Mat descriptors; };
enum class MatcherType { BF, FLANN, LIGHTERGLUE, GPU_BF };
```

### Core Modules

**Feature Detection & Matching**
- `XFeatONNX` — Fast local feature detector/descriptor (64-dim) via ONNX Runtime. Primary class for detect+compute and direct image-pair matching.
- `XFeatCV` — OpenCV `Feature2D` wrapper around XFeatONNX for drop-in compatibility with standard OpenCV pipelines.
- `LighterGlueONNX` — Deep learning matcher; takes keypoints+descriptors from two frames and returns match pairs with confidence.

**Place Recognition**
- `NetVLADONNX` — Computes global 512-D image descriptors for loop closure. Input: `[batch, 256, 30, 40]`.
- `JistONNX` — Sequence-based place recognition (ResNet + SeqGeM). Supports streaming via `add_frame()` with a rolling buffer, or batch `infer_batch()`. Default sequence length: 5 frames, output: 512-D descriptor.
- `XFeatNetVLADONNX` — Unified model combining feature detection and place recognition.

**Vector Search**
- `FaissDatabase` — Wraps FAISS for descriptor indexing and kNN search. Supports flat (L2/cosine) and IVF-Flat indexes with optional GPU float16 compression.

**Stereo Depth** (all inherit from abstract `StereoDepth`)
- `OpenCVStereoDepth` — CPU BM/SGBM algorithms.
- `LibSGMStereoDepth` — GPU-accelerated SGM via libSGM (CUDA). Faster than CPU methods.
- `OnnxStereoDepth` — ONNX Runtime inference (FastACVNet and similar networks).
- `LightStereoDepth` — TensorRT-based deep stereo; only compiled when TensorRT is found (`HAVE_TENSORRT` defined).

**Segmentation**
- `SkySegTRT` — TensorRT sky segmentation, used to handle sky/cloud in M2DGR dataset sequences.

**Utilities**
- `helpers.h` — Keypoint uncertainty estimation (Hessian-based, Sobel, softmax), RANSAC homography.
- `nms/anms/` — Adaptive Non-Maximum Suppression using range trees.

### Typical Data Flows

Feature matching pipeline:
```
Image → XFeatONNX::detect_and_compute() → (keypoints, descriptors, scores)
      → LighterGlueONNX::match() → match pairs
```

Place recognition pipeline:
```
Video frames → JistONNX::add_frame() → 512-D descriptor
             → FaissDatabase::add() / search() → loop closure candidates
```

Stereo depth pipeline:
```
Rectified stereo pair → StereoDepth::compute() → disparity (CV_16S or CV_32F)
                      → depth map (CV_32F, meters)
```

### Conditional Compilation
TensorRT detection is automatic. If not found, `LightStereoDepth` and `SkySegTRT` are excluded and a warning is printed. Check for `#ifdef HAVE_TENSORRT` guards in stereo/segmentation code.

## ONNX Models
Models are stored in `onnx_model/` (not tracked by git). Key models:
- `xfeat_nv_fc_512.onnx` — XFeat + NetVLAD combined
- `netvlad_fc_512.onnx` — NetVLAD standalone
- `lg_640x480_dyn.onnx` — LighterGlue for 640×480
- LightStereo engines: `LightStereo-{S,M,L}-*.engine` (TensorRT)
- `skyseg.engine` — Sky segmentation (TensorRT)

Use `python/scripts/print_onnx.py` to inspect ONNX model structure.

## Code Style
A `.clang-format` config is present at the root — use it for formatting.
