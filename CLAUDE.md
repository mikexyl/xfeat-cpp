# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**xfeat-cpp** is a C++17 computer vision library providing GPU-accelerated feature detection, matching, place recognition, and stereo depth estimation. It wraps deep learning models via ONNX Runtime and TensorRT. All public classes live in `namespace xfeat`.

## Build System

### Prerequisites
- ONNX Runtime GPU (1.22.0) at `../onnxruntime-linux-x64-gpu-1.22.0`
- CUDA, OpenCV, FAISS, TBB, OpenMP
- TensorRT (optional — enables LightStereo deep stereo and sky segmentation)
- GTest (for tests), Boost + PCL (for examples)

### Build Commands
```bash
# Using presets (default: Ninja, Debug, compile_commands.json generated)
cmake --preset default
cmake --build --preset default

# Run tests
ctest --preset default

# Run individual tests (test images expected at image/sample1.jpg, image/sample2.jpg)
./build/tests/test_xfeat_cv
./build/tests/test_faiss
```

The `CMakePresets.json` sets `ONNXRUNTIME_ROOTDIR` and exports compile commands automatically. CUDA architecture is hardcoded to SM 89 (RTX 4090) in `CMakeLists.txt`.

### Install (for downstream CMake projects)
```bash
cmake --install build --prefix /path/to/install
# Downstream: find_package(xfeat-cpp REQUIRED)
```

### Submodules
The project uses tracked submodules under `thirdparty/`, including `gms`, `lightstereo`, and Fast-FoundationStereo. Run `git submodule update --init --recursive` before first build.

### CUDA MPS (multi-process GPU)
If running multiple GPU processes simultaneously:
```bash
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
export CUDA_VISIBLE_DEVICES=0
sudo nvidia-cuda-mps-control -d
```

## Architecture

### Library Target
`xfeat-cpp` is a static library. All modules share a common type system defined in `include/xfeat-cpp/types.h`:
```cpp
struct DetectionResult { cv::Mat keypoints; cv::Mat scores; cv::Mat descriptors; };
enum class MatcherType { BF, FLANN, LIGHTERGLUE, GPU_BF };
```

### Core Modules

**Feature Detection & Matching**
- `XFeatONNX` — Fast local feature detector/descriptor (64-dim) via ONNX Runtime. Primary class for detect+compute and direct image-pair matching.
- `XFeatCV` — OpenCV `Feature2D` wrapper around XFeatONNX for drop-in compatibility with standard OpenCV pipelines.
- `LighterGlueONNX` — Deep learning matcher; takes keypoints+descriptors from two frames and returns match pairs with confidence.
- `LighterGlueCV` — OpenCV-style wrapper around LighterGlueONNX; default image size 640×480.

**Place Recognition**

All VPR models inherit from the abstract base class `PlaceRecognizer` (`include/xfeat-cpp/place_recognition/place_recognizer.h`). It provides:
- `infer(images)` / `infer(image)` — pure virtual; returns a descriptor `cv::Mat` (1 × dim, CV_32F)
- `infer_batch(batch)` — default loops over `infer()`; override for GPU batching
- `add_frame(image, descriptor)` / `reset_buffer()` — concrete streaming API backed by a rolling `frame_buffer_` deque
- `get_seq_length()` / `get_descriptor_dim()` — pure virtual; single-image models return seq_length=1

Concrete implementations:
- `JistONNX` — Sequence-based (ResNet + SeqGeM). Default: 5 frames, 288×512 input, 512-D output.
- `PatchNetVLADONNX` — Single-image (VGG-16 + NetVLAD). `infer()` returns a 4096-D global descriptor; `extract()` returns global + per-scale local patch descriptors for re-ranking. Default: 480×640 input, 4096-D output.
- `PatchNetVLADMatcher` — Local patch RANSAC re-ranker. Ports `PatchMatcher.compare_two_ransac()` from Python. Computes mutual nearest-neighbor matches on 3 patch scales then runs RANSAC homography; returns a weighted inlier-ratio score. Default patch sizes: {2, 5, 8}, weights: {0.45, 0.15, 0.40}.
- `MixVPRONNX` — Single-image (ResNet-50 + MLP mixer). `infer()` returns a 4096-D L2-normalized global descriptor. Default: 320×320 input, 4096-D output.

**Vector Search**
- `FaissDatabase` — Wraps FAISS for descriptor indexing and kNN search.
  - `IndexMode::kFlat` — flat inner-product index (cosine similarity for L2-normalized vectors); created in-memory with `dim` parameter.
  - `IndexMode::kIVFFlat` — pre-trained IVF-Flat index loaded from file; supports GPU float16 compression.
  - Both modes support optional GPU acceleration.

**Stereo Depth** (all inherit from abstract `StereoDepth`)
- `OpenCVStereoDepth` — CPU BM/SGBM algorithms.
- `OnnxStereoDepth` — ONNX Runtime inference (FastACVNet and similar networks).
- `LightStereoDepth` — TensorRT-based deep stereo; only compiled when TensorRT is found (`HAVE_TENSORRT` defined).

**Segmentation**
- `SkySegTRT` — TensorRT sky segmentation, used to handle sky/cloud in M2DGR dataset sequences. Guarded by `#ifdef HAVE_TENSORRT`.

**Utilities**
- `helpers.h` — Keypoint uncertainty estimation (Hessian-based, Sobel, softmax), RANSAC homography.
- `nms/anms/` — Adaptive Non-Maximum Suppression using range trees.

### Conditional Compilation
TensorRT detection is automatic. If not found, `LightStereoDepth` and `SkySegTRT` are excluded and a warning is printed. Check for `#ifdef HAVE_TENSORRT` guards in stereo/segmentation code.

### Typical Data Flows

Feature matching pipeline:
```
Image → XFeatONNX::detect_and_compute() → (keypoints, descriptors, scores)
      → LighterGlueONNX::match() → match pairs
```

Place recognition pipeline (JIST, sequence-based):
```
Video frames → JistONNX::add_frame() → 512-D descriptor
             → FaissDatabase::add() / search() → loop closure candidates
```

Place recognition pipeline (PatchNetVLAD, single-image + re-ranking):
```
Image → PatchNetVLADONNX::extract() → Features{global_desc [1,4096], local_descs [3 scales]}
      → FaissDatabase::search() using global_desc → top-k candidates
      → PatchNetVLADMatcher::match(query_features, db_features) → MatchResult::score → re-ranked list
```

Place recognition pipeline (MixVPR, single-image):
```
Image → MixVPRONNX::infer() → global_desc [1,4096] (L2-normalized)
      → FaissDatabase::add() / search() → loop closure candidates
```

Stereo depth pipeline:
```
Rectified stereo pair → StereoDepth::compute() → disparity (CV_16S or CV_32F)
                      → depth map (CV_32F, meters)
```

## Examples

Built under `examples/`; require Boost + PCL. Demonstrate real usage patterns:
- `main` — XFeat feature detection and matching
- `feature_nms` — NMS on detected keypoints
- `stereo_depth_example` / `onnx_stereo_depth_example` — stereo depth pipelines
- `patchnetvlad_example` / `mixvpr_example` — place recognition pipelines
- `skyseg_example` — sky segmentation (TensorRT)

## Scripts & Utilities

C++ utility executables built under `scripts/`:
- `train_faiss_ivfflat` — Train a FAISS IVF-Flat index from saved descriptors.

Python utilities:
- `python/scripts/print_onnx.py` — Inspect ONNX model inputs/outputs/shapes.
- `python/convert_skyseg_to_tensorrt.py` — Convert sky segmentation ONNX model to TensorRT engine.

## ONNX Models
Models are stored in `onnx_model/` (not tracked by git). Key models:
- `lg_640x480_dyn.onnx` — LighterGlue for 640×480
- `patchnetvlad_trt.onnx` — PatchNetVLAD; input `[1,3,480,640]`, outputs `global_feat [1,4096]` + `local_0/1/2 [1,4096,P_i]`
- `mixvpr_resnet50_4096d.onnx` — MixVPR; input `image [1,3,320,320]`, output `descriptor [1,4096]`
- LightStereo engines: `LightStereo-{S,M,L}-*.engine` (TensorRT)
- `skyseg.engine` — Sky segmentation (TensorRT)

## Code Style
A `.clang-format` config is present at the root — use it for formatting.
