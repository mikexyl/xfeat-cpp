# Monocular Depth

This module adds an OpenCV-only public API for monocular depth estimation and a TensorRT backend for Depth Anything V3.
The C++ runtime loads TensorRT `.engine` files only; ONNX-to-engine conversion is handled by the Python helper.

Reference behavior follows the Depth Anything V3 TensorRT ROS 2 project:
https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt

## Convert DA3 ONNX to TensorRT

Place your DA3 ONNX file under the ignored model directory, then build an engine:

```bash
pixi run -e export convert-da3
```

Equivalent direct command:

```bash
python python/convert_depth_anything_v3_to_tensorrt.py \
  --onnx onnx_model/mono_depth/depth_anything_v3/depth_anything_v3.onnx \
  --output onnx_model/mono_depth/depth_anything_v3/depth_anything_v3.engine \
  --height 280 --width 504
```

For grouped multi-view engines with a dynamic view axis:

```bash
python python/convert_depth_anything_v3_to_tensorrt.py \
  --onnx onnx_model/mono_depth/depth_anything_v3/depth_anything_v3.onnx \
  --output onnx_model/mono_depth/depth_anything_v3/depth_anything_v3_v4.engine \
  --dynamic-views --min-views 1 --opt-views 4 --max-views 4
```

## C++ API

```cpp
#include "xfeat-cpp/mono_depth/depth_anything_v3_trt.h"

xfeat::DepthAnythingV3TRT::Params params;
params.engine_path = "onnx_model/mono_depth/depth_anything_v3/depth_anything_v3.engine";

xfeat::DepthAnythingV3TRT depth(params);

xfeat::CameraIntrinsics k;
k.fx = 721.0;
k.fy = 721.0;
k.cx = 640.0;
k.cy = 360.0;
k.width = image.cols;
k.height = image.rows;

xfeat::MonoDepthResult result = depth.infer(image, k);
cv::Mat metric_depth = result.depth;     // CV_32FC1 at input image size
cv::Mat raw_model = result.raw_depth;    // CV_32FC1 at model output size
cv::Mat sky_mask = result.sky_mask;      // CV_8UC1, non-zero means sky, may be empty
```

Grouped multi-view inference sends one set of views through one DA3 engine call:

```cpp
std::vector<cv::Mat> views = {front, left, right, rear};
std::vector<xfeat::MonoDepthResult> results = depth.infer_multi_view(views);
```

Pose-conditioned DA3 engines expose `input_extrinsics` and `input_intrinsics`. The base `MonoDepth` API stays image-only,
but `DepthAnythingV3TRT` has a class-specific overload for calibrated multi-view calls:

```cpp
std::vector<xfeat::CameraIntrinsics> intrinsics = {k0, k1, k2};
std::vector<cv::Matx44f> world_to_camera = {T_c0_w, T_c1_w, T_c2_w};
std::vector<xfeat::MonoDepthResult> results = depth.infer_multi_view(views, intrinsics, world_to_camera);
```

The runtime scales intrinsics to the resized model image before inference. For pose-conditioned engines, it also reads
DA3's predicted extrinsics output and applies a pose-scale depth correction against the input extrinsics.

## Example CLI

Single image:

```bash
build/examples/mono_depth_example \
  --engine onnx_model/mono_depth/depth_anything_v3/depth_anything_v3.engine \
  --fx 721 --fy 721 \
  --out-dir output/mono_depth \
  image/sample1.jpg
```

Grouped multi-view:

```bash
build/examples/mono_depth_example \
  --engine onnx_model/mono_depth/depth_anything_v3/depth_anything_v3_v4.engine \
  --out-dir output/mono_depth_group \
  view0.jpg view1.jpg view2.jpg view3.jpg
```

Random strided sequence example, using five frames from the Graco ground-03 sequence with a 10-frame interval:

```bash
build/examples/mono_depth_multiview_sequence_example \
  --engine onnx_model/mono_depth/depth_anything_v3/DA3METRIC-LARGE_280x504_v5_fp16.engine \
  --sequence-dir /data/graco/ground-03_images/camera_left_image_raw \
  --views 5 --interval 10 \
  --fx 940.862825677534 --fy 938.554923506332 \
  --cx 799.1626975233576 --cy 559.295406893583 \
  --out-dir output/mono_depth_graco_ground03_multiview
```

This example writes `selected_views.txt`, per-view depth/mask outputs, and `multiview_summary.png`.
The engine must accept the requested view count, either as a fixed `[5, 3, H, W]` engine or as a dynamic-view engine built with `--dynamic-views --min-views 1 --opt-views 5 --max-views 5`.

For fixed three-view pose-conditioned DA3 engines:

```bash
build/examples/mono_depth_multiview_sequence_example \
  --engine onnx_model/mono_depth/depth_anything_v3/DA3-SMALL_pose_v3_350x504_fp16.engine \
  --sequence-dir /data/graco/ground-03_images/camera_left_image_raw \
  --poses /data/graco/ground-03.txt \
  --views 3 --interval 10 --seed 1 \
  --fx 940.862825677534 --fy 938.554923506332 \
  --cx 799.1626975233576 --cy 559.295406893583 \
  --t-body-camera "0.99985436,-0.00116148,-0.01702670,-0.11655291,\
0.01702167,-0.00421530,0.99984624,0.01614558,\
-0.00123307,-0.99999044,-0.00419492,0.07950961,\
0,0,0,1" \
  --min-confidence 1.0 \
  --out-dir output/mono_depth_graco_ground03_pose_conditioned_da3_small_v3_t_imu_cam0_conf1
```

When the engine has camera inputs, this example feeds calibrated intrinsics plus per-frame world-to-camera extrinsics
from `--poses`. For GRACO ground sequences, `--poses` is an IMU/body trajectory, so `--t-body-camera` should be
`T_Imu_cam0` from `/data/graco/ground-calibration/stereo-imu.yaml` for left-camera images. It still writes per-frame
camera clouds and one fused `map_points.ply`.

The example writes `*_depth.tiff` and `*_raw_depth.tiff` as OpenCV-readable `CV_32FC1` float images. Read them with `cv::imread(path, cv::IMREAD_UNCHANGED)`. It also writes `*_depth_vis.png`, `*_metadata.yml`, `*_intrinsics.yml`, and `*_sky_mask.png` when the engine exposes a sky output. `*_intrinsics.yml` contains `fx`, `fy`, `cx`, `cy`, `width`, `height`, and the 3x3 `K` matrix needed to backproject the depth image.
It also writes a colored point cloud by default:

```text
*_cloud.ply
*_cloud_preview.png
```

The grouped-view sequence example writes each view under `point_clouds/<view_prefix>/` so `depth.tiff`, `raw_depth.tiff`,
`confidence.tiff`, `raw_confidence.tiff`, `confidence_vis.png`, `raw_confidence_vis.png`,
`confidence_filter_mask.png`, `raw_confidence_filter_mask.png`, `intrinsics.yml`, and any generated
`points_camera.ply` for that view live in the same directory. Confidence TIFFs, visualizations, and filter masks are
present only when the engine exposes a confidence output such as `depth_conf`. The confidence visualization uses a fixed
0 to 4 colorbar so threshold values can be compared across frames. In the filter masks, white pixels pass
`--min-confidence` and black pixels are rejected by the confidence threshold.

Point cloud options:

```bash
--no-cloud                 skip PLY and preview output
--cloud-stride 4           sample every N pixels
--cloud-max-depth 200      discard farther depth values
--cloud-include-sky        keep sky-mask pixels in the point cloud
--min-confidence VALUE     drop points below this absolute confidence value; 0 disables confidence filtering
```

When `--fx/--fy` are provided, the cloud uses those intrinsics. Without intrinsics, the preview and PLY use an approximate focal length based on the image size.

Ground-truth pose map from a nearby frame window:

```bash
build/examples/mono_depth_pose_map_example \
  --engine onnx_model/mono_depth/depth_anything_v3/DA3METRIC-LARGE_280x504_fp16.engine \
  --sequence-dir /data/graco/ground-03_images/camera_left_image_raw \
  --poses /data/graco/ground-03.txt \
  --out-dir output/mono_depth_graco_ground03_pose_map \
  --start-index 4247 --count 12 --interval 2 \
  --fx 940.862825677534 --fy 938.554923506332 \
  --cx 799.1626975233576 --cy 559.295406893583 \
  --t-body-camera "0.99985436,-0.00116148,-0.01702670,-0.11655291,\
0.01702167,-0.00421530,0.99984624,0.01614558,\
-0.00123307,-0.99999044,-0.00419492,0.07950961,\
0,0,0,1" \
  --cloud-stride 4 --max-depth 200
```

This demo samples a nearby strided image window, runs DA3 metric depth per image, backprojects colored points with the camera intrinsics, transforms each local cloud by `T_world_body * T_body_camera`, and writes `map_points.ply`. Poses are read as `timestamp tx ty tz qx qy qz qw` and interpreted as body-to-world by default. Pass `--poses-are-world-to-body` to invert them. The implementation uses PCL point clouds, `pcl::transformPointCloud`, and binary PLY output.

When `mono_depth_pose_map_example` is given a pose-conditioned multi-view DA3 engine, it runs selected frames in grouped
multi-view batches instead of single-image inference. Use `--multi-view-size 3` for the fixed three-view BASE/SMALL/LARGE
pose engines and `--min-confidence VALUE` to filter low-confidence DA3 points when `depth_conf` is available.
Add `--save-component-clouds` to write ICP-ready component outputs under `component_clouds/`: each
`*_points_camera.ply` is in that frame's camera coordinates, `poses_tum.txt` contains matching `T_world_camera` poses,
`poses_matrices.txt` contains row-major 4x4 `T_world_camera` matrices, and `component_index.tsv` maps clouds back to
image paths.

Full `ground-03` map with one image every 10 meters of GT path length:

```bash
build/examples/mono_depth_pose_map_example \
  --engine onnx_model/mono_depth/depth_anything_v3/DA3METRIC-LARGE_280x504_fp16.engine \
  --sequence-dir /data/graco/ground-03_images/camera_left_image_raw \
  --poses /data/graco/ground-03.txt \
  --out-dir output/mono_depth_graco_ground03_pose_map_10m_full \
  --sample-distance 10 --include-last \
  --fx 940.862825677534 --fy 938.554923506332 \
  --cx 799.1626975233576 --cy 559.295406893583 \
  --t-body-camera "0.99985436,-0.00116148,-0.01702670,-0.11655291,\
0.01702167,-0.00421530,0.99984624,0.01614558,\
-0.00123307,-0.99999044,-0.00419492,0.07950961,\
0,0,0,1" \
  --cloud-stride 4 --max-depth 200
```

Set `--sample-distance` to another value to change the spacing. Add `--voxel-size 0.2` or similar to run a PCL voxel-grid filter before writing the map.

## Benchmarking

TensorRT single-view and fixed-batch single-view benchmarking:

```bash
build/examples/mono_depth_benchmark \
  --engine onnx_model/mono_depth/depth_anything_v3/DA3METRIC-LARGE_280x504_fp16.engine \
  --mode single --warmup 10 --runs 100 \
  --fx 940.862825677534 --fy 938.554923506332 \
  --cx 799.1626975233576 --cy 559.295406893583 \
  --json-out output/mono_depth_benchmarks/single_trt.json \
  /data/graco/ground-03_images/camera_left_image_raw/1661304637150000095.png
```

```bash
build/examples/mono_depth_benchmark \
  --engine onnx_model/mono_depth/depth_anything_v3/DA3METRIC-LARGE_280x504_v5_fp16.engine \
  --mode batch --warmup 10 --runs 100 \
  --fx 940.862825677534 --fy 938.554923506332 \
  --cx 799.1626975233576 --cy 559.295406893583 \
  --json-out output/mono_depth_benchmarks/batched_single_trt_v5.json \
  view0.png view1.png view2.png view3.png view4.png
```

The C++ benchmark reports end-to-end API time and samples CUDA device memory before model load, after model load, after warmup, after benchmark, and during timed iterations.

Real DA3 multi-view benchmarking uses the upstream PyTorch API:

```bash
pixi run -e export python python/benchmark_depth_anything_v3_multiview.py \
  --da3-repo /tmp/depth-anything-3 \
  --model depth-anything/DA3-SMALL \
  --warmup 3 --runs 20 \
  --ref-view-strategy middle \
  --json-out output/mono_depth_benchmarks/real_multiview_da3_small.json \
  view0.png view1.png view2.png view3.png view4.png
```

This path calls `DepthAnything3.inference()` on all views together and reports sampled CUDA device memory plus PyTorch allocator peak/reserved memory.

## Model I/O Expectations

Expected input is BGR `CV_8UC3` from OpenCV. The runtime converts to RGB, resizes to the engine input size, converts to `[0,1]`, applies ImageNet normalization, and packs NCHW.

Common DA3 TensorRT model shape:

```text
images:           [1, 3, 280, 504] or grouped [V, 3, H, W] / [1, V, 3, H, W]
input_extrinsics: optional [1, V, 4, 4]
input_intrinsics: optional [1, V, 3, 3]
depth:            [V, 1, H, W] or [1, V, 1, H, W]
depth_conf:       optional [V, H, W] or [1, V, H, W]
extrinsics:       optional [1, V, 3, 4]
sky:              optional, same spatial shape as depth
```

The runtime prefers output tensor names containing `depth` and `sky`. If no sky output exists, sky handling is skipped and `MonoDepthResult::sky_mask` is empty.

Depth postprocessing:

- raw depth is converted to `CV_32FC1`
- negative, NaN, and infinite values are clamped to zero
- when intrinsics are provided, metric depth is scaled by `((fx * scale_x + fy * scale_y) / 2) / 300`
- when pose inputs and predicted extrinsics are present, depth is additionally divided by the estimated input-to-predicted pose scale
- when `depth_conf` is present, point-cloud examples can filter low-confidence points without changing the depth TIFFs
- sky pixels are `sky > sky_threshold`
- sky pixels are filled with `min(99th_percentile(non_sky_depth), sky_depth_cap)`
- final depth and sky mask are resized to the input image size
