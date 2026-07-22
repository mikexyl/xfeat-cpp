# TensorRT XFeat, LighterGlue, and JIST

The native TensorRT backends load serialized engines directly and do not use
ONNX Runtime during inference:

- `xfeat::XFeatTRT` runs the XFeat network and performs heatmap, NMS, score,
  and descriptor interpolation in C++.
- `xfeat::LighterGlueTRT` supports different keypoint counts for the two images
  through a TensorRT optimization profile and handles data-dependent match
  output shapes.
- `xfeat::JistTRT` implements the `PlaceRecognizer` interface for the fixed
  five-frame JIST engine.

## Build engines

TensorRT engines are tied to the TensorRT release and target GPU. Build them
on the deployment machine:

```bash
python python/convert_feature_models_to_tensorrt.py \
  --xfeat onnx_model/xfeat_320x224.onnx \
  --lighterglue onnx_model/lg_320x224_dyn.onnx \
  --jist onnx_model/JIST_r18_512_seqgem_simplified.onnx \
  --output-dir onnx_model/trt \
  --min-keypoints 1 \
  --opt-keypoints 500 \
  --max-keypoints 1024
```

The LighterGlue engine only accepts keypoint counts within the profile range.
Increase `--max-keypoints` if the extractor is configured above that limit;
memory use grows approximately quadratically with the number of keypoints.

## C++ usage

```cpp
#include "xfeat-cpp/lighterglue_trt.h"
#include "xfeat-cpp/place_recognition/jist_trt.h"
#include "xfeat-cpp/xfeat_trt.h"

xfeat::XFeatTRT extractor({
    .engine_path = "onnx_model/trt/xfeat_320x224_fp16.engine",
});
auto features0 = extractor.detect_and_compute(image0, 500);
auto features1 = extractor.detect_and_compute(image1, 500);

xfeat::LighterGlueTRT matcher(
    "onnx_model/trt/lg_320x224_dyn_fp16.engine");
std::array<float, 2> size0 = {
    static_cast<float>(image0.cols), static_cast<float>(image0.rows)};
std::array<float, 2> size1 = {
    static_cast<float>(image1.cols), static_cast<float>(image1.rows)};
auto match_indices = matcher.match(features0, size0, features1, size1);

xfeat::JistTRT::Params jist_params;
jist_params.model_path =
    "onnx_model/trt/JIST_r18_512_seqgem_simplified_fp16.engine";
xfeat::JistTRT jist(jist_params);
cv::Mat descriptor = jist.infer(five_frame_sequence);
```

These classes are available when the project is built with TensorRT and the
`HAVE_TENSORRT` compile definition is enabled.

## Smoke tests

Engine-backed tests are opt-in because engine files are platform-specific:

```bash
XFEAT_TRT_ENGINE=/path/to/xfeat.engine \
LIGHTERGLUE_TRT_ENGINE=/path/to/lighterglue.engine \
JIST_TRT_ENGINE=/path/to/jist.engine \
./build/tests/test_tensorrt_features
```
