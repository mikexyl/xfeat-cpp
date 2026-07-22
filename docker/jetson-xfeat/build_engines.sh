#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(realpath "$script_dir/../..")
artifact_dir=$(realpath -m "${ARTIFACT_DIR:-$repo_root/artifacts/xfeat-lightglue}")
container_image=${TRT_IMAGE:-nvcr.io/nvidia/tensorrt:26.05-py3}
workspace_mib=${WORKSPACE_MIB:-4096}

xfeat_onnx=xfeat_320x224.onnx
lighterglue_onnx=lg_320x224_dyn.onnx
xfeat_engine=xfeat_320x224_fp16.engine
lighterglue_engine=lg_320x224_dyn_n1_o500_m1024_fp16.engine
timing_cache=orin-nx-trt-10.16.timing.cache

mkdir -p "$artifact_dir"
for model in "$xfeat_onnx" "$lighterglue_onnx"; do
  if [[ ! -f "$artifact_dir/$model" ]]; then
    echo "ONNX model not found: $artifact_dir/$model" >&2
    exit 2
  fi
done

echo "Building XFeat FP16 engine on the Jetson"
docker run --rm \
  --runtime=nvidia \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$artifact_dir:/artifacts" \
  --entrypoint /opt/tensorrt/bin/trtexec \
  "$container_image" \
  --onnx="/artifacts/$xfeat_onnx" \
  --saveEngine="/artifacts/$xfeat_engine" \
  --timingCacheFile="/artifacts/$timing_cache" \
  --fp16 \
  --builderOptimizationLevel=5 \
  --avgTiming=8 \
  --memPoolSize="workspace:${workspace_mib}M" \
  --skipInference \
  2>&1 | tee "$artifact_dir/${xfeat_engine%.engine}.build.log"

echo "Building LighterGlue FP16 engine with a 1/500/1024-keypoint profile"
docker run --rm \
  --runtime=nvidia \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$artifact_dir:/artifacts" \
  --entrypoint /opt/tensorrt/bin/trtexec \
  "$container_image" \
  --onnx="/artifacts/$lighterglue_onnx" \
  --saveEngine="/artifacts/$lighterglue_engine" \
  --timingCacheFile="/artifacts/$timing_cache" \
  --minShapes=mkpts0:1x1x2,feats0:1x1x64,mkpts1:1x1x2,feats1:1x1x64 \
  --optShapes=mkpts0:1x500x2,feats0:1x500x64,mkpts1:1x500x2,feats1:1x500x64 \
  --maxShapes=mkpts0:1x1024x2,feats0:1x1024x64,mkpts1:1x1024x2,feats1:1x1024x64 \
  --fp16 \
  --builderOptimizationLevel=5 \
  --avgTiming=8 \
  --memPoolSize="workspace:${workspace_mib}M" \
  --skipInference \
  2>&1 | tee "$artifact_dir/${lighterglue_engine%.engine}.build.log"

for engine in "$xfeat_engine" "$lighterglue_engine"; do
  if [[ ! -s "$artifact_dir/$engine" ]]; then
    echo "TensorRT did not create a non-empty engine: $artifact_dir/$engine" >&2
    exit 1
  fi
done

sha256sum \
  "$artifact_dir/$xfeat_onnx" \
  "$artifact_dir/$xfeat_engine" \
  "$artifact_dir/$lighterglue_onnx" \
  "$artifact_dir/$lighterglue_engine"
