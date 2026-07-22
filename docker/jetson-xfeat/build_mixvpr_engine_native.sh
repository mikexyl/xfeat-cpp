#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(realpath "$script_dir/../..")
artifact_dir=$(realpath -m "${ARTIFACT_DIR:-$repo_root/artifacts/xfeat-lightglue}")
trtexec_bin=${TRTEXEC:-/usr/bin/trtexec}
workspace_mib=${WORKSPACE_MIB:-4096}
onnx_name=mixvpr_resnet50_4096d.onnx
engine_name=mixvpr_resnet50_4096d_fp16.engine
timing_cache=orin-nx-trt-10.16.timing.cache

mkdir -p "$artifact_dir"
if [[ ! -f "$artifact_dir/$onnx_name" ]]; then
  echo "ONNX model not found: $artifact_dir/$onnx_name" >&2
  exit 2
fi

if [[ -s "$artifact_dir/$engine_name" && "${FORCE_REBUILD:-0}" != 1 ]]; then
  echo "MixVPR engine already exists: $artifact_dir/$engine_name"
  sha256sum "$artifact_dir/$onnx_name" "$artifact_dir/$engine_name"
  exit 0
fi

"$trtexec_bin" \
  --onnx="$artifact_dir/$onnx_name" \
  --saveEngine="$artifact_dir/$engine_name" \
  --timingCacheFile="$artifact_dir/$timing_cache" \
  --fp16 \
  --builderOptimizationLevel=5 \
  --avgTiming=8 \
  --memPoolSize="workspace:${workspace_mib}M" \
  --skipInference \
  2>&1 | tee "$artifact_dir/${engine_name%.engine}.build.log"

if [[ ! -s "$artifact_dir/$engine_name" ]]; then
  echo "TensorRT did not create a non-empty MixVPR engine" >&2
  exit 1
fi

sha256sum "$artifact_dir/$onnx_name" "$artifact_dir/$engine_name"
