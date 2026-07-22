#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(realpath "$script_dir/../..")
artifact_dir=$(realpath -m "${ARTIFACT_DIR:-$repo_root/artifacts}")

onnx_name=DA3-LARGE-1.1_multiview_v2_350x504.onnx
engine_name=DA3-LARGE-1.1_multiview_v2_350x504_fp16.engine
timing_cache_name=orin-nx-trt-10.16.timing.cache
container_image=${TRT_IMAGE:-nvcr.io/nvidia/tensorrt:26.05-py3}
workspace_mib=${WORKSPACE_MIB:-8192}

cpu_freq_dir=/sys/devices/system/cpu/cpu0/cpufreq
gpu_freq_dir=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
clocks_are_locked() {
  [[ -r "$cpu_freq_dir/scaling_min_freq" && -r "$cpu_freq_dir/scaling_max_freq" ]] \
    && [[ -r "$gpu_freq_dir/min_freq" && -r "$gpu_freq_dir/max_freq" ]] \
    && [[ $(<"$cpu_freq_dir/scaling_min_freq") == $(<"$cpu_freq_dir/scaling_max_freq") ]] \
    && [[ $(<"$gpu_freq_dir/min_freq") == $(<"$gpu_freq_dir/max_freq") ]]
}

onnx_path="$artifact_dir/$onnx_name"
engine_path="$artifact_dir/$engine_name"
build_log="$artifact_dir/${engine_name%.engine}.build.log"

if [[ ! -f "$onnx_path" ]]; then
  echo "ONNX model not found: $onnx_path" >&2
  exit 2
fi

power_mode=$(nvpmodel -q 2>/dev/null | head -n 1 || true)
if [[ "$power_mode" != *MAXN* ]]; then
  echo "Current power mode is '${power_mode:-unknown}', not MAXN." >&2
  echo "Run docker/jetson-da3/performance.sh enable before building." >&2
  exit 2
fi
if ! clocks_are_locked; then
  echo "CPU/GPU clocks are not statically locked at their maxima." >&2
  echo "Run docker/jetson-da3/performance.sh enable before building." >&2
  exit 2
fi

mkdir -p "$artifact_dir"
echo "Building $engine_name"
echo "  ONNX:       $onnx_path"
echo "  TensorRT:   $container_image"
echo "  Workspace:  ${workspace_mib} MiB"
echo "  Power mode: $power_mode"

docker run --rm \
  --runtime=nvidia \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$artifact_dir:/artifacts" \
  --entrypoint /opt/tensorrt/bin/trtexec \
  "$container_image" \
  --onnx="/artifacts/$onnx_name" \
  --saveEngine="/artifacts/$engine_name" \
  --timingCacheFile="/artifacts/$timing_cache_name" \
  --fp16 \
  --builderOptimizationLevel=5 \
  --avgTiming=8 \
  --memPoolSize="workspace:${workspace_mib}M" \
  --skipInference \
  2>&1 | tee "$build_log"

if [[ ! -s "$engine_path" ]]; then
  echo "TensorRT did not create a non-empty engine: $engine_path" >&2
  exit 1
fi

sha256sum "$onnx_path" "$engine_path"
echo "Engine:    $engine_path"
echo "Build log: $build_log"
