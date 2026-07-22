#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: benchmark.sh [--image IMAGE] [--warmup N] [--runs N] ENGINE VIEW0 VIEW1

Runs one grouped two-view call per iteration for the fixed
DA3-LARGE-1.1_multiview_v2_350x504_fp16 engine. The TensorRT engine must have
been built on this Jetson with the same TensorRT container release.

Environment:
  RESULTS_DIR  Host output directory (default: ./benchmark-results)
EOF
}

container_image="xfeat-da3:jetson"
warmup=10
runs=100

cpu_freq_dir=/sys/devices/system/cpu/cpu0/cpufreq
gpu_freq_dir=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
clocks_are_locked() {
  [[ -r "$cpu_freq_dir/scaling_min_freq" && -r "$cpu_freq_dir/scaling_max_freq" ]] \
    && [[ -r "$gpu_freq_dir/min_freq" && -r "$gpu_freq_dir/max_freq" ]] \
    && [[ $(<"$cpu_freq_dir/scaling_min_freq") == $(<"$cpu_freq_dir/scaling_max_freq") ]] \
    && [[ $(<"$gpu_freq_dir/min_freq") == $(<"$gpu_freq_dir/max_freq") ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      container_image=${2:?missing value for --image}
      shift 2
      ;;
    --warmup)
      warmup=${2:?missing value for --warmup}
      shift 2
      ;;
    --runs)
      runs=${2:?missing value for --runs}
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -ne 3 ]]; then
  usage >&2
  exit 2
fi

engine=$(realpath "$1")
shift
if [[ ! -f "$engine" ]]; then
  echo "Engine not found: $engine" >&2
  exit 2
fi

docker_mounts=(-v "$engine:/models/da3.engine:ro")
container_images=()
index=0
for image in "$@"; do
  image=$(realpath "$image")
  if [[ ! -f "$image" ]]; then
    echo "Image not found: $image" >&2
    exit 2
  fi
  extension=${image##*.}
  container_path="/inputs/view_${index}.${extension}"
  docker_mounts+=(-v "$image:$container_path:ro")
  container_images+=("$container_path")
  index=$((index + 1))
done

mode=batch

results_dir=$(realpath -m "${RESULTS_DIR:-./benchmark-results}")
mkdir -p "$results_dir"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
json_path="$results_dir/da3-large-1.1-v2-fp16-${timestamp}.json"
tegrastats_path="$results_dir/da3-large-1.1-v2-fp16-tegrastats-${timestamp}.log"

power_mode=$(nvpmodel -q 2>/dev/null | head -n 1 || true)
if [[ "$power_mode" != *MAXN* ]]; then
  echo "Current power mode is '${power_mode:-unknown}', not MAXN." >&2
  echo "Run docker/jetson-da3/performance.sh enable before benchmarking." >&2
  exit 2
fi
if ! clocks_are_locked; then
  echo "CPU/GPU clocks are not statically locked at their maxima." >&2
  echo "Run docker/jetson-da3/performance.sh enable before benchmarking." >&2
  exit 2
fi

tegrastats --interval 100 >"$tegrastats_path" &
tegrastats_pid=$!
cleanup() {
  kill "$tegrastats_pid" 2>/dev/null || true
  wait "$tegrastats_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

docker run --rm \
  --runtime=nvidia \
  --ipc=host \
  "${docker_mounts[@]}" \
  -v "$results_dir:/results" \
  "$container_image" \
  --engine /models/da3.engine \
  --mode "$mode" \
  --warmup "$warmup" \
  --runs "$runs" \
  --json-out "/results/$(basename "$json_path")" \
  "${container_images[@]}"

echo "Benchmark JSON: $json_path"
echo "tegrastats log: $tegrastats_path"
