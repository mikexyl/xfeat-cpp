#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: benchmark.sh [--image IMAGE] [--top-k N] [--warmup N] [--runs N] XFEAT_ENGINE LIGHTERGLUE_ENGINE IMAGE0 IMAGE1

Runs the native C++ XFeat/LighterGlue TensorRT benchmark on a Jetson. Set
ALLOW_UNLOCKED_CLOCKS=1 to run in MAXN without static clock locking.
EOF
}

container_image=xfeat-tensorrt-features:jetson
top_k=500
warmup=20
runs=200

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      container_image=${2:?missing value for --image}
      shift 2
      ;;
    --top-k)
      top_k=${2:?missing value for --top-k}
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

if [[ $# -ne 4 ]]; then
  usage >&2
  exit 2
fi

paths=()
for path in "$@"; do
  absolute=$(realpath "$path")
  if [[ ! -f "$absolute" ]]; then
    echo "File not found: $absolute" >&2
    exit 2
  fi
  paths+=("$absolute")
done

power_mode=$(nvpmodel -q 2>/dev/null | head -n 1 || true)
if [[ "$power_mode" != *MAXN* ]]; then
  echo "Current power mode is '${power_mode:-unknown}', not MAXN." >&2
  exit 2
fi

cpu_freq_dir=/sys/devices/system/cpu/cpu0/cpufreq
gpu_freq_dir=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
clocks_locked=false
if [[ -r "$cpu_freq_dir/scaling_min_freq" && -r "$cpu_freq_dir/scaling_max_freq" \
      && -r "$gpu_freq_dir/min_freq" && -r "$gpu_freq_dir/max_freq" ]] \
    && [[ $(<"$cpu_freq_dir/scaling_min_freq") == $(<"$cpu_freq_dir/scaling_max_freq") ]] \
    && [[ $(<"$gpu_freq_dir/min_freq") == $(<"$gpu_freq_dir/max_freq") ]]; then
  clocks_locked=true
fi
if [[ "$clocks_locked" != true && "${ALLOW_UNLOCKED_CLOCKS:-0}" != 1 ]]; then
  echo "CPU/GPU clocks are not locked. Run docker/jetson-da3/performance.sh enable first." >&2
  echo "To accept dynamic clocks, rerun with ALLOW_UNLOCKED_CLOCKS=1." >&2
  exit 2
fi

results_dir=$(realpath -m "${RESULTS_DIR:-./benchmark-results}")
mkdir -p "$results_dir"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
json_path="$results_dir/xfeat-lightglue-fp16-${timestamp}.json"
tegrastats_path="$results_dir/xfeat-lightglue-fp16-tegrastats-${timestamp}.log"

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
  -v "${paths[0]}:/models/xfeat.engine:ro" \
  -v "${paths[1]}:/models/lighterglue.engine:ro" \
  -v "${paths[2]}:/inputs/image0.jpg:ro" \
  -v "${paths[3]}:/inputs/image1.jpg:ro" \
  -v "$results_dir:/results" \
  "$container_image" \
  --xfeat-engine /models/xfeat.engine \
  --lighterglue-engine /models/lighterglue.engine \
  --top-k "$top_k" \
  --warmup "$warmup" \
  --runs "$runs" \
  --json-out "/results/$(basename "$json_path")" \
  /inputs/image0.jpg \
  /inputs/image1.jpg

echo "Power mode: $power_mode"
echo "Clocks locked: $clocks_locked"
echo "Benchmark JSON: $json_path"
echo "tegrastats log: $tegrastats_path"
