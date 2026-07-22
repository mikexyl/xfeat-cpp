#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(realpath "$script_dir/../..")
artifact_dir=$(realpath -m "${ARTIFACT_DIR:-$repo_root/artifacts/xfeat-lightglue}")
results_dir=$(realpath -m "${RESULTS_DIR:-$repo_root/benchmark-results}")
benchmark_bin=${BENCHMARK_BIN:-$artifact_dir/tensorrt_features_benchmark}
top_k=${TOP_K:-500}
warmup=${WARMUP:-20}
runs=${RUNS:-200}

if [[ $# -ne 2 ]]; then
  echo "Usage: benchmark_native.sh IMAGE0 IMAGE1" >&2
  exit 2
fi
for path in "$benchmark_bin" \
  "$artifact_dir/xfeat_320x224_fp16.engine" \
  "$artifact_dir/lg_320x224_dyn_n1_o500_m1024_fp16.engine" \
  "$1" "$2"; do
  if [[ ! -f "$path" ]]; then
    echo "File not found: $path" >&2
    exit 2
  fi
done

jist_args=()
jist_engine="$artifact_dir/JIST_r18_512_seqgem_simplified_fp16.engine"
if [[ -f "$jist_engine" ]]; then
  jist_args=(--jist-engine "$jist_engine")
fi

mixvpr_args=()
mixvpr_engine="$artifact_dir/mixvpr_resnet50_4096d_fp16.engine"
if [[ -f "$mixvpr_engine" ]]; then
  mixvpr_args=(--mixvpr-engine "$mixvpr_engine")
fi

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

"$benchmark_bin" \
  --xfeat-engine "$artifact_dir/xfeat_320x224_fp16.engine" \
  --lighterglue-engine "$artifact_dir/lg_320x224_dyn_n1_o500_m1024_fp16.engine" \
  --top-k "$top_k" \
  --warmup "$warmup" \
  --runs "$runs" \
  --json-out "$json_path" \
  "${jist_args[@]}" \
  "${mixvpr_args[@]}" \
  "$1" "$2"

echo "Benchmark JSON: $json_path"
echo "tegrastats log: $tegrastats_path"
