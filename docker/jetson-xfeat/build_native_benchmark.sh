#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(realpath "$script_dir/../..")
artifact_dir=$(realpath -m "${ARTIFACT_DIR:-$repo_root/artifacts/xfeat-lightglue}")
output="$artifact_dir/tensorrt_features_benchmark"

mkdir -p "$artifact_dir"
cd "$repo_root"

g++ \
  -std=c++17 -O3 -DNDEBUG -DHAVE_TENSORRT -mcpu=cortex-a78 -flto \
  -Iinclude -I/usr/local/cuda/include -I/usr/include/opencv4 \
  $(pkg-config --cflags opencv4) \
  src/tensorrt/trt_engine.cpp \
  src/xfeat_trt.cpp \
  src/lighterglue_trt.cpp \
  src/place_recognition/place_recognizer.cpp \
  src/place_recognition/jist_trt.cpp \
  src/place_recognition/mixvpr_trt.cpp \
  include/xfeat-cpp/nms/anms/anms.cpp \
  examples/tensorrt_features_benchmark.cpp \
  -o "$output" \
  $(pkg-config --libs opencv4) \
  -L/usr/local/cuda/lib64 \
  -lnvinfer -lcudart -pthread \
  -Wl,-O1,--as-needed

strip "$output"
echo "Benchmark binary: $output"
