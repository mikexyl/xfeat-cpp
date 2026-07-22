# Jetson XFeat/LighterGlue/JIST/MixVPR TensorRT benchmark

This benchmark targets JetPack 7.2 / L4T R39.2 on Jetson Orin and uses the
TensorRT 26.05 container (CUDA 13.2, TensorRT 10.16). It reports warmup-excluded
latency distributions for XFeat extraction, cached-feature LighterGlue
matching, MixVPR place descriptors, and the complete two-image
feature-and-match pipeline.

Build the small native C++ benchmark image from the repository root:

```bash
docker build --file docker/jetson-xfeat/Dockerfile --tag xfeat-tensorrt-features:jetson .
```

On a Jetson with the TensorRT and OpenCV development packages already present,
the fully offline host-native path is also available:

```bash
docker/jetson-xfeat/build_native_benchmark.sh
docker/jetson-xfeat/build_engines_native.sh
docker/jetson-xfeat/build_jist_engine_native.sh
docker/jetson-xfeat/build_mixvpr_engine_native.sh
docker/jetson-xfeat/benchmark_native.sh image/sample1.jpg image/sample2.jpg
```

When `JIST_r18_512_seqgem_simplified_fp16.engine` is present in the artifact
directory, the native benchmark also measures one five-frame JIST sequence per
call. The two input images are alternated to form the test sequence.
When `mixvpr_resnet50_4096d_fp16.engine` is present, it also measures one
320×320, 4096-dimensional MixVPR descriptor per call.

Copy these portable models into `artifacts/xfeat-lightglue/`:

```text
xfeat_320x224.onnx
lg_320x224_dyn.onnx
JIST_r18_512_seqgem_simplified.onnx
mixvpr_resnet50_4096d.onnx
```

For repeatable peak-performance results, enable the same MAXN and static-clock
configuration used by the DA3 benchmark, then build the engines natively:

```bash
docker/jetson-da3/performance.sh enable
docker/jetson-xfeat/build_engines.sh
```

Run the application-level benchmark with two representative images:

```bash
docker/jetson-xfeat/benchmark.sh \
  artifacts/xfeat-lightglue/xfeat_320x224_fp16.engine \
  artifacts/xfeat-lightglue/lg_320x224_dyn_n1_o500_m1024_fp16.engine \
  image/sample1.jpg \
  image/sample2.jpg
```

The LighterGlue engine uses minimum, optimum, and maximum keypoint counts of 1,
500, and 1024. Keep `--top-k` at or below 1024. The output directory contains a
JSON result and a 100 ms `tegrastats` log.

TensorRT engines are specific to their TensorRT release and GPU architecture;
build the engines on the target Jetson rather than copying desktop engines.
