# Jetson Orin NX XFeat/LighterGlue/JIST/MixVPR TensorRT benchmark

## Summary

XFeat, LighterGlue, JIST, and MixVPR were built as FP16 TensorRT engines directly on
the target Jetson Orin NX and tested with the native C++ backends. Across three
trials of 200 measured calls per stage, after 20 warmup calls, the complete
two-image extraction-and-matching pipeline averaged **22.96 ms per pair** or
**43.55 pairs/s**. A five-frame JIST sequence averaged **24.85 ms** or
**40.23 sequences/s**. The new single-image MixVPR backend averaged
**7.690 ms per image** or **130.04 images/s** across three additional trials.

| Stage | Three-trial mean | Trial-mean range | Throughput from mean |
|---|---:|---:|---:|
| XFeat, one image | 8.246 ms | 7.776–8.796 ms | 121.28 images/s |
| LighterGlue, one cached-feature pair | 8.289 ms | 7.793–8.729 ms | 120.64 pairs/s |
| Two XFeat calls plus LighterGlue | 22.961 ms | 22.656–23.345 ms | 43.55 pairs/s |
| JIST, one five-frame sequence | 24.855 ms | 24.457–25.274 ms | 40.23 sequences/s |
| MixVPR, one image | 7.690 ms | 7.486–7.846 ms | 130.04 images/s |

The average full-pipeline p95 was 23.587 ms. Each input image was 640×480,
resized by XFeat to 320×224, and limited to 500 keypoints. Both images produced
500 keypoints and LighterGlue returned 212 matches.

The JIST test alternated the same two 640×480 images to form a five-frame
sequence. JIST resized each frame to 512×288 and returned a 512-dimensional
descriptor with an L2 norm of 1.0.

MixVPR used the same 640×480 inputs, resized them to 320×320 with ImageNet RGB
normalization, and returned a 4096-dimensional descriptor with an L2 norm of
1.0.

## MAXN_SUPER JIST/MixVPR rerun

JIST and MixVPR were rerun on 2026-07-23 after the board was changed to
`MAXN_SUPER`. Fresh FP16 engines were built directly on the Jetson with a fresh
timing cache. Merely changing an `nvpmodel` power mode does not ordinarily
require an engine rebuild. In this case, however, TensorRT warned that the
regular-MAXN plans came from a different device model after the board changed
to the Super hardware configuration. The old plans passed a smoke test, but
fresh plans were used so the benchmark did not rely on that unsupported path
and TensorRT could select tactics at the new clock limits.

The focused benchmark loaded only JIST and MixVPR. It used the same two
640×480 sample images and native C++ backends as the original test, with 20
warmup calls and 200 measured calls per model in each of three trials. JIST
again received a five-frame sequence alternating the two images. Clocks
remained dynamic: CPU governors were `schedutil`, the GPU ranged from 306 to
1173 MHz, and EMC ranged from 204 to 3199 MHz. `jetson_clocks` was not enabled.

### MAXN_SUPER application results

| Trial | JIST mean / median / p95 (ms) | MixVPR mean / median / p95 (ms) |
|---:|---:|---:|
| 1 | 25.065 / 25.200 / 26.838 | 6.375 / 6.495 / 6.639 |
| 2 | 25.156 / 25.327 / 26.662 | 6.509 / 6.521 / 6.595 |
| 3 | 25.813 / 25.777 / 27.391 | 6.339 / 6.263 / 6.575 |
| Three-trial aggregate | **25.345 / 25.435 / 26.964** | **6.408 / 6.427 / 6.603** |

The aggregate throughputs were **39.46 JIST sequences/s** and **156.06
MixVPR images/s**. Both outputs had the expected dimensions and an L2 norm of
1.0.

The following comparison uses the original dynamic-clock regular-MAXN results
as the baseline. Lower latency is better.

| Metric | Regular MAXN | MAXN_SUPER | Change |
|---|---:|---:|---:|
| JIST C++ mean latency | 24.855 ms | 25.345 ms | +2.0% |
| MixVPR C++ mean latency | 7.690 ms | 6.408 ms | -16.7% |
| JIST raw host latency | 6.932 ms | 5.808 ms | -16.2% |
| JIST raw GPU compute | 6.446 ms | 5.266 ms | -18.3% |
| MixVPR raw host latency | 3.590 ms | 2.954 ms | -17.7% |
| MixVPR raw GPU compute | 3.519 ms | 2.881 ms | -18.1% |

The raw engines are about 18% faster and deliver about 22% more throughput in
MAXN_SUPER. MixVPR carries most of that improvement through its full C++ path.
JIST's end-to-end latency is effectively unchanged because five-image CPU
preprocessing plus the current wrapper's per-call allocation and transfers
remain the larger part of the call. The application comparison is also not a
perfect isolated power-mode experiment: the original binary loaded all four
models, whereas this rerun deliberately used a focused two-model process.

### MAXN_SUPER raw TensorRT results

Each engine ran for 10 seconds after a 2-second warmup with spin-wait and normal
host/device transfers.

| Metric | JIST five-frame | MixVPR |
|---|---:|---:|
| Throughput | 189.76 queries/s | 346.86 queries/s |
| Mean / median / p95 host latency | 5.808 / 5.808 / 5.824 ms | 2.954 / 2.954 / 2.962 ms |
| Mean / p95 GPU compute time | 5.266 / 5.280 ms | 2.881 / 2.887 ms |
| Mean H2D latency | 0.535 ms | 0.067 ms |
| Mean D2H latency | 0.007 ms | 0.006 ms |

Both plans deserialized and completed without TensorRT's cross-device warning.

### MAXN_SUPER CUDA-visible memory usage

Jetson Orin uses unified system memory, so these are CUDA-visible RAM deltas,
not allocations from a separate pool of dedicated VRAM. Each model was
measured in three isolated processes relative to a baseline taken after CUDA
context initialization. The steady value was sampled after 50 inference calls;
the peak was polled concurrently during those calls.

| Model | After engine load, mean (range) | After warmup, mean (range) | Maximum sampled delta |
|---|---:|---:|---:|
| JIST | 49.0 MiB (42.4–53.6) | 106.4 MiB (104.2–110.3) | 114.5 MiB |
| MixVPR | 55.7 MiB (53.1–59.0) | 142.4 MiB (140.0–146.2) | 146.2 MiB |

The post-warmup figures are the practical full-process footprints and include
TensorRT, CUDA's lazily loaded kernels and caches, and driver allocations. For
comparison, `trtexec` reported only 33.75 MiB of TensorRT execution-context
device memory for JIST and 8.20 MiB for MixVPR. The per-call float32 input
buffers are 8.44 MiB and 1.17 MiB, respectively, and are included in the
sampled peaks. The board exposed 15,598 MiB total to CUDA during these tests.

### MAXN_SUPER engine metadata

The engines are stored on the Jetson under
`/home/mikexyl/xfeat-trt-benchmark/artifacts/xfeat-lightglue-maxn-super/`.
They used FP16, builder optimization level 5, eight timing samples, and a 4096
MiB workspace.

| Engine | Size | SHA-256 | Build time |
|---|---:|---|---:|
| `JIST_r18_512_seqgem_simplified_fp16.engine` | 23,328,716 bytes | `023ac7bf52db9f13794d2d30f02d1d3c56c045ba318c147067b7fa7a0d62c439` | 272.1 s |
| `mixvpr_resnet50_4096d_fp16.engine` | 22,733,316 bytes | `bb79656624d805ac94fc4fb0653ccbf546f683b95987cbeb16da0a12997eda4e` | 220.1 s |

The final timing cache contained 2,245 entries and occupied 2,842,380 bytes.

### MAXN_SUPER power and thermal telemetry

`tegrastats` was sampled every 100 ms. Application telemetry covers all three
trials and includes model loading and warmup; each raw row covers one isolated
`trtexec` run.

| Test | Mean / peak board input power | Mean / peak GPU load | Peak GPU temperature |
|---|---:|---:|---:|
| Three application trials | 9.88 / 14.56 W | 36.1 / 89% | 53.3 °C |
| Raw JIST | 27.30 / 28.51 W | 97.7 / 99% | 59.1 °C |
| Raw MixVPR | 23.06 / 23.38 W | 95.5 / 98% | 59.8 °C |

The raw, per-trial application, and memory logs are retained on the Jetson at
`/home/mikexyl/xfeat-trt-benchmark/benchmark-results/maxn-super-vpr-20260723/`.

## Test system

| Component | Configuration |
|---|---|
| Device | NVIDIA Jetson Orin NX, 16 GB |
| GPU | Orin, compute capability 8.7, 8 SMs |
| OS | Ubuntu 24.04 |
| JetPack/L4T | JetPack 7.2, L4T R39.2 |
| TensorRT | 10.16.2.10, CUDA 13.2 package |
| Power profile | MAXN |
| CPU frequency range | 729.6–1984 MHz |
| GPU frequency range | 306–918 MHz |
| Clock policy | Dynamic; CPU and GPU clocks were not statically locked |

The clock policy is an important limitation. Passwordless sudo was not
available, so `jetson_clocks` was not enabled. These results represent warmed-up
MAXN operation with the normal dynamic governors, not a statically locked
peak-performance run. For direct comparison with the earlier DA3 benchmark,
enable the DA3 performance configuration and repeat the test:

```bash
docker/jetson-da3/performance.sh enable
```

## Engines built on the Jetson

The engines are stored on the Jetson at
`/home/mikexyl/xfeat-trt-benchmark/artifacts/xfeat-lightglue/`.

MixVPR was exported from the official ResNet-50/4096-D checkpoint with
SHA-256 `97528606773e9920e93ca4d211daeabf1c7312480f38b45379ee5551a1b4dd24`.
An RTX 4070 preflight test confirmed cosine agreement greater than 0.999
between the FP16 TensorRT descriptor and the CPU ONNX Runtime descriptor.
The laptop engine is stored at
`onnx_model/trt/mixvpr_resnet50_4096d_fp16.engine`; it is 24,059,356 bytes and
has SHA-256 `b52b02d3336c8d760a8fcb606c8702c9434eea223f314b1354ce13d88e9f08dd`.

| Engine | Size | SHA-256 |
|---|---:|---|
| `xfeat_320x224_fp16.engine` | 1,897,756 bytes (1.81 MiB) | `50e5adf2be14fb3d7d6cab165bde06112ae5ab986ad842d016accc6a464ac973` |
| `lg_320x224_dyn_n1_o500_m1024_fp16.engine` | 9,984,468 bytes (9.52 MiB) | `fd6f56082280a4bc46d0d05ee44e0408f845097a70b22203ac918549e5e09ff3` |
| `JIST_r18_512_seqgem_simplified_fp16.engine` | 23,335,204 bytes (22.25 MiB) | `3c2e026d806fbb68ce8578cbdb56e0d85eafe0639105e90b48a8c241281c47e6` |
| `mixvpr_resnet50_4096d_fp16.engine` | 22,759,204 bytes (21.70 MiB) | `1d4da85db06bc21e0d6dd31f92f2b93e635db2e44468d6a26214ecbfcd4381cd` |

The LighterGlue optimization profile accepts 1–1024 keypoints per image and is
optimized for 500. All four engines used FP16, builder optimization level 5, eight
timing samples, and a 4096 MiB workspace. XFeat took 503.8 seconds to build and
LighterGlue took 390.9 seconds. JIST reused their timing cache and took 283.9
seconds. The build produced a reusable TensorRT timing cache with 10,301
entries after all three engines were complete. MixVPR then took 227.6 seconds
to build and expanded the cache to 11,212 entries. Its source ONNX file is
43,493,553 bytes with SHA-256
`f285decfa6b970b02b0f87fd1c47f921890b190a6eb472743d8b8657ccd541e8`.

TensorRT engines are tied to the target GPU and TensorRT release. These Jetson
TensorRT 10.16 engines must not be mixed with the separate TensorRT 10.13
engines built for the RTX 4070 laptop.

## Application-level results

All timings below include the native C++ runtime work for the named stage and
exclude warmup. Pipeline timing is measured directly; it should not be inferred
by adding separately measured stages because clock and thermal state vary
between phases.

| Trial | XFeat mean / median / p95 (ms) | LighterGlue mean / median / p95 (ms) | Pipeline mean / median / p95 (ms) |
|---:|---:|---:|---:|
| 1 | 8.796 / 9.162 / 9.563 | 7.793 / 7.669 / 8.155 | 22.656 / 22.583 / 23.146 |
| 2 | 8.165 / 8.132 / 9.137 | 8.345 / 7.768 / 13.109 | 22.881 / 22.764 / 23.624 |
| 3 | 7.776 / 7.475 / 8.876 | 8.729 / 8.071 / 13.186 | 23.345 / 23.304 / 23.990 |

CUDA-reported used-memory growth after loading both engines was approximately
82–94 MiB. After warmup, the increase over the pre-model baseline was
approximately 98–102 MiB. Jetson uses unified memory, so these figures describe
CUDA-visible system memory rather than dedicated VRAM.

### JIST application results

| Trial | Mean (ms) | Median (ms) | p95 (ms) | Throughput (sequences/s) |
|---:|---:|---:|---:|---:|
| 1 | 25.274 | 25.444 | 27.367 | 39.57 |
| 2 | 24.834 | 25.069 | 26.451 | 40.27 |
| 3 | 24.457 | 24.530 | 26.234 | 40.89 |
| Three-trial aggregate | **24.855** | **25.015** | **26.684** | **40.23** |

Loading all three models increased CUDA-visible used memory by approximately
102–105 MiB. After all stages were warmed up, the increase over the pre-model
baseline was approximately 173–189 MiB.

### MixVPR application results

Each trial used 20 warmup calls followed by 200 measured calls through the
native `MixVPRTRT` C++ interface.

| Trial | Mean (ms) | Median (ms) | p95 (ms) | Throughput (images/s) |
|---:|---:|---:|---:|---:|
| 1 | 7.738 | 7.753 | 7.879 | 129.23 |
| 2 | 7.486 | 7.338 | 7.789 | 133.59 |
| 3 | 7.846 | 7.818 | 8.060 | 127.46 |
| Three-trial aggregate | **7.690** | **7.636** | **7.909** | **130.04** |

The benchmark loaded all four engines in the same process. Loading them grew
CUDA-visible used memory by 109–118 MiB. After all stages were warmed up, the
increase over the pre-model baseline was 276–284 MiB.

## Raw TensorRT engine results

`trtexec` measured each serialized engine for 10 seconds after a 2-second
warmup, using spin-wait and normal host/device transfers. LighterGlue used the
500/500-keypoint shape.

| Metric | XFeat | LighterGlue 500/500 | JIST five-frame | MixVPR |
|---|---:|---:|---:|---:|
| Throughput | 1243.69 queries/s | 130.49 queries/s | 155.01 queries/s | 283.96 queries/s |
| Mean end-to-end engine latency | 0.933 ms | 7.632 ms | 6.932 ms | 3.590 ms |
| Median end-to-end engine latency | 0.933 ms | 7.604 ms | 6.930 ms | 3.590 ms |
| p95 end-to-end engine latency | 0.942 ms | 8.030 ms | 6.942 ms | 3.598 ms |
| Mean GPU compute time | 0.802 ms | 7.584 ms | 6.446 ms | 3.519 ms |
| p95 GPU compute time | 0.811 ms | 7.977 ms | 6.455 ms | 3.526 ms |
| Mean H2D latency | 0.072 ms | 0.032 ms | 0.480 ms | 0.066 ms |
| Mean D2H latency | 0.059 ms | 0.016 ms | 0.006 ms | 0.006 ms |

The key performance finding is that raw XFeat GPU inference takes only about
0.80 ms, while the full XFeat C++ call averages 8.25 ms. Most XFeat latency is
therefore in image preprocessing and CPU postprocessing: heatmap softmax, NMS,
sorting, descriptor interpolation, normalization, copies, and allocations.
LighterGlue is primarily GPU-bound: its 7.58 ms raw GPU time is close to its
8.29 ms application-level mean.

JIST shows a similar application/runtime gap to XFeat. Raw GPU compute takes
6.45 ms, while the complete five-frame C++ call averages 24.85 ms. The gap is
mainly five-image preprocessing and the current TensorRT wrapper's per-call
allocation and transfer strategy. Its 8.4 MiB float32 input also explains the
higher 0.48 ms H2D time in `trtexec`.

MixVPR's raw host latency was 3.590 ms, compared with 7.690 ms through the
complete C++ call. The latter also includes 640×480-to-320×320 resize,
BGR-to-RGB conversion, ImageNet normalization, NCHW packing, per-call TensorRT
buffer allocation and transfers, output copy, and descriptor normalization.

### JIST C++ timing breakdown

A separate 200-run test reproduced the JIST call using the same operations but
timed each component inside every iteration. This avoids attributing phase-level
clock changes to one component. The sum of the component means equals the
measured 24.980 ms breakdown total.

| Component | Mean latency | Share of breakdown total |
|---|---:|---:|
| Five-frame preprocessing and tensor packing | 13.237 ms | 53.0% |
| TensorRT wrapper with a prepacked tensor | 11.716 ms | 46.9% |
| Copy 512 floats and L2-normalize descriptor | 0.026 ms | 0.1% |

The descriptor postprocessing is negligible. Preprocessing performs five
BGR-to-RGB conversions, five 640×480-to-512×288 resizes, five uint8-to-float
conversions, 15 channel splits, and packing of an 8.4 MiB tensor. The wrapper
then allocates and frees device input/output buffers, copies from pageable host
memory, and synchronizes on every call. Its 11.72 ms is higher than the 6.93 ms
`trtexec` engine latency because `trtexec` reuses preallocated buffers and uses
a tuned inference loop. Persistent device buffers, pinned host buffers, and a
fused or GPU preprocessing path are the main optimization opportunities.

## Power and thermal telemetry

`tegrastats` was sampled every 100 ms. Application-level averages include model
loading, warmup, and all measured stages, so they are not isolated
per-stage power measurements.

| Trial | Mean / peak board input power | Mean / peak GPU load | Peak GPU temperature |
|---:|---:|---:|---:|
| XFeat/LighterGlue trial 1 | 7.64 / 9.49 W | 34.4 / 94% | 48.2 °C |
| XFeat/LighterGlue trial 2 | 7.90 / 9.42 W | 33.6 / 81% | 48.3 °C |
| XFeat/LighterGlue trial 3 | 7.93 / 9.35 W | 31.9 / 45% | 48.7 °C |
| JIST-inclusive trial 1 | 8.23 / 9.91 W | 33.8 / 87% | 48.9 °C |
| JIST-inclusive trial 2 | 8.42 / 9.59 W | 33.1 / 74% | 48.7 °C |
| JIST-inclusive trial 3 | 8.36 / 9.55 W | 31.3 / 62% | 48.8 °C |
| Raw XFeat/LighterGlue `trtexec` sequence | 10.79 / 12.70 W | 68.8 / 95% | 49.9 °C |
| Raw JIST `trtexec` | 23.32 / 24.48 W | 97.0 / 99% | 57.0 °C |
| All-model MixVPR trial 1 | 9.03 / 10.47 W | 34.3 / 72% | 48.8 °C |
| All-model MixVPR trial 2 | 8.91 / 10.71 W | 34.9 / 71% | 49.5 °C |
| All-model MixVPR trial 3 | 8.63 / 10.39 W | 36.4 / 99% | 49.6 °C |
| Raw MixVPR `trtexec` | 19.61 / 20.75 W | 96.5 / 98% | 54.3 °C |

No thermal throttling was observed in these short runs.

## Reproduction

From the repository copied to the Jetson:

```bash
cd /home/mikexyl/xfeat-trt-benchmark

./docker/jetson-xfeat/build_native_benchmark.sh
./docker/jetson-xfeat/build_engines_native.sh
./docker/jetson-xfeat/build_jist_engine_native.sh
./docker/jetson-xfeat/build_mixvpr_engine_native.sh

WARMUP=20 RUNS=200 TOP_K=500 \
  ./docker/jetson-xfeat/benchmark_native.sh \
  image/sample1.jpg \
  image/sample2.jpg
```

The benchmark emits JSON results and 100 ms `tegrastats` logs under
`benchmark-results/`. The implementation and alternative container workflow
are documented in [`docker/jetson-xfeat/README.md`](../docker/jetson-xfeat/README.md).
