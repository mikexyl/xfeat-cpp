# Jetson Orin NX XFeat/LighterGlue TensorRT benchmark

## Summary

XFeat and LighterGlue were built as FP16 TensorRT engines directly on the
target Jetson Orin NX and tested with the native C++ backend. Across three
trials of 200 measured calls per stage, after 20 warmup calls, the complete
two-image extraction-and-matching pipeline averaged **22.96 ms per pair** or
**43.55 pairs/s**.

| Stage | Three-trial mean | Trial-mean range | Throughput from mean |
|---|---:|---:|---:|
| XFeat, one image | 8.246 ms | 7.776–8.796 ms | 121.28 images/s |
| LighterGlue, one cached-feature pair | 8.289 ms | 7.793–8.729 ms | 120.64 pairs/s |
| Two XFeat calls plus LighterGlue | 22.961 ms | 22.656–23.345 ms | 43.55 pairs/s |

The average full-pipeline p95 was 23.587 ms. Each input image was 640×480,
resized by XFeat to 320×224, and limited to 500 keypoints. Both images produced
500 keypoints and LighterGlue returned 212 matches.

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

| Engine | Size | SHA-256 |
|---|---:|---|
| `xfeat_320x224_fp16.engine` | 1,897,756 bytes (1.81 MiB) | `50e5adf2be14fb3d7d6cab165bde06112ae5ab986ad842d016accc6a464ac973` |
| `lg_320x224_dyn_n1_o500_m1024_fp16.engine` | 9,984,468 bytes (9.52 MiB) | `fd6f56082280a4bc46d0d05ee44e0408f845097a70b22203ac918549e5e09ff3` |

The LighterGlue optimization profile accepts 1–1024 keypoints per image and is
optimized for 500. Both engines used FP16, builder optimization level 5, eight
timing samples, and a 4096 MiB workspace. XFeat took 503.8 seconds to build and
LighterGlue took 390.9 seconds. The build produced a reusable TensorRT timing
cache.

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

## Raw TensorRT engine results

`trtexec` measured each serialized engine for 10 seconds after a 2-second
warmup, using spin-wait and normal host/device transfers. LighterGlue used the
500/500-keypoint shape.

| Metric | XFeat | LighterGlue 500/500 |
|---|---:|---:|
| Throughput | 1243.69 queries/s | 130.49 queries/s |
| Mean end-to-end engine latency | 0.933 ms | 7.632 ms |
| Median end-to-end engine latency | 0.933 ms | 7.604 ms |
| p95 end-to-end engine latency | 0.942 ms | 8.030 ms |
| Mean GPU compute time | 0.802 ms | 7.584 ms |
| p95 GPU compute time | 0.811 ms | 7.977 ms |
| Mean H2D latency | 0.072 ms | 0.032 ms |
| Mean D2H latency | 0.059 ms | 0.016 ms |

The key performance finding is that raw XFeat GPU inference takes only about
0.80 ms, while the full XFeat C++ call averages 8.25 ms. Most XFeat latency is
therefore in image preprocessing and CPU postprocessing: heatmap softmax, NMS,
sorting, descriptor interpolation, normalization, copies, and allocations.
LighterGlue is primarily GPU-bound: its 7.58 ms raw GPU time is close to its
8.29 ms application-level mean.

## Power and thermal telemetry

`tegrastats` was sampled every 100 ms. Application-level averages include model
loading, warmup, and all three measured stages, so they are not isolated
per-stage power measurements.

| Trial | Mean / peak board input power | Mean / peak GPU load | Peak GPU temperature |
|---:|---:|---:|---:|
| 1 | 7.64 / 9.49 W | 34.4 / 94% | 48.2 °C |
| 2 | 7.90 / 9.42 W | 33.6 / 81% | 48.3 °C |
| 3 | 7.93 / 9.35 W | 31.9 / 45% | 48.7 °C |
| Raw `trtexec` sequence | 10.79 / 12.70 W | 68.8 / 95% | 49.9 °C |

No thermal throttling was observed in these short runs.

## Reproduction

From the repository copied to the Jetson:

```bash
cd /home/mikexyl/xfeat-trt-benchmark

./docker/jetson-xfeat/build_native_benchmark.sh
./docker/jetson-xfeat/build_engines_native.sh

WARMUP=20 RUNS=200 TOP_K=500 \
  ./docker/jetson-xfeat/benchmark_native.sh \
  image/sample1.jpg \
  image/sample2.jpg
```

The benchmark emits JSON results and 100 ms `tegrastats` logs under
`benchmark-results/`. The implementation and alternative container workflow
are documented in [`docker/jetson-xfeat/README.md`](../docker/jetson-xfeat/README.md).
