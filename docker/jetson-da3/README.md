# Jetson DA3-LARGE-1.1 two-view TensorRT C++ benchmark

This image builds only the project's DA3 TensorRT runtime and benchmark. It
targets JetPack 7.2 / L4T R39.2 on Jetson Orin and uses NVIDIA's TensorRT 26.05
container (CUDA 13.2, TensorRT 10.16), the newest container line compatible
with the device's CUDA 13.2 driver. The benchmark target is fixed to:

```text
model:     depth-anything/DA3-LARGE-1.1
input:     images [1, 2, 3, 350, 504]
precision: FP16
engine:    DA3-LARGE-1.1_multiview_v2_350x504_fp16.engine
```

Build natively on the Jetson from the repository root:

```bash
docker build \
  --file docker/jetson-da3/Dockerfile \
  --tag xfeat-da3:jetson \
  .
```

Prepare the host for a short peak-performance benchmark:

```bash
docker/jetson-da3/performance.sh enable
```

If `nvpmodel` requests a reboot, reboot first and run the command again. MAXN
is unconstrained and can thermally throttle, so use adequate cooling. The
script locks clocks and requests full fan speed to improve repeatability.

Place the portable ONNX graph at:

```text
artifacts/DA3-LARGE-1.1_multiview_v2_350x504.onnx
```

Build the engine natively on the Jetson. Builder optimization level 5 searches
more tactics than TensorRT's default level 3, and the timing cache makes a
subsequent rebuild faster:

```bash
docker/jetson-da3/build_engine.sh
```

Run the two-view C++ benchmark:

```bash
docker/jetson-da3/benchmark.sh \
  artifacts/DA3-LARGE-1.1_multiview_v2_350x504_fp16.engine \
  /path/to/view0.jpg \
  /path/to/view1.jpg
```

Each timed iteration is one grouped TensorRT call containing both views.
Results are written to `benchmark-results/` as benchmark JSON and a 100 ms
`tegrastats` log.

TensorRT engines are not generally portable across TensorRT releases or GPU
architectures. Build the FP16 engine on this Jetson using TensorRT 26.05; an
engine built for a desktop Ada GPU or with a different TensorRT release is not
a valid Jetson benchmark artifact.

`DA3-LARGE-1.1` is distributed under CC BY-NC 4.0; confirm that its
non-commercial terms fit the intended deployment.
