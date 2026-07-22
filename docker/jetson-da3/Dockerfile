# syntax=docker/dockerfile:1.7

# 26.05 is the newest NVIDIA TensorRT image built on CUDA 13.2, matching
# JetPack 7.2 / L4T R39.2 on the target Jetson. 26.06 requires CUDA 13.3.
ARG TENSORRT_IMAGE=nvcr.io/nvidia/tensorrt:26.05-py3

FROM ${TENSORRT_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        g++ \
        libboost-program-options-dev \
        libeigen3-dev \
        libopencv-core-dev \
        libopencv-imgcodecs-dev \
        libopencv-imgproc-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY include/xfeat-cpp/mono_depth/ include/xfeat-cpp/mono_depth/
COPY src/mono_depth/depth_anything_v3_trt.cpp src/mono_depth/depth_anything_v3_trt.cpp
COPY examples/mono_depth_benchmark.cpp examples/mono_depth_benchmark.cpp

# The image is built natively on Orin. Optimizing for Cortex-A78 improves the
# OpenCV preprocessing/postprocessing around the TensorRT inference call.
RUN c++ \
      -std=c++17 -O3 -DNDEBUG -DHAVE_TENSORRT -mcpu=cortex-a78 -flto \
      -Iinclude -I/usr/include/eigen3 -I/usr/include/opencv4 -I/usr/local/cuda/include \
      src/mono_depth/depth_anything_v3_trt.cpp \
      examples/mono_depth_benchmark.cpp \
      -o /usr/local/bin/mono_depth_benchmark \
      -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
      -L/usr/local/cuda/lib64 \
      -lboost_program_options -lnvinfer -lcudart -pthread \
      -Wl,-O1,--as-needed \
    && strip /usr/local/bin/mono_depth_benchmark

FROM ${TENSORRT_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libboost-program-options1.83.0 \
        libopencv-core406t64 \
        libopencv-imgcodecs406t64 \
        libopencv-imgproc406t64 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/mono_depth_benchmark /usr/local/bin/mono_depth_benchmark

ENTRYPOINT ["/usr/local/bin/mono_depth_benchmark"]
CMD ["--help"]
