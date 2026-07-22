ARG TENSORRT_IMAGE=nvcr.io/nvidia/tensorrt:26.05-py3

FROM ${TENSORRT_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        g++ \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY include/xfeat-cpp/ include/xfeat-cpp/
COPY src/tensorrt/trt_engine.cpp src/tensorrt/trt_engine.cpp
COPY src/xfeat_trt.cpp src/xfeat_trt.cpp
COPY src/lighterglue_trt.cpp src/lighterglue_trt.cpp
COPY src/place_recognition/place_recognizer.cpp src/place_recognition/place_recognizer.cpp
COPY src/place_recognition/jist_trt.cpp src/place_recognition/jist_trt.cpp
COPY examples/tensorrt_features_benchmark.cpp examples/tensorrt_features_benchmark.cpp

RUN c++ \
      -std=c++17 -O3 -DNDEBUG -DHAVE_TENSORRT -mcpu=cortex-a78 -flto \
      -Iinclude -I/usr/include/opencv4 -I/usr/local/cuda/include \
      src/tensorrt/trt_engine.cpp \
      src/xfeat_trt.cpp \
      src/lighterglue_trt.cpp \
      src/place_recognition/place_recognizer.cpp \
      src/place_recognition/jist_trt.cpp \
      include/xfeat-cpp/nms/anms/anms.cpp \
      examples/tensorrt_features_benchmark.cpp \
      -o /usr/local/bin/tensorrt_features_benchmark \
      -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_features2d -lopencv_calib3d -lopencv_highgui \
      -L/usr/local/cuda/lib64 \
      -lnvinfer -lcudart -pthread \
      -Wl,-O1,--as-needed \
    && strip /usr/local/bin/tensorrt_features_benchmark

FROM ${TENSORRT_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libopencv-calib3d406t64 \
        libopencv-core406t64 \
        libopencv-features2d406t64 \
        libopencv-highgui406t64 \
        libopencv-imgcodecs406t64 \
        libopencv-imgproc406t64 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/tensorrt_features_benchmark /usr/local/bin/tensorrt_features_benchmark

ENTRYPOINT ["/usr/local/bin/tensorrt_features_benchmark"]
CMD ["--help"]
