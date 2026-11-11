# TensorRT inference service image.
#   The NGC TensorRT image provides CUDA, TensorRT headers/libs and trtexec.
#   Pick a tag whose TensorRT major version matches what you validated (10.x):
#   docker build --build-arg TRT_IMAGE=nvcr.io/nvidia/tensorrt:<tag> -t wildfire-vit-trt .
ARG TRT_IMAGE=nvcr.io/nvidia/tensorrt:25.08-py3

FROM ${TRT_IMAGE} AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config curl ca-certificates \
      libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
      libopencv-core-dev libopencv-imgproc-dev libopencv-imgcodecs-dev \
    && rm -rf /var/lib/apt/lists/*
# single-header HTTP server library
RUN curl -fsSL -o /usr/local/include/httplib.h \
      https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.18.0/httplib.h
WORKDIR /src
COPY CMakeLists.txt .
COPY proto proto
COPY src src
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"$(nproc)"

# Runtime stage: same base so TensorRT/CUDA versions match the build exactly.
FROM ${TRT_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
# (dev packages pull in the matching runtime libs without hard-coding distro-specific names)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgrpc++-dev libopencv-core-dev libopencv-imgproc-dev libopencv-imgcodecs-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/build/wildfire_service /app/wildfire_service
COPY entrypoint.sh build_engine.sh /app/
RUN sed -i 's/\r$//' /app/*.sh && chmod +x /app/*.sh
# The ONNX file is baked in; the TensorRT engine is built on the target GPU at first start
# (an engine is only valid for the GPU model + TensorRT version that built it).
COPY model/vit_wildfire.onnx /models/vit_wildfire.onnx

ENV ENGINE_PATH=/engine/vit_fp16.plan
EXPOSE 8080 50051
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/app/wildfire_service", "--engine", "/engine/vit_fp16.plan", "--contexts", "2"]
