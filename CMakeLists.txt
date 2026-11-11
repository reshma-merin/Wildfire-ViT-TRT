cmake_minimum_required(VERSION 3.18)
project(vit_bench LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# --- ONNX Runtime runner: cmake -DORT_DIR=/path/to/onnxruntime-<platform>-<ver> ---
if(ORT_DIR)
  find_library(ORT_LIB onnxruntime HINTS ${ORT_DIR}/lib REQUIRED)
  add_executable(bench_ort src/bench_ort.cpp)
  target_include_directories(bench_ort PRIVATE ${ORT_DIR}/include ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(bench_ort PRIVATE ${ORT_LIB} ${OpenCV_LIBS})
endif()

# --- TensorRT runner: cmake -DTENSORRT_DIR=/path/to/TensorRT-10.x (L40 machine only) ---
if(TENSORRT_DIR)
  find_package(CUDAToolkit REQUIRED)
  find_library(NVINFER_LIB NAMES nvinfer nvinfer_10 HINTS ${TENSORRT_DIR}/lib REQUIRED)
  add_executable(bench_trt src/bench_trt.cpp)
  target_include_directories(bench_trt PRIVATE ${TENSORRT_DIR}/include ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(bench_trt PRIVATE ${NVINFER_LIB} CUDA::cudart ${OpenCV_LIBS})
endif()

if(NOT ORT_DIR AND NOT TENSORRT_DIR)
  message(WARNING "Set -DORT_DIR=... and/or -DTENSORRT_DIR=... to build a runner.")
endif()