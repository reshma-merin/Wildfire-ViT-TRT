cmake_minimum_required(VERSION 3.18)
project(wildfire_service LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPCPP REQUIRED IMPORTED_TARGET grpc++)
pkg_check_modules(PROTOBUF REQUIRED IMPORTED_TARGET protobuf)

# TensorRT (headers/libs ship with the NGC tensorrt image; override with -DTENSORRT_DIR=...)
find_path(NVINFER_INCLUDE NvInfer.h HINTS ${TENSORRT_DIR}/include)
find_library(NVINFER_LIB nvinfer HINTS ${TENSORRT_DIR}/lib)
if(NOT NVINFER_INCLUDE OR NOT NVINFER_LIB)
  message(FATAL_ERROR "TensorRT not found; pass -DTENSORRT_DIR=<path>")
endif()

# gRPC/protobuf code generation (plain protoc, so it works with distro gRPC packages)
find_program(PROTOC protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
set(GEN ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${GEN})
add_custom_command(
  OUTPUT ${GEN}/classifier.pb.cc ${GEN}/classifier.pb.h ${GEN}/classifier.grpc.pb.cc ${GEN}/classifier.grpc.pb.h
  COMMAND ${PROTOC} -I ${CMAKE_CURRENT_SOURCE_DIR}/proto --cpp_out=${GEN} --grpc_out=${GEN}
          --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN} ${CMAKE_CURRENT_SOURCE_DIR}/proto/classifier.proto
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/proto/classifier.proto)

add_executable(wildfire_service src/server.cpp
  ${GEN}/classifier.pb.cc ${GEN}/classifier.grpc.pb.cc)
target_include_directories(wildfire_service PRIVATE ${GEN} ${NVINFER_INCLUDE} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(wildfire_service PRIVATE
  Threads::Threads CUDA::cudart ${NVINFER_LIB} ${OpenCV_LIBS} PkgConfig::GRPCPP PkgConfig::PROTOBUF)
