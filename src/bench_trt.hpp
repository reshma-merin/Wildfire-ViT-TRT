// Native TensorRT (10.x) C++ runner using enqueueV3.
//   bench_trt --engine vit_fp16.plan --image tile.png --batches 1,8,32 --dump-logits trt_logits.txt
// Engine must be built with an optimization profile covering every batch you ask for
// (see README: trtexec --minShapes/--optShapes/--maxShapes).
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>

#include "bench_common.hpp"

#define CUDA_CHECK(x)                                                                 \
  do {                                                                                \
    cudaError_t e = (x);                                                              \
    if (e != cudaSuccess) {                                                           \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::fprintf(stderr, "[TRT] %s\n", msg);
  }
};

int main(int argc, char** argv) {
  Args a(argc, argv);
  const int size = a.num("size", 512);

  // --- load plan ---
  std::ifstream f(a.str("engine", "vit_fp16.plan"), std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot open engine\n"); return 1; }
  std::vector<char> blob((size_t)f.tellg());
  f.seekg(0);
  f.read(blob.data(), blob.size());

  Logger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
  std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine) { std::fprintf(stderr, "engine deserialization failed (built on a different GPU/TRT version?)\n"); return 1; }
  std::unique_ptr<nvinfer1::IExecutionContext> ctx(engine->createExecutionContext());

  const char *in_name = nullptr, *out_name = nullptr;
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char* n = engine->getIOTensorName(i);
    (engine->getTensorIOMode(n) == nvinfer1::TensorIOMode::kINPUT ? in_name : out_name) = n;
  }

  double prep_ms = 0;
  std::vector<float> one = load_input(a, size, prep_ms);
  std::vector<int> batches = a.ints("batches", {1, 8, 32});
  int max_batch = *std::max_element(batches.begin(), batches.end());

  // Buffers sized for the largest batch; logits count taken from the engine at run time.
  const size_t in_elems_max = (size_t)max_batch * one.size();
  const size_t max_out_elems = (size_t)max_batch * 4096;  // generous upper bound on classes
  float *h_in, *h_out, *d_in, *d_out;
  CUDA_CHECK(cudaMallocHost(&h_in, in_elems_max * sizeof(float)));  // pinned -> faster H2D
  CUDA_CHECK(cudaMallocHost(&h_out, max_out_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_in, in_elems_max * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, max_out_elems * sizeof(float)));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  for (int batch : batches) {
    for (int b = 0; b < batch; ++b) std::copy(one.begin(), one.end(), h_in + (size_t)b * one.size());
    if (!ctx->setInputShape(in_name, nvinfer1::Dims4{batch, 3, size, size})) {
      std::fprintf(stderr, "batch %d is outside the engine's optimization profile\n", batch);
      continue;
    }
    nvinfer1::Dims od = ctx->getTensorShape(out_name);
    size_t out_elems = 1;
    for (int i = 0; i < od.nbDims; ++i) out_elems *= (size_t)od.d[i];
    ctx->setTensorAddress(in_name, d_in);
    ctx->setTensorAddress(out_name, d_out);

    const size_t in_bytes = (size_t)batch * one.size() * sizeof(float);
    // H2D + inference + D2H + sync: the full "give me logits" latency, comparable to ORT.
    auto run = [&] {
      CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, in_bytes, cudaMemcpyHostToDevice, stream));
      if (!ctx->enqueueV3(stream)) { std::fprintf(stderr, "enqueueV3 failed\n"); std::exit(1); }
      CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));  // must sync before the timer stops
    };

    if (batch == 1) {
      run();
      dump_logits(a.str("dump-logits"), h_out, out_elems);
    }
    benchmark("trt_fp16", batch, a, prep_ms, run);
  }

  cudaFreeHost(h_in); cudaFreeHost(h_out); cudaFree(d_in); cudaFree(d_out);
  cudaStreamDestroy(stream);
  return 0;
}