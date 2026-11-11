// ONNX Runtime C++ runner (CPU or CUDA).
//   bench_ort --model vit_wildfire.onnx --image tile.png --provider cpu|cuda
//             --batches 1,8,32 --warmup 50 --iters 500 --dump-logits ort_logits.txt
#include <onnxruntime_cxx_api.h>

#include <array>
#include <filesystem>

#include "bench_common.hpp"

int main(int argc, char** argv) {
  Args a(argc, argv);
  const int size = a.num("size", 512);
  const std::string provider = a.str("provider", "cpu");

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vit");
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (provider == "cuda") {
    OrtCUDAProviderOptions cuda{};
    opts.AppendExecutionProvider_CUDA(cuda);
  } else {
    opts.SetIntraOpNumThreads(a.num("threads", 0));  // 0 = ORT default
  }
  // ORTCHAR_T is wchar_t on Windows, char elsewhere.
  Ort::Session session(env, std::filesystem::path(a.str("model", "vit_wildfire.onnx")).c_str(), opts);

  double prep_ms = 0;
  std::vector<float> one = load_input(a, size, prep_ms);

  const char* in_names[] = {"image"};
  const char* out_names[] = {"logits"};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  for (int batch : a.ints("batches", {1, 8, 32})) {
    std::vector<float> in((size_t)batch * one.size());
    for (int b = 0; b < batch; ++b) std::copy(one.begin(), one.end(), in.begin() + (size_t)b * one.size());
    std::array<int64_t, 4> shape{batch, 3, size, size};
    Ort::Value input = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(), shape.data(), shape.size());

    // First run: dump logits for the parity check (first image of the batch).
    if (batch == 1) {
      auto out = session.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
      size_t n = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
      dump_logits(a.str("dump-logits"), out[0].GetTensorData<float>(), n);
    }

    benchmark("ort_" + provider, batch, a, prep_ms, [&] {
      auto out = session.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    });
  }
  return 0;
}