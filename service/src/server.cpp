// TensorRT inference service: REST (cpp-httplib) + gRPC, Prometheus metrics, k8s-style probes.
//
//   wildfire_service --engine /engine/vit_fp16.plan --http-port 8080 --grpc-port 50051
//                    --contexts 2 --size 512 --mean 0.485,0.456,0.406 --std 0.229,0.224,0.225
//                    --labels no_fire,fire
//
// Endpoints:  POST /predict (raw PNG/JPEG body)   GET /health (liveness)
//             GET /ready (readiness)              GET /metrics (Prometheus text format)
// gRPC:       wildfire.Classifier/Predict
//
// Preprocessing is identical to the benchmark runners and to export/check_cpp_parity.py:
// BGR->RGB, bilinear resize (only if needed), /255, (x-mean)/std, HWC->CHW.
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <grpcpp/grpcpp.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "classifier.grpc.pb.h"
#include "httplib.h"

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

#define CUDA_CHECK(x)                                                                          \
  do {                                                                                         \
    cudaError_t e_ = (x);                                                                      \
    if (e_ != cudaSuccess) {                                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(1); /* let Kubernetes restart the pod */                                       \
    }                                                                                          \
  } while (0)

// ---------------------------------------------------------------- config ---
static std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string t; std::getline(ss, t, sep);) out.push_back(t);
  return out;
}

struct Config {
  std::string engine = "/engine/vit_fp16.plan";
  int size = 512;
  int http_port = 8080;
  int grpc_port = 50051;
  int contexts = 2;  // concurrent TensorRT execution contexts (each with its own stream/buffers)
  int warmup = 10;
  std::vector<float> mean{0.485f, 0.456f, 0.406f};
  std::vector<float> stdv{0.229f, 0.224f, 0.225f};
  std::vector<std::string> labels;
};

static Config parse_args(int argc, char** argv) {
  Config c;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--engine") c.engine = v;
    else if (k == "--size") c.size = std::stoi(v);
    else if (k == "--http-port") c.http_port = std::stoi(v);
    else if (k == "--grpc-port") c.grpc_port = std::stoi(v);
    else if (k == "--contexts") c.contexts = std::stoi(v);
    else if (k == "--warmup") c.warmup = std::stoi(v);
    else if (k == "--labels") c.labels = split(v, ',');
    else if (k == "--mean" || k == "--std") {
      std::vector<float> f;
      for (auto& t : split(v, ',')) f.push_back(std::stof(t));
      if (f.size() != 3) { std::fprintf(stderr, "%s needs 3 comma-separated values\n", k.c_str()); std::exit(2); }
      (k == "--mean" ? c.mean : c.stdv) = f;
    } else {
      std::fprintf(stderr, "unknown argument %s\n", k.c_str());
      std::exit(2);
    }
  }
  return c;
}

// --------------------------------------------------------------- metrics ---
struct Histogram {
  std::vector<double> bounds{0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5};
  std::vector<uint64_t> counts = std::vector<uint64_t>(bounds.size() + 1, 0);  // last = overflow
  double sum = 0;
  uint64_t count = 0;
  std::mutex m;
  void observe(double seconds) {
    std::lock_guard<std::mutex> lk(m);
    size_t i = 0;
    while (i < bounds.size() && seconds > bounds[i]) ++i;
    counts[i]++;
    sum += seconds;
    count++;
  }
};

static void render_hist(std::ostringstream& o, const std::string& name, const std::string& help, Histogram& h) {
  std::lock_guard<std::mutex> lk(h.m);
  o << "# HELP " << name << ' ' << help << "\n# TYPE " << name << " histogram\n";
  uint64_t cum = 0;
  for (size_t i = 0; i < h.bounds.size(); ++i) {
    cum += h.counts[i];
    o << name << "_bucket{le=\"" << h.bounds[i] << "\"} " << cum << '\n';
  }
  cum += h.counts.back();
  o << name << "_bucket{le=\"+Inf\"} " << cum << '\n';
  o << name << "_sum " << h.sum << '\n' << name << "_count " << h.count << '\n';
}

struct Metrics {
  std::mutex m;
  std::map<std::string, uint64_t> requests;  // key "protocol|status"
  Histogram infer;  // GPU path only
  Histogram total;  // decode + preprocess + GPU path
  std::atomic<int> inflight{0};
  Clock::time_point started = Clock::now();

  void count(const std::string& proto, const std::string& status) {
    std::lock_guard<std::mutex> lk(m);
    requests[proto + "|" + status]++;
  }

  std::string render(bool ready) {
    std::ostringstream o;
    o << "# HELP wildfire_requests_total Requests handled, by protocol and outcome.\n"
         "# TYPE wildfire_requests_total counter\n";
    {
      std::lock_guard<std::mutex> lk(m);
      for (auto& kv : requests) {
        auto p = kv.first.find('|');
        o << "wildfire_requests_total{protocol=\"" << kv.first.substr(0, p) << "\",status=\""
          << kv.first.substr(p + 1) << "\"} " << kv.second << '\n';
      }
    }
    render_hist(o, "wildfire_inference_seconds", "H2D + TensorRT + D2H time per request.", infer);
    render_hist(o, "wildfire_request_seconds", "Decode + preprocess + inference time per request.", total);
    o << "# HELP wildfire_inflight_requests Requests currently being processed.\n"
         "# TYPE wildfire_inflight_requests gauge\nwildfire_inflight_requests "
      << inflight.load() << '\n';
    o << "# HELP wildfire_ready 1 when the engine is loaded and warmed up.\n"
         "# TYPE wildfire_ready gauge\nwildfire_ready " << (ready ? 1 : 0) << '\n';
    o << "# HELP wildfire_uptime_seconds Process uptime.\n# TYPE wildfire_uptime_seconds gauge\n"
         "wildfire_uptime_seconds "
      << ms_since(started) / 1e3 << '\n';
    return o.str();
  }
};

// ---------------------------------------------------------------- engine ---
class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::fprintf(stderr, "[TRT] %s\n", msg);
  }
};

class Engine {
 public:
  ~Engine() {
    for (auto& w : workers_) {
      if (w.h_in) cudaFreeHost(w.h_in);
      if (w.h_out) cudaFreeHost(w.h_out);
      if (w.d_in) cudaFree(w.d_in);
      if (w.d_out) cudaFree(w.d_out);
      if (w.stream) cudaStreamDestroy(w.stream);
    }
  }

  bool load(const Config& c) {
    std::ifstream f(c.engine, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open engine %s\n", c.engine.c_str()); return false; }
    std::vector<char> blob((size_t)f.tellg());
    f.seekg(0);
    f.read(blob.data(), blob.size());

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) {
      std::fprintf(stderr, "engine deserialization failed (built on a different GPU/TensorRT version?)\n");
      return false;
    }
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char* n = engine_->getIOTensorName(i);
      (engine_->getTensorIOMode(n) == nvinfer1::TensorIOMode::kINPUT ? in_name_ : out_name_) = n;
    }
    in_elems_ = (size_t)3 * c.size * c.size;

    for (int k = 0; k < c.contexts; ++k) {
      Worker w;
      w.ctx.reset(engine_->createExecutionContext());
      if (!w.ctx || !w.ctx->setInputShape(in_name_.c_str(), nvinfer1::Dims4{1, 3, c.size, c.size})) {
        std::fprintf(stderr, "cannot set input shape 1x3x%dx%d on context %d\n", c.size, c.size, k);
        return false;
      }
      nvinfer1::Dims od = w.ctx->getTensorShape(out_name_.c_str());
      size_t n = 1;
      for (int i = 0; i < od.nbDims; ++i) n *= (size_t)od.d[i];
      num_classes_ = (int)n;
      CUDA_CHECK(cudaMallocHost(&w.h_in, in_elems_ * sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&w.h_out, n * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&w.d_in, in_elems_ * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&w.d_out, n * sizeof(float)));
      CUDA_CHECK(cudaStreamCreate(&w.stream));
      w.ctx->setTensorAddress(in_name_.c_str(), w.d_in);
      w.ctx->setTensorAddress(out_name_.c_str(), w.d_out);
      workers_.push_back(std::move(w));
      free_.push_back(k);
    }
    return true;
  }

  int num_classes() const { return num_classes_; }

  // Thread-safe: blocks until one of the execution contexts is free.
  bool infer(const std::vector<float>& chw, std::vector<float>& logits, double& ms) {
    int k = acquire();
    Worker& w = workers_[k];
    auto t0 = Clock::now();
    std::memcpy(w.h_in, chw.data(), in_elems_ * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(w.d_in, w.h_in, in_elems_ * sizeof(float), cudaMemcpyHostToDevice, w.stream));
    bool ok = w.ctx->enqueueV3(w.stream);
    CUDA_CHECK(cudaMemcpyAsync(w.h_out, w.d_out, num_classes_ * sizeof(float), cudaMemcpyDeviceToHost, w.stream));
    CUDA_CHECK(cudaStreamSynchronize(w.stream));  // must sync before the timer stops
    ms = ms_since(t0);
    logits.assign(w.h_out, w.h_out + num_classes_);
    release(k);
    return ok;
  }

 private:
  struct Worker {
    std::unique_ptr<nvinfer1::IExecutionContext> ctx;
    cudaStream_t stream = nullptr;
    float *h_in = nullptr, *h_out = nullptr, *d_in = nullptr, *d_out = nullptr;
  };
  int acquire() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return !free_.empty(); });
    int k = free_.back();
    free_.pop_back();
    return k;
  }
  void release(int k) {
    { std::lock_guard<std::mutex> lk(mu_); free_.push_back(k); }
    cv_.notify_one();
  }

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::vector<Worker> workers_;  // declared after engine_ -> contexts die before the engine
  std::vector<int> free_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::string in_name_, out_name_;
  size_t in_elems_ = 0;
  int num_classes_ = 0;
};

// ---------------------------------------------------------------- globals ---
static Config g_cfg;
static Engine g_engine;
static Metrics g_metrics;
static std::atomic<bool> g_ready{false};
static volatile std::sig_atomic_t g_stop = 0;

// ----------------------------------------------------------- prediction ---
static bool preprocess(const std::string& bytes, std::vector<float>& out, std::string& err) {
  if (bytes.empty()) { err = "empty request body"; return false; }
  cv::Mat raw(1, (int)bytes.size(), CV_8UC1, const_cast<char*>(bytes.data()));
  cv::Mat bgr = cv::imdecode(raw, cv::IMREAD_COLOR);
  if (bgr.empty()) { err = "could not decode image (send PNG or JPEG bytes)"; return false; }
  const int size = g_cfg.size;
  cv::Mat resized, rgb, f32;
  if (bgr.rows != size || bgr.cols != size)
    cv::resize(bgr, resized, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);
  else
    resized = bgr;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
  std::vector<cv::Mat> ch(3);
  cv::split(f32, ch);
  out.resize((size_t)3 * size * size);
  for (int c = 0; c < 3; ++c) {
    cv::Mat dst(size, size, CV_32F, out.data() + (size_t)c * size * size);
    ch[c].convertTo(dst, CV_32F, 1.0 / g_cfg.stdv[c], -g_cfg.mean[c] / g_cfg.stdv[c]);
  }
  return true;
}

struct Prediction {
  std::vector<float> logits, probs;
  int cls = 0;
  double infer_ms = 0;
};

static std::string label_of(int i) {
  return i < (int)g_cfg.labels.size() ? g_cfg.labels[i] : std::to_string(i);
}

// Returns an HTTP-style status code: 200 ok, 400 bad input, 503 not ready, 500 failure.
static int run_prediction(const std::string& bytes, Prediction& p, std::string& err) {
  if (!g_ready) { err = "not ready"; return 503; }
  std::vector<float> chw;
  if (!preprocess(bytes, chw, err)) return 400;
  if (!g_engine.infer(chw, p.logits, p.infer_ms)) { err = "inference failed"; return 500; }
  float mx = *std::max_element(p.logits.begin(), p.logits.end());
  double z = 0;
  for (float v : p.logits) z += std::exp(v - mx);
  for (float v : p.logits) p.probs.push_back((float)(std::exp(v - mx) / z));
  p.cls = (int)(std::max_element(p.logits.begin(), p.logits.end()) - p.logits.begin());
  return 200;
}

static int handle(const std::string& proto, const std::string& bytes, Prediction& p, std::string& err) {
  g_metrics.inflight++;
  auto t0 = Clock::now();
  int code = run_prediction(bytes, p, err);
  double total_s = ms_since(t0) / 1e3;
  g_metrics.inflight--;
  g_metrics.count(proto, code == 200 ? "ok" : code == 400 ? "bad_request" : "error");
  if (code == 200) {
    g_metrics.infer.observe(p.infer_ms / 1e3);
    g_metrics.total.observe(total_s);
  }
  return code;
}

static std::string to_json(const Prediction& p) {
  std::ostringstream o;
  o.precision(9);
  auto arr = [&](const std::vector<float>& v) {
    o << '[';
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << ']';
  };
  o << "{\"predicted_class\":" << p.cls << ",\"predicted_label\":\"" << label_of(p.cls)
    << "\",\"probabilities\":";
  arr(p.probs);
  o << ",\"logits\":";
  arr(p.logits);
  o << ",\"inference_ms\":" << p.infer_ms << "}";
  return o.str();
}

// ------------------------------------------------------------------ gRPC ---
class ClassifierImpl final : public wildfire::Classifier::Service {
  grpc::Status Predict(grpc::ServerContext*, const wildfire::PredictRequest* req,
                       wildfire::PredictResponse* resp) override {
    Prediction p;
    std::string err;
    int code = handle("grpc", req->image(), p, err);
    if (code == 400) return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, err);
    if (code == 503) return grpc::Status(grpc::StatusCode::UNAVAILABLE, err);
    if (code != 200) return grpc::Status(grpc::StatusCode::INTERNAL, err);
    for (float v : p.logits) resp->add_logits(v);
    for (float v : p.probs) resp->add_probabilities(v);
    resp->set_predicted_class(p.cls);
    resp->set_predicted_label(label_of(p.cls));
    resp->set_inference_ms(p.infer_ms);
    return grpc::Status::OK;
  }
};

static void on_signal(int) { g_stop = 1; }

int main(int argc, char** argv) {
  g_cfg = parse_args(argc, argv);
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  // HTTP server starts first so /health answers (liveness) while the engine loads.
  httplib::Server svr;
  svr.set_payload_max_length(64ULL * 1024 * 1024);
  svr.new_task_queue = [] { return new httplib::ThreadPool(16); };
  svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content("ok", "text/plain");
  });
  svr.Get("/ready", [](const httplib::Request&, httplib::Response& res) {
    res.status = g_ready ? 200 : 503;
    res.set_content(g_ready ? "ready" : "loading", "text/plain");
  });
  svr.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(g_metrics.render(g_ready), "text/plain; version=0.0.4");
  });
  svr.Post("/predict", [](const httplib::Request& req, httplib::Response& res) {
    Prediction p;
    std::string err;
    int code = handle("rest", req.body, p, err);
    if (code != 200) {
      res.status = code;
      res.set_content("{\"error\":\"" + err + "\"}", "application/json");
      return;
    }
    res.set_content(to_json(p), "application/json");
  });
  std::thread http_thread([&] { svr.listen("0.0.0.0", g_cfg.http_port); });

  if (!g_engine.load(g_cfg)) { svr.stop(); http_thread.join(); return 1; }
  {  // warmup: first runs pay one-time CUDA/TensorRT initialisation costs
    std::vector<float> zeros((size_t)3 * g_cfg.size * g_cfg.size, 0.f), logits;
    double ms = 0;
    for (int i = 0; i < g_cfg.warmup; ++i) g_engine.infer(zeros, logits, ms);
  }

  ClassifierImpl service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort("0.0.0.0:" + std::to_string(g_cfg.grpc_port), grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
  std::unique_ptr<grpc::Server> grpc_server = builder.BuildAndStart();
  if (!grpc_server) { std::fprintf(stderr, "failed to start gRPC server\n"); svr.stop(); http_thread.join(); return 1; }

  g_ready = true;
  std::printf("ready: %d classes, %d contexts, http :%d, grpc :%d\n", g_engine.num_classes(),
              g_cfg.contexts, g_cfg.http_port, g_cfg.grpc_port);
  std::fflush(stdout);

  while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));

  std::printf("shutting down\n");
  g_ready = false;  // readiness probe fails first so no new traffic arrives
  grpc_server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));
  svr.stop();
  http_thread.join();
  return 0;
}
