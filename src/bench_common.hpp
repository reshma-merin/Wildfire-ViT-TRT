// Shared pieces for both runners: arg parsing, preprocessing, timing, reporting.
#pragma once
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---- minimal "--key value" parser ----------------------------------------
struct Args {
  std::map<std::string, std::string> kv;
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string k = argv[i];
      if (k.rfind("--", 0) != 0) continue;
      k = k.substr(2);
      kv[k] = (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) ? argv[++i] : "1";
    }
  }
  std::string str(const std::string& k, const std::string& d = "") const {
    auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
  int num(const std::string& k, int d) const {
    auto it = kv.find(k);
    return it == kv.end() ? d : std::stoi(it->second);
  }
  std::vector<float> floats(const std::string& k, std::vector<float> d) const {
    auto it = kv.find(k);
    if (it == kv.end()) return d;
    std::vector<float> v;
    std::stringstream ss(it->second);
    for (std::string t; std::getline(ss, t, ',');) v.push_back(std::stof(t));
    return v;
  }
  std::vector<int> ints(const std::string& k, std::vector<int> d) const {
    auto f = floats(k, std::vector<float>(d.begin(), d.end()));
    return std::vector<int>(f.begin(), f.end());
  }
};

// ---- preprocessing -------------------------------------------------------
// MUST match training (and export/check_cpp_parity.py):
// BGR->RGB, resize (bilinear, only if needed), /255, (x-mean)/std, HWC->CHW.
// Defaults are ImageNet stats; pass --mean/--std if THOMAS was pretrained differently.
inline void preprocess(const cv::Mat& bgr, int size, const std::vector<float>& mean,
                       const std::vector<float>& stdv, float* out_chw) {
  // Distinct Mats on purpose: in-place cvtColor on a header that shares `bgr`'s buffer
  // would swap channels in the caller's image on every call.
  cv::Mat resized, rgb, f32;
  if (bgr.rows != size || bgr.cols != size)
    cv::resize(bgr, resized, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);
  else
    resized = bgr;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
  std::vector<cv::Mat> ch(3);
  cv::split(f32, ch);
  for (int c = 0; c < 3; ++c) {
    cv::Mat dst(size, size, CV_32F, out_chw + (size_t)c * size * size);
    ch[c].convertTo(dst, CV_32F, 1.0 / stdv[c], -mean[c] / stdv[c]);
  }
}

// Loads --image (or random noise if absent) and returns one preprocessed CHW image.
// Also reports preprocessing cost per image (ms, median of 50 runs) via prep_ms.
inline std::vector<float> load_input(const Args& a, int size, double& prep_ms) {
  std::vector<float> mean = a.floats("mean", {0.485f, 0.456f, 0.406f});
  std::vector<float> stdv = a.floats("std", {0.229f, 0.224f, 0.225f});
  std::string path = a.str("image");
  cv::Mat bgr;
  if (!path.empty()) {
    bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  } else {
    bgr = cv::Mat(size, size, CV_8UC3);
    cv::randu(bgr, 0, 255);
    std::fprintf(stderr, "[warn] no --image given, using random pixels (latency only)\n");
  }
  std::vector<float> buf((size_t)3 * size * size);
  std::vector<double> t;
  for (int i = 0; i < 50; ++i) {
    auto t0 = Clock::now();
    preprocess(bgr, size, mean, stdv, buf.data());
    t.push_back(ms_since(t0));
  }
  std::sort(t.begin(), t.end());
  prep_ms = t[t.size() / 2];
  return buf;
}

// ---- timing + reporting --------------------------------------------------
inline double percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  size_t i = (size_t)std::ceil(p / 100.0 * v.size());
  return v[std::min(v.size() - 1, i == 0 ? 0 : i - 1)];
}

// `run` must be fully synchronous (GPU work finished) when it returns.
inline void benchmark(const std::string& backend, int batch, const Args& a, double prep_ms,
                      const std::function<void()>& run) {
  int warmup = a.num("warmup", 50), iters = a.num("iters", 500);
  for (int i = 0; i < warmup; ++i) run();
  std::vector<double> lat;
  lat.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    run();
    lat.push_back(ms_since(t0));
  }
  double mean = 0;
  for (double x : lat) mean += x;
  mean /= lat.size();
  double p50 = percentile(lat, 50), p95 = percentile(lat, 95), p99 = percentile(lat, 99);
  double ips = batch / (mean / 1e3);
  std::printf("%-14s batch=%-3d p50=%.3f p95=%.3f p99=%.3f ms | %.1f img/s | prep=%.3f ms/img\n",
              backend.c_str(), batch, p50, p95, p99, ips, prep_ms);
  std::ofstream f(a.str("out", "results/results.jsonl"), std::ios::app);
  f << "{\"backend\":\"" << backend << "\",\"batch\":" << batch << ",\"iters\":" << iters
    << ",\"p50_ms\":" << p50 << ",\"p95_ms\":" << p95 << ",\"p99_ms\":" << p99
    << ",\"mean_ms\":" << mean << ",\"throughput_ips\":" << ips
    << ",\"prep_ms_per_img\":" << prep_ms << "}\n";
}

inline void dump_logits(const std::string& path, const float* logits, size_t n) {
  if (path.empty()) return;
  std::ofstream f(path);
  f.precision(9);
  for (size_t i = 0; i < n; ++i) f << logits[i] << "\n";
}