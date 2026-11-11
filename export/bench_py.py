"""Python runners for ONNX Runtime (CPU/CUDA) and the TensorRT engine -- same preprocessing,
timing loop, console line and results.jsonl format as the C++ runners (no compiler needed).

  python export/bench_py.py --backend ort-cpu  --image tile.png --dump-logits ort_logits.txt
  python export/bench_py.py --backend ort-cuda --image tile.png
  python export/bench_py.py --backend trt      --image tile.png --dump-logits trt_logits.txt

Timed region = "give me logits": host input -> device -> inference -> host logits -> sync
(preprocessing is measured separately and reported as prep_ms, as in the C++ runners).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess(bgr, size, mean, std):
    # BGR -> resize (bilinear, only if needed) -> RGB -> /255 -> (x-mean)/std -> CHW
    if bgr.shape[0] != size or bgr.shape[1] != size:
        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(mean, np.float32)) / np.array(std, np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def load_input(a):
    if a.image:
        bgr = cv2.imread(a.image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"cannot read {a.image}")
    else:
        print("[warn] no --image given, using random pixels (latency only)")
        bgr = np.random.randint(0, 256, (a.size, a.size, 3), dtype=np.uint8)
    ts = []
    for _ in range(50):
        t0 = time.perf_counter()
        x = preprocess(bgr, a.size, a.mean, a.std)
        ts.append((time.perf_counter() - t0) * 1e3)
    return x, float(np.median(ts))


def benchmark(name, batch, a, prep_ms, run):
    for _ in range(a.warmup):
        run()
    lat = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        run()
        lat.append((time.perf_counter() - t0) * 1e3)
    lat = np.array(lat)
    row = dict(backend=name, batch=batch, iters=a.iters,
               p50_ms=float(np.percentile(lat, 50)), p95_ms=float(np.percentile(lat, 95)),
               p99_ms=float(np.percentile(lat, 99)), mean_ms=float(lat.mean()),
               throughput_ips=float(batch / (lat.mean() / 1e3)), prep_ms_per_img=prep_ms)
    print(f"{name:<14s} batch={batch:<3d} p50={row['p50_ms']:.3f} p95={row['p95_ms']:.3f} "
          f"p99={row['p99_ms']:.3f} ms | {row['throughput_ips']:.1f} img/s | prep={prep_ms:.3f} ms/img")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")


def dump_logits(path, arr):
    if path:
        np.savetxt(path, np.asarray(arr, np.float32).ravel(), fmt="%.9g")


def run_ort(a, one, prep_ms):
    import onnxruntime as ort
    if a.backend == "ort-cuda":
        try:
            import torch  # noqa: F401  (puts torch's bundled CUDA/cuDNN DLLs on the search path)
            ort.preload_dlls()
        except Exception:
            pass
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        name = "ort_cuda"
    else:
        providers = ["CPUExecutionProvider"]
        name = "ort_cpu"
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(a.model, so, providers=providers)
    print("providers in use:", sess.get_providers())
    if a.backend == "ort-cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
        raise SystemExit("CUDAExecutionProvider did not load (missing CUDA/cuDNN DLLs?) -- "
                         "would silently benchmark the CPU. Aborting.")
    for b in a.batches:
        x = np.ascontiguousarray(np.repeat(one[None], b, axis=0))
        if b == 1:
            dump_logits(a.dump_logits, sess.run(None, {"image": x})[0][0])
        benchmark(name, b, a, prep_ms, lambda: sess.run(None, {"image": x}))


def run_trt(a, one, prep_ms):
    import tensorrt as trt
    import torch
    logger = trt.Logger(trt.Logger.WARNING)
    with open(a.engine, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine is None:
        raise SystemExit("engine deserialization failed (built on a different GPU/TRT version?)")
    ctx = engine.create_execution_context()
    in_name = out_name = None
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            in_name = n
        else:
            out_name = n
    stream = torch.cuda.Stream()
    for b in a.batches:
        if not ctx.set_input_shape(in_name, (b, 3, a.size, a.size)):
            print(f"batch {b} is outside the engine's optimization profile, skipping")
            continue
        out_shape = tuple(ctx.get_tensor_shape(out_name))
        h_in = torch.from_numpy(np.repeat(one[None], b, axis=0)).pin_memory()
        d_in = torch.empty(h_in.shape, dtype=torch.float32, device="cuda")
        d_out = torch.empty(out_shape, dtype=torch.float32, device="cuda")
        h_out = torch.empty(out_shape, dtype=torch.float32).pin_memory()
        ctx.set_tensor_address(in_name, d_in.data_ptr())
        ctx.set_tensor_address(out_name, d_out.data_ptr())

        def run():
            with torch.cuda.stream(stream):
                d_in.copy_(h_in, non_blocking=True)
                if not ctx.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("execute_async_v3 failed")
                h_out.copy_(d_out, non_blocking=True)
            stream.synchronize()

        if b == 1:
            run()
            dump_logits(a.dump_logits, h_out[0].numpy())
        benchmark("trt_fp16_py", b, a, prep_ms, run)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["ort-cpu", "ort-cuda", "trt"])
    p.add_argument("--model", default="vit_wildfire.onnx")
    p.add_argument("--engine", default="vit_fp16.plan")
    p.add_argument("--image", default="")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batches", default="1,8,32")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--mean", type=float, nargs=3, default=IMAGENET_MEAN)
    p.add_argument("--std", type=float, nargs=3, default=IMAGENET_STD)
    p.add_argument("--dump-logits", default="")
    p.add_argument("--out", default="results/results.jsonl")
    a = p.parse_args()
    a.batches = [int(s) for s in a.batches.split(",")]

    one, prep_ms = load_input(a)
    if a.backend == "trt":
        run_trt(a, one, prep_ms)
    else:
        run_ort(a, one, prep_ms)


if __name__ == "__main__":
    main()
