"""PyTorch eager baseline, same stats/format as the C++ runners (model time only)."""
import argparse, json, time
import numpy as np
import torch
from model_def import DEFAULT_REPO, build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--head-ckpt", default=None)
    p.add_argument("--pool", default="mean")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batches", default="1,8,32")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/results.jsonl")
    a = p.parse_args()

    model = build_model(a.repo, a.num_classes, a.head_ckpt, a.pool).to(a.device)
    if a.dtype == "fp16":
        model = model.half()
    dt = torch.float16 if a.dtype == "fp16" else torch.float32
    sync = torch.cuda.synchronize if a.device == "cuda" else (lambda: None)
    name = f"pytorch_{a.device}_{a.dtype}"

    for b in [int(s) for s in a.batches.split(",")]:
        x = torch.randn(b, 3, a.size, a.size, device=a.device, dtype=dt)
        with torch.inference_mode():
            for _ in range(a.warmup):
                model(x)
            sync()
            lat = []
            for _ in range(a.iters):
                t0 = time.perf_counter()
                model(x)
                sync()  # GPU: must sync before stopping the clock
                lat.append((time.perf_counter() - t0) * 1e3)
        lat = np.array(lat)
        row = dict(backend=name, batch=b, iters=a.iters,
                   p50_ms=float(np.percentile(lat, 50)), p95_ms=float(np.percentile(lat, 95)),
                   p99_ms=float(np.percentile(lat, 99)), mean_ms=float(lat.mean()),
                   throughput_ips=float(b / (lat.mean() / 1e3)))
        print(row)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()