# Wildfire-ViT-TRT: ViT wildfire classifier, ONNX / TensorRT deployment and benchmark

Inference optimization of a fine-tuned **MAE-pretrained ViT-Small** wildfire classifier
(Sentinel-2 imagery, 512x512 input, 2 classes, 21.59M parameters). The model is exported to ONNX,
compiled to a **TensorRT 10 FP16** engine, and benchmarked against PyTorch and ONNX Runtime on an
**NVIDIA L40** with numeric parity checks. A C++ REST/gRPC inference service with Docker and Kubernetes
manifests is included.

## Model

| | |
|---|---|
| Architecture | ViT-Small: patch 16, embed 384, depth 12, 6 heads, CLS-token pooling, fixed 2D sin-cos positional embedding |
| Input | `image` float32 `[B, 3, 512, 512]` (dynamic batch) |
| Output | `logits` float32 `[B, 2]` |
| Weights | MAE pretraining on Sentinel-2 tiles, then supervised fine-tuning of the full classifier |

`export/model_def.py` is a self-contained re-implementation of the training classifier. Its parameter names
match the trained checkpoint, which is loaded with `strict=True`.

## Pipeline

```
checkpoint (.safetensors)
   -> export/export_onnx.py      ONNX opset 17, dynamic batch, ORT parity check
   -> trtexec                    TensorRT FP16 engine, shapes min/opt/max = 1/8/32
   -> benchmarks                 PyTorch FP32/FP16, ONNX Runtime CPU/CUDA, TensorRT
   -> export/check_cpp_parity.py cross-backend logit comparison on a real tile
   -> export/make_table.py       results table
```

### Reproduce

```powershell
# 1. Export (checks PyTorch vs ONNX Runtime parity at the end)
python export\export_onnx.py --head-ckpt $ckpt --out vit_wildfire.onnx

# 2. Build the TensorRT FP16 engine (build it on the GPU that will run it)
trtexec --onnx=vit_wildfire.onnx --saveEngine=vit_fp16.plan --fp16 `
  --minShapes=image:1x3x512x512 --optShapes=image:8x3x512x512 --maxShapes=image:32x3x512x512

# 3. Benchmarks (each appends rows to results\results.jsonl)
python export\bench_pytorch.py --head-ckpt $ckpt
python export\bench_py.py --backend ort-cpu  --model vit_wildfire.onnx --image tile.png
python export\bench_py.py --backend ort-cuda --model vit_wildfire.onnx --image tile.png
python export\bench_py.py --backend trt      --engine vit_fp16.plan   --image tile.png

# 4. Table
python export\make_table.py results\results.jsonl
```

Environment: Windows VM with vGPU NVIDIA L40-24Q, driver 572.60, CUDA 12.1 toolkit, TensorRT 10.13.0.35,
Python 3.10, PyTorch 2.11 (cu128), ONNX Runtime GPU.

## Results

All runs on the same L40 (virtual `L40-24Q`), same preprocessing, warmup excluded. Latency is p50 per
batch; throughput is images/s. Throughput for ONNX Runtime rows is computed as batch / p50.

| Backend | Precision | Batch 1 p50 (ms) | Batch 8 p50 (ms) | Batch 32 p50 (ms) | img/s (b1 / b8 / b32) |
|---|---|---|---|---|---|
| PyTorch eager (CUDA) | FP32 | 9.36 | 49.63 | 211.65 | 117 / 160 / 151 |
| PyTorch eager (CUDA) | FP16 | 7.94 | 19.77 | 82.71 | 140 / 404 / 389 |
| ONNX Runtime CUDA | FP32 | 11.84 | 32.93 | 149.30 | ~84 / ~243 / ~214 |
| ONNX Runtime CPU | FP32 | 119.9 | 689.9 | 2,573 | ~8 / ~12 / ~12 |
| **TensorRT (Python runtime)** | **FP16** | **1.47** | **5.37** | **26.07** | **636 / 1,477 / 1,235** |

TensorRT p95 latency: 1.89 ms (batch 1), 5.70 ms (batch 8), 27.18 ms (batch 32).

**Speedup of TensorRT FP16 over eager PyTorch FP16 (p50):** 5.4x at batch 1, 3.7x at batch 8, 3.2x at batch 32.
Versus ONNX Runtime CUDA (FP32): about 8x, 6x and 5.7x. That comparison is not like-for-like (FP32 vs FP16).

Cross-check with `trtexec` (spin-wait, 10 s): host latency 1.69 / 5.58 / 28.63 ms and GPU compute
1.57 / 4.63 / 24.80 ms for batch 1 / 8 / 32, consistent with the Python runner.

**Numeric parity** (max abs logit difference vs PyTorch FP32)

| Comparison | Max abs diff | Predicted class |
|---|---|---|
| PyTorch vs ONNX (CPU, batch 2, export check) | 9.5e-07 | same |
| ONNX Runtime CPU | 1.5e-06 | same |
| ONNX Runtime CUDA | 1.1e-04 | same |
| TensorRT FP16 | 1.6e-03 | same |

**Caveats**
- The PyTorch baseline is eager mode with hand-written attention, with no `torch.compile` or fused
  attention, so a tuned PyTorch setup would narrow the gap.
- TensorRT runs through the Python runtime here (not a native C++ runner); the numbers match `trtexec`.
- ONNX Runtime CUDA is FP32 and may use TF32 matmuls; it includes host-device copies in each call.
- Batch 32 is slower per image than batch 8 because the engine profile's optimal shape is batch 8.
- Image decode and normalization take about 6 ms per image on the CPU and are not included above.
- The ONNX Runtime CPU rows used 30 iterations on the VM CPU and only show that GPU inference matters.

### Notes on method
- ONNX is exported with the legacy TorchScript exporter (`dynamo=False`, opset 17) because the dynamo
  exporter emitted opset-18 `Split` nodes that onnxsim could not handle. The simplifier is optional.
- A TensorRT engine is only valid for the GPU model and TensorRT version that built it, so the engine is
  rebuilt on the target GPU rather than shipped.
- Preprocessing is identical in every runner: BGR to RGB, bilinear resize if needed, divide by 255,
  normalize with mean/std, HWC to CHW. Verify the mean/std match the values used in fine-tuning.
- The engine's optimization profile uses opt batch 8; rebuild with `--optShapes=image:32x3x512x512` if
  large-batch throughput is the target.

## Inference service (C++ / REST / gRPC)

`service/` contains a C++ TensorRT service (TensorRT 10 `enqueueV3`, pinned host memory, one CUDA stream
per execution context so requests run concurrently).

| Endpoint | Purpose |
|---|---|
| `POST /predict` (body = PNG/JPEG bytes) | classify one image, JSON response |
| gRPC `wildfire.Classifier/Predict` (:50051) | same prediction over protobuf |
| `GET /health` | liveness |
| `GET /ready` | readiness: true only after the engine is loaded and warmed up |
| `GET /metrics` | Prometheus: request counters, inference and request latency histograms, in-flight gauge, uptime |

Throughput: `rate(wildfire_requests_total[1m])`.
p95 latency: `histogram_quantile(0.95, rate(wildfire_request_seconds_bucket[1m]))`.

### Docker

The multi-stage `Dockerfile` compiles the service inside an NGC TensorRT image (no local compiler
needed) and bakes in the ONNX model. The engine is built on first start on the GPU the container sees,
into a mounted `/engine` volume.

```bash
mkdir -p service/model && cp vit_wildfire.onnx service/model/
cd service
docker build -t wildfire-vit-trt:latest .
docker run --rm --gpus all -p 8080:8080 -p 50051:50051 -v wildfire-engine:/engine wildfire-vit-trt:latest

curl http://localhost:8080/ready
curl -X POST --data-binary @../tile.png -H "Content-Type: application/octet-stream" http://localhost:8080/predict
curl http://localhost:8080/metrics
python client_bench.py --image ../tile.png --mode rest --concurrency 8 --requests 2000
```

Requires Docker with the NVIDIA Container Toolkit on a Linux host with an NVIDIA GPU.

### Kubernetes

`service/k8s/deployment.yaml` defines a Deployment and Service:
- **GPU scheduling:** `resources.limits: nvidia.com/gpu: 1` (needs the NVIDIA device plugin or GPU Operator),
  plus a toleration for GPU-tainted nodes.
- **Init container** builds the TensorRT engine on the GPU the pod is scheduled to.
- **Probes:** startup (allows for CUDA init, engine load and warmup), readiness on `/ready`, liveness on `/health`.
- **Monitoring:** Prometheus scrape annotations pointing at `/metrics`.
- **Graceful shutdown:** on SIGTERM readiness fails first, then the servers drain.

```bash
kubectl apply -f service/k8s/deployment.yaml
kubectl get pods -w
kubectl port-forward svc/wildfire-vit 8080:8080 50051:50051
```

For a single machine use k3s, minikube or kind with the NVIDIA runtime, and make the local image visible to
the cluster (`minikube image load`, `kind load docker-image`, or push to a registry and edit `image:`).



## Layout

```
export/     model_def.py, export_onnx.py, bench_pytorch.py, bench_py.py, check_cpp_parity.py, make_table.py
src/        standalone C++ benchmark runners (ORT, TensorRT; not compiled)
service/    C++ REST/gRPC service, Dockerfile, k8s/deployment.yaml, client_bench.py
results/    results.jsonl and generated tables
```
