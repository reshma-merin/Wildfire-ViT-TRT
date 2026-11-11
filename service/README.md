# Wildfire ViT TensorRT inference service

C++ service (TensorRT 10 `enqueueV3`, pinned memory, one CUDA stream per execution context) exposing
**REST** and **gRPC**, with Prometheus metrics and Kubernetes readiness/liveness probes.

| Endpoint | Purpose |
|---|---|
| `POST /predict` (body = PNG/JPEG bytes) | classify one image, JSON out |
| `GET /health` | liveness (process up) |
| `GET /ready` | readiness (engine loaded + warmed up) |
| `GET /metrics` | Prometheus: request counters, inference/request latency histograms, in-flight, uptime |
| gRPC `wildfire.Classifier/Predict` on :50051 | same prediction, protobuf |

Throughput in Prometheus = `rate(wildfire_requests_total[1m])`; p95 latency =
`histogram_quantile(0.95, rate(wildfire_request_seconds_bucket[1m]))`.

## 0. Can this machine run it?

Run these in PowerShell on the L40 VM:

```powershell
docker --version
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
wsl --status
```

The third command should print the L40. If `docker` is missing, or the GPU container fails, Docker cannot use
the GPU on this machine (typical for a virtual GPU without WSL2 GPU passthrough or without admin rights).
In that case run steps 1-6 on any Linux machine with an NVIDIA GPU, Docker and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

## 1. Put the model in the build context

```powershell
mkdir service\model
copy vit_wildfire.onnx service\model\
cd service
```

(Do not commit the .onnx to git.)

## 2. Build the image (compiles the C++ inside Docker -- no local compiler needed)

```powershell
docker build -t wildfire-vit-trt:latest .
```

If the base tag is not available, pick a current `nvcr.io/nvidia/tensorrt` tag with TensorRT 10.x and pass
`--build-arg TRT_IMAGE=...`.

## 3. Run it with Docker

```powershell
docker run --rm --gpus all -p 8080:8080 -p 50051:50051 -v wildfire-engine:/engine wildfire-vit-trt:latest
```

First start builds the engine (~1 min) into the `wildfire-engine` volume, later starts reuse it.

## 4. Test it

```powershell
curl.exe http://localhost:8080/ready
curl.exe -X POST --data-binary "@..\tile.png" -H "Content-Type: application/octet-stream" http://localhost:8080/predict
curl.exe http://localhost:8080/metrics
```

The returned `probabilities` should match the earlier PyTorch/ONNX result for the same tile
(logits ~ `[1.8135, -1.7660]` for the benchmark tile; FP16 allows ~1e-3 drift).

gRPC (with [grpcurl](https://github.com/fullstorydev/grpcurl); the image field is base64):

```
grpcurl -plaintext -import-path proto -proto classifier.proto -d "{\"image\":\"<base64 of tile.png>\"}" localhost:50051 wildfire.Classifier/Predict
```

## 5. Load test (latency / throughput)

```powershell
python client_bench.py --image ..\tile.png --mode rest --concurrency 8 --requests 2000 --out ..\results\service.jsonl
```

## 6. Kubernetes with GPU scheduling

Needs a cluster whose node exposes `nvidia.com/gpu` (NVIDIA device plugin or GPU Operator). For a single machine:
k3s/minikube/kind with the NVIDIA runtime. Make the local image visible to the cluster
(`minikube image load wildfire-vit-trt:latest`, `kind load docker-image ...`, or push to a registry and edit `image:`).

```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods -w                         # init container builds the engine, then server goes Ready
kubectl describe pod -l app=wildfire-vit    # shows the nvidia.com/gpu allocation and probe status
kubectl port-forward svc/wildfire-vit 8080:8080 50051:50051
```

The pod requests `nvidia.com/gpu: 1`, builds the engine in an init container, uses a startup probe while loading,
gates traffic on `/ready`, and exposes Prometheus annotations for scraping.

## Status

Written but **not yet built or run** at the time it was generated. Fill in your own results below before using any
of it on a resume:

- [ ] `docker build` succeeds
- [ ] container serves `/predict` and the logits match PyTorch
- [ ] gRPC call succeeds
- [ ] pod reaches Ready on a GPU node; `/metrics` shows requests
- [ ] load-test numbers recorded
