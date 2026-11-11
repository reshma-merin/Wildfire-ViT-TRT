"""Load-test client for the service (REST or gRPC): concurrent requests, p50/p95/p99, throughput.

  python client_bench.py --image tile.png --mode rest --concurrency 8 --requests 2000
  python client_bench.py --image tile.png --mode grpc --concurrency 8 --requests 2000
    (gRPC mode: pip install grpcio grpcio-tools, then once:
     python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/classifier.proto)
"""
import argparse
import http.client
import json
import threading
import time

import numpy as np


def rest_worker(host, port, body, n, lat, errs):
    conn = http.client.HTTPConnection(host, port, timeout=60)
    for _ in range(n):
        t0 = time.perf_counter()
        conn.request("POST", "/predict", body, {"Content-Type": "application/octet-stream"})
        r = conn.getresponse()
        r.read()
        lat.append((time.perf_counter() - t0) * 1e3)
        if r.status != 200:
            errs.append(r.status)


def grpc_worker(host, port, body, n, lat, errs):
    import grpc
    import classifier_pb2
    import classifier_pb2_grpc
    stub = classifier_pb2_grpc.ClassifierStub(grpc.insecure_channel(f"{host}:{port}"))
    req = classifier_pb2.PredictRequest(image=body)
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            stub.Predict(req, timeout=60)
        except Exception as e:  # noqa: BLE001
            errs.append(str(e))
        lat.append((time.perf_counter() - t0) * 1e3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--mode", choices=["rest", "grpc"], default="rest")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=0, help="default 8080 (rest) / 50051 (grpc)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--requests", type=int, default=2000, help="total requests")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--out", default="", help="append a jsonl result row here")
    a = p.parse_args()
    port = a.port or (8080 if a.mode == "rest" else 50051)
    worker = rest_worker if a.mode == "rest" else grpc_worker
    body = open(a.image, "rb").read()

    worker(a.host, port, body, a.warmup, [], [])  # warmup, single connection

    per = a.requests // a.concurrency
    lats = [[] for _ in range(a.concurrency)]
    errs = [[] for _ in range(a.concurrency)]
    threads = [threading.Thread(target=worker, args=(a.host, port, body, per, lats[i], errs[i]))
               for i in range(a.concurrency)]
    t0 = time.perf_counter()
    [t.start() for t in threads]
    [t.join() for t in threads]
    wall = time.perf_counter() - t0

    lat = np.array([x for l in lats for x in l])
    n_err = sum(len(e) for e in errs)
    row = dict(backend=f"service_{a.mode}", concurrency=a.concurrency, requests=int(lat.size),
               errors=n_err, p50_ms=float(np.percentile(lat, 50)), p95_ms=float(np.percentile(lat, 95)),
               p99_ms=float(np.percentile(lat, 99)), throughput_rps=float(lat.size / wall))
    print(row)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
