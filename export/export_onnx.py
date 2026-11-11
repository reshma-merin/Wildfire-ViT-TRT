"""Step 1: export the classifier to ONNX, simplify, and check parity vs PyTorch.

  python export/export_onnx.py --num-classes 2 --head-ckpt model.safetensors --out vit_wildfire.onnx
"""
import argparse
import numpy as np
import torch
from model_def import DEFAULT_REPO, build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--head-ckpt", default=None)
    p.add_argument("--pool", default="cls", choices=["mean", "cls"])
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--out", default="vit_wildfire.onnx")
    p.add_argument("--no-simplify", action="store_true")
    a = p.parse_args()

    model = build_model(a.repo, a.num_classes, a.head_ckpt, a.pool, a.size)
    dummy = torch.randn(1, 3, a.size, a.size)

    # dynamo=False -> classic TorchScript exporter: emits opset 17 directly, so onnxsim /
    # TensorRT see a consistent graph (the new dynamo exporter writes opset-18 ops, then
    # down-converts, leaving attributes like Split.num_outputs that opset 17 doesn't allow).
    torch.onnx.export(
        model, dummy, a.out,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print("exported", a.out)

    if not a.no_simplify:
        try:
            import onnx
            from onnxsim import simplify
            m, ok = simplify(onnx.load(a.out))
            assert ok, "onnxsim could not validate the simplified graph"
            onnx.save(m, a.out)
            print("simplified with onnxsim")
        except Exception as e:  # simplification is optional; TensorRT optimizes the graph anyway
            print(f"[warn] onnxsim skipped ({type(e).__name__}: {e}); keeping the unsimplified graph")

    # Parity check at a DIFFERENT batch size than the dummy, to prove dynamic batch works.
    import onnxruntime as ort
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    x = np.random.randn(2, 3, a.size, a.size).astype(np.float32)
    onnx_out = sess.run(None, {"image": x})[0]
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(onnx_out, torch_out, rtol=1e-3, atol=1e-4)
    print("parity OK, max abs diff:", float(np.abs(onnx_out - torch_out).max()))


if __name__ == "__main__":
    main()
