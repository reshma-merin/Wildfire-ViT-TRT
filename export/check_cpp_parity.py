"""Compare C++ logits (from --dump-logits) with PyTorch on the SAME image.

The preprocessing here mirrors src/bench_common.hpp exactly:
cv2.imread (BGR) -> resize (INTER_LINEAR, only if size differs) -> RGB -> /255 -> (x-mean)/std -> CHW.

  python export/check_cpp_parity.py --image tile.png --logits ort_logits.txt --num-classes 2
"""
import argparse
import cv2
import numpy as np
import torch
from model_def import DEFAULT_REPO, build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess(path, size, mean, std):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(mean, np.float32)) / np.array(std, np.float32)
    return img.transpose(2, 0, 1)[None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--logits", required=True)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--head-ckpt", default=None)
    p.add_argument("--pool", default="mean")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--mean", type=float, nargs=3, default=IMAGENET_MEAN)
    p.add_argument("--std", type=float, nargs=3, default=IMAGENET_STD)
    p.add_argument("--atol", type=float, default=1e-3, help="use ~1e-2 for TensorRT FP16")
    a = p.parse_args()

    model = build_model(a.repo, a.num_classes, a.head_ckpt, a.pool)
    with torch.no_grad():
        ref = model(torch.from_numpy(preprocess(a.image, a.size, a.mean, a.std))).numpy().ravel()
    cpp = np.loadtxt(a.logits, dtype=np.float32).ravel()
    print("pytorch:", ref)
    print("c++    :", cpp)
    print("max abs diff:", float(np.abs(ref - cpp).max()), "| argmax match:", ref.argmax() == cpp.argmax())
    np.testing.assert_allclose(cpp, ref, atol=a.atol, rtol=1e-3)
    print("C++ parity OK")


if __name__ == "__main__":
    main()