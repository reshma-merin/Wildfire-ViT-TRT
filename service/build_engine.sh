#!/bin/sh
# Builds the FP16 TensorRT engine for the GPU this container sees (skips if it already exists).
set -e
ENGINE="${ENGINE_PATH:-/engine/vit_fp16.plan}"
mkdir -p "$(dirname "$ENGINE")"
if [ -f "$ENGINE" ]; then
  echo "engine already present: $ENGINE"
  exit 0
fi
TRTEXEC="$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)"
echo "building TensorRT engine -> $ENGINE"
"$TRTEXEC" --onnx=/models/vit_wildfire.onnx --saveEngine="$ENGINE" --fp16 \
  --minShapes=image:1x3x512x512 --optShapes=image:8x3x512x512 --maxShapes=image:32x3x512x512 \
  --skipInference
echo "engine built"
