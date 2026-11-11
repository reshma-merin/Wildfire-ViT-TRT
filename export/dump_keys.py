"""Print the architecture hidden in a checkpoint: key names (block index collapsed) + shapes.
  python dump_keys.py "C:\\path\\to\\model.safetensors"
"""
import re
import sys

from safetensors.torch import load_file

sd = load_file(sys.argv[1])
seen = {}
for k, v in sd.items():
    n = re.sub(r"blocks\.\d+", "blocks.N", k)
    seen.setdefault(n, tuple(v.shape))
for n, s in seen.items():
    print(f"{n:55s} {s}")
blocks = {int(m.group(1)) for k in sd if (m := re.match(r"blocks\.(\d+)\.", k))}
print("\nnum blocks:", len(blocks), "| total keys:", len(sd),
      "| params (M):", round(sum(v.numel() for v in sd.values()) / 1e6, 2))
