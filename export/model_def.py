"""Deployable classifier = exact re-implementation of Wildfire/pretraining_MAE/classifier.py
(ViTClassifier: ViT-Small, 512x512, patch 16, CLS-token head), loadable from the
fine-tuned model.safetensors. Parameter names match the checkpoint 1:1 (151 keys).

Differences from training code, chosen so the model exports cleanly to ONNX/TensorRT
(numerically equivalent):
  * attention is written out with F.linear/softmax using the same in_proj_* / out_proj
    parameters as nn.MultiheadAttention (avoids MHA's fused "fast path" op that ONNX can't export)
  * forward(x) takes a tensor and returns logits (not a dict)
The classifier ALWAYS pools the CLS token (after the final LayerNorm), exactly as in training;
the `pool` argument is accepted only for script compatibility and ignored.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_REPO = ""  # unused: no Hugging Face access needed


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    def sincos_1d(dim, pos):
        omega = np.arange(dim // 2, dtype=np.float32)
        omega /= dim / 2.0
        omega = 1.0 / (10000 ** omega)
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb_h = sincos_1d(embed_dim // 2, grid[0])
    emb_w = sincos_1d(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Same parameters as nn.MultiheadAttention(batch_first=True)."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        h = self.num_heads
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(b, n, h, d // h).transpose(1, 2)
        k = k.reshape(b, n, h, d // h).transpose(1, 2)
        v = v.reshape(b, n, h, d // h).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d // h), dim=-1)
        out = (att @ v).transpose(1, 2).reshape(b, n, d)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTClassifier(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=384,
                 depth=12, num_heads=6, num_classes=2):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        pos = get_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid_size)
        self.register_buffer(
            "pos_embed",
            torch.cat([torch.zeros(1, embed_dim), torch.from_numpy(pos).float()], dim=0).unsqueeze(0),
            persistent=False,
        )
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])


def _load_state(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    sd = torch.load(path, map_location="cpu")
    return sd.get("state_dict", sd)


def build_model(repo=DEFAULT_REPO, num_classes=2, head_ckpt=None, pool="cls", size=512):
    if pool != "cls":
        print(f"[note] pool='{pool}' ignored: the trained classifier uses the CLS token.")
    model = ViTClassifier(img_size=size, num_classes=num_classes)
    if head_ckpt:
        sd = _load_state(head_ckpt)
        model.load_state_dict(sd, strict=True)  # raises on ANY missing/unexpected/shape mismatch
        print(f"loaded {head_ckpt}: all {len(sd)} tensors matched (strict)")
    else:
        print("[warn] no --head-ckpt: random weights (fine for latency, not for accuracy)")
    return model.eval().cpu()