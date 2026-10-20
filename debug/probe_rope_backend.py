"""Which implementation does the production `rms_rope_split_half` actually run?

The in-model A/B says the single-pass kernel changes the image (PSNR 23 dB, i.e.
a different image, not a rounding difference), while the offline probe says it is
bit-identical to `comfy_kitchen.backends.triton.rms_rope.rms_rope_split_half` on
0/12.2M values.  Both can be true only if the *production* op is not the Triton
kernel -- the registry has an `eager` backend too, and the op dispatches through
`torch.ops.comfy_kitchen.rms_rope_split_half`.

This runs the op, the two backends directly, and `comfy.xpu_rope` on one set of
tensors built the way `model.py` builds them (q/k are `chunk` views of one qkv
projection, so they are strided, and `pe` is fp32 from `rope_fp32`).

    python debug/probe_rope_backend.py
"""
import sys

import torch

sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")

import comfy_kitchen as ck  # noqa: E402
from comfy_kitchen.backends.eager.rope import (  # noqa: E402
    rms_rope_split_half as eager_rope,
)
from comfy_kitchen.backends.triton.rms_rope import (  # noqa: E402
    rms_rope_split_half as triton_rope,
)

from comfy.xpu_rope import rms_rope_split_half as mine_rope  # noqa: E402

dev = torch.device("xpu", 0)
B, S, H, D = 1, 3973, 24, 128
EPS = 1e-6

torch.manual_seed(0)
# in-model layout: one fused qkv projection, chunked (strided views)
qkv = (torch.randn(B, S, 3 * H * D, device=dev) * 0.3).to(torch.bfloat16)
q, k, _v = qkv.chunk(3, dim=-1)
q = q.view(B, S, H, D)
k = k.view(B, S, H, D)
scale = torch.rand(D, device=dev, dtype=torch.float32) * 0.2 + 0.9
ang = torch.rand(B, S, 1, D // 2, device=dev, dtype=torch.float32) * 3.0
pe = torch.stack([torch.stack([ang.cos(), -ang.sin()], -1),
                  torch.stack([ang.sin(), ang.cos()], -1)], -2)

print(f"q{tuple(q.shape)} strides={q.stride()} contig={q.is_contiguous()}")
print(f"pe{tuple(pe.shape)} {pe.dtype}  scale {scale.dtype}")


def cmp(name, a, b):
    bad = (a.view(torch.int16) != b.view(torch.int16)).sum().item()
    d = (a.float() - b.float()).abs().max().item()
    print(f"  {name:<40} mismatches {bad:>9d}/{b.numel():<9d} maxabs {d:.3e}")


op_q, op_k = ck.rms_rope_split_half(q, k, pe, scale, None, EPS)
tr_q, tr_k = triton_rope(q, k, pe, scale, None, EPS)
eg_q, eg_k = eager_rope(q, k, pe, scale, None, EPS)
my_q, my_k = mine_rope(q, k, pe, scale, None, EPS)
torch.xpu.synchronize()

print("\nvs the production op (what the image actually comes from):")
cmp("triton backend", tr_q, op_q)
cmp("eager backend", eg_q, op_q)
cmp("comfy.xpu_rope (mine)", my_q, op_q)

print("\nbackends against each other:")
cmp("eager vs triton", eg_q, tr_q)
cmp("mine vs triton", my_q, tr_q)

print("\nother outputs (v is unused, sanity):")
cmp("op k vs triton k", op_k, tr_k)
