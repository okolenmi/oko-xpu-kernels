"""The Triton attention's launch configuration: which (BLOCK_M, BLOCK_N, num_warps, num_stages) is fastest at SDXL's shapes.

``tl.dot`` on this card was two orders of magnitude below oneDNN with small tiles, so the default table is a guess.  This sweeps the
forward and the two backward kernels separately (device time) and prints the best per shape; the winner goes into
``DEFAULT_CONFIG`` in ``python/xpu_kernels/triton/attention.py`` or into ``XPU_KERNELS_ATTENTION_CONFIG``.  A configuration the
backend refuses (scratch space, register pressure) is a result too and is printed as failed.

    python debug/bench_attention.py [batch]        # default 4
"""
import itertools
import math
import os
import sys

import torch

from common import device_ms
from xpu_kernels.triton import attention as A

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 4
dev, dt = "xpu", torch.bfloat16
SHAPES = (("self 1024, 10 heads", (batch, 10, 1024, 64), (batch, 10, 1024, 64)),
          ("cross 77, 10 heads", (batch, 10, 1024, 64), (batch, 10, 77, 64)),
          ("self 256, 20 heads", (batch, 20, 256, 64), (batch, 20, 256, 64)))
BLOCKS_M, BLOCKS_N, WARPS, STAGES = (64, 128, 256), (32, 64, 128), (4, 8, 16), (1, 2, 3)

for label, qs, ks in SHAPES:
    gen = torch.Generator(device=dev).manual_seed(0)
    q, k, v, dout = (torch.randn(s, generator=gen, device=dev).to(dt) for s in (qs, ks, ks, qs))
    scale = 1.0 / math.sqrt(qs[-1])
    print(f"--- {label}: q {qs}, k {ks}")
    print(f"{'BM':>4s} {'BN':>4s} {'warps':>5s} {'stages':>6s} {'fwd ms':>8s} {'bwd ms':>8s} {'sum ms':>8s}")
    rows = []
    for bm, bn, w, s in itertools.product(BLOCKS_M, BLOCKS_N, WARPS, STAGES):
        os.environ["XPU_KERNELS_ATTENTION_CONFIG"] = f"{bm},{bn},{w},{s}"
        try:
            out, lse = A._forward(q, k, v, scale)
            fwd = device_ms(lambda: A._forward(q, k, v, scale))
            bwd = device_ms(lambda: A._backward(q, k, v, out, lse, dout, scale))
        except Exception as exc:  # noqa: BLE001
            print(f"{bm:4d} {bn:4d} {w:5d} {s:6d}   failed: {str(exc).splitlines()[0][:70] if str(exc) else type(exc).__name__}")
            continue
        rows.append((fwd + bwd, bm, bn, w, s, fwd, bwd))
        print(f"{bm:4d} {bn:4d} {w:5d} {s:6d} {fwd:8.3f} {bwd:8.3f} {fwd + bwd:8.3f}")
    if rows:
        best = min(rows)
        print(f"best for {label}: XPU_KERNELS_ATTENTION_CONFIG={best[1]},{best[2]},{best[3]},{best[4]}  ({best[0]:.3f} ms)")
os.environ.pop("XPU_KERNELS_ATTENTION_CONFIG", None)
