"""The Triton attention's launch configuration: which (BLOCK_M, BLOCK_N, num_warps, num_stages) is fastest at SDXL's shapes.

The forward and the backward's two kernels do different work, so they are swept separately and each has its own table in
``python/xpu_kernels/triton/attention.py`` (``DEFAULT_CONFIG``, ``BWD_DQ_CONFIG``, ``BWD_DKDV_CONFIG``).  For dQ, BLOCK_M is the query
tile and BLOCK_N the key tile; for dK/dV it is the other way round, and the grid is over keys.  A configuration the backend refuses
(scratch space, register pressure) is a result too and is printed as failed.

The kernels build their descriptors on the device, so nothing here monkeypatches the library's tables: each candidate is passed
straight to the kernel.  Whether the result is worth using at all is part 4 of ``debug/verify_attention_capture.py``.

    python debug/bench_attention.py [batch]        # default 4
"""
import math
import sys

import torch

from common import device_ms
from xpu_kernels.triton import attention as A

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 4
dev, dt = "xpu", torch.bfloat16
SHAPES = (("self 1024, 10 heads", (batch, 10, 1024, 64), (batch, 10, 1024, 64)),
          ("cross 77, 10 heads", (batch, 10, 1024, 64), (batch, 10, 77, 64)),
          ("self 256, 20 heads", (batch, 20, 256, 64), (batch, 20, 256, 64)))
FWD = ((64, 32, 4, 2), (128, 32, 8, 2), (64, 64, 4, 2), (128, 64, 8, 2), (128, 64, 8, 3))
DQ = ((128, 16, 8, 2), (128, 32, 8, 2), (64, 32, 4, 2), (128, 64, 8, 2))
DKDV = ((16, 64, 8, 2), (16, 128, 8, 2), (32, 64, 8, 2), (32, 128, 8, 2))


def best(fn):
    fn()
    torch.xpu.synchronize()
    return min(device_ms(fn) for _ in range(3))


for label, qs, ks in SHAPES:
    gen = torch.Generator(device=dev).manual_seed(0)
    q, k, v, dout = (torch.randn(s, generator=gen, device=dev).to(dt) for s in (qs, ks, ks, qs))
    scale = 1.0 / math.sqrt(qs[-1])
    batch_, heads, lq, d = q.shape
    bh, lk = batch_ * heads, ks[2]
    out, lse = A._forward(q, k, v, scale)
    delta = torch.empty_like(lse)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    A._kernels[1][(math.ceil(lq / 128), bh)](out, dout, delta, lq, D=d, BLOCK_M=128, num_warps=4)
    torch.xpu.synchronize()
    print(f"--- {label}: q {qs}, k {ks}")
    print(f"{'kernel':>6s} {'BM':>4s} {'BN':>4s} {'warps':>5s} {'stages':>6s} {'ms':>8s}")
    rows = {}
    for bm, bn, w, s in FWD:
        fn = lambda bm=bm, bn=bn, w=w, s=s: A._kernels[0][(math.ceil(lq / bm), bh)](
            q, k, v, out, lse, scale, lq, lk, D=d, BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=s)
        rows.setdefault("fwd", []).append((best(fn), bm, bn, w, s))
    for bm, bn, w, s in DQ:
        fn = lambda bm=bm, bn=bn, w=w, s=s: A._kernels[2][(math.ceil(lq / bm), bh)](
            q, k, v, dout, lse, delta, dq, scale, lq, lk, D=d, BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=s)
        rows.setdefault("dq", []).append((best(fn), bm, bn, w, s))
    for bm, bn, w, s in DKDV:
        fn = lambda bm=bm, bn=bn, w=w, s=s: A._kernels[3][(math.ceil(lk / bn), bh)](
            q, k, v, dout, lse, delta, dk, dv, scale, lq, lk, D=d, BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=s)
        rows.setdefault("dkdv", []).append((best(fn), bm, bn, w, s))
    for kernel, found in rows.items():
        for t, bm, bn, w, s in found:
            print(f"{kernel:>6s} {bm:4d} {bn:4d} {w:5d} {s:6d} {t:8.3f}")
        t, bm, bn, w, s = min(found)
        print(f"best {kernel} for {label}: ({bm}, {bn}, {w}, {s})  {t:.3f} ms")
