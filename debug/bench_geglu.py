"""The fused GEGLU piece by piece, and which launch configuration is fastest.

Part 1: forward and backward separately, the eager chain against the fused kernel, against the time to copy the same bytes.
Part 2: the fused kernels over BLOCK and num_warps (nothing here is a reduction, so neither changes the numbers).

    LEVEL_ZERO_V1_SDK_PATH=... python debug/bench_geglu.py [rows]       # rows = batch * tokens, default 16384
"""
import sys

import torch

from common import bench
from xpu_kernels.triton import geglu as G

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
dev, dt = "xpu", torch.bfloat16
GBPS = 397.0  # the copy ceiling measured on this card


def floor_ms(elements):
    return elements * 2 / (GBPS * 1e9) * 1000


for inner in (5120, 2560):
    gen = torch.Generator(device=dev).manual_seed(0)
    h = (torch.randn(rows, 2 * inner, generator=gen, device=dev) * 2).to(dt)
    dy = torch.randn(rows, inner, generator=gen, device=dev).to(dt)
    leaf = h.clone().requires_grad_(True)
    out = G.chain(leaf)
    print(f"--- rows {rows}, inner {inner}: gates {[g.describe()[:40] for g in G.gates().values()]}")
    fwd_floor, bwd_floor = floor_ms(rows * inner * 3), floor_ms(rows * inner * 5)
    print(f"forward : chain {bench(lambda: G.chain(h)):.3f} ms, fused {bench(lambda: G._forward(h)):.3f} ms, floor {fwd_floor:.3f} ms")
    print(f"backward: chain {bench(lambda: torch.autograd.grad(out, leaf, dy, retain_graph=True)):.3f} ms, "
          f"fused {bench(lambda: G._backward(dy, h)):.3f} ms, floor {bwd_floor:.3f} ms")
    print(f"copy of the forward's bytes: {bench(lambda: h.clone()):.3f} ms for {h.numel() * 2 / 1e6:.0f} MB read and as much written")
    dh = torch.empty_like(h)
    y = torch.empty(rows, inner, dtype=dt, device=dev)
    print("block  warps   forward ms   backward ms")
    for block in (256, 512, 1024, 2048, 4096):
        blocks = (inner + block - 1) // block
        for warps in (2, 4, 8, 16):
            try:
                f = bench(lambda: G._forward_kernel[(rows * blocks,)](h, y, inner, blocks, BLOCK=block, num_warps=warps), n=10)
                b = bench(lambda: G._backward_kernel[(rows * blocks,)](dy, h, dh, inner, blocks, BLOCK=block, num_warps=warps), n=10)
                print(f"{block:5d} {warps:6d} {f:12.3f} {b:13.3f}")
            except Exception as exc:  # noqa: BLE001 - a configuration the backend refuses is a result too
                print(f"{block:5d} {warps:6d}   failed: {str(exc)[:60]}")
