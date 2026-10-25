"""The fused GEGLU against the eager chain: how many values differ, and what it is worth.

Prints, at SDXL's feed-forward widths, the exact count of differing bf16 values in the forward and in the backward (the gate
tolerates the backward's last-bit differences, the forward must be zero), then the time of forward+backward for the chain and
the fused op and the bandwidth floor of each.

    python debug/verify_geglu.py [rows]          # rows = batch * tokens, default 16384
"""
import sys

import torch

from common import bench
from xpu_kernels.triton import geglu as G

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
dev, dt = "xpu", torch.bfloat16
gen = torch.Generator(device=dev).manual_seed(0)

for inner in (5120, 2560):
    h = (torch.randn(rows, 2 * inner, generator=gen, device=dev) * 2).to(dt)
    dy = torch.randn(rows, inner, generator=gen, device=dev).to(dt)

    def chain_pass():
        leaf = h.clone().requires_grad_(True)
        out = G.chain(leaf)
        out.backward(dy)
        return out, leaf.grad

    def fused_pass():
        leaf = h.clone().requires_grad_(True)
        out = G.geglu(leaf)
        out.backward(dy)
        return out, leaf.grad

    out_c, dh_c = chain_pass()
    out_f, dh_f = fused_pass()
    fwd_bad = int((out_c.view(torch.int16) != out_f.view(torch.int16)).sum())
    bwd_bad = int((dh_c.view(torch.int16) != dh_f.view(torch.int16)).sum())
    rel = float((dh_f.double() - dh_c.double()).norm() / dh_c.double().norm())
    t_chain, t_fused = bench(chain_pass), bench(fused_pass)
    floor = (h.numel() + out_c.numel() + dy.numel() + h.numel() * 2) * 2 / 400e9 * 1000  # fwd 3I, bwd 5I elements, ~400 GB/s
    print(f"inner {inner}: forward differs {fwd_bad}/{out_c.numel()}, backward differs {bwd_bad}/{dh_c.numel()} (rel {rel:.1e})")
    print(f"           chain {t_chain:.3f} ms, fused {t_fused:.3f} ms ({t_chain / t_fused:.2f}x), bandwidth floor ~{floor:.3f} ms")
print("gates:", {k: g.describe() for k, g in G.gates().items()})
