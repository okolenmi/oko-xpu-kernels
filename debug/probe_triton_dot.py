"""Why the Triton weight dequantizer was 30-400x slower than a 3.4 GFLOP product should be: a ladder, one change per rung.

The dequantizer's rotation is ``C = A @ R`` with A the weight viewed as ``[N*K/256, 256]`` rows of one 256-element group and R
the 256x256 rotation: 3.4 GFLOP for a 1280x5120 weight, 0.07 ms at 50 TFLOPS.  Measured, the first version took 84 ms and most
launch configurations did not build ("total scratch space exceeds HW supported limit": the compiler spilled the operands).
This probe starts from a plain bf16 product and adds, one at a time, what the dequantizer does, so the rung where the time
falls off a cliff names the cause:

  torch.mm            oneDNN on the same bf16 operands: what the matrix unit does with this shape
  L0  plain pointers, bf16 A, K looped in chunks (the textbook Triton product)
  L1  block pointers instead (what Intel's backend maps to 2-D block loads)
  L2  A is int8 and widened to bf16 in registers
  L3  as L2 with the row scale and the bf16 rounding (the dequantizer's arithmetic)
  L4  the first version's structure: the whole 256-wide operand in one product, no K loop

    LEVEL_ZERO_V1_SDK_PATH=... python debug/probe_triton_dot.py [N K]      # about 10 kernels to compile: a minute or two of CPU
"""
import sys

import torch
from xpu_kernels import _levelzero

_levelzero.ensure_headers()  # before the first Triton compile

import triton
import triton.language as tl

from common import bench

dev = "xpu"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1280
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5120
GPR = K // 256            # groups per weight row
M = N * GPR               # rows of the [M, 256] view


@triton.jit
def _product(a_ptr, r_ptr, c_ptr, scale_ptr, M, gpr, KIND: tl.constexpr, BM: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    mask = rows < M
    cols = tl.program_id(1) * BJ + tl.arange(0, BJ)
    base = rows.to(tl.int64)[:, None] * 256
    acc = tl.zeros((BM, BJ), dtype=tl.float32)
    for k in range(0, 256, BK):
        ks = k + tl.arange(0, BK)
        a = tl.load(a_ptr + base + ks[None, :], mask=mask[:, None], other=0)
        if KIND >= 1:
            a = a.to(tl.float32)
        if KIND == 2:
            s = tl.load(scale_ptr + rows // gpr, mask=mask, other=0.0)
            a = a * s[:, None]
        if KIND >= 1:
            a = a.to(tl.bfloat16)
        r = tl.load(r_ptr + ks[:, None] * 256 + cols[None, :])
        acc = tl.dot(a, r, acc)
    tl.store(c_ptr + base + cols[None, :], acc.to(tl.bfloat16), mask=mask[:, None])


@triton.jit
def _product_block(a_ptr, r_ptr, c_ptr, M, BM: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr):
    m0, j0 = tl.program_id(0) * BM, tl.program_id(1) * BJ
    a_blk = tl.make_block_ptr(a_ptr, (M, 256), (256, 1), (m0, 0), (BM, BK), (1, 0))
    r_blk = tl.make_block_ptr(r_ptr, (256, 256), (256, 1), (0, j0), (BK, BJ), (1, 0))
    acc = tl.zeros((BM, BJ), dtype=tl.float32)
    for _ in range(0, 256, BK):
        acc = tl.dot(tl.load(a_blk, boundary_check=(0,)), tl.load(r_blk), acc)
        a_blk = tl.advance(a_blk, (0, BK))
        r_blk = tl.advance(r_blk, (BK, 0))
    c_blk = tl.make_block_ptr(c_ptr, (M, 256), (256, 1), (m0, j0), (BM, BJ), (1, 0))
    tl.store(c_blk, acc.to(tl.bfloat16), boundary_check=(0,))


@triton.jit
def _first_version(w8_ptr, scale_ptr, r_ptr, out_ptr, M, gpr, BM: tl.constexpr, BJ: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    mask = rows < M
    cols = tl.arange(0, 256)
    base = rows.to(tl.int64)[:, None] * 256
    a = tl.load(w8_ptr + base + cols[None, :], mask=mask[:, None], other=0)
    s = tl.load(scale_ptr + rows // gpr, mask=mask, other=0.0)
    a = (a.to(tl.float32) * s[:, None]).to(tl.bfloat16)
    for j in tl.static_range(0, 256, BJ):
        oc = j + tl.arange(0, BJ)
        r = tl.load(r_ptr + cols[:, None] * 256 + oc[None, :])
        tl.store(out_ptr + base + oc[None, :], tl.dot(a, r).to(tl.bfloat16), mask=mask[:, None])


def tflops(ms):
    return 2 * M * 256 * 256 / (ms / 1000) / 1e12


gen = torch.Generator(device=dev).manual_seed(0)
h = torch.ones(1, 1, dtype=torch.float32)
while h.shape[0] < 256:
    h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
rot = (h / 16).to(dev, torch.bfloat16).contiguous()
a16 = torch.randn(M, 256, generator=gen, device=dev).to(torch.bfloat16)
a8 = torch.randint(-127, 128, (M, 256), generator=gen, device=dev, dtype=torch.int8)
scale = (torch.rand(N, generator=gen, device=dev) * 0.02 + 0.001)
out = torch.empty(M, 256, dtype=torch.bfloat16, device=dev)
print(f"N={N} K={K}: the product is [{M}, 256] x [256, 256], {2 * M * 256 * 256 / 1e9:.2f} GFLOP; floor of the whole dequantizer "
      f"{N * K * 3 / 397e9 * 1000:.3f} ms at 397 GB/s\n")


def report(name, fn):
    try:
        ms = bench(fn, n=10)
        print(f"{name:62s} {ms:9.3f} ms {tflops(ms):7.1f} TFLOPS")
    except Exception as exc:  # noqa: BLE001 - a rung that does not build is a result
        lines = [l for l in str(exc).splitlines() if "error:" in l.lower()] or str(exc).splitlines()
        print(f"{name:62s}   failed: {lines[-1].strip()[:90]}")


report("torch.mm (oneDNN), bf16 A", lambda: torch.mm(a16, rot))
report("torch chain: int8 -> *scale -> bf16, then torch.mm", lambda: torch.mm((a8.float() * scale.repeat_interleave(GPR)[:, None]).bfloat16(), rot))
for bm, bj, bk, warps in ((64, 64, 32, 8), (128, 64, 32, 8), (64, 64, 32, 4)):
    grid = (triton.cdiv(M, bm), 256 // bj)
    tag = f"BM {bm} BJ {bj} BK {bk} warps {warps}"
    report(f"L0 plain pointers, bf16 A ({tag})", lambda: _product[grid](a16, rot, out, scale, M, GPR, KIND=0, BM=bm, BJ=bj, BK=bk, num_warps=warps))
bm, bj, bk, warps = 64, 64, 32, 8
grid = (triton.cdiv(M, bm), 256 // bj)
report("L1 block pointers, bf16 A (BM 64 BJ 64 BK 32 warps 8)", lambda: _product_block[grid](a16, rot, out, M, BM=bm, BJ=bj, BK=bk, num_warps=warps))
report("L2 int8 A widened in registers", lambda: _product[grid](a8, rot, out, scale, M, GPR, KIND=1, BM=bm, BJ=bj, BK=bk, num_warps=warps))
report("L3 + row scale and bf16 rounding (the dequantizer's arithmetic)", lambda: _product[grid](a8, rot, out, scale, M, GPR, KIND=2, BM=bm, BJ=bj, BK=bk, num_warps=warps))
g4 = (triton.cdiv(M, 32),)
report("L4 first version: whole 256-wide operand in one product (BM 32 BJ 64)", lambda: _first_version[g4](a8, scale, rot, out, M, GPR, BM=32, BJ=64, num_warps=8))
