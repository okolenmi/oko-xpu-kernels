"""A single-pass replacement for comfy-kitchen's Triton `rms_adaln`.

`backends/triton/adaln.py` walks the row twice: pass 1 accumulates the sum of
squares, pass 2 reloads `x` to normalize and modulate.  For the MossImage shape
(d = 3072, `block_d = min(next_power_of_2(d), 4096) = 4096`) that loop is a single
iteration, so `x` is read twice and written once -- 72 MB per call instead of 48.
(RMSNorm pins the mean to 0, so the first pass needs *only* the sum of squares,
which makes this a pure dependency: nothing in pass 1 needs the reload.)

This probe keeps the arithmetic identical -- same `tl.sum` over the same
`block_d`-wide block, same fp32 expression `((x - mean) * rstd) * (1 + sc) + sh`
with `mean = 0.0` -- and keeps the row in registers, so it must be bit-exact.
`subtract_mean=True` (true LayerNorm) is not covered: it needs both passes'
statistics and is not on this path.

    python debug/probe_adaln_singlepass.py [rows] [dim]
"""
import sys
import time

import torch
import triton
import triton.language as tl

from comfy_kitchen.backends._modulation import adaln_prep_modulation
from comfy_kitchen.backends.triton.adaln import rms_adaln as shipped_rms_adaln


@triton.jit
def rms_adaln_rows_kernel(
    x_ptr, scale_ptr, shift_ptr, y_ptr,
    d, scale_group, shift_group,
    stride_xr, stride_sr, stride_shr, stride_yr,
    eps,
    block_d: tl.constexpr, dtype: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_d)
    mask = cols < d

    # one load of the row: the sum of squares and the normalize pass share it
    x = tl.load(x_ptr + row * stride_xr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x) / d + eps)

    sc = tl.load(scale_ptr + (row // scale_group) * stride_sr + cols, mask=mask,
                 other=0.0).to(tl.float32)
    sh = tl.load(shift_ptr + (row // shift_group) * stride_shr + cols, mask=mask,
                 other=0.0).to(tl.float32)

    out = (x - 0.0) * rstd * (1.0 + sc) + sh
    tl.store(y_ptr + row * stride_yr + cols, out.to(dtype), mask=mask)


def rms_adaln_rows(x, scale, shift, eps=1e-6, n_warps=None):
    orig_shape = x.shape
    d = x.shape[-1]
    n = x.numel() // d
    assert d <= 4096, "single-pass path only covers the single-iteration shapes"
    x_flat = x.reshape(n, d)
    if not x_flat.is_contiguous():
        x_flat = x_flat.contiguous()
    scale_flat, scale_group = adaln_prep_modulation(scale, x, n, d)
    shift_flat, shift_group = adaln_prep_modulation(shift, x, n, d)
    out = torch.empty_like(x_flat)
    kwargs = {} if n_warps is None else {"num_warps": n_warps}
    rms_adaln_rows_kernel[(n,)](
        x_flat, scale_flat, shift_flat, out,
        d, scale_group, shift_group,
        x_flat.stride(0), scale_flat.stride(0), shift_flat.stride(0), out.stride(0),
        eps,
        block_d=min(triton.next_power_of_2(d), 4096),
        dtype=tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float32,
        **kwargs)
    return out.reshape(orig_shape)


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


dev = torch.device("xpu", 0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3973
D = int(sys.argv[2]) if len(sys.argv) > 2 else 3072
EPS = 1e-6

torch.manual_seed(0)
x = (torch.randn(N, D, device=dev) * 0.5).to(torch.bfloat16)
scale = torch.randn(1, D, device=dev, dtype=torch.float32) * 0.1
shift = torch.randn(1, D, device=dev, dtype=torch.float32) * 0.1

ref = shipped_rms_adaln(x, scale, shift, EPS)
torch.xpu.synchronize()
print(f"shape N={N} D={D}  bf16 in/out, fp32 modulation broadcast over rows")

print("\n--- bit-exactness (must be 0) ---")
for nw in (1, 2, 4, 8):
    out = rms_adaln_rows(x, scale, shift, EPS, n_warps=nw)
    bad = (out.view(torch.int16) != ref.view(torch.int16)).sum().item()
    d = (out.float() - ref.float()).abs().max().item()
    print(f"  n_warps={nw}: mismatches {bad}/{ref.numel()}  maxabs {d:.3e}")

mb_ref = (x.numel() * 2 * 2 + x.numel() * 2) / 1e6   # read twice + write
mb_new = (x.numel() * 2 + x.numel() * 2) / 1e6        # read once + write
print("\n--- timing ---")
t_ref = bench(lambda: shipped_rms_adaln(x, scale, shift, EPS))
print(f"  shipped two-pass          {t_ref:7.4f} ms   ({mb_ref:.1f} MB moved)")
for nw in (1, 2, 4, 8):
    t_new = bench(lambda: rms_adaln_rows(x, scale, shift, EPS, n_warps=nw))
    print(f"  single-pass n_warps={nw}     {t_new:7.4f} ms   ({mb_new:.1f} MB moved)"
          f"   speedup {t_ref / t_new:.2f}x")
