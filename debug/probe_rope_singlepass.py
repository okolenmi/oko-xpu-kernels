"""A single-pass replacement for comfy-kitchen's Triton `rms_rope_split_half`.

The shipped kernel (comfy_kitchen/backends/triton/rms_rope.py) has a grid of
(batch, head, seq) and, per program:

  * loads the whole head row (head_dim values) for the sum of squares,
  * loads the two halves *again* to rotate them,
  * loads the four `pe` planes again -- `pe` is fp32, so that is 1 KB per
    program, re-read once per head for a table that is only a couple of MB.

This probe keeps the arithmetic *identical* (same `tl.sum` over the same
128-element block, same bf16 rounding of the norm output, same fp32 rotation
expression) but changes only the schedule: one program per (batch, seq), the
head loop inside, `pe` loaded once and reused for all heads, and the row read
once.  The acceptance test is a bit-for-bit match against the shipped kernel.

    python debug/probe_rope_singlepass.py [seq] [heads] [dim]
"""
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")
from comfy_kitchen.backends.triton.rms_rope import (  # noqa: E402
    rms_rope_split_half as triton_rope,
)


@triton.jit
def rms_rope_rows_kernel(
    x_ptr, freqs_ptr, scale_ptr, out_ptr,
    head_dim, n_heads, freqs_batch, freqs_seq,
    stride_x_b, stride_x_s, stride_x_h, stride_x_d,
    stride_o_b, stride_o_s, stride_o_h, stride_o_d,
    stride_f_b, stride_f_s, stride_f_dim, stride_f_rot, stride_f_pair,
    epsilon,
    block_dim: tl.constexpr, n_pairs: tl.constexpr, split_half: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    offs = tl.arange(0, n_pairs)
    full = tl.arange(0, block_dim)
    full_mask = full < head_dim

    freqs_b = tl.where(freqs_batch == 1, 0, batch_idx)
    freqs_s = tl.where(freqs_seq == 1, 0, seq_idx)
    fbase = freqs_ptr + freqs_b * stride_f_b + freqs_s * stride_f_s
    freqs_00 = tl.load(fbase + offs * stride_f_dim)
    freqs_01 = tl.load(fbase + offs * stride_f_dim + stride_f_pair)
    freqs_10 = tl.load(fbase + offs * stride_f_dim + stride_f_rot)
    freqs_11 = tl.load(fbase + offs * stride_f_dim + stride_f_rot + stride_f_pair)

    if split_half:
        dim_idx_0 = offs
        dim_idx_1 = offs + n_pairs
    else:
        dim_idx_0 = offs * 2
        dim_idx_1 = offs * 2 + 1

    scale_0 = tl.load(scale_ptr + dim_idx_0).to(tl.float32)
    scale_1 = tl.load(scale_ptr + dim_idx_1).to(tl.float32)

    x_row = x_ptr + batch_idx * stride_x_b + seq_idx * stride_x_s
    o_row = out_ptr + batch_idx * stride_o_b + seq_idx * stride_o_s

    for h in range(n_heads):
        x_base = x_row + h * stride_x_h
        x_full = tl.load(x_base + full * stride_x_d, mask=full_mask,
                         other=0.0).to(tl.float32)
        inv_rms = tl.math.rsqrt(tl.sum(x_full * x_full, axis=0) / head_dim + epsilon)

        # the two halves come out of the register block, not a second load
        if split_half:
            pair = tl.trans(tl.reshape(x_full, (2, n_pairs)))
        else:
            pair = tl.reshape(x_full, (n_pairs, 2))
        x_0, x_1 = tl.split(pair)

        x_0 = (x_0 * inv_rms * scale_0).to(tl.bfloat16).to(tl.float32)
        x_1 = (x_1 * inv_rms * scale_1).to(tl.bfloat16).to(tl.float32)

        out_0 = freqs_00 * x_0 + freqs_01 * x_1
        out_1 = freqs_10 * x_0 + freqs_11 * x_1

        o_base = o_row + h * stride_o_h
        tl.store(o_base + dim_idx_0 * stride_o_d, out_0)
        tl.store(o_base + dim_idx_1 * stride_o_d, out_1)


def rope_rows(x, freqs_cis, scale, epsilon=1e-6, split_half=True, n_warps=None):
    """Same signature/semantics as _rms_rope for the [B,S,H,D] (bnhd) layout."""
    assert x.ndim == 4 and freqs_cis.ndim == 6
    B, S, H, D = x.shape
    if not scale.is_contiguous():
        scale = scale.contiguous()
    out = torch.empty_like(x)
    kwargs = {}
    if n_warps is not None:
        kwargs["num_warps"] = n_warps
    rms_rope_rows_kernel[(S, B)](
        x, freqs_cis, scale, out,
        D, H, freqs_cis.shape[0], freqs_cis.shape[1],
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        freqs_cis.stride(0), freqs_cis.stride(1), freqs_cis.stride(3),
        freqs_cis.stride(4), freqs_cis.stride(5),
        epsilon,
        block_dim=triton.next_power_of_2(D), n_pairs=D // 2,
        split_half=split_half, **kwargs)
    return out


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
S = int(sys.argv[1]) if len(sys.argv) > 1 else 3973
H = int(sys.argv[2]) if len(sys.argv) > 2 else 24
D = int(sys.argv[3]) if len(sys.argv) > 3 else 128
B, EPS, N = 1, 1e-6, D // 2

torch.manual_seed(0)
q = (torch.randn(B, S, H, D, device=dev) * 0.3).to(torch.bfloat16)
k = (torch.randn(B, S, H, D, device=dev) * 0.3).to(torch.bfloat16)
scale = torch.rand(D, device=dev, dtype=torch.float32) * 0.2 + 0.9
ang = torch.rand(B, S, 1, N, device=dev, dtype=torch.float32) * 3.0
pe = torch.stack([torch.stack([ang.cos(), -ang.sin()], -1),
                  torch.stack([ang.sin(), ang.cos()], -1)], -2)
print(f"shape B={B} S={S} H={H} D={D}  pe{tuple(pe.shape)} {pe.dtype}")

print("\n--- bit-exactness (must be 0) ---")
for split_half in (True, False):
    ref = triton_rope(q, k, pe, scale, None, EPS, ) if split_half else None
    if split_half:
        ref_q, ref_k = ref
    else:
        from comfy_kitchen.backends.triton.rms_rope import (
            rms_rope1 as t_rope1, rms_rope1_ as _unused)
        ref_q, ref_k = t_rope1(q, pe, scale, EPS), t_rope1(k, pe, scale, EPS)
    for tag, X, R in (("q", q, ref_q), ("k", k, ref_k)):
        out = rope_rows(X, pe, scale, EPS, split_half=split_half)
        bad = (out.view(torch.int16) != R.view(torch.int16)).sum().item()
        d = (out.float() - R.float()).abs().max().item()
        print(f"  split_half={int(split_half)} {tag}: mismatches {bad}/{R.numel()} "
              f"maxabs {d:.3e}")

print("\n--- timing, q+k together (ms) ---")
mb_ref = (q.numel() * 2 * 2 + q.numel() * 2) / 1e6      # read twice + write
mb_new = (q.numel() * 2 + q.numel() * 2) / 1e6          # read once + write
t_ref = bench(lambda: triton_rope(q, k, pe, scale, None, EPS))
print(f"  shipped triton kernel   {t_ref:7.4f} ms   {mb_ref / t_ref * 1e3:7.0f} GB/s  "
      f"({mb_ref:.1f} MB)")
for nw in (1, 2, 4, 8):
    t_new = bench(lambda: (rope_rows(q, pe, scale, EPS, n_warps=nw),
                           rope_rows(k, pe, scale, EPS, n_warps=nw)))
    print(f"  single-pass n_warps={nw}   {t_new:7.4f} ms   {mb_new / t_new * 1e3:7.0f} GB/s  "
          f"({mb_new:.1f} MB)   speedup {t_ref / t_new:.2f}x")
