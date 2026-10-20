"""Which part of the single-pass rewrite breaks bit-exactness, and what is it worth?

`probe_rope_singlepass.py` showed the head-loop rewrite matching the shipped
kernel on 0/12.2M values -- but it fed it *contiguous* tensors.  In the model,
q and k are `chunk` views of one qkv projection, so they are strided, and there
`probe_rope_backend.py` shows 36/12.2M mismatches.  Those are a reduction-order
difference in `tl.sum(x*x)`: squares of bf16 values are exact, but summing 128 of
them is only *almost* always exact in fp32, and where it is not, the order that
Triton's layout gives decides the last bit.  1 ULP in inv_rms moves a bf16
rounding, and 36 moved values in q are enough to change the whole image
(PSNR 23 dB), so exactness here means reproducing Triton's own reduction.

This probe separates the two changes:

  V1  same grid as the shipped kernel, x loaded once (halves from the block)
  V2  head-loop restructure (fewer programs, pe once per row), num_warps=1
  V3  head-loop restructure, num_warps=4 (the shipped kernel's default)
  V4  head-loop restructure but x re-loaded for the halves (keeps the reduction
      shape identical, only the schedule changes)

    python debug/probe_rope_variants.py
"""
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")
from comfy_kitchen.backends.triton.rms_rope import (  # noqa: E402
    rms_rope_split_half as shipped,
)


# grid (batch, head, seq): the shipped decomposition, single x load
@triton.jit
def _k_bhs(x_ptr, freqs_ptr, scale_ptr, out_ptr, head_dim, freqs_batch, freqs_seq,
           stride_x_b, stride_x_s, stride_x_h, stride_x_d,
           stride_o_b, stride_o_s, stride_o_h, stride_o_d,
           stride_f_b, stride_f_s, stride_f_dim, stride_f_rot, stride_f_pair,
           epsilon, block_dim: tl.constexpr, n_pairs: tl.constexpr):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    offs = tl.arange(0, n_pairs)
    full = tl.arange(0, block_dim)

    fb = tl.where(freqs_batch == 1, 0, batch_idx)
    fs = tl.where(freqs_seq == 1, 0, seq_idx)
    fbase = freqs_ptr + fb * stride_f_b + fs * stride_f_s
    f00 = tl.load(fbase + offs * stride_f_dim)
    f01 = tl.load(fbase + offs * stride_f_dim + stride_f_pair)
    f10 = tl.load(fbase + offs * stride_f_dim + stride_f_rot)
    f11 = tl.load(fbase + offs * stride_f_dim + stride_f_rot + stride_f_pair)
    s0 = tl.load(scale_ptr + offs).to(tl.float32)
    s1 = tl.load(scale_ptr + offs + n_pairs).to(tl.float32)

    x_base = x_ptr + batch_idx * stride_x_b + seq_idx * stride_x_s + head_idx * stride_x_h
    x_full = tl.load(x_base + full * stride_x_d).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x_full * x_full, axis=0) / head_dim + epsilon)
    x_0, x_1 = tl.split(tl.trans(tl.reshape(x_full, (2, n_pairs))))
    x_0 = (x_0 * inv * s0).to(tl.bfloat16).to(tl.float32)
    x_1 = (x_1 * inv * s1).to(tl.bfloat16).to(tl.float32)

    o_base = out_ptr + batch_idx * stride_o_b + seq_idx * stride_o_s + head_idx * stride_o_h
    tl.store(o_base + offs * stride_o_d, f00 * x_0 + f01 * x_1)
    tl.store(o_base + (offs + n_pairs) * stride_o_d, f10 * x_0 + f11 * x_1)


# grid (seq, batch), head loop inside
@triton.jit
def _k_rows(x_ptr, freqs_ptr, scale_ptr, out_ptr, head_dim, n_heads, freqs_batch,
            freqs_seq, stride_x_b, stride_x_s, stride_x_h, stride_x_d,
            stride_o_b, stride_o_s, stride_o_h, stride_o_d,
            stride_f_b, stride_f_s, stride_f_dim, stride_f_rot, stride_f_pair,
            epsilon, block_dim: tl.constexpr, n_pairs: tl.constexpr,
            REUSE: tl.constexpr):
    seq_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    offs = tl.arange(0, n_pairs)
    full = tl.arange(0, block_dim)

    fb = tl.where(freqs_batch == 1, 0, batch_idx)
    fs = tl.where(freqs_seq == 1, 0, seq_idx)
    fbase = freqs_ptr + fb * stride_f_b + fs * stride_f_s
    f00 = tl.load(fbase + offs * stride_f_dim)
    f01 = tl.load(fbase + offs * stride_f_dim + stride_f_pair)
    f10 = tl.load(fbase + offs * stride_f_dim + stride_f_rot)
    f11 = tl.load(fbase + offs * stride_f_dim + stride_f_rot + stride_f_pair)
    s0 = tl.load(scale_ptr + offs).to(tl.float32)
    s1 = tl.load(scale_ptr + offs + n_pairs).to(tl.float32)

    x_row = x_ptr + batch_idx * stride_x_b + seq_idx * stride_x_s
    o_row = out_ptr + batch_idx * stride_o_b + seq_idx * stride_o_s
    for h in range(n_heads):
        x_base = x_row + h * stride_x_h
        x_full = tl.load(x_base + full * stride_x_d).to(tl.float32)
        inv = tl.math.rsqrt(tl.sum(x_full * x_full, axis=0) / head_dim + epsilon)
        if REUSE:
            x_0, x_1 = tl.split(tl.trans(tl.reshape(x_full, (2, n_pairs))))
        else:
            x_0 = tl.load(x_base + offs * stride_x_d).to(tl.float32)
            x_1 = tl.load(x_base + (offs + n_pairs) * stride_x_d).to(tl.float32)
        x_0 = (x_0 * inv * s0).to(tl.bfloat16).to(tl.float32)
        x_1 = (x_1 * inv * s1).to(tl.bfloat16).to(tl.float32)
        o_base = o_row + h * stride_o_h
        tl.store(o_base + offs * stride_o_d, f00 * x_0 + f01 * x_1)
        tl.store(o_base + (offs + n_pairs) * stride_o_d, f10 * x_0 + f11 * x_1)


def run(x, freqs, scale, eps, variant, n_warps=None):
    B, S, H, D = x.shape
    out = torch.empty_like(x)
    kw = {} if n_warps is None else {"num_warps": n_warps}
    common = (D, freqs.shape[0], freqs.shape[1],
              x.stride(0), x.stride(1), x.stride(2), x.stride(3),
              out.stride(0), out.stride(1), out.stride(2), out.stride(3),
              freqs.stride(0), freqs.stride(1), freqs.stride(3),
              freqs.stride(4), freqs.stride(5), eps)
    if variant == "V1":
        _k_bhs[(S, H, B)](x, freqs, scale, out, *common,
                          block_dim=triton.next_power_of_2(D), n_pairs=D // 2, **kw)
    else:
        head_dim, fb, fs = D, freqs.shape[0], freqs.shape[1]
        _k_rows[(S, B)](x, freqs, scale, out, head_dim, H, fb, fs,
                        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                        freqs.stride(0), freqs.stride(1), freqs.stride(3),
                        freqs.stride(4), freqs.stride(5), eps,
                        block_dim=triton.next_power_of_2(D), n_pairs=D // 2,
                        REUSE=(variant != "V4"), **kw)
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
B, S, H, D = 1, 3973, 24, 128
EPS = 1e-6
torch.manual_seed(0)

qkv = (torch.randn(B, S, 3 * H * D, device=dev) * 0.3).to(torch.bfloat16)
q, k, _v = qkv.chunk(3, dim=-1)
q = q.view(B, S, H, D)
k = k.view(B, S, H, D)
scale = torch.rand(D, device=dev, dtype=torch.float32) * 0.2 + 0.9
ang = torch.rand(B, S, 1, D // 2, device=dev, dtype=torch.float32) * 3.0
pe = torch.stack([torch.stack([ang.cos(), -ang.sin()], -1),
                  torch.stack([ang.sin(), ang.cos()], -1)], -2)

ref_q, ref_k = shipped(q, k, pe, scale, None, EPS)
torch.xpu.synchronize()
print(f"strided q: shape {tuple(q.shape)} strides {q.stride()}")
print(f"shipped q+k: {bench(lambda: shipped(q, k, pe, scale, None, EPS)):.4f} ms\n")

print(f"{'variant':<34}{'mismatches q':>14}{'mismatches k':>14}{'ms q+k':>10}")
for variant, nw in (("V1", None), ("V2", 1), ("V3", 4), ("V3", 8),
                    ("V4", 4), ("V4", 1)):
    out_q = run(q, pe, scale, EPS, variant, nw)
    bad_q = (out_q.view(torch.int16) != ref_q.view(torch.int16)).sum().item()
    out_k = run(k, pe, scale, EPS, variant, nw)
    bad_k = (out_k.view(torch.int16) != ref_k.view(torch.int16)).sum().item()
    ms = bench(lambda: (run(q, pe, scale, EPS, variant, nw),
                        run(k, pe, scale, EPS, variant, nw)))
    tag = f"{variant} (num_warps={'default' if nw is None else nw})"
    print(f"{tag:<34}{bad_q:>14}{bad_k:>14}{ms:>10.4f}")
