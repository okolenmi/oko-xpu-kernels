"""``python -m xpu_kernels``'s INT8 half: does the library work on this machine?

Not a bit-exactness suite -- the kernel-level suites live in ``debug/``.  This
checks that the pieces fit together on this card: weight prep, the quantize
round trip, the int32 accumulator path, the strided slice, and the fused
dequant+activation against the unfused chain it replaces.
"""

from __future__ import annotations

import torch

from . import _lib as L
from ._lib import available, device_name
from .ops import (
    dequant,
    dequant_act_quantize,
    dequant_slice,
    dequant_weight,
    linear_dx_int8,
    linear,
    linear_acc,
    linear_dx,
    prepare_weight,
    quantize_act,
    quantize_row,
    quantize_row_torch,
)

def selftest(verbose: bool = True) -> bool:
    """Quick end-to-end check on this machine.  Returns True when it passes.

    Not a bit-exactness suite -- it checks that the pieces fit together (weight
    prep, quantization round trip, GEMM, dequant, the fused activation path)
    and that the INT8 path is close to bf16, as an INT8 quantization must be.
    The bit-exactness suite for the fused kernels lives in the project's
    ``debug/`` directory (``verify_situ.py``, ``verify_qkv_rope_fuse.py``).
    """
    if not available():
        if verbose:
            print(f"libxpuint8 unavailable: {L._ERR}")
        return False
    dev = torch.device("xpu:0")
    torch.xpu.set_device(0)
    g = torch.Generator(device=dev).manual_seed(0)
    M, K, N = 512, 1024, 512
    x = (torch.randn(M, K, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    w = (torch.randn(N, K, generator=g, device=dev) * 0.05).to(torch.bfloat16)
    bias = torch.randn(N, generator=g, device=dev) * 0.1

    w8, ws = prepare_weight(w)
    y = linear(x, w8, ws, bias)
    ref = (x.float() @ w.to(torch.float32).t() + bias).to(torch.bfloat16)
    rel = ((y.float() - ref.float()).norm() / ref.float().norm()).item()

    # the accumulator path must agree with linear() exactly
    acc, xs = linear_acc(x, w8, ws)
    same = dequant(acc, xs, ws, bias).view(torch.int16).equal(y.view(torch.int16))

    # a column slice of a wider accumulator, read in place with its row stride
    acc_w, xs_w = linear_acc(x, torch.cat([w8, w8], 0), torch.cat([ws, ws]))
    half = acc_w[:, :N]
    sl = dequant_slice(half, xs_w, ws, bias)
    sl_ref = dequant(half.contiguous(), xs_w, ws, bias)
    slice_same = sl.view(torch.int16).equal(sl_ref.view(torch.int16))

    # the fused situ epilogue against the documented unfused equivalent:
    # dequant -> activation + rotate + quantize (that chain's numbers, not the
    # same bit pattern, because the fused one keeps the accumulator in fp32)
    kff = 1024
    wup = (torch.randn(2 * kff, K, generator=g, device=dev) * 0.05).to(torch.bfloat16)
    wup8, wup_s = prepare_weight(wup)
    xup = (torch.randn(M, K, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    acc2, xs2 = linear_acc(xup, wup8, wup_s)
    q2, s2 = dequant_act_quantize(acc2, xs2, wup_s)
    d = dequant(acc2, xs2, wup_s)
    q2_ref, s2_ref = quantize_act(d, "situ")
    code_diff = (q2.to(torch.int16) - q2_ref.to(torch.int16)).abs().max().item()
    scale_rel = ((s2 - s2_ref).abs() / s2_ref.clamp_min(1e-20)).max().item()
    # the backward's activation gradient: the convention must be exact (same
    # GEMM as autograd's, so anything else means a transpose is missing), and
    # the INT8-weight gradient must be as close as the weight itself is
    dy = (torch.randn(M, N, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    xr = x.detach().clone().requires_grad_(True)
    torch.nn.functional.linear(xr, w).backward(dy)
    grad = xr.grad
    dx_bf16 = linear_dx(dy, w)
    dx_rel = ((dx_bf16.float() - grad.float()).norm() / grad.float().norm()).item()
    w_deq = dequant_weight(w8, ws)
    w_rel = ((w_deq.float() - w.float()).norm() / w.float().norm()).item()
    dx_int8 = linear_dx(dy, w_deq)
    dx_int8_rel = ((dx_int8.float() - grad.float()).norm() / grad.float().norm()).item()
    # the quantized-gradient variant: the weight stays int8, `dy` is quantized
    dx_q = linear_dx_int8(dy, w8, ws)
    dx_q_rel = ((dx_q.float() - grad.float()).norm() / grad.float().norm()).item()

    # the fused row-quantizer: the same numbers as the chain to within a unit,
    # and the same scale exactly.  It is the difference between the int8
    # backward costing 3.4x the bf16 one and costing about the same, which is
    # what makes the "keep the weight in int8" option affordable.
    q_fused, s_fused = quantize_row(dy)
    q_chain, s_chain = quantize_row_torch(dy)
    quant_diff = int((q_fused.view(torch.int16) != q_chain.view(torch.int16)).sum())
    quant_rel = float((q_fused.float() - q_chain.float()).norm()
                       / q_chain.float().norm())
    scale_same = bool(s_fused.view(torch.int32).equal(s_chain.view(torch.int32)))

    ok = (same and slice_same and rel < 0.02 and q2.dtype == torch.int8
          and code_diff <= 1 and scale_rel < 0.05
          and dx_rel < 1e-6 and w_rel < 0.02 and dx_int8_rel < 0.03
          and dx_q_rel < 0.04
          and quant_rel < 2e-3 and scale_same)
    if verbose:
        print(f"device            : {device_name()}")
        print(f"INT8 vs bf16      : relative error {rel:.4f} (expect < 0.02)")
        print(f"acc path == linear: {same}")
        print(f"strided slice     : {slice_same}")
        print(f"fused situ vs chain: max code diff {code_diff}, "
              f"max scale rel {scale_rel:.4f}")
        print(f"backward dx vs autograd: rel {dx_rel:.2e} (expect 0)")
        print(f"dequant_weight    : rel {w_rel:.4f} (the INT8 weight itself)")
        print(f"backward dx, INT8 : rel {dx_int8_rel:.4f}")
        print(f"backward dx, int8 GEMM (weight stays quantized): rel {dx_q_rel:.4f}")
        print(f"fused quantizer vs chain: {quant_diff} of {q_fused.numel()} values "
              f"differ, rel {quant_rel:.2e}, scales exact: {scale_same}")
        print("PASS" if ok else "FAIL")
    return ok
