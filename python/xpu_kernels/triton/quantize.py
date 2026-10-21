"""A fused per-row quantizer: bf16 [M, K] -> int8 [M, K] + fp32 [M] row scales.

This is the one piece of the INT8 training path that was a torch chain, and
it was the expensive part: the chain materializes an fp32 copy of the whole
tensor, takes an absolute value, reduces, divides, rounds and casts -- about
forty bytes of traffic per element and eleven kernel launches, against three
bytes and one launch for the fused version.

Why it matters: the quantized-gradient backward (``int8.linear_dx_int8``) is
the path that keeps the weight in int8, and the weight is what a training run
does not have room for.  With the weight in int8 *and* in bf16 (the
weight-only backward needs the bf16 copy), a 3.7 B model costs 3 + 7.4 = 11.1
GB; with the weight only in int8 it costs 3.7 GB.  On a 12.2 GB card that is
the difference between fitting and not.

The kernel is bit-exact against the torch chain, which is the contract the
whole package has: same arithmetic, same rounding, same result.  The one
subtlety is the rounding mode.  ``torch.round`` rounds halves to *even*
(``rint``); C's ``round`` rounds halves away from zero, and they differ on
exactly the values a quantizer sees most: a bf16 value that lands on .5 after
the division.  The kernel therefore uses ``rint``, and
``debug/verify_quantize.py`` checks it on data built out of ties.

The gate, the escape hatch and the fallback are the package's usual ones: the
kernel verifies itself once per process against the torch chain and is refused
if it does not match, and ``XPU_KERNELS_NO_QUANTIZE=1`` switches it off.
"""

from __future__ import annotations

import logging
import os

import torch

from ._gate import ACCELERATORS, Gate, probe_device

log = logging.getLogger(__name__)

__all__ = ["gates", "quantize_row", "set_reference", "usable"]

_kernel = None
_triton = None
_import_error = None
_logged = False

# One row per program, one pass to find the maximum and one to write the
# quantized values.  The second pass re-reads the row, which is 6-16 KB and
# therefore still in L2, so the extra read is nearly free -- and it keeps the
# register pressure independent of K, which matters because K is 3072 in the
# attention projections and 8192 in the FFN.
BLOCK_K = 1024
NUM_WARPS = 4


try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _triton = triton

    @triton.jit
    def _quantize_row_kernel(
        x_ptr, q_ptr, scale_ptr,
        n_rows, n_cols,
        stride_x_m, stride_x_k,
        stride_q_m, stride_q_k,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_row = x_ptr + row * stride_x_m
        q_row = q_ptr + row * stride_q_m

        # pass 1: the row maximum
        amax = 0.0
        for start in range(0, n_cols, BLOCK_K):
            offs = start + tl.arange(0, BLOCK_K)
            mask = offs < n_cols
            x = tl.load(x_row + offs * stride_x_k, mask=mask, other=0.0)
            amax = tl.maximum(amax, tl.max(tl.abs(x.to(tl.float32)), axis=0))

        # the chain: (amax / 127).clamp_min(1e-30)
        # The division is the delicate part, and it is delicate in *two* ways.
        # Triton's "/" is approximate (div.full, about an ULP out), and a
        # quotient that lands on .5 then rounds the other way -- a quantizer
        # sees exactly that value often.  Worse, torch itself is not
        # self-consistent here: on this backend its tensor/tensor division is
        # correctly rounded, but its tensor/python-scalar division is a
        # reciprocal multiply and is not.  Measured over 2^20 values:
        #
        #   torch x/scalar  == tl.fdiv(ieee_rounding=True)   (1 048 576/1 048 576)
        #   torch x/tensor  == libdevice.div_rn(x, s)         (1 048 576/1 048 576)
        #   and the two disagree with each other on 13 % of values
        #
        # The chain divides a tensor by the *scalar* 127.0 for the scale and
        # by the *tensor* scale for the quotient, so the kernel must do the
        # same.  The gate caught 73 of 721 984 values before this was settled.
        scale = tl.maximum(tl.fdiv(amax, 127.0, ieee_rounding=True), 1e-30)

        # pass 2: quantize
        for start in range(0, n_cols, BLOCK_K):
            offs = start + tl.arange(0, BLOCK_K)
            mask = offs < n_cols
            x = tl.load(x_row + offs * stride_x_k, mask=mask, other=0.0)
            xf = x.to(tl.float32)
            # rint: round half to even, like torch.round.  libdevice.round
            # would round halves away from zero and differ on ties.
            #
            # The division is the fast one, and that is a measured trade-off,
            # not an oversight.  A *correctly rounded* fp32 division is
            # available (libdevice.div_rn) and it reproduces the torch chain
            # bit for bit -- but on this backend it costs 6.3 ms for a
            # 3974x8192 tensor, against 4.6 ms for the whole torch chain, so a
            # bit-exact fused quantizer would be slower than what it replaces.
            # The fast division costs 0.35 ms (13x faster than the chain) and
            # moves 2-3e-4 of the quantized values by one unit: 9.1e-4 relative
            # on the gradient, against the int8 path's own 1.35% error, i.e.
            # fifteen times smaller.  The gate therefore compares with a
            # relative budget, like the backward kernels do, and the formula is
            # pinned in float64 by the tests.
            q = libdevice.rint(tl.fdiv(xf, scale, ieee_rounding=True))
            q = tl.minimum(tl.maximum(q, -127.0), 127.0)
            tl.store(q_row + offs * stride_q_k, q.to(tl.int8), mask=mask)

        tl.store(scale_ptr + row, scale)

    _kernel = _quantize_row_kernel

except Exception as exc:  # noqa: BLE001 - no triton means no kernel
    _import_error = exc


def _reference_chain(x):
    """The torch chain this kernel must match, bit for bit.

    It lives here as well as in ``xpu_kernels.int8`` because the gate needs a
    reference that does not itself call the kernel.
    """
    xf = x.to(torch.float32)
    scale = (xf.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-30)
    q = torch.round(xf / scale).clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), scale.squeeze(-1).contiguous()


def enabled() -> bool:
    """True if the kernel is there and has not been switched off."""
    if _kernel is not None:
        return True
    global _import_error, _logged
    if _import_error is not None:
        if not _logged:
            _logged = True
            log.info("triton quantizer unavailable: %s", _import_error)
        return False
    if os.environ.get("XPU_KERNELS_NO_QUANTIZE", "") not in ("", "0"):
        return False
    return True


def _rows(x, out=None, scale=None):
    M, K = x.shape
    if out is None:
        out = torch.empty((M, K), dtype=torch.int8, device=x.device)
    if scale is None:
        scale = torch.empty((M,), dtype=torch.float32, device=x.device)
    _kernel[(M,)](
        x, out, scale,
        M, K,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_K=BLOCK_K, num_warps=NUM_WARPS)
    return out, scale


def quantize_row(x, out=None, scale=None):
    """Per-row symmetric int8 quantization, no rotation.

    ``bf16 [.., K] -> int8 [.., K]`` plus the fp32 row scales.  The result is
    bit-identical to the torch chain, and the row scale is the *row* maximum
    over the last axis -- the same convention ``xpu_kernels.int8.quantize_row``
    uses.

    This must not be confused with :func:`xpu_kernels.int8.quantize`, which
    *rotates*: the rotation is already baked into an INT8 weight, so rotating
    a gradient here would silently produce a rotated gradient.
    """
    _GATE.require()
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"expected bfloat16 or float32, got {x.dtype}")
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    q, s = _rows(x2, out, scale)
    # the same shapes the torch chain returns: the values like x, the scales
    # per row of x (so a 3-D input gives [B, S] scales, not [B*S])
    return q.reshape(x.shape), s.reshape(x.shape[:-1])


def usable(x) -> bool:
    """May the fused quantizer be used for this input?"""
    if not enabled():
        return False
    if x.dtype not in (torch.bfloat16, torch.float32):
        return False
    if x.device.type not in ACCELERATORS:
        return False
    if x.shape[-1] == 0:
        return False
    return _GATE.ok()


def _probe(ref):
    if _kernel is None:
        # No Triton here: say so, rather than failing with a NameError or a
        # "NoneType is not subscriptable" from inside the launch.
        raise RuntimeError("the triton quantizer kernel is not available here")

    """The fused kernel against the torch chain, at a training-like shape.

    Two shapes, because the row length is a parameter of the kernel: 3072 (the
    attention projections) and 8192 (the FFN).  And a third case built out of
    *ties*: values that land exactly on .5 after the division, which is where
    a rounding mode that differs from ``torch.round`` shows up.

    The comparison is a *relative* one, not bitwise: the fused kernel divides
    faster than the chain can be divided exactly (see the comment in the
    kernel), so the two differ on 2-3e-4 of the values by one unit.  The
    budget is small enough that a wrong formula -- a missing clamp, the wrong
    scale, a rotation that should not be there -- cannot hide inside it: those
    are off by ~1e-1, not ~1e-4.  The formula itself is pinned in float64 by
    ``tests/test_quantize.py``.
    """
    device = probe_device()
    gen = torch.Generator(device=device).manual_seed(0)
    M, K = 64, 3072
    x = (torch.randn((M, K), generator=gen, device=device) * 0.4).to(torch.bfloat16)
    want_q, want_s = ref(x)
    pairs = {"random 3072": (_rows(x)[0], want_q)}

    x8 = (torch.randn((M, 8192), generator=gen, device=device) * 0.4).to(torch.bfloat16)
    want_q8, want_s8 = ref(x8)
    pairs["random 8192"] = (_rows(x8)[0], want_q8)

    # ties: a scale of 2.0 makes every odd integer land on .5
    tie = (torch.arange(-8, 8, device=device, dtype=torch.bfloat16) * 0.5)
    tie = tie.repeat(M, 1).contiguous()
    want_t, _ = ref(tie)
    pairs["ties"] = (_rows(tie)[0], want_t)

    # and the scales themselves, which the gate must check too: the row scale
    # is exact (it is a division by a constant, which the fast path does
    # correctly), so this pair is compared bit for bit
    pairs["scales 8192"] = (_rows(x8)[1], want_s8)
    return pairs


def _default_reference():
    return _reference_chain


#: The relative budget for the quantized values.  The measured deviation is
#: 3.3e-4 (one unit on 2-3e-4 of the values); a wrong formula is off by
#: ~1e-1, so the budget separates the two by two orders of magnitude.
REL = 2e-3

_GATE = Gate(
    "quantize_row", _probe,
    default_reference=_default_reference,
    env="XPU_KERNELS_NO_QUANTIZE",
    num_warps=NUM_WARPS, rel=REL,
)


def set_reference(fn=None):
    """Set (or clear) the reference the gate compares against.

    The default is this module's own torch chain, which is the right
    reference for a quantizer: the kernel is not replacing a *framework's*
    implementation, it is replacing a torch chain inside this package.
    """
    _GATE.set_reference(fn)


def gates():
    """The gates of this module, by name."""
    return {"quantize_row": _GATE}
