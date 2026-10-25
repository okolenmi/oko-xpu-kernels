"""A fused GEGLU for the bf16 path: ``h [.., 2I] -> h[.., :I] * gelu(h[.., I:])``, forward and backward.

SDXL's feed-forward is ``Linear -> chunk -> x * gelu(gate) -> Linear``.  Eager torch runs the middle as a ``gelu`` and a
``mul`` forward (five tensors' worth of traffic per element pair), and the backward as a ``mul`` twice, a ``gelu_backward``
and a ``cat`` that glues the two gradients back into the layout of ``h`` -- about twelve.  The kernels here read ``h`` once
and write the result once in each direction, and the backward writes both halves of ``dh`` itself, so the ``cat`` does not
exist.  It also removes a saved activation: the chain keeps ``gelu(gate)`` for the ``mul``'s backward, the fused op keeps
only ``h``, which the preceding linear's backward needs anyway.

The arithmetic is the chain's, step for step, including the roundings it does between kernels: ``gelu`` is rounded to the
activation dtype before the multiply, and the backward rounds ``dy * x`` before ``gelu_backward`` sees it.  Skipping those
roundings would be faster to write and would be a different function.

Two gates, because the two halves have different contracts.  The forward is compared bit for bit: it is the same formula
over the same operands.  The backward is compared with a small relative budget, like the other backward kernels, because
``exp`` is the one function whose last bit the backend does not promise to agree on.  If the backward gate is closed the
forward still runs and the backward is autograd of the eager chain, recomputed from ``h``.

``XPU_KERNELS_NO_GEGLU=1`` switches both off.
"""

from __future__ import annotations

import logging
import os

import torch

from ._gate import ACCELERATORS, Gate, probe_device

log = logging.getLogger(__name__)

__all__ = ["gates", "geglu", "set_reference", "usable"]

_triton = None
_import_error = None
_forward_kernel = None
_backward_kernel = None

# One program per (row, block of columns).  Nothing here is a reduction, so the warp count is not part of the numerics;
# 1024 columns per program keeps every load a full 16-byte vector per lane.
BLOCK = 1024
NUM_WARPS = 4
DTYPES = (torch.bfloat16, torch.float16, torch.float32)

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _triton = triton

    @triton.jit
    def _gelu(x):
        # torch's order of operations: (x * 0.5) * (1 + erf(x * sqrt(1/2)))
        return x * 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))

    @triton.jit
    def _geglu_forward_kernel(h_ptr, out_ptr, inner, blocks, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        row = (pid // blocks).to(tl.int64)
        col = (pid % blocks) * BLOCK + tl.arange(0, BLOCK)
        mask = col < inner
        x = tl.load(h_ptr + row * (2 * inner) + col, mask=mask, other=0.0)
        gate = tl.load(h_ptr + row * (2 * inner) + inner + col, mask=mask, other=0.0)
        g = _gelu(gate.to(tl.float32)).to(x.dtype)
        y = (x.to(tl.float32) * g.to(tl.float32)).to(x.dtype)
        tl.store(out_ptr + row * inner + col, y, mask=mask)

    @triton.jit
    def _geglu_backward_kernel(dy_ptr, h_ptr, dh_ptr, inner, blocks, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        row = (pid // blocks).to(tl.int64)
        col = (pid % blocks) * BLOCK + tl.arange(0, BLOCK)
        mask = col < inner
        base = row * (2 * inner)
        dy = tl.load(dy_ptr + row * inner + col, mask=mask, other=0.0)
        x = tl.load(h_ptr + base + col, mask=mask, other=0.0)
        gate = tl.load(h_ptr + base + inner + col, mask=mask, other=0.0)
        dyf, xf, gf = dy.to(tl.float32), x.to(tl.float32), gate.to(tl.float32)
        # d(x * g): each factor's gradient is rounded to the activation dtype, as the two mul kernels do
        g = _gelu(gf).to(x.dtype).to(tl.float32)
        dx = (dyf * g).to(x.dtype)
        dg = (dyf * xf).to(x.dtype).to(tl.float32)
        # gelu_backward: dg * (cdf + x * pdf)
        cdf = 0.5 * (1.0 + libdevice.erf(gf * 0.7071067811865476))
        pdf = libdevice.exp(gf * gf * -0.5) * 0.3989422804014327
        dgate = (dg * (cdf + gf * pdf)).to(x.dtype)
        tl.store(dh_ptr + base + col, dx, mask=mask)
        tl.store(dh_ptr + base + inner + col, dgate, mask=mask)

    _forward_kernel = _geglu_forward_kernel
    _backward_kernel = _geglu_backward_kernel

except Exception as exc:  # noqa: BLE001 - no triton means no kernel
    _import_error = exc


def chain(h):
    """The eager chain the kernels replace (and the default reference of the forward gate)."""
    x, gate = h.chunk(2, dim=-1)
    return x * torch.nn.functional.gelu(gate)


def chain_backward(dy, h):
    """Autograd of the eager chain, from ``h``: the fallback when the backward gate is closed."""
    with torch.enable_grad():
        leaf = h.detach().requires_grad_(True)
        out = chain(leaf)
        (dh,) = torch.autograd.grad(out, leaf, dy)
    return dh


def enabled() -> bool:
    if _forward_kernel is None:
        return False
    return os.environ.get("XPU_KERNELS_NO_GEGLU", "") in ("", "0")


def _launch_shape(h):
    inner = h.shape[-1] // 2
    rows = h.numel() // (2 * inner)
    blocks = (inner + BLOCK - 1) // BLOCK
    return inner, rows, blocks


def _forward(h):
    inner, rows, blocks = _launch_shape(h)
    out = torch.empty(h.shape[:-1] + (inner,), dtype=h.dtype, device=h.device)
    _forward_kernel[(rows * blocks,)](h, out, inner, blocks, BLOCK=BLOCK, num_warps=NUM_WARPS)
    return out


def _backward(dy, h):
    inner, rows, blocks = _launch_shape(h)
    dh = torch.empty_like(h)
    _backward_kernel[(rows * blocks,)](dy, h, dh, inner, blocks, BLOCK=BLOCK, num_warps=NUM_WARPS)
    return dh


class _Geglu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h):
        _FORWARD.require()
        ctx.fused_backward = _BACKWARD.ok()
        ctx.save_for_backward(h)
        return _forward(h)

    @staticmethod
    def backward(ctx, dy):
        (h,) = ctx.saved_tensors
        if ctx.fused_backward:
            return _backward(dy.contiguous(), h)
        return chain_backward(dy, h)


def geglu(h):
    """``h[.., :I] * gelu(h[.., I:])`` for a contiguous ``h`` of shape ``[.., 2I]``; differentiable."""
    if not h.is_contiguous():
        h = h.contiguous()
    return _Geglu.apply(h)


def usable(h) -> bool:
    """May the fused GEGLU be used for this input?"""
    if not enabled() or h.dtype not in DTYPES or h.device.type not in ACCELERATORS:
        return False
    if h.shape[-1] == 0 or h.shape[-1] % 2:
        return False
    return _FORWARD.ok()


def _inputs(rows, inner, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    # wide enough that the erf tails and the sign flips of gelu are all exercised
    return (torch.randn((rows, 2 * inner), generator=gen, device=device) * 2.0).to(torch.bfloat16)


# 5120 is SDXL's FFN width, 2560 its 640-channel one, 1288 is not a multiple of the block and tests the mask
_SHAPES = ((64, 5120), (64, 2560), (33, 1288))


def _probe_forward(ref):
    if _forward_kernel is None:
        raise RuntimeError("the triton GEGLU kernel is not available here")
    device = probe_device()
    return {f"forward {inner}": (_forward(h), ref(h))
            for rows, inner in _SHAPES for h in (_inputs(rows, inner, device, 0),)}


def _probe_backward(ref):
    if _backward_kernel is None:
        raise RuntimeError("the triton GEGLU kernel is not available here")
    device = probe_device()
    pairs = {}
    for rows, inner in _SHAPES:
        h = _inputs(rows, inner, device, 1)
        gen = torch.Generator(device=device).manual_seed(2)
        dy = torch.randn((rows, inner), generator=gen, device=device).to(torch.bfloat16)
        leaf = h.clone().requires_grad_(True)
        (want,) = torch.autograd.grad(ref(leaf), leaf, dy)
        pairs[f"backward {inner}"] = (_backward(dy, h), want)
    return pairs


#: The backward's relative budget.  A wrong constant or a missing rounding is off by ~1e-2 or more; one flipped bf16 ULP in
#: a few elements is ~1e-4.
REL = 2e-3

_FORWARD = Gate("geglu", _probe_forward, default_reference=lambda: chain, env="XPU_KERNELS_NO_GEGLU", num_warps=NUM_WARPS)
_BACKWARD = Gate("geglu_backward", _probe_backward, default_reference=lambda: chain, env="XPU_KERNELS_NO_GEGLU",
                 num_warps=NUM_WARPS, rel=REL)


def set_reference(fn=None):
    """Set (or clear) the chain both gates compare against: ``fn(h) -> x * gelu(gate)`` and differentiable."""
    _FORWARD.set_reference(fn)
    _BACKWARD.set_reference(fn)


def gates():
    return {"geglu": _FORWARD, "geglu_backward": _BACKWARD}
