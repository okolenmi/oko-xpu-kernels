"""The INT8 API: weight preparation, the linear, and the accumulator paths.

Every function here takes and returns torch tensors on the XPU device; the
kernels are enqueued on the caller's current stream, so nothing in this module
synchronises.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import torch

from ._lib import (
    ACTIVATIONS,
    _err,
    _flat2d,
    _fp32,
    _lib,
    _stream,
)

def rotate(x):
    """ConvRot rotation only: bf16 [.., K] in, bf16 [.., K] out, no quantization."""
    lib = _lib()
    x2, shape = _flat2d(x)
    out = torch.empty_like(x2)
    m, k = x2.shape
    rc = lib.xpuint8_convrot_rotate_bf16_stream(
        _stream(x.device), ctypes.c_void_p(x2.data_ptr()),
        ctypes.c_void_p(out.data_ptr()), m, k)
    if rc != 0:
        _err(lib, "rotate")
    return out.reshape(shape)


def quantize(x):
    """Rotate and quantize an activation: bf16 [.., K] -> (int8 [.., K], fp32 [..])."""
    lib = _lib()
    x2, shape = _flat2d(x)
    m, k = x2.shape
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    scale = torch.empty((m,), device=x.device, dtype=torch.float32)
    rc = lib.xpuint8_convrot_quantize_stream(
        _stream(x.device), ctypes.c_void_p(x2.data_ptr()),
        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(scale.data_ptr()), m, k)
    if rc != 0:
        _err(lib, "quantize")
    return q.reshape(*shape[:-1], k), scale.reshape(shape[:-1])


def quantize_act(x, activation: str):
    """Rotate+quantize with an activation fused into the load.

    ``gelu_tanh`` consumes [.., K], ``swiglu`` and ``situ`` consume [.., 2K]
    laid out as [hidden, gate]; the result is [.., K].  ``K`` is the *output*
    width in all cases -- the C API takes it that way, because for the pair
    activations it doubles as the stride between the two halves.
    """
    lib = _lib()
    fn = getattr(lib, "xpuint8_convrot_quantize_act_stream", None)
    if fn is None:
        raise RuntimeError("this build has no fused activation quantizer")
    x2, shape = _flat2d(x)
    m, k2 = x2.shape
    act = ACTIVATIONS[activation]
    k = k2 // 2 if act != 1 else k2
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    scale = torch.empty((m,), device=x.device, dtype=torch.float32)
    rc = fn(_stream(x.device), ctypes.c_void_p(x2.data_ptr()),
            ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(scale.data_ptr()),
            m, k, act)
    if rc != 0:
        _err(lib, "quantize_act")
    return q.reshape(*shape[:-1], k), scale.reshape(shape[:-1])


def prepare_weight(w, group: int = 256):
    """Prepare an INT8 weight for :func:`linear`.

    ``w`` is bf16 or fp32 [N, K] in the *unrotated* orientation, i.e. the
    matrix a plain ``x @ w.T`` would use.  Returns ``(int8 [N, K], fp32 [N])``:
    the row-rotated weights and the per-output-channel scale the dequant
    epilogue expects.  Round to nearest, symmetric, clipped to +-127.
    """
    if w.shape[1] % group:
        raise ValueError(f"K={w.shape[1]} is not a multiple of the group size {group}")
    wr = rotate(w.detach().to(device=w.device, dtype=torch.bfloat16).contiguous())
    amax = wr.abs().amax(dim=1).clamp_min(1e-30)
    scale = amax / 127.0
    q = torch.round(wr / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale.contiguous()


def dequant_weight(w8, w_scale):
    """The unrotated bf16 weight behind an INT8 ConvRot weight.

    ``prepare_weight`` stores ``rotate(W) / s`` rounded to int8, so
    ``W = rotate(w8 * s)`` -- the rotation is its own inverse (``rotate(rotate(x))
    == x`` up to bf16 rounding), which is what makes the forward's
    ``(x R) @ (W R)^T == x @ W^T`` work in the first place.  One ``rotate`` call
    therefore undoes it.

    This is the weight-only path for a backward pass: dequantize once and keep
    the result, then ``dx = linear_dx(dy, w)`` is a plain bf16 GEMM.  It is
    called once per weight, not per step -- see ``docs/todo.md``.

    The result is bf16 because that is what the GEMM consumes.  Storing the
    dequantized weight costs one extra ULP per element on top of the int8
    quantization the forward already applies; ``debug/verify_linear_dx.py``
    measures the effect on a gradient (~0.4% relative, the same order as the
    weight quantization itself).
    """
    if w8.dtype != torch.int8:
        raise TypeError(f"w8 must be int8, got {w8.dtype}")
    w = w8.to(torch.float32) * w_scale.to(torch.float32).reshape(-1, 1)
    return rotate(w.to(torch.bfloat16))


def linear_dx(dy, w):
    """``dL/dx`` for a linear with weight ``w``: ``dy [.., N] @ w [N, K] -> [.., K]``.

    ``w`` is the *unrotated* ``[N, K]`` weight -- ``dequant_weight(...)`` of an
    INT8 weight, or the bf16 weight itself.  No transpose and no bias: a bias
    does not depend on ``x``, and ``torch.nn.functional.linear(x, w)`` is
    ``x @ w.T``, so its vjp is ``dy @ w``.
    """
    if dy.shape[-1] != w.shape[0]:
        raise ValueError(f"dy has {dy.shape[-1]} columns but w has {w.shape[0]} rows")
    return dy @ w


def quantize_row(x):
    """Per-row symmetric int8 quantization, no rotation: bf16 [.., K] -> (int8, fp32).

    The ConvRot quantizer (:func:`quantize`) must not be used for a gradient: it
    rotates, and the rotation is already baked into the INT8 weight.  Rotating a
    `dy` here would produce a silently rotated gradient -- wrong, and plausible
    enough to reach a loss curve before anyone noticed.

    This is a torch chain rather than a kernel because it exists to run the
    quantized-gradient A/B (``debug/probe_linear_dx_variants.py``); the probe
    reports what the chain costs, which is the number a fused kernel would have
    to beat before it is worth writing.
    """
    xf = x.to(torch.float32)
    scale = (xf.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-30)
    q = torch.round(xf / scale).clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), scale.squeeze(-1).contiguous()


def linear_dx_int8(dy, w8, w_scale):
    """``dL/dx`` against the weight *still in int8* -- the quantized-gradient path.

    Exact in the weight: since ``W[n, k] = W8[n, k] * s[n]`` and ``s[n]`` is the
    contraction index of ``dx = dy @ W``, it is a *column* scale on ``dy``:

        dx = dy @ diag(s) @ W8 = (dy * s) @ W8      (then undo the ConvRot rotation)

    so the weight never leaves int8 (no dequant, half the memory of the bf16
    copy) and the only new error is quantizing ``dy``.  That is a training
    decision, not a kernel one: A/B it against :func:`linear_dx` on a loss curve
    before making it a default (`docs/todo.md`).
    """
    d = dy.to(torch.bfloat16) * w_scale.to(torch.device(dy.device), torch.bfloat16)
    d2 = d.reshape(-1, d.shape[-1])
    q, a = quantize_row(d2)
    # dy [M, N] @ W8 [N, K]: the weight enters as stored (no transpose -- that is
    # the forward's convention, where x [M, K] meets w8 [N, K].t())
    acc = torch._int_mm(q, w8)
    # the epilogue's column scales are indexed by the accumulator's *columns*,
    # which here are K, not N -- a longer buffer is read only in part, a shorter
    # one reads past the end
    dx = dequant(acc, a, torch.ones(acc.shape[1], device=acc.device,
                                    dtype=torch.float32))
    return rotate(dx).reshape(*dy.shape[:-1], w8.shape[1])


def dequant(acc, x_scale, w_scale, bias=None):
    """bf16 dequant of an int32 [M, N] accumulator with per-row and per-column scales."""
    lib = _lib()
    if acc.dim() != 2 or acc.dtype != torch.int32:
        raise TypeError("acc must be an int32 [M, N] tensor")
    m, n = acc.shape
    out = torch.empty((m, n), device=acc.device, dtype=torch.bfloat16)
    xs, ws = _fp32(x_scale), _fp32(w_scale)
    b = None if bias is None else _fp32(bias)
    rc = lib.xpuint8_dequant_epilogue_stream(
        _stream(acc.device), ctypes.c_void_p(acc.data_ptr()),
        ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(xs.data_ptr()),
        ctypes.c_void_p(ws.data_ptr()),
        ctypes.c_void_p(0 if b is None else b.data_ptr()), m, n)
    if rc != 0:
        _err(lib, "dequant")
    return out


def dequant_slice(acc, x_scale, w_scale, bias=None):
    """As :func:`dequant`, for a [M, N] column slice of a wider [M, S] accumulator.

    ``acc`` may be a strided view (``big[:, off:off+N]``); the kernel walks
    ``acc.stride(0)`` between rows, so the slice is read in place instead of
    copied.  Useful when one GEMM produces q|k|v and only one of them needs
    bf16.
    """
    lib = _lib()
    fn = getattr(lib, "xpuint8_dequant_epilogue_slice_stream", None)
    if fn is None:
        raise RuntimeError("this build has no row-strided dequant")
    if acc.dim() != 2 or acc.dtype != torch.int32:
        raise TypeError("acc must be an int32 [M, N] tensor")
    m, n = acc.shape
    out = torch.empty((m, n), device=acc.device, dtype=torch.bfloat16)
    xs, ws = _fp32(x_scale), _fp32(w_scale)
    b = None if bias is None else _fp32(bias)
    rc = fn(_stream(acc.device), ctypes.c_void_p(acc.data_ptr()),
            ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(xs.data_ptr()),
            ctypes.c_void_p(ws.data_ptr()),
            ctypes.c_void_p(0 if b is None else b.data_ptr()),
            m, n, acc.stride(0))
    if rc != 0:
        _err(lib, "dequant_slice")
    return out


def dequant_act_quantize(acc, x_scale, w_scale, activation="situ"):
    """Dequant, activate, rotate and quantize an up|gate accumulator in one pass.

    ``acc`` is int32 [M, 2K] holding [gate | up].  Returns ``(int8 [M, K],
    fp32 [M])`` -- the quantized activation for the *next* INT8 linear, with
    the bf16 intermediate never written.  This is what makes a
    ``down(situ(up(x)))`` block two kernels instead of eight.
    """
    lib = _lib()
    fn = getattr(lib, "xpuint8_dequant_act_quantize_stream", None)
    if fn is None:
        raise RuntimeError("this build has no fused dequant+activation+quantize")
    if acc.dim() != 2 or acc.dtype != torch.int32 or acc.shape[1] % 2:
        raise TypeError("acc must be an int32 [M, 2K] tensor")
    m, k2 = acc.shape
    k = k2 // 2
    q = torch.empty((m, k), device=acc.device, dtype=torch.int8)
    scale = torch.empty((m,), device=acc.device, dtype=torch.float32)
    xs, ws = _fp32(x_scale), _fp32(w_scale)
    rc = fn(_stream(acc.device), ctypes.c_void_p(acc.data_ptr()),
            ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(scale.data_ptr()),
            ctypes.c_void_p(xs.data_ptr()), ctypes.c_void_p(ws.data_ptr()),
            m, k, ACTIVATIONS[activation])
    if rc != 0:
        _err(lib, "dequant_act_quantize")
    return q, scale


def linear_acc(x, w8, w_scale, act: Optional[str] = None):
    """The GEMM without the output dequant: ``(int32 [M, N], fp32 [M])``.

    Use this when the consumer can walk the accumulator itself -- a fused
    dequant+activation, a rope/norm kernel, or a strided dequant of one slice
    of a fused projection.  ``act`` fuses an activation into the *input*
    quantizer (``gelu_tanh`` on [.., K], ``swiglu``/``situ`` on [.., 2K]).
    """
    if act is None:
        xq, xs = quantize(x)
    else:
        xq, xs = quantize_act(x, act)
    if w8.dtype != torch.int8 or w8.dim() != 2:
        raise TypeError("w8 must be an int8 [N, K] tensor")
    if xq.shape[-1] != w8.shape[1]:
        raise ValueError(f"K mismatch: activation {xq.shape[-1]} vs weight {w8.shape[1]}")
    acc = torch._int_mm(xq.reshape(-1, xq.shape[-1]), w8.t())
    return acc, xs.reshape(-1)


def linear(x, w8, w_scale, bias=None, act: Optional[str] = None):
    """INT8 ConvRot linear: bf16 [.., K] -> bf16 [.., N] (N = w8.shape[0])."""
    acc, xs = linear_acc(x, w8, w_scale, act)
    out = dequant(acc, xs, w_scale, bias)
    shape = (*x.shape[:-1], w8.shape[0])
    return out.reshape(shape)


class Int8Linear(torch.nn.Module):
    """Drop-in ``nn.Linear`` replacement for inference with prepared weights.

        mod = Int8Linear.from_linear(model.some_linear)
        model.some_linear = mod
    """

    def __init__(self, w8, w_scale, bias=None):
        super().__init__()
        self.register_buffer("w8", w8.detach())
        self.register_buffer("w_scale", w_scale.detach().to(torch.float32))
        self.bias = None if bias is None else torch.nn.Parameter(
            bias.detach().to(torch.float32), requires_grad=False)

    @classmethod
    def from_linear(cls, mod):
        q, scale = prepare_weight(mod.weight.detach())
        return cls(q, scale, getattr(mod, "bias", None))

    def forward(self, x):
        return linear(x, self.w8, self.w_scale, self.bias)

    def dense_weight(self, refresh=False):
        """The unrotated bf16 weight, dequantized once and cached.

        Not a registered buffer: it is derived state, it doubles this module's
        memory, and a training loop only needs it when it takes a backward.  So
        ``.to()`` does not move it either -- call with ``refresh=True`` after
        re-quantizing ``w8``/``w_scale`` or moving the module.
        """
        cached = getattr(self, "_dense", None)
        if cached is None or refresh:
            cached = dequant_weight(self.w8, self.w_scale)
            self._dense = cached
        return cached

    def dx(self, dy):
        """``dL/dx`` given ``dL/dy`` -- the backward's activation gradient."""
        return linear_dx(dy, self.dense_weight())

    def extra_repr(self):
        n, k = self.w8.shape
        return f"in_features={k}, out_features={n}, bias={self.bias is not None}"
