"""The INT8 API: weight preparation, the linear, and the accumulator paths.

Every function here takes and returns torch tensors on the XPU device; the
kernels are enqueued on the caller's current stream, so nothing in this module
synchronises.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import torch

from .._capture import is_capturing
from ._lib import (
    ACTIVATIONS,
    _err,
    _flat2d,
    _lib,
    _stream,
    load,
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
    # fp32 on purpose: the kernel reads an fp32 scale, and an 8-bit-mantissa
    # bf16 scale would quantize the *scale* on top of the weight.  ``wr / scale``
    # then promotes to fp32, so the round-to-int8 decision uses the exact scale.
    scale = amax.float() / 127.0
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
    if is_capturing():
        # A full-precision copy of the weight, inside a device-graph capture,
        # would be captured and re-computed on every replay -- ~0.4 s per step
        # for a 3.7 B model.  Materialize it before capturing; the INT8
        # backward ("mode='int8'") needs no copy at all.
        raise RuntimeError(
            "dequant_weight() materializes a full-precision copy of the "
            "weight; inside a device-graph capture it would be captured and "
            "re-computed on every replay. Call it before capturing, or use "
            "the int8 backward (xpu_kernels.int8_linear(..., mode='int8')), "
            "which needs no copy at all.")
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


def linear_dw(dy, x):
    """``dL/dW`` for a linear: ``dy [.., N]^T @ x [.., K] -> [N, K]``.

    The surprising part is how little of the INT8 machinery shows up here.

    The forward is *mathematically* the bf16 forward with the dequantized
    weight: the ConvRot rotation is orthogonal, so it cancels in the
    contraction, and the activation's quantization is exact for the same
    reason.  So ``y = x @ dequant(w8, s)^T`` as an identity, and the weight
    gradient of the INT8 model is the weight gradient of that bf16 model:

        dL/dW = dy^T @ x

    No scale, no rotation, and -- unlike :func:`linear_dx` -- no dequantized
    weight at all: *the weight does not appear in its own gradient*.  The
    weight's quantization still shows up in the gradient's **value**, because
    the forward used the dequantized weight; that is the same 0.9 % the dx
    path reports, and the convention is the straight-through one: the gradient
    is the gradient of the model as it was computed, and the parameter is
    updated as if the quantization were the identity.

    ``x`` is the *unrotated* activation, which is what the caller has: the
    residual stream, in bf16.  If the caller saved the *quantized* activation
    instead, the gradient picks up the activation's quantization error and the
    rotation has to be undone afterwards; that variant is not here.
    """
    if dy.shape[-2] != x.shape[-2]:
        raise ValueError(
            f"dy has {dy.shape[-2]} rows but x has {x.shape[-2]}: the batch "
            f"dims must match for dy^T @ x")
    return dy.reshape(-1, dy.shape[-1]).transpose(0, 1) @ \
        x.reshape(-1, x.shape[-1])


class _Int8LinearFn(torch.autograd.Function):
    """The INT8 linear as an autograd op, so a training graph can use it.

    Without this the gradient would *silently vanish*: the kernels are not
    differentiable, so ``linear()`` returns a tensor with no ``grad_fn`` and a
    model that drops an INT8 layer into a training graph trains nothing
    through it.  (This is the failure the trainer's own ``Int8LinearFn``
    exists to prevent.)

    The backward is :func:`linear_dx` -- the weight is dequantized for it --
    or :func:`linear_dx_int8`, which keeps the weight in int8 and quantizes
    ``dy`` instead: no full-precision copy of the weight at all, at the price
    of one quantizer's worth of gradient error (1.35 % against the bf16
    gradient, against 0.93 % for the dequantized-weight path).

    The weight gradient is not returned here: the INT8 weight is a frozen
    int8 buffer, so its gradient is meaningless.  For a trainable weight the
    parameter is bf16 and the forward quantizes it per step; the gradient of
    *that* parameter is :func:`linear_dw`, and the caller computes it (it
    needs the activation, which the caller has).
    """

    @staticmethod
    def forward(ctx, x, w8, w_scale, bias, mode):
        ctx.save_for_backward(w8, w_scale)
        ctx.mode = mode
        ctx.input_dtype = x.dtype
        return linear(x, w8, w_scale, bias)

    @staticmethod
    def backward(ctx, dy):
        w8, w_scale = ctx.saved_tensors
        dy = dy.to(torch.bfloat16)
        if ctx.mode == "int8":
            dx = linear_dx_int8(dy, w8, w_scale)
        elif ctx.mode == "dense":
            # the weight is materialized here, once, and dropped again: the
            # memory the quantization saved stays saved.  Inside a device-graph
            # capture this raises (dequant_weight refuses), because a captured
            # dequant would be replayed on every step.
            dx = linear_dx(dy, dequant_weight(w8, w_scale))
        else:
            raise ValueError(f"unknown backward mode {ctx.mode!r}; "
                             f"use 'dense' or 'int8'")
        return dx.to(ctx.input_dtype), None, None, None, None


def int8_linear(x, w8, w_scale, bias=None, *, mode="dense"):
    """The INT8 linear, differentiable.

    Same forward as :func:`linear`; the backward is the vjp of the INT8 model
    (:func:`linear_dx` or :func:`linear_dx_int8`).

    ``mode``:

    * ``"dense"`` -- the weight is dequantized in the backward, so the
      gradient is exact against the dequantized weight (0.93 % from the
      unquantized one).  The full-precision copy exists only for the duration
      of that one layer's backward.
    * ``"int8"`` -- the weight stays int8: no copy at all, one byte per
      parameter, and the gradient is 1.35 % from the unquantized model
      because ``dy`` is quantized.  Needs the loss-curve A/B before it is a
      default; ``docs/todo.md`` has it.

    The weight gradient is not returned: the INT8 weight is a frozen int8
    buffer, so its gradient is meaningless.  For a trainable weight, the
    parameter is bf16 and the forward quantizes it per step; the gradient of
    *that* parameter is :func:`linear_dw`, computed by the caller (it needs
    the activation, which the caller has).
    """
    if mode not in ("dense", "int8"):
        raise ValueError(f"unknown backward mode {mode!r}; use 'dense' or 'int8'")
    return _Int8LinearFn.apply(x, w8, w_scale, bias, mode)


def quantize_row_torch(x):
    """The per-row quantizer as a torch chain -- the reference, always there.

    This is what the fused Triton kernel is compared against, bit for bit, so
    it must not call the kernel.  :func:`quantize_row` is the fast path and
    falls back to this.
    """
    xf = x.to(torch.float32)
    scale = (xf.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-30)
    q = torch.round(xf / scale).clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), scale.squeeze(-1).contiguous()


def quantize_row(x):
    """Per-row symmetric int8 quantization, no rotation: bf16 [.., K] -> (int8, fp32).

    The ConvRot quantizer (:func:`quantize`) must not be used for a gradient: it
    rotates, and the rotation is already baked into the INT8 weight.  Rotating a
    `dy` here would produce a silently rotated gradient -- wrong, and plausible
    enough to reach a loss curve before anyone noticed.

    The fused Triton kernel does this in one pass (three bytes of traffic per
    element, one launch); the torch chain needs about forty and eleven.  The
    two agree bit for bit -- the gate proves it on every process, on data that
    includes the rounding ties -- so this is the same number, faster.
    """
    from ..triton import quantize as _q

    if _q.usable(x):
        return _q.quantize_row(x)
    return quantize_row_torch(x)


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


def expand_scale(scale, n, device=None):
    """A flat fp32 ``[n]`` scale, broadcasting a per-tensor scalar.

    The kernels index ``scale[i]`` for every ``i < n``, so a scalar *must* be
    materialised: a 1-element buffer where ``n`` values are read is an
    out-of-bounds read, not a broadcast.  That is why this exists -- a
    per-tensor weight scale (``[1]``, ``[N, 1]`` or a python float) is legal
    input to :func:`linear`, and getting it wrong is a silent wrong result.

    The fast path returns ``scale`` unchanged when it is already a flat fp32
    ``[n]`` tensor on the right device, which is the case for a per-channel
    scale after :func:`prepare_weight` -- no allocation, no copy.  Otherwise it
    allocates, so call it once per weight and keep the result rather than
    re-expanding inside a hot loop.
    """
    if not isinstance(scale, torch.Tensor):
        if not isinstance(scale, (float, int)):
            raise TypeError(f"scale must be a tensor or a python float, "
                            f"got {type(scale).__name__}")
        if device is None:
            raise TypeError("a python-scalar scale needs an explicit device")
        return torch.full((n,), float(scale), device=device, dtype=torch.float32)
    if (scale.dtype == torch.float32 and scale.dim() == 1 and scale.numel() == n
            and scale.is_contiguous() and (device is None or scale.device == device)):
        return scale
    s = scale.reshape(-1).to(device=device or scale.device, dtype=torch.float32)
    if s.numel() == 1 and n != 1:
        s = s.expand(n)
    return s.contiguous()


def _dims(x, w8, act):
    m = 1
    for d in x.shape[:-1]:
        m *= int(d)
    return m, int(w8.shape[0]), int(w8.shape[1])


def unsupported_reason(x, w8, w_scale, act=None):
    """Why this call cannot use the fused kernels, or ``None`` if it can.

    Returned instead of a bare bool so a caller can log *why* a layer fell back;
    ``None`` from the kernel layer is indistinguishable from a typo.  The
    checks are ordered cheapest-and-most-likely-first, and are the single source
    of truth: :func:`linear` raises exactly this message, so the two cannot
    drift apart.

    A caller that wants fallback semantics (ComfyUI does, per layer) writes::

        if int8.unsupported_reason(x, w8, ws, act):
            return fallback(...)
        return int8.linear(x, w8, ws, act=act)
    """
    if not isinstance(x, torch.Tensor) or not isinstance(w8, torch.Tensor):
        return "activation and weight must be torch tensors"
    if x.dtype not in (torch.bfloat16, torch.float32):
        return (f"activation dtype {x.dtype} is not bfloat16 or float32 "
                f"(the kernels consume bf16)")
    if w8.dtype != torch.int8 or w8.dim() != 2:
        return f"weight must be an int8 [N, K] tensor, got {w8.dtype} {tuple(w8.shape)}"
    if act not in (None, "gelu_tanh", "swiglu", "situ"):
        return (f"unknown activation {act!r}; expected None, 'gelu_tanh', "
                f"'swiglu' or 'situ'")
    k = int(w8.shape[1])
    want = k * 2 if act in ("swiglu", "situ") else k
    if x.shape[-1] != want:
        return (f"activation has {x.shape[-1]} columns but this weight wants "
                f"{want} (K={k}{' with two halves' if want != k else ''})")
    # 256 is a multiple of 16, so this covers the vector width too
    if k % 256:
        return f"K={k} is not a multiple of the ConvRot group size (256)"
    if w8.shape[0] % 16:
        return (f"N={w8.shape[0]} is not a multiple of 16; the dequant epilogue "
                f"reads the output 16 columns at a time and would drop the tail")
    if isinstance(w_scale, torch.Tensor):
        # any float dtype is fine: it is converted to fp32 exactly, which is
        # what the kernel reads.  Integer scales are not -- they would be a
        # different quantity, not a rounded one.
        if not w_scale.is_floating_point():
            return (f"weight scale dtype {w_scale.dtype} is not floating point; "
                    f"a per-channel scale is amax/127")
        if w_scale.dim() > 2:
            return (f"weight scale has {w_scale.dim()} dims; expected a scalar, "
                    f"[N] or [N, 1]")
        if w_scale.numel() not in (1, w8.shape[0]):
            return (f"weight scale has {w_scale.numel()} values; expected 1 or "
                    f"N={w8.shape[0]}")
    elif not isinstance(w_scale, (float, int)):
        return "weight scale must be a tensor or a python float"
    if load() is None:
        return "libxpuint8.so is not available (see xpu_kernels.int8._lib)"
    if x.device.type != "xpu":
        return f"activation is on {x.device}; the kernels need an XPU tensor"
    if w8.device != x.device:
        return (f"weight is on {w8.device} but the activation is on {x.device}; "
                f"the kernels take raw pointers into one queue")
    return None


def dequant(acc, x_scale, w_scale, bias=None):
    """bf16 dequant of an int32 [M, N] accumulator with per-row and per-column scales."""
    lib = _lib()
    if acc.dim() != 2 or acc.dtype != torch.int32:
        raise TypeError("acc must be an int32 [M, N] tensor")
    m, n = acc.shape
    out = torch.empty((m, n), device=acc.device, dtype=torch.bfloat16)
    # the kernel reads x_scale[m] and w_scale[n] for every row/column, so both
    # have to be flat and n long -- a scalar is broadcast here, not by the C side
    xs, ws = expand_scale(x_scale, m, acc.device), expand_scale(w_scale, n, acc.device)
    b = None if bias is None else expand_scale(bias, n, acc.device)
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
    xs, ws = expand_scale(x_scale, m, acc.device), expand_scale(w_scale, n, acc.device)
    b = None if bias is None else expand_scale(bias, n, acc.device)
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
    # both scales are indexed per accumulator column/row, so both must be flat
    xs = expand_scale(x_scale, m, acc.device)
    ws = expand_scale(w_scale, k2, acc.device)
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


def linear(x, w8, w_scale, bias=None, act: Optional[str] = None, residual=None,
           residual_scale=None, return_acc: bool = False):
    """INT8 ConvRot linear: bf16/fp32 [.., K] -> bf16/fp32 [.., N].

    The one entry point for a projection, covering what the kernels fuse:

    * ``act`` folds an activation into the *input* quantizer, so ``act(x)`` is
      never materialised: ``gelu_tanh`` on [.., K], ``swiglu``/``situ`` on
      [.., 2K] laid out as ``[up | gate]``, both producing [.., K].
    * ``residual``/``residual_scale`` apply ``residual + residual_scale * out``
      after the dequant, which is where a pre-norm block's addcmul belongs.
    * ``return_acc`` skips the output epilogue and returns
      ``(int32 [M, N], fp32 [M], fp32 [N])`` -- the accumulator plus the two
      scales it still needs.  Use it when the consumer walks the accumulator
      itself (:func:`dequant_slice`, :func:`dequant_act_quantize`).  Only for a
      plain, unbiased projection: there is no epilogue to add a bias or residual
      to.

    Raises ``TypeError`` with a specific message when the shape, dtype or device
    is outside what the kernels handle; :func:`unsupported_reason` returns that
    message without raising, for callers that fall back per layer.

    ``x`` may be float32 (CLIP text encoders pass it); the compute precision is
    bf16 either way, and the result is returned in the caller's dtype.
    """
    reason = unsupported_reason(x, w8, w_scale, act)
    if reason is not None:
        raise TypeError(reason)
    if return_acc and (bias is not None or residual is not None):
        raise ValueError("return_acc skips the epilogue, so bias and residual "
                         "cannot be applied")
    m, n, _ = _dims(x, w8, act)
    input_dtype = x.dtype
    x2 = x.reshape(m, -1)
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(dtype=torch.bfloat16)
    if not x2.is_contiguous():
        x2 = x2.contiguous()

    if act is None:
        xq, xs = quantize(x2)
    else:
        xq, xs = quantize_act(x2, act)
    acc = torch._int_mm(xq, w8.t())
    if return_acc:
        return acc, xs.reshape(-1), expand_scale(w_scale, n, acc.device)

    out = dequant(acc, xs, w_scale, bias)
    if residual is not None:
        residual = residual.reshape(m, n).to(
            out.dtype if input_dtype == torch.bfloat16 else input_dtype)
        if input_dtype != torch.bfloat16:
            out = out.to(dtype=input_dtype)
        out = torch.addcmul(residual, out, residual_scale)
    out = out.reshape(*x.shape[:-1], n)
    return out if input_dtype == torch.bfloat16 else out.to(dtype=input_dtype)


def ffn_situ(x, up_w8, up_scale, down_w8, down_scale, down_bias=None):
    """``situ(up(x)) @ down.T`` with the bf16 intermediate never written.

    ``up_w8`` is [2*ffn, K] holding ``[up | gate]`` and ``down_w8`` is
    [K, ffn].  The up-projection's int32 accumulator goes straight into the
    fused dequant+situ+rotate+quantize kernel (:func:`dequant_act_quantize`), so
    the [M, 2*ffn] bf16 tile that the two-op form writes and reads again does
    not exist.  The down-projection's epilogue is the ordinary one, so
    ``down_bias`` is free but the up-projection must be unbiased (its bias would
    have to be added before the activation, inside the fused kernel).

    Raises ``TypeError`` like :func:`linear`; see :func:`unsupported_reason`.
    """
    reason = unsupported_reason(x, up_w8, up_scale, None)
    if reason is not None:
        raise TypeError(reason)
    m, n, k = _dims(x, up_w8, None)
    if n % 2 or (n // 2) % 256 or (n // 2) % 16:
        raise TypeError(f"up-projection N={n} must be 2*ffn with ffn a multiple "
                        f"of 256 (and of 16) for the fused situ epilogue")
    if down_w8.dtype != torch.int8 or down_w8.dim() != 2:
        raise TypeError(f"down weight must be an int8 [K, ffn] tensor, got "
                        f"{down_w8.dtype} {tuple(down_w8.shape)}")
    if tuple(down_w8.shape) != (k, n // 2):
        raise TypeError(f"down weight is {tuple(down_w8.shape)}, expected "
                        f"{(k, n // 2)}")
    if down_w8.device != x.device:
        raise TypeError(f"down weight is on {down_w8.device} but the activation "
                        f"is on {x.device}")
    if isinstance(down_scale, torch.Tensor):
        if down_scale.dtype != torch.float32 or down_scale.dim() > 2:
            raise TypeError("down weight scale must be a float32 scalar or [K]")
        if down_scale.numel() not in (1, k):
            raise TypeError(f"down weight scale has {down_scale.numel()} values; "
                            f"expected 1 or K={k}")
    elif not isinstance(down_scale, (float, int)):
        raise TypeError("down weight scale must be a tensor or a python float")

    device = x.device
    input_dtype = x.dtype
    x2 = x.reshape(m, k)
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(dtype=torch.bfloat16)
    if not x2.is_contiguous():
        x2 = x2.contiguous()

    xq, xs = quantize(x2)
    acc = torch._int_mm(xq, up_w8.t())
    q2, xs2 = dequant_act_quantize(acc, xs, expand_scale(up_scale, n, device), "situ")
    acc2 = torch._int_mm(q2, down_w8.t())
    out = dequant(acc2, xs2, down_scale, down_bias)
    out = out.reshape(*x.shape[:-1], k)
    return out if input_dtype == torch.bfloat16 else out.to(dtype=input_dtype)


class Int8Linear(torch.nn.Module):
    """Drop-in ``nn.Linear`` replacement for inference with prepared weights.

        mod = Int8Linear.from_linear(model.some_linear)
        model.some_linear = mod
    """

    def __init__(self, w8, w_scale, bias=None, *, backward="dense"):
        super().__init__()
        self.register_buffer("w8", w8.detach())
        self.register_buffer("w_scale", w_scale.detach().to(torch.float32))
        self.bias = None if bias is None else torch.nn.Parameter(
            bias.detach().to(torch.float32), requires_grad=False)
        if backward not in ("dense", "int8"):
            raise ValueError(f"unknown backward mode {backward!r}; "
                             f"use 'dense' or 'int8'")
        #: how the backward gets the weight: ``"dense"`` dequantizes it for
        #: that one layer's backward (exact, no resident copy), ``"int8"``
        #: keeps it in int8 and quantizes ``dy`` instead (one byte per
        #: parameter, one quantizer's worth of gradient error).
        self.backward = backward

    @classmethod
    def from_linear(cls, mod, *, backward="dense"):
        q, scale = prepare_weight(mod.weight.detach())
        return cls(q, scale, getattr(mod, "bias", None), backward=backward)

    def forward(self, x):
        """Differentiable: the backward is the vjp of the INT8 model.

        Without the autograd op the gradient would silently vanish (the
        kernels are not differentiable), which is the one thing a training
        graph must not do.
        """
        return int8_linear(x, self.w8, self.w_scale, self.bias,
                           mode=self.backward)

    def dense_weight(self, refresh=False):
        """The unrotated bf16 weight, dequantized once and cached.

        Not a registered buffer: it is derived state, it doubles this module's
        memory, and a training loop only needs it when it takes a backward.  So
        ``.to()`` does not move it either -- call with ``refresh=True`` after
        re-quantizing ``w8``/``w_scale`` or moving the module.

        This must happen **before** a device-graph capture, and the package
        refuses to do it inside one: the dequant is a full-precision copy of
        the weight (~0.4 s for a 3.7 B model), and a captured kernel is
        replayed on every step.  Call ``xpu_kernels.prepare_for_capture(model)``
        once, outside the capture.
        """
        cached = getattr(self, "_dense", None)
        if cached is None or refresh:
            if is_capturing():
                raise RuntimeError(
                    "Int8Linear.dense_weight() materializes a second, "
                    "full-precision copy of the weight; inside a device-graph "
                    "capture it would be captured and re-computed on every "
                    "replay. Call it before capturing -- "
                    "xpu_kernels.prepare_for_capture(model) does it for a whole "
                    "model.")
            cached = dequant_weight(self.w8, self.w_scale)
            self._dense = cached
        return cached

    def dx(self, dy):
        """``dL/dx`` given ``dL/dy`` -- the backward's activation gradient."""
        return linear_dx(dy, self.dense_weight())

    def extra_repr(self):
        n, k = self.w8.shape
        return f"in_features={k}, out_features={n}, bias={self.bias is not None}"
