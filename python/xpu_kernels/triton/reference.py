"""The eager chain, in torch: the fallback, and the backward's reference.

The Triton kernels in this package are schedule changes over a chain of torch
ops.  This module is that chain, written out:

    y = rope_split_half(rms_norm(x) * scale)
    dx = rms_rope_backward(dy, x, freqs_cis, scale)

It exists for three reasons, in order of importance:

* **A caller without ComfyUI needs something to inject as the reference.** The
  kernels verify themselves against the path the caller falls back to; outside
  ComfyUI there is no ``comfy_kitchen``, so this is it.
* **The backward needs a differentiable reference.** ``torch.autograd`` of
  :func:`rms_rope_split_half` is what the fused backward is measured against
  (see ``rope._GATE_BWD``), and it is what a caller gets when the backward
  kernel is inactive.
* **The formula is pinned in float64 by ``tests/test_rope_backward.py``**,
  against ``torch.autograd.gradcheck``. The Triton versions are then verified as
  transcriptions of this arithmetic.

Same arithmetic, different reduction order: this code sums a row with torch's
tree reduction, the kernels with Triton's block reduction, so the results differ
in the last bit.  That is why a kernel verified against this module uses the
gate's tolerance mode (:data:`SAME_ARITHMETIC_REL`), while the same kernel
verified against ``comfy_kitchen`` -- the same schedule -- is bit-exact.
"""

from __future__ import annotations

import torch

__all__ = [
    "SAME_ARITHMETIC_REL",
    "rms_adaln",
    "rms_adaln_backward",
    "rms_adaln_normalized",
    "rms_norm",
    "rms_rope_backward",
    "rms_rope_split_half",
    "rope_split_half",
]

#: Budget for "same arithmetic, different reduction order".
#:
#: The two implementations differ only in how a sum of a row is ordered, which
#: moves a few bf16 results by one ULP (~4e-3 relative each, on a small fraction
#: of elements). A wrong formula is off by ~1e-1, so 1e-3 separates the two
#: cases with room to spare on both sides.
SAME_ARITHMETIC_REL = 1e-3


def _work(x: torch.Tensor) -> torch.Tensor:
    """Promote to at least fp32 for the arithmetic, keeping float64 if asked.

    The kernels' intermediates are fp32; the tests run the same code in float64
    to check the formula rather than the rounding, and ``.float()`` would
    silently defeat that.
    """
    return x.to(torch.promote_types(x.dtype, torch.float32))


def rms_norm(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm with the mean pinned to zero, then the per-dim scale, in x's dtype.

    Mirrors the shipped chain: mean of squares in fp32, ``rsqrt(mean + eps)``,
    multiply by the scale in fp32, then one cast back to the input dtype.
    """
    xf = _work(x)
    inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv_rms * _work(scale)).to(x.dtype)


def _freq_planes(freqs_cis: torch.Tensor):
    """The four rotation planes, shaped to broadcast over the head dimension.

    ``freqs_cis`` is ``[B, S, 1, D/2, 2, 2]`` holding ``[[cos, -sin],
    [sin, cos]]`` per pair (the layout ``comfy/ldm/flux/math.py`` builds).
    """
    return (freqs_cis[..., 0, 0], freqs_cis[..., 0, 1],
            freqs_cis[..., 1, 0], freqs_cis[..., 1, 1])


def rope_split_half(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Split-half rotation: the first half rotates against the second half."""
    xf = _work(x)
    half = xf.shape[-1] // 2
    x0, x1 = xf[..., :half], xf[..., half:]
    f00, f01, f10, f11 = _freq_planes(freqs_cis)
    out0 = f00 * x0 + f01 * x1
    out1 = f10 * x0 + f11 * x1
    return torch.cat((out0, out1), dim=-1).to(x.dtype)


def rms_rope_split_half(x: torch.Tensor, freqs_cis: torch.Tensor,
                        scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """The chain the fused kernel replaces, differentiable end to end."""
    return rope_split_half(rms_norm(x, scale, eps), freqs_cis)


def rms_rope_backward(dy: torch.Tensor, x: torch.Tensor,
                      freqs_cis: torch.Tensor, scale: torch.Tensor,
                      eps: float = 1e-6) -> torch.Tensor:
    """Gradient of :func:`rms_rope_split_half` with respect to ``x``.

    The rotation is orthogonal, so its vjp is the rotation by the transposed
    plane.  The norm's vjp uses the identity

        dx = inv_rms * (dw - (w . dw) / D * w),   w = x * inv_rms

    which needs one dot product per head row -- the only whole-row quantity in
    the backward, and the reason it is a single pass: one program holds a row.

    The cast in ``rms_norm`` is treated as an identity (straight-through), which
    is what ``torch.autograd`` does for a cast; the reference chain and this
    function therefore agree in float64.
    """
    xf = _work(x)
    inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    w = xf * inv_rms

    d = _work(dy)
    half = d.shape[-1] // 2
    dy0, dy1 = d[..., :half], d[..., half:]
    f00, f01, f10, f11 = _freq_planes(freqs_cis)
    # R^T applied to dy, then undo the per-dim scale
    dw = torch.cat((f00 * dy0 + f10 * dy1,
                    f01 * dy0 + f11 * dy1), dim=-1) * _work(scale)

    dot = (dw * w).sum(dim=-1, keepdim=True)
    dx = inv_rms * (dw - (dot / d.shape[-1]) * w)
    return dx.to(x.dtype)


def _reduce_like(grad: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Sum ``grad`` over the dimensions ``ref`` was broadcast along.

    ``torch.autograd`` gives a broadcast input the shape it was passed in, not
    the shape of the output, so the vjp of ``y = ... + shift`` is ``dy`` summed
    over every axis where ``shift`` had size 1 (including any leading axes it
    did not have).  This is that reduction, and it is what makes
    :func:`rms_adaln_backward` usable with a per-block modulation ``[1, 1, D]``
    as well as with a full ``[B, S, D]`` one.
    """
    if grad.dim() < ref.dim():
        # a grouped reduction returns one row per distinct vector, which is
        # fewer axes than the modulation it came from
        grad = grad.reshape(ref.shape)
    n_lead = grad.dim() - ref.dim()
    dims = list(range(n_lead))
    for i, size in enumerate(ref.shape):
        if size == 1 and grad.shape[i + n_lead] != 1:
            dims.append(i + n_lead)
    out = grad.sum(dim=tuple(dims), keepdim=True) if dims else grad
    return out.reshape(ref.shape)


def rms_adaln_normalized(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``x * rstd`` -- the value the modulation is applied to.

    ``dscale`` is ``sum(dy * v)`` over the broadcast axes, so the backward needs
    this whether or not it recomputes the kernel's intermediate.
    """
    xf = _work(x)
    return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)


def rms_adaln(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    """The adaLN chain: ``x * rstd * (1 + scale) + shift``, differentiable.

    ``scale`` and ``shift`` may have any shape broadcastable to ``x`` -- a
    per-block ``[1, 1, D]`` modulation is the usual case, and
    :func:`_reduce_like` collects their gradients back into that shape.
    """
    out = rms_adaln_normalized(x, eps) * (1 + _work(scale)) + _work(shift)
    return out.to(x.dtype)


def rms_adaln_backward(dy: torch.Tensor, x: torch.Tensor, scale: torch.Tensor,
                       shift: torch.Tensor = None, eps: float = 1e-6):
    """``(dx, dscale, dshift)`` for :func:`rms_adaln`, analytically.

    With ``v = x * rstd`` and ``dv = dy * (1 + scale)``:

        dscale = sum(dy * v)        over the axes the modulation was broadcast
        dshift = sum(dy)            down those same axes
        dx     = rstd * (dv - (v . dv) / D * v)

    the last line being the norm's vjp (the same identity as the rope backward).
    ``dx`` does not depend on ``shift``, which is why the kernel never loads it.
    The cast in :func:`rms_adaln` is straight-through, as for the other chains.
    """
    xf = _work(x)
    v = rms_adaln_normalized(x, eps)
    d = _work(dy)
    dv = d * (1 + _work(scale))
    proj = (v * dv).sum(dim=-1, keepdim=True) / xf.shape[-1]
    dx = (torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
          * (dv - proj * v))
    dscale = _reduce_like(d * v, scale).to(scale.dtype)
    dshift = None if shift is None else _reduce_like(d, shift).to(shift.dtype)
    return dx.to(x.dtype), dscale, dshift
