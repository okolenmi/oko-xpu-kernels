"""Single-pass Triton kernels for the non-quantized path.

Both kernels replace a multi-kernel torch chain with one pass over the data and
are meant to be *bit-identical* to it, not merely close:

``rope.rms_rope_split_half(q, k, freqs_cis, q_scale, k_scale=None, eps=1e-6)``
    RMSNorm (mean pinned to 0) + rope, applied to q and k in place of
    ``norm(x)`` followed by the rotation.  Split-half layout, one warp count
    (``num_warps=4``) — a different warp count changes the reduction order and
    the output, so do not "tune" it without re-running the gate.

``adaln.rms_adaln(x, scale, shift, eps=1e-6)``
    RMSNorm + adaLN modulation (``x * rsqrt(mean(x^2)+eps) * (1+scale) + shift``).

``dequant.dequant_weight(w8, scale)``
    The INT8 weight dequantizer as a Triton scale-and-round pass plus one oneDNN product for the group rotation.

``geglu.geglu(h)``
    SDXL's feed-forward activation ``h[.., :I] * gelu(h[.., I:])``, forward and backward in one pass each.

The rule every kernel here follows, and the reason they are safe to drop into a
model: each one verifies itself once per process against a reference
implementation, and the verdict is latched.  A kernel whose gate is closed is
refused at the entry point -- the fused functions ask their own gate, so a
caller that forgets to call :func:`usable` gets an exception rather than an
unverified kernel -- and the backward falls back to the torch gradient, which
*is* the eager chain's gradient.  See
:class:`xpu_kernels.triton._gate.Gate` for the contract, and
``set_reference`` below for supplying the reference.

    import xpu_kernels.triton as k
    k.set_reference(k.rope, my_rms_rope)      # your fallback path, once
    print(k.status())                         # active / inactive + why
"""

from __future__ import annotations

from . import adaln, dequant, geglu, nf4, quantize, rope
from ._gate import Gate

__all__ = ["Gate", "adaln", "dequant", "gates", "geglu", "nf4", "quantize", "rope",
           "set_reference", "status"]

_MODULES = (rope, adaln, quantize, nf4, geglu, dequant)


def gates():
    """Every gate in the package, by kernel name."""
    out = {}
    for module in _MODULES:
        out.update(module.gates())
    return out


def set_reference(module, fn=None):
    """Set (or clear) the reference implementation for one kernel module.

    ``module`` is ``rope``, ``adaln``, ``quantize``, ``nf4``, ``geglu``, ``dequant``, or their name as a
    string;
    ``fn`` is the implementation that module falls back to, so the gate compares
    like with like.  Without a reference the kernels stay inactive.
    """
    if isinstance(module, str):
        module = {"rope": rope, "adaln": adaln, "quantize": quantize,
                  "nf4": nf4, "geglu": geglu, "dequant": dequant}[module]
    module.set_reference(fn)


def status():
    """One line per kernel: whether it is in use on this machine, and why not.

    Runs the verification of anything that has not been verified yet in this
    process, so the answer is about the kernels as they would run now.
    """
    return [f"{name:18s}: {gate.describe()}" for name, gate in gates().items()]
