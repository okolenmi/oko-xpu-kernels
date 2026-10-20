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

The discipline both follow, and the reason they are safe to drop into a model:
they verify themselves once per process against an established reference and
then either engage or fall back, permanently, for that process.  See
``set_reference`` in each module and ``status()`` below.

    import xpu_kernels.triton as k
    k.set_reference(k.rope, my_reference)      # your fallback path, once
    print(k.status())                          # active / inactive + why
"""

from __future__ import annotations

from . import adaln, rope

__all__ = ["adaln", "rope", "status", "set_reference"]


def set_reference(module, fn=None):
    """Set (or clear) the reference implementation for one kernel module.

    ``module`` is ``rope`` or ``adaln``; ``fn`` is the implementation that
    module falls back to, so the gate compares like with like.
    """
    module.set_reference(fn)


def status():
    """One line per kernel: whether it is in use on this machine, and why not."""
    return [f"rms_rope  : {rope.describe()}", f"rms_adaln : {adaln.describe()}"]
