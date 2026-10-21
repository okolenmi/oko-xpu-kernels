"""The bit-exactness gate every fused kernel in this package goes through.

A fused kernel is only allowed to run if it has been proved, on *this machine
and this process*, to produce the same bits as the implementation it replaces.
That is a stronger requirement than a numerical tolerance, and it is what makes
a package like this safe to drop into somebody else's model: an image pipeline
is chaotic in its q/k values (one ULP in a norm scale flips a bf16 rounding and
the picture with it), and a training run accumulates the error into every
weight instead.

The rule the gate encodes:

* **Compare against the thing being replaced, not against a tidy formula.**
  ``torch.autograd`` of the eager chain for a backward, the established kernel
  for a forward. A hand-written reference can be wrong in the same way the
  kernel is wrong, and then the comparison proves nothing.
* **Verify once, latch the verdict, fall back hard.** A mismatch disables the
  kernel for the rest of the process; it does not retry on the next shape,
  because the failure mode we care about is intermittent.
* **Never re-verify silently after a config change.** The gate compares whatever
  is deployed *now*, so changing ``num_warps`` or the reduction order is a code
  change that the gate will catch on the next run — it is not something to tune
  by eye.

A gate has no device assumptions: the probe hands it tensors and it compares
bit patterns, so the gate logic itself can be unit-tested on CPU (``tests/``).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Callable, Mapping

import torch

log = logging.getLogger(__name__)

__all__ = ["ACCELERATORS", "Gate", "bit_equal_count", "probe_device",
           "relative_error"]

_UNSET = object()

# Devices these kernels are willing to run on.  The Triton kernels are not
# XPU-specific code -- they are schedule changes over the same arithmetic --
# but only an accelerator has a Triton backend, so the guards accept the two
# torch accelerator types and let the gate do the rest.
ACCELERATORS = ("xpu", "cuda")

_BITS = {
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
    torch.float32: torch.int32,
    torch.int32: torch.int32,
    torch.float64: torch.int64,
    torch.int64: torch.int64,
}


@lru_cache(maxsize=1)
def probe_device() -> str:
    """The device a probe builds its own inputs on.

    Probes must not depend on the caller's tensors (they run once, before the
    first call), so they pick the accelerator this process has.  CPU is the
    last resort: the probes then report a launch failure and the gates stay
    closed, which is the correct outcome for a Triton kernel with no GPU.
    """
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _bits(t: torch.Tensor) -> torch.Tensor:
    """The raw bit pattern of a tensor, so equality means equality."""
    dtype = _BITS.get(t.dtype)
    if dtype is None:
        raise TypeError(f"gate: cannot compare {t.dtype} bitwise")
    return t.reshape(-1).view(dtype)


def bit_equal_count(got: torch.Tensor, want: torch.Tensor) -> tuple[int, int]:
    """``(differing, total)`` elements, comparing bit patterns.

    Shapes must match: a kernel that returns the right numbers in the wrong
    order is not correct, and silently broadcasting the comparison would hide it.
    """
    if got.shape != want.shape:
        raise ValueError(f"gate: shape {tuple(got.shape)} vs {tuple(want.shape)}")
    if got.dtype != want.dtype:
        raise TypeError(f"gate: dtype {got.dtype} vs {want.dtype}")
    a, b = _bits(got), _bits(want)
    if a.numel() != b.numel():
        raise ValueError("gate: element count differs")
    return int((a != b).sum()), int(a.numel())


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    """L2 relative error ``||got - want|| / ||want||``, in float64.

    The criterion for a reference that is the same arithmetic with a different
    reduction order (a torch implementation of a Triton kernel, or
    ``torch.autograd`` of the eager chain): bitwise equality is not attainable,
    so the gate needs a budget.  It is deliberately *relative to the whole
    tensor*, not per element: a per-element tolerance has to be loose enough for
    the values near zero, which makes it blind to a systematic error elsewhere.
    """
    if got.shape != want.shape:
        raise ValueError(f"gate: shape {tuple(got.shape)} vs {tuple(want.shape)}")
    a, b = got.double(), want.double()
    norm = float(b.norm())
    diff = float((a - b).norm())
    return diff / norm if norm > 0 else diff


class Gate:
    """One fused kernel's permission slip.

    ``probe`` is called as ``probe(reference)`` and must return a mapping of
    ``name -> (got, want)``.  Returning several pairs is normal -- a forward and
    a backward, or q and k -- and the gate fails if *any* pair differs, so the
    two halves of an autograd pair are verified together rather than separately.

    ``default_reference`` resolves the reference when nothing was injected: for
    kernels extracted from ComfyUI that is the ``comfy_kitchen`` implementation,
    which is what the kernel was tuned against.  Outside that environment the
    caller supplies one with :meth:`set_reference`; without any reference the
    gate stays closed and :meth:`describe` says so.

    The gate is bit-exact by default.  When the reference is the same arithmetic
    with a *different reduction order* -- a torch implementation of a Triton
    kernel, or ``torch.autograd`` of the eager chain for a backward -- pass
    ``rel=<budget>`` (to the constructor or to :meth:`set_reference`) and the gate
    compares the L2 relative error instead.  That is a transcription check, not a
    proof: the formula itself is pinned in float64 by the tests, and the budget
    must be small enough that a wrong formula (which is off by ~1e-1, not 1e-3)
    cannot hide inside it.
    """

    def __init__(
        self,
        name: str,
        probe: Callable[[Callable], Mapping[str, tuple]],
        *,
        default_reference: Callable[[], Callable] | None = None,
        env: str | None = None,
        legacy_env: str | None = None,
        num_warps: int | None = None,
        rel: float | None = None,
    ):
        self.name = name
        self.probe = probe
        self.default_reference = default_reference
        self.num_warps = num_warps
        self.rel = rel
        self.reference: Callable | None = None
        self._ok: bool | None = None
        self._why: str | None = None
        self.disabled_by: str | None = None
        for flag in (env, legacy_env):
            if flag and os.environ.get(flag, "") not in ("", "0"):
                self.disabled_by = f"{flag} is set"
                break

    # -- reference ---------------------------------------------------------

    def set_reference(self, fn: Callable | None, rel: float | None = _UNSET) -> None:
        """Inject the implementation this kernel must match, and re-arm the gate.

        The reference must be what the caller falls back to, with the same
        signature as the fused kernel -- otherwise a pass says nothing.  Pass
        ``rel`` when the reference computes the same arithmetic with a different
        reduction order (a torch implementation, ``torch.autograd`` of the eager
        chain): bitwise equality is then not attainable and the gate compares
        with that relative budget instead.  Leaving it out keeps the gate
        bit-exact, which is the default and the stronger claim.
        """
        self.reference, self._ok, self._why = fn, None, None
        if rel is not _UNSET:
            self.rel = rel

    # -- verdict -----------------------------------------------------------

    def ok(self) -> bool:
        """Verify on first call, then return the latched verdict."""
        if self.disabled_by:
            return False
        if self._ok is not None:
            return self._ok
        self._ok = False
        try:
            ref = self.reference or self.default_reference()
        except Exception as exc:  # noqa: BLE001 - any import failure means "none"
            self._why = f"unverified: no reference implementation ({exc})"
            log.warning("%s: %s", self.name, self._why)
            return False
        if ref is None:
            self._why = "unverified: no reference implementation (call set_reference)"
            log.warning("%s: %s", self.name, self._why)
            return False
        # The comparison lives inside the same guard as the probe: a probe that
        # returns the wrong shape or dtype has not run, and that is a failure,
        # not an exception the caller has to know how to handle.
        try:
            pairs = self.probe(ref)
            bad = total = 0
            worst = 0.0
            for got, want in pairs.values():
                if self.rel is None:
                    differing, n = bit_equal_count(got, want)
                    bad += differing
                    total += n
                else:
                    worst = max(worst, relative_error(got, want))
                    total += got.numel()
        except Exception as exc:  # noqa: BLE001 - a broken probe is not a pass
            self._why = f"probe failed: {exc}"
            log.warning("%s: self-test failed (%s)", self.name, exc)
            return False
        if bad:
            self._why = f"mismatch: {bad}/{total} values differ"
            log.warning("%s: self-test found %d differing values, falling back",
                        self.name, bad)
            return False
        if self.rel is not None and worst > self.rel:
            self._why = (f"mismatch: worst relative error {worst:.2e} "
                         f"> tolerance {self.rel:g}")
            log.warning("%s: %s, falling back", self.name, self._why)
            return False
        self._ok = True
        if self.rel is None:
            self._why = f"verified against the reference ({total} values)"
        else:
            self._why = (f"verified within {self.rel:g} "
                         f"(worst relative error {worst:.1e}, {total} values)")
        log.info("%s: %s", self.name, self._why)
        return True

    def describe(self) -> str:
        """One line for a status report; runs the verification if needed."""
        self.ok()
        if self.disabled_by:
            return f"disabled: {self.disabled_by}"
        verdict = "active" if self._ok else "inactive"
        warps = f", num_warps={self.num_warps}" if self.num_warps else ""
        return f"{verdict}: {self._why}{warps}"

    def reset(self) -> None:
        """Forget the verdict (tests only; production latches it)."""
        self._ok = self._why = None
