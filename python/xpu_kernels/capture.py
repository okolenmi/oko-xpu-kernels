"""Capture readiness: what must be true before a step is captured.

A device-graph capture freezes a region of the step and replays it.  Three
things in this package can be wrong inside such a region, and all three fail
*quietly* or expensively rather than loudly:

1. **A kernel's self-test inside the capture.**  Each fused kernel verifies
   itself once per process against a reference, and that probe synchronizes
   and reads values back -- which is illegal during a capture.  The gate
   therefore refuses to run inside one (:mod:`xpu_kernels.triton._gate`), and
   the fix is to let it latch its verdict *before* the capture starts.

2. **A fallback inside the capture.**  If the gate is closed, the eager
   reference runs instead.  That is correct, and it is also what nobody wants
   inside a graph: the rope backward allocates about ten tensors, and the
   allocation is not free in a graph either -- it comes from the graph's
   private pool and is replayed with the graph.

3. **The INT8 backward's dequantized weight.**  ``Int8Linear.dx`` needs the
   weight in bf16, which means a second copy of it (a full-precision copy of
   every quantized weight).  Materializing that copy *inside* the capture is
   not a correctness bug -- the copy is deterministic -- but it is a large
   allocation that is then re-computed on **every replay**: ~0.4 s per step
   for a 3.7 B model, against a step that is otherwise ~1 s.  So the copy is
   made before the capture, and the package refuses to make it inside one.

The API is three functions:

    import xpu_kernels

    xpu_kernels.prepare_for_capture(model, need=("rms_rope", "rms_adaln"))
    xpu_kernels.require_capture_ready(model, need=("rms_rope", "rms_adaln"))

``prepare_for_capture`` does the work (materialize the dense weights, latch the
gates' verdicts), ``require_capture_ready`` is the check to call right before
capturing, and ``capture_status`` returns the same information as a list of
problems instead of raising.

Everything here must be called **outside** a capture: it is the thing you call
to get ready for one.
"""

from __future__ import annotations

from typing import Iterable

from ._capture import is_capturing

__all__ = ["capture_status", "is_capturing", "prepare_for_capture",
           "require_capture_ready"]


def _triton_gates():
    from . import triton

    return triton.gates()


def _int8_modules(model):
    """Every ``Int8Linear`` in a model, by dotted name."""
    from .int8 import Int8Linear

    if model is None:
        return []
    if isinstance(model, Int8Linear):
        return [(getattr(model, "_name", "int8_linear"), model)]
    out = []
    for name, module in model.named_modules():
        if isinstance(module, Int8Linear):
            out.append((name, module))
    return out


def capture_status(need: Iterable[str] | None = None,
                   model=None) -> list[str]:
    """Everything that would make a captured step wrong, slow, or wasteful.

    Returns a list of problem strings; an empty list means the step is ready to
    be captured.  ``need`` names the fused kernels the step intends to use
    (``"rms_rope"``, ``"rms_adaln"``, their ``*_backward`` variants); ``None``
    means all of them.  ``model`` is an optional ``nn.Module`` whose
    ``Int8Linear`` layers are checked for a materialized dense weight.

    This *verifies* the kernels it is asked about, which is the point: the
    answer is about the kernels as they would run now, not as they were
    configured.  It must therefore be called outside a capture -- inside one,
    the verification itself would be illegal, and the kernels are reported as
    unverified instead.
    """
    problems: list[str] = []

    gates = _triton_gates()
    if need is None:
        wanted = list(gates)
    else:
        wanted = list(need)
        unknown = [n for n in wanted if n not in gates]
        if unknown:
            # A typo must not read as "everything is fine".
            raise ValueError(
                f"unknown kernel(s) {unknown!r}; this package has "
                f"{sorted(gates)}")

    capturing = is_capturing()
    for name in wanted:
        gate = gates[name]
        if gate.verified is None and capturing:
            # Do not probe inside a capture: the probe synchronizes.
            problems.append(
                f"{name}: unverified -- the self-test cannot run inside a "
                f"capture; call prepare_for_capture() before capturing")
            continue
        if not gate.ok():
            problems.append(f"{name}: {gate.describe()}")

    for name, module in _int8_modules(model):
        if getattr(module, "_dense", None) is None:
            problems.append(
                f"{name}: the dense weight is not materialized -- the dequant "
                f"would be captured and re-run on every replay; call "
                f"prepare_for_capture(model) first")

    return problems


def require_capture_ready(need: Iterable[str] | None = None,
                          model=None) -> None:
    """Raise ``RuntimeError`` unless the step is ready to be captured.

    The check the trainer puts in its init: a captured step that silently fell
    back to the eager chain is slower than an eager step and looks like a
    regression nobody can explain.
    """
    problems = capture_status(need=need, model=model)
    if problems:
        raise RuntimeError(
            "not ready to capture a step:\n  " + "\n  ".join(problems))


def prepare_for_capture(model=None, need: Iterable[str] | None = None) -> dict:
    """Get a step ready to be captured, and report what was done.

    * materializes the dense (bf16) weight of every ``Int8Linear`` in
      ``model``, so the dequant is not captured and not replayed;
    * runs the self-test of every kernel in ``need``, so the gate's verdict is
      latched before the capture starts.

    Returns a small summary dict.  Call it outside a capture.
    """
    gates = _triton_gates()
    if need is None:
        wanted = list(gates)
    else:
        wanted = list(need)
        unknown = [n for n in wanted if n not in gates]
        if unknown:
            raise ValueError(
                f"unknown kernel(s) {unknown!r}; this package has "
                f"{sorted(gates)}")

    verified = []
    for name in wanted:
        if gates[name].ok():
            verified.append(name)

    materialized = 0
    for _name, module in _int8_modules(model):
        module.dense_weight()
        materialized += 1

    return {"verified": verified, "dense_weights": materialized,
            "problems": capture_status(need=wanted, model=model)}
