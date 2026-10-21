"""Do the capture guards actually fire inside a real capture?

The package refuses two things inside a device-graph capture: a kernel's
self-test (it synchronizes, which is illegal during a capture) and the INT8
backward's dequantized weight (a full-precision copy of the weight, which
would be re-computed on every replay).  Both refusals are unit-tested on CPU
(``tests/test_capture.py``), but a CPU test cannot capture anything -- so this
script captures on the card and checks that the guard is what stops it, not
some unrelated synchronisation error.

It also checks the other half: that after ``prepare_for_capture`` the same
capture *does* work, which is the property the trainer depends on.

    PYTHONPATH=python LEVEL_ZERO_V1_SDK_PATH=~/.local/levelzero \\
        python debug/verify_capture_guards.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import xpu_kernels  # noqa: E402
from xpu_kernels import int8  # noqa: E402
from xpu_kernels.triton import rope  # noqa: E402

DEV = "xpu"


def _rope_inputs():
    q = torch.randn(1, 64, 24, 128, device=DEV).to(torch.bfloat16)
    k = torch.randn_like(q)
    pe = torch.randn(1, 64, 1, 64, 2, 2, device=DEV).to(torch.bfloat16)
    scale = torch.randn(24, 128, device=DEV).to(torch.bfloat16)
    return q, k, pe, scale


def main():
    print(f"device: {torch.xpu.get_device_name(0)}")
    failures = []

    # -- 1. the gate must not probe inside a capture ------------------------
    # The path a caller actually takes: ask the module whether the kernel is
    # usable.  That call consults the gate, and the gate must refuse to probe
    # here (the probe synchronizes).
    gate = rope.gates()["rms_rope"]
    gate.reset()                     # unverified again: the next call probes
    q, k, pe, scale = _rope_inputs()
    g = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(g):
            rope.usable(q, pe)
        failures.append("the gate probed inside a capture without refusing")
        print("  usable() inside a capture -> NO REFUSAL (bad)")
    except RuntimeError as exc:
        msg = str(exc)
        if "capture" in msg:
            print(f"  usable() inside a capture -> RuntimeError: {msg}")
        else:
            failures.append(f"the capture failed for the wrong reason: {msg}")
            print(f"  usable() inside a capture -> RuntimeError (wrong reason): {msg}")

    # A finding, not a failure: the fused entry points do not consult their
    # gate themselves -- the *caller* is expected to ask `usable()` first.  A
    # caller that skips that question gets an unverified kernel, silently.
    # `prepare_for_capture` is the supported way to be sure.
    print("  note: rope.rms_rope_split_half() does not consult the gate "
          "itself; the caller must ask usable() first (see docs/todo.md)")

    # -- 2. the dense weight must not be materialized inside a capture ------
    w = torch.randn(32, 512, device=DEV).to(torch.bfloat16) * 0.02
    w8, w_scale = xpu_kernels.prepare_weight(w)
    mod = int8.Int8Linear(w8, w_scale)
    g2 = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(g2):
            mod.dense_weight()
        failures.append("the dense weight was allocated inside a capture")
        print("  dense weight inside a capture -> NO REFUSAL (bad)")
    except RuntimeError as exc:
        msg = str(exc)
        if "capture" in msg:
            print(f"  dense weight inside a capture -> RuntimeError: {msg}")
        else:
            failures.append(f"the capture failed for the wrong reason: {msg}")
            print(f"  dense weight inside a capture -> RuntimeError (wrong reason): {msg}")

    # -- 3. after preparing, the same capture works -------------------------
    model = torch.nn.Module()
    model.proj = mod
    summary = xpu_kernels.prepare_for_capture(model, need=("rms_rope",))
    print(f"  prepare_for_capture: verified={summary['verified']} "
          f"dense_weights={summary['dense_weights']} "
          f"problems={summary['problems']}")
    if summary["problems"]:
        failures.append(f"prepare_for_capture left problems: {summary['problems']}")

    dy = torch.randn(8, 32, device=DEV).to(torch.bfloat16)
    dx_eager = mod.dx(dy)
    g3 = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(g3):
            dx = mod.dx(dy)
        g3.replay()
        torch.xpu.synchronize()
        same = dx.view(torch.int16).equal(dx_eager.view(torch.int16))
        print(f"  capture of Int8Linear.dx: ok, dx {tuple(dx.shape)} {dx.dtype}, "
              f"replay {'identical' if same else 'DIFFERS'} to eager")
        if not same:
            failures.append("the captured INT8 backward differs from the eager one")
    except Exception as exc:  # noqa: BLE001 - the failure mode is the finding
        failures.append(f"the INT8 backward could not be captured: {exc}")
        print(f"  capture of Int8Linear.dx -> {type(exc).__name__}: {exc}")

    # -- 4. and the status is clean ----------------------------------------
    problems = xpu_kernels.capture_status(need=("rms_rope",), model=model)
    print(f"  capture_status: {problems if problems else 'clean'}")
    if problems:
        failures.append(f"capture_status is not clean: {problems}")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        raise SystemExit(1)
    print("all capture guards behave as documented")


if __name__ == "__main__":
    main()
