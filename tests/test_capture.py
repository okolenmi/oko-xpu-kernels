"""Capture readiness: the checks that keep a device graph honest.

A captured step replays whatever was inside it, every step, forever.  Three
things in this package can be wrong inside a capture, and each of them is
silent or expensive rather than fatal:

* a kernel's self-test (it synchronizes -- illegal during a capture),
* a fallback to the eager chain (correct, and slow, and it allocates),
* the INT8 backward's dequantized weight (a full-precision copy of every
  quantized weight, re-computed on every replay).

These tests are CPU-only: they check the *contract* (what raises, what is
reported), not the capture itself.  The capture itself is verified on the
card by ``debug/verify_graph_capture.py``.
"""

from unittest import mock

import pytest
import torch

import xpu_kernels
from xpu_kernels import capture, int8, triton


def _model():
    """A model with one INT8 linear in it.

    The weight is built by hand rather than with ``prepare_weight``: that
    needs the device library, and this is a CPU test.
    """
    w8 = torch.randint(-127, 128, (32, 512), dtype=torch.int8)
    scale = torch.full((32,), 0.01, dtype=torch.float32)
    mod = torch.nn.Module()
    mod.proj = int8.Int8Linear(w8, scale)
    return mod


def _disabled(name):
    """Make one package gate look switched off, and give it back."""
    gate = triton.gates()[name]
    saved = gate.disabled_by
    gate.disabled_by = "test"
    try:
        yield gate
    finally:
        gate.disabled_by = saved


# -- the one question everything else asks ---------------------------------

def test_is_capturing_is_false_and_never_raises():
    # no capture is running here, and the answer must not depend on a device
    assert xpu_kernels.is_capturing() is False


# -- the report ------------------------------------------------------------

def test_status_is_empty_when_the_kernels_are_active():
    # the Triton kernels verify on CPU as well, so this is the honest baseline:
    # nothing to report
    assert capture.capture_status(need=("rms_rope", "rms_adaln")) == []


def test_status_reports_a_disabled_kernel():
    gate = triton.gates()["rms_rope"]
    saved = gate.disabled_by
    gate.disabled_by = "test"
    try:
        problems = capture.capture_status(need=("rms_rope",))
    finally:
        gate.disabled_by = saved
    assert problems, "a switched-off kernel must be reported"
    assert any("rms_rope" in p and "test" in p for p in problems)


def test_status_rejects_an_unknown_kernel_name():
    # a typo must not read as "everything is fine"
    with pytest.raises(ValueError) as exc:
        capture.capture_status(need=("rms_ropee",))
    assert "rms_ropee" in str(exc.value)


def test_require_ready_raises_with_the_problems():
    gate = triton.gates()["rms_rope"]
    saved = gate.disabled_by
    gate.disabled_by = "test"
    try:
        with pytest.raises(RuntimeError) as exc:
            capture.require_capture_ready(need=("rms_rope",))
    finally:
        gate.disabled_by = saved
    assert "rms_rope" in str(exc.value)


def test_status_reports_a_missing_dense_weight():
    model = _model()
    problems = capture.capture_status(need=(), model=model)
    assert any("dense weight" in p for p in problems)


def test_prepare_materializes_the_dense_weight():
    model = _model()
    # the real dequant needs the device library; what is under test here is
    # the plumbing -- that prepare visits every Int8Linear and that the
    # status stops complaining afterwards
    def fake_dequant(w8, w_scale):
        return torch.zeros((w8.shape[0], w8.shape[1]), dtype=torch.bfloat16)

    with mock.patch("xpu_kernels.int8.ops.dequant_weight", fake_dequant):
        summary = capture.prepare_for_capture(model, need=())
    assert summary["dense_weights"] == 1
    problems = capture.capture_status(need=(), model=model)
    assert not [p for p in problems if "dense weight" in p]


# -- the refusals ----------------------------------------------------------

def test_the_gate_refuses_to_probe_inside_a_capture():
    gate = triton.gates()["rms_rope"]
    gate.reset()
    try:
        with mock.patch("xpu_kernels.triton._gate.is_capturing", return_value=True):
            with pytest.raises(RuntimeError) as exc:
                gate.ok()
        assert "capture" in str(exc.value)
        # and the verdict is still unverified, so a later call outside a
        # capture can still latch it
        assert gate.verified is None
    finally:
        gate.ok()          # re-latch, so the rest of the suite sees a verdict


def test_dense_weight_refuses_to_allocate_inside_a_capture():
    model = _model()
    with mock.patch("xpu_kernels.int8.ops.is_capturing", return_value=True):
        with pytest.raises(RuntimeError) as exc:
            model.proj.dense_weight()
    assert "capture" in str(exc.value)


def test_status_inside_a_capture_does_not_probe():
    # the probe synchronizes, so inside a capture an unverified kernel is
    # reported as unverified instead of being verified
    gate = triton.gates()["rms_rope"]
    gate.reset()
    try:
        with mock.patch("xpu_kernels.capture.is_capturing", return_value=True):
            problems = capture.capture_status(need=("rms_rope",))
        assert any("unverified" in p for p in problems)
    finally:
        gate.ok()
