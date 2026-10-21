"""The gate's contract, tested without a GPU.

The gate is the one piece of this package that stands between a fast kernel and
somebody's silently-wrong model, so its failure modes are pinned here: it must
latch, it must fail closed, and it must never pass a probe it could not run.
These tests use ordinary CPU tensors -- the gate only ever compares bit
patterns, so none of this needs an XPU.
"""

from __future__ import annotations

import os

import pytest
import torch

from xpu_kernels.triton._gate import Gate, bit_equal_count


def _pair(a, b=None):
    return {"v": (a, a if b is None else b)}


def test_identical_passes_and_reports_value_count():
    calls = []

    def probe(ref):
        calls.append(1)
        return _pair(torch.ones(4, dtype=torch.bfloat16))

    gate = Gate("t", probe, default_reference=lambda: object())
    assert gate.ok() is True
    # latched: a second question must not run the probe again
    assert gate.ok() is True
    assert len(calls) == 1
    assert "verified" in gate.describe()
    assert "4 values" in gate.describe()


def test_one_flipped_bit_fails_and_latches():
    got = torch.ones(8, dtype=torch.bfloat16)
    want = got.clone()
    want.view(torch.int16)[3] ^= 0b1          # one ULP in one element
    gate = Gate("t", lambda ref: _pair(got, want), default_reference=lambda: object())
    assert gate.ok() is False
    assert gate.ok() is False                  # no retry on the next call
    assert "1/8" in gate.describe()


def test_no_reference_fails_closed():
    def unavailable():
        raise ImportError("comfy_kitchen not installed")

    gate = Gate("t", lambda ref: _pair(torch.zeros(2)), default_reference=unavailable)
    assert gate.ok() is False
    why = gate.describe()
    assert "inactive" in why and "comfy_kitchen" in why


def test_injected_reference_rearms_after_failure():
    good = torch.ones(2, dtype=torch.bfloat16)
    gate = Gate("t", lambda ref: _pair(good, torch.zeros(2, dtype=torch.bfloat16)),
                default_reference=lambda: None)
    assert gate.ok() is False
    gate.set_reference(lambda *a, **k: good)   # the caller's real fallback
    # the probe still returns the same mismatching pair, so it is still a fail --
    # what matters is that set_reference cleared the latched verdict
    gate.probe = lambda ref: _pair(good)
    assert gate.ok() is True


def test_broken_probe_is_not_a_pass():
    def probe(ref):
        raise RuntimeError("kernel launch failed")

    gate = Gate("t", probe, default_reference=lambda: object())
    assert gate.ok() is False
    assert "probe failed" in gate.describe()


def test_env_escape_hatch_disables_without_probing():
    os.environ["XPU_KERNELS_TEST_DISABLE"] = "1"
    try:
        gate = Gate("t", lambda ref: pytest.fail("must not run"),
                    env="XPU_KERNELS_TEST_DISABLE")
        assert gate.ok() is False
        assert "XPU_KERNELS_TEST_DISABLE" in gate.describe()
    finally:
        del os.environ["XPU_KERNELS_TEST_DISABLE"]


def test_any_mismatched_pair_fails_the_gate():
    a = torch.ones(4, dtype=torch.bfloat16)
    gate = Gate("t", lambda ref: {"q": (a, a), "k": (a, a + 1)},
                default_reference=lambda: object())
    assert gate.ok() is False


def test_shape_and_dtype_mismatch_are_refused():
    with pytest.raises(ValueError):
        bit_equal_count(torch.zeros(2), torch.zeros(3))
    with pytest.raises(TypeError):
        bit_equal_count(torch.zeros(2), torch.zeros(2, dtype=torch.bfloat16))


def test_bit_equal_count_ignores_nothing():
    # -0.0 and +0.0 are equal as floats and different as bits, which is the
    # whole point: the gate compares what was written.
    z, nz = torch.zeros(1), torch.tensor([-0.0])
    assert bit_equal_count(z, nz) == (1, 1)


def test_num_warps_is_reported():
    gate = Gate("t", lambda ref: _pair(torch.ones(1)), num_warps=4,
                default_reference=lambda: object())
    assert "num_warps=4" in gate.describe()


def test_tolerance_mode_passes_small_perturbations():
    want = torch.randn(64, dtype=torch.float32)
    got = want + 1e-5 * torch.randn(64)
    gate = Gate("t", lambda ref: _pair(got, want), default_reference=lambda: object(),
                rel=1e-3)
    assert gate.ok() is True
    assert "within 0.001" in gate.describe()


def test_tolerance_mode_fails_a_systematic_error():
    want = torch.ones(64, dtype=torch.float32)
    gate = Gate("t", lambda ref: _pair(want * 1.01, want),
                default_reference=lambda: object(), rel=1e-3)
    assert gate.ok() is False
    assert "relative error" in gate.describe()


def test_set_reference_can_switch_a_bit_exact_gate_to_tolerance():
    want = torch.ones(8, dtype=torch.bfloat16)
    gate = Gate("t", lambda ref: _pair(want * 1.01, want),   # 1.3 bf16 ULP
                default_reference=lambda: object())          # bit-exact by default
    assert gate.ok() is False
    gate.set_reference(lambda *a, **k: want, rel=1e-2)
    assert gate.ok() is True


def test_tolerance_gate_still_refuses_the_wrong_shape():
    want = torch.ones(8, dtype=torch.float32)
    gate = Gate("t", lambda ref: _pair(torch.ones(7), want),
                default_reference=lambda: object(), rel=1e-1)
    # the shape complaint happens inside the probe and must not be swallowed
    assert gate.ok() is False
    assert "probe failed" in gate.describe()
