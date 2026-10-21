"""The fused row-quantizer: the formula, pinned in float64.

The kernel replaces a torch chain that is the reference for the quantized
gradient path, so it has to compute the same thing.  It cannot be *bitwise*
the same -- the correctly rounded fp32 division that the chain uses costs 18x
more on this backend than the fast one the kernel uses, and a bit-exact fused
quantizer would be slower than the chain it replaces (``docs/performance.md``).

So the contract is the other way round: the **formula** is pinned exactly, in
float64, and the gate compares the kernel against the chain with a small
relative budget.  A wrong formula -- a missing clamp, the wrong scale, a
rotation that should not be there -- is off by ~1e-1, and cannot hide inside a
budget of 2e-3.

The row *scale* is a separate matter: it is exact, because it is a division
by a constant, and it is checked bit for bit.

The tests that need the kernel need a device: Triton's CPU backend wants the
Level Zero headers when torch has an XPU, and there is no XPU in CI.  The
device path is checked by ``python -m xpu_kernels`` (the self-test) instead.
"""

import pytest
import torch

from xpu_kernels import int8
from xpu_kernels.triton import quantize as Q

needs_a_device = pytest.mark.skipif(
    not (torch.xpu.is_available() or torch.cuda.is_available()),
    reason="the fused quantizer is a Triton kernel: it needs a device")


def _rows(M=8, K=64, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(M, K, generator=g) * 0.4).to(torch.bfloat16).to(device)


# -- what can be checked anywhere ------------------------------------------

def test_the_chain_is_the_formula():
    # the reference the gate compares against: the row maximum over 127, then
    # round to nearest, clamped to int8
    x = _rows()
    q, s = int8.quantize_row_torch(x)
    xf = x.double()
    scale = (xf.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-30)
    want = torch.round(xf / scale).clamp(-127, 127)
    assert torch.equal(q.to(torch.int16), want.to(torch.int16).to(torch.int16))
    assert int(q.min()) >= -127 and int(q.max()) <= 127
    assert s.shape == (x.shape[0],)


def test_the_budget_is_small():
    # the gate's budget must stay far below the error a wrong formula has
    assert 0 < Q.REL < 1e-2


# -- what needs a device ---------------------------------------------------

@needs_a_device
def test_the_row_scale_is_bit_exact():
    x = _rows(device="xpu")
    _q, s = Q.quantize_row(x)
    _wq, ws = int8.quantize_row_torch(x)
    assert s.view(torch.int32).equal(ws.view(torch.int32))


@needs_a_device
def test_the_values_are_within_one_unit_of_the_float64_formula():
    x = _rows(device="xpu")
    q, _s = Q.quantize_row(x)
    xf = x.double()
    scale = (xf.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-30)
    want = torch.round(xf / scale).clamp(-127, 127)     # float64: half to even
    d = (q.double() - want).abs()
    assert float(d.max()) <= 1.0
    # and the difference is rare -- the fast division only moves the values
    # that sit on a rounding boundary
    assert float((d > 0).to(torch.float32).mean()) < 1e-2


@needs_a_device
def test_ties_round_to_even():
    # a row whose maximum is 254 gives a scale of 2.0, so 127 lands on .5
    tie = torch.tensor([[127.0, 254.0, -254.0, 63.0]], dtype=torch.bfloat16,
                       device="xpu")
    q, _s = Q.quantize_row(tie)
    # 127/2 = 63.5 -> 64 (even), 254/2 = 127, -254/2 = -127, 63/2 = 31.5 -> 32
    assert q[0].tolist() == [64, 127, -127, 32]


@needs_a_device
def test_the_int8_api_agrees_with_the_chain_to_within_a_unit():
    x = _rows(M=16, K=256, seed=3, device="xpu")
    q, s = int8.quantize_row(x)
    wq, ws = int8.quantize_row_torch(x)
    assert s.view(torch.int32).equal(ws.view(torch.int32))
    d = (q.to(torch.int16) - wq.to(torch.int16)).abs()
    assert int(d.max()) <= 1


@needs_a_device
def test_the_gate_is_active():
    gate = Q.gates()["quantize_row"]
    assert gate.ok(), gate.describe()
