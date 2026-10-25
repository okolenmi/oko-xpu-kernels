"""The fused GEGLU: the formula its backward kernel writes out, pinned in float64, and the gates' refusal on a machine without a device.

The kernel's backward spells the derivative of ``x * gelu(gate)`` out by hand (``cdf + x * pdf`` with its own constants), so a
wrong constant would be a wrong gradient that still looks plausible.  The transcription below uses the kernel's constants and is
compared with autograd of the eager chain in float64, where no rounding can hide a mistake.  The device path is checked by the
gates (``python -m xpu_kernels``) and ``debug/verify_geglu.py``.
"""

import math

import pytest
import torch

from xpu_kernels.triton import geglu as G

SQRT_HALF = 0.7071067811865476
BETA = 0.3989422804014327  # the kernel's pdf constant: 1 / sqrt(2 pi)


def _kernel_formula_backward(dy, h):
    """What ``_geglu_backward_kernel`` computes, in the dtype of ``h`` (no intermediate rounding)."""
    x, gate = h.chunk(2, dim=-1)
    g = gate * 0.5 * (1.0 + torch.erf(gate * SQRT_HALF))
    cdf = 0.5 * (1.0 + torch.erf(gate * SQRT_HALF))
    pdf = torch.exp(gate * gate * -0.5) * BETA
    return torch.cat([dy * g, dy * x * (cdf + gate * pdf)], dim=-1)


def test_constants_are_the_gaussian_ones():
    assert SQRT_HALF == pytest.approx(math.sqrt(0.5), rel=1e-15)
    assert BETA == pytest.approx(1 / math.sqrt(2 * math.pi), rel=1e-15)


def test_backward_formula_matches_autograd_in_float64():
    gen = torch.Generator().manual_seed(0)
    h = torch.randn(16, 64, generator=gen, dtype=torch.float64) * 2
    dy = torch.randn(16, 32, generator=gen, dtype=torch.float64)
    want = G.chain_backward(dy, h)
    got = _kernel_formula_backward(dy, h)
    assert torch.allclose(got, want, rtol=1e-12, atol=1e-12)


def test_chain_backward_is_the_gradient_of_the_chain():
    h = torch.randn(4, 16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(G.chain, (h,))


def test_without_a_device_the_fused_op_is_refused():
    h = torch.randn(2, 8, dtype=torch.bfloat16)
    assert not G.usable(h)
    with pytest.raises(RuntimeError):
        G.geglu(h)


def test_odd_width_is_not_usable():
    assert not G.usable(torch.randn(2, 7))
