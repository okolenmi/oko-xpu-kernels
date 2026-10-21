"""The rope backward's formula, checked in float64 without a GPU.

The fused backward cannot be bit-identical to ``torch.autograd`` of the eager
chain -- it sums a row in a different order -- so its acceptance test is two
levels:

* here, in float64, the analytic backward (:func:`reference.rms_rope_backward`)
  is checked against ``torch.autograd`` and against finite differences, which is
  a test of the *formula*;
* on the device, ``rope._GATE_BWD`` checks that the Triton kernels transcribe
  that formula (tolerance mode, worst relative error reported).

A wrong formula is off by ~1e-1; a transcription error is off by a reduction
order (~1e-5). The two levels are what separates those cases.
"""

from __future__ import annotations

import pytest
import torch

from xpu_kernels.triton import reference


def _pe(batch, seq, pairs, dtype=torch.float64, seed=0):
    gen = torch.Generator().manual_seed(seed)
    ang = torch.rand(batch, seq, 1, pairs, generator=gen, dtype=torch.float64) * 3.0
    ang = ang.to(dtype)
    return torch.stack([torch.stack([ang.cos(), -ang.sin()], -1),
                        torch.stack([ang.sin(), ang.cos()], -1)], -2)


def _rand(shape, dtype=torch.float64, seed=0, scale=0.5):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=gen, dtype=torch.float64) * scale).to(dtype)


@pytest.mark.parametrize("batch,seq,heads,dim", [
    (2, 3, 2, 8),        # several heads
    (1, 4, 3, 16),       # batch-1 freqs broadcast over the batch
    (3, 1, 1, 4),        # single head, dim 4
])
def test_analytic_backward_matches_autograd(batch, seq, heads, dim):
    x = _rand((batch, seq, heads, dim), seed=1).requires_grad_(True)
    pe = _pe(batch, seq, dim // 2, seed=2)
    scale = _rand((dim,), seed=3, scale=0.1) + 1.0

    y = reference.rms_rope_split_half(x, pe, scale)
    dy = _rand(y.shape, seed=4)
    y.backward(dy)

    dx = reference.rms_rope_backward(dy, x.detach(), pe, scale)
    torch.testing.assert_close(dx, x.grad, rtol=1e-12, atol=1e-12)


def test_analytic_backward_matches_finite_differences():
    """``gradcheck``: the formula against numerical differentiation, not itself."""

    class _WithAnalyticBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, pe, scale):
            ctx.save_for_backward(x, pe, scale)
            return reference.rms_rope_split_half(x, pe, scale)

        @staticmethod
        def backward(ctx, dy):
            x, pe, scale = ctx.saved_tensors
            return reference.rms_rope_backward(dy, x, pe, scale), None, None

    x = _rand((2, 2, 2, 8), seed=5).requires_grad_(True)
    pe = _pe(2, 2, 4, seed=6)
    scale = _rand((8,), seed=7, scale=0.2) + 0.9
    assert torch.autograd.gradcheck(
        lambda t: _WithAnalyticBackward.apply(t, pe, scale), (x,),
        eps=1e-6, atol=1e-9, rtol=1e-6)


def test_rotation_preserves_the_row_norm():
    """A property of the forward, independent of both implementations."""
    x = _rand((2, 3, 2, 8), seed=8)
    pe = _pe(2, 3, 4, seed=9)
    y = reference.rope_split_half(x, pe)
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), rtol=1e-12, atol=1e-12)


def test_normalised_rows_have_unit_rms():
    x = _rand((4, 32), seed=10)
    scale = torch.ones(32, dtype=torch.float64)
    # eps would make this sqrt(m/(m+eps)) instead of exactly 1, by ~eps/2m
    w = reference.rms_norm(x, scale, 0.0)
    torch.testing.assert_close(w.pow(2).mean(dim=-1).sqrt(),
                               torch.ones(4, dtype=torch.float64),
                               rtol=1e-6, atol=1e-9)


def test_backward_is_orthogonal_to_the_input_row():
    """The (w . dw) projection term must be there.

    For a pure scaling of x by 1/||x||, the Jacobian has x in its null space, so
    x . dx == 0 for any dy. Dropping the projection term leaves
    inv_rms * dw, and x . dw is not zero -- so this catches a missing term that
    the gate's tolerance would let through.
    """
    x = _rand((1, 2, 1, 8), seed=11)
    pe = _pe(1, 2, 4, seed=12)
    scale = _rand((8,), seed=13, scale=0.1) + 1.0
    dy = _rand(x.shape, seed=14)
    # eps=0 is the exact statement: with eps > 0 the identity picks up a
    # dot * eps / (m + eps) term, which is the argument for keeping eps tiny
    dx = reference.rms_rope_backward(dy, x, pe, scale, eps=0.0)
    assert dx.dtype == torch.float64
    assert (x * dx).sum(dim=-1).abs().max() < 1e-15
    # and the term is not trivially absent: the plain scaled dw is visible
    assert dx.abs().max() > 0
