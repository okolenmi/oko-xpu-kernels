"""The adaLN backward's formula, checked in float64 without a GPU.

Same two levels as the rope backward (`tests/test_rope_backward.py`): here the
analytic gradient in `reference` is checked against ``torch.autograd`` and
finite differences, which tests the *formula*; on the device,
``adaln._GATE_BWD`` checks that the Triton kernel transcribes it.

adaLN adds two things the rope backward does not have, and they are what these
tests are mostly about:

* ``dscale`` and ``dshift`` are reductions over the rows the modulation was
  broadcast across, and ``torch.autograd`` gives a broadcast input the shape it
  was *passed* in. A kernel that returns the wrong shape, or that forgets to
  sum, produces an optimizer update that silently trains the wrong thing.
* ``dx`` must not depend on ``shift`` (a shift only translates the output), and
  ``dx`` must be orthogonal to ``v = x * rstd`` -- the norm's vjp property. Both
  catch a missing or doubled term.
"""

from __future__ import annotations

import pytest
import torch

from xpu_kernels.triton import reference

#: shapes a trainer actually passes: a per-block modulation, a per-dim one, a
#: full one, and a broadcast over the batch
SHAPES = [(1, 1, 16), (16,), (2, 5, 16), (1, 5, 16), (2, 1, 16)]


def _rand(shape, seed=0, scale=0.5):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64) * scale


def _max_abs(a, b):
    # detach: some callers pass tensors that require grad, and float() on one
    # is only a warning -- but a warning in the suite hides the next one
    return float((a.detach() - b.detach()).abs().max())


@pytest.mark.parametrize("mod_shape", SHAPES)
def test_analytic_backward_matches_autograd(mod_shape):
    x = _rand((2, 5, 16), seed=1).requires_grad_(True)
    scale = _rand(mod_shape, seed=2, scale=0.1).requires_grad_(True)
    shift = _rand(mod_shape, seed=3, scale=0.1).requires_grad_(True)
    dy = _rand((2, 5, 16), seed=4)

    gx, gs, gh = torch.autograd.grad(reference.rms_adaln(x, scale, shift), (x, scale, shift), dy)
    dx, dscale, dshift = reference.rms_adaln_backward(dy, x, scale, shift)

    assert _max_abs(dx, gx) < 1e-12
    assert _max_abs(dscale, gs) < 1e-12
    assert _max_abs(dshift, gh) < 1e-12


@pytest.mark.parametrize("mod_shape", SHAPES)
def test_gradient_shapes_follow_the_input_shapes(mod_shape):
    x = _rand((2, 5, 16), seed=1)
    scale = _rand(mod_shape, seed=2, scale=0.1)
    shift = _rand(mod_shape, seed=3, scale=0.1)
    dy = _rand(x.shape, seed=4)

    dx, dscale, dshift = reference.rms_adaln_backward(dy, x, scale, shift)
    assert dx.shape == x.shape
    assert dscale.shape == torch.Size(mod_shape)
    assert dshift.shape == torch.Size(mod_shape)


def test_gradcheck_against_finite_differences():
    """The formula against finite differences, not just against autograd.

    autograd of the chain and the analytic backward could agree and both be
    wrong (a misread of which tensor is the modulation); finite differences
    cannot.
    """
    x = _rand((2, 3, 8), seed=5).requires_grad_(True)
    scale = _rand((1, 1, 8), seed=6, scale=0.1).requires_grad_(True)
    shift = _rand((1, 1, 8), seed=7, scale=0.1).requires_grad_(True)

    def f(x_, s_, h_):
        return reference.rms_adaln(x_, s_, h_)

    assert torch.autograd.gradcheck(f, (x, scale, shift), eps=1e-6, atol=1e-8)


@pytest.mark.parametrize("mod_shape", SHAPES)
def test_dx_is_orthogonal_to_the_normalized_value(mod_shape):
    """dx . v == 0 for every row: the norm's vjp removes the radial part.

    Only exact with ``eps = 0``.  With eps the normalised row is not exactly
    unit RMS (``|v|^2 = D |x|^2 / (|x|^2 + D eps)``), so a residual of order
    ``eps * D / |x|^2`` remains -- which is why the eps=0 assertion below is
    tight and the eps=1e-6 one is only relative.
    """
    x = _rand((2, 5, 16), seed=8)
    scale = _rand(mod_shape, seed=9, scale=0.1)
    shift = _rand(mod_shape, seed=10, scale=0.1)
    dy = _rand(x.shape, seed=11)

    dx, _, _ = reference.rms_adaln_backward(dy, x, scale, shift, eps=0.0)
    v = reference.rms_adaln_normalized(x, eps=0.0)
    # 1e-12: roundoff at the scale of dx itself (~1), still six orders below a
    # missing or doubled projection term
    assert _max_abs((dx * v).sum(dim=-1), torch.zeros(2, 5)) < 1e-12
    assert dx.abs().max() > 0           # and the term is not trivially absent

    # with the production eps the same quantity is small but not zero
    dx_e, _, _ = reference.rms_adaln_backward(dy, x, scale, shift, eps=1e-6)
    v_e = reference.rms_adaln_normalized(x, eps=1e-6)
    resid = (dx_e * v_e).sum(dim=-1).abs().max()
    assert resid < 1e-3 * dx_e.abs().max() * v_e.abs().max()


def test_shift_does_not_affect_dx_or_dscale():
    x = _rand((2, 5, 16), seed=12)
    scale = _rand((1, 1, 16), seed=13, scale=0.1)
    dy = _rand(x.shape, seed=14)
    a = _rand((1, 1, 16), seed=15, scale=0.1)
    b = _rand((1, 1, 16), seed=16, scale=2.0)

    dx_a, ds_a, dh_a = reference.rms_adaln_backward(dy, x, scale, a)
    dx_b, ds_b, dh_b = reference.rms_adaln_backward(dy, x, scale, b)
    assert _max_abs(dx_a, dx_b) == 0.0
    assert _max_abs(ds_a, ds_b) == 0.0
    # ... but dshift is dy, summed, regardless of shift
    assert _max_abs(dh_a, dh_b) == 0.0
    ref = _rand(x.shape, seed=14)
    assert _max_abs(dh_a, ref.sum(dim=(0, 1), keepdim=True)) < 1e-12


def test_dshift_is_dy_summed_over_the_broadcast_axes():
    x = _rand((3, 4, 8), seed=17)
    scale = _rand((1, 1, 8), seed=18, scale=0.1)
    shift = _rand((1, 1, 8), seed=19, scale=0.1)
    dy = _rand(x.shape, seed=20)

    _, _, dshift = reference.rms_adaln_backward(dy, x, scale, shift)
    assert _max_abs(dshift, dy.sum(dim=(0, 1), keepdim=True)) < 1e-12

    # a full-shape modulation gets a per-element gradient, not a reduction
    full = _rand((3, 4, 8), seed=21, scale=0.1)
    _, _, dshift_full = reference.rms_adaln_backward(dy, x, scale, full)
    assert dshift_full.shape == full.shape
    assert _max_abs(dshift_full, dy) < 1e-12


def test_scale_gradient_uses_the_normalized_value_not_x():
    """dscale = sum(dy * v), with v = x * rstd -- not sum(dy * x)."""
    x = _rand((2, 3, 8), seed=22)
    scale = _rand((8,), seed=23, scale=0.1)
    shift = _rand((8,), seed=24, scale=0.1)
    dy = _rand(x.shape, seed=25)

    _, dscale, _ = reference.rms_adaln_backward(dy, x, scale, shift)
    v = reference.rms_adaln_normalized(x)
    assert _max_abs(dscale, (dy * v).sum(dim=(0, 1))) < 1e-12
    assert _max_abs(dscale, (dy * x).sum(dim=(0, 1))) > 1e-3
