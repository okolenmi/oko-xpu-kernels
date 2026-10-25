"""The two-pass weight dequantizer without a device: the identity it rests on, and that it is refused where it cannot be verified.

Rotating each 256-element group is a product with one fixed orthogonal symmetric matrix, so ``view(-1, 256) @ R`` is the
group-wise rotation of the whole weight (checked on a Sylvester-Hadamard stand-in; the real matrix is the SYCL rotation applied to
the identity on a device and is gated there).
"""

import pytest
import torch

from xpu_kernels.triton import dequant as D


def _hadamard(n):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n ** 0.5


def test_rotation_matrix_is_an_involution():
    r = _hadamard(256)
    assert torch.allclose(r, r.T)
    assert torch.allclose(r @ r, torch.eye(256, dtype=torch.float64), atol=1e-12)


def test_one_product_rotates_every_group():
    r = _hadamard(256)
    w = torch.randn(4, 1024, dtype=torch.float64)
    by_view = (w.reshape(-1, 256) @ r).reshape(4, 1024)
    by_groups = torch.cat([w[:, g * 256:(g + 1) * 256] @ r for g in range(4)], dim=1)
    assert torch.allclose(by_view, by_groups, atol=1e-12)


def test_without_a_device_it_is_refused():
    w8 = torch.zeros(16, 256, dtype=torch.int8)
    assert not D.usable(w8)
    with pytest.raises(RuntimeError):
        D.dequant_weight(w8, torch.ones(16))
