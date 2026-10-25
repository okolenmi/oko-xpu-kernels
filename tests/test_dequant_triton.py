"""The Triton weight dequantizer: what can be checked without a device (it is refused there), and the matrix identity it rests on.

The kernel multiplies each 256-element group by the library's rotation matrix instead of running the butterfly.  That is the same
function only if the rotation is a linear map that is its own inverse, so the matrix is symmetric and orthogonal; this checks the
identity on a Sylvester-Hadamard stand-in (the real matrix comes from the SYCL rotation on a device and is gated there).
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


def test_group_product_is_the_rotation_of_the_group():
    r = _hadamard(256)
    w = torch.randn(4, 512, dtype=torch.float64)
    by_groups = (w.reshape(4, 2, 256) @ r).reshape(4, 512)
    by_blocks = torch.block_diag(r, r)
    assert torch.allclose(by_groups, w @ by_blocks, atol=1e-12)


def test_without_a_device_it_is_refused():
    w8 = torch.zeros(16, 256, dtype=torch.int8)
    assert not D.usable(w8)
    with pytest.raises(RuntimeError):
        D.dequant_weight(w8, torch.ones(16))
