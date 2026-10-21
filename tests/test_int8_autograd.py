"""The differentiable INT8 linear: the contract that can be checked anywhere.

The op itself needs a device (the forward is a SYCL kernel), but the parts
that decide *whether* it is called correctly are pure:

* the backward mode is validated before anything is touched, so a typo is an
  error rather than a silent fallback;
* ``linear_dw`` is a plain torch op, so its shape contract is checkable on
  CPU -- and it is the one that is easy to get wrong, because the batch
  dimension is the *contraction* of ``dy^T @ x`` and not the last axis of
  either operand.
"""

import pytest
import torch

from xpu_kernels import int8


def test_an_unknown_backward_mode_is_refused():
    with pytest.raises(ValueError) as exc:
        int8.int8_linear(torch.zeros(2, 4, dtype=torch.bfloat16),
                          torch.zeros(4, 4, dtype=torch.int8),
                          torch.ones(4, dtype=torch.float32),
                          mode="nope")
    assert "nope" in str(exc.value)


def test_an_unknown_mode_is_refused_by_the_module():
    with pytest.raises(ValueError):
        int8.Int8Linear(torch.zeros(4, 4, dtype=torch.int8),
                         torch.ones(4, dtype=torch.float32),
                         backward="nope")


def test_linear_dw_contracts_over_the_batch():
    # dy [M, N] and x [M, K] -> [N, K]: the contraction is over M, so the
    # *batch* dims have to agree -- checking the last axis of either operand
    # would be the wrong check.
    dy = torch.randn(8, 16, dtype=torch.bfloat16)
    x = torch.randn(8, 32, dtype=torch.bfloat16)
    dw = int8.linear_dw(dy, x)
    assert dw.shape == (16, 32)
    assert torch.equal(dw, dy.transpose(0, 1) @ x)


def test_linear_dw_rejects_mismatched_batches():
    dy = torch.randn(8, 16, dtype=torch.bfloat16)
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError) as exc:
        int8.linear_dw(dy, x)
    assert "rows" in str(exc.value)


def test_linear_dw_flattens_leading_dims():
    dy = torch.randn(2, 3, 8, 16, dtype=torch.bfloat16)
    x = torch.randn(2, 3, 8, 32, dtype=torch.bfloat16)
    dw = int8.linear_dw(dy, x)
    assert dw.shape == (16, 32)
    assert torch.equal(dw, dy.reshape(-1, 16).transpose(0, 1)
                        @ x.reshape(-1, 32))
