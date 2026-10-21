"""The INT8 API contract that can be checked without a device.

``unsupported_reason`` is the library's answer to "your card cannot do this" vs
"you passed the wrong thing": it returns the specific reason, and ``linear``
raises with that same string.  A caller that falls back per layer (ComfyUI does)
branches on it, so the messages are part of the API and are pinned here.

The kernels need an XPU and ``libxpuint8.so``; both are absent in CI, which is
why only the argument checks -- the ones that run before either is touched -- are
tested.  The device path lives in ``python -m xpu_kernels`` (a real self-test on
the card).
"""

import pytest
import torch

from xpu_kernels import int8


def _w(n=32, k=512):
    return torch.zeros((n, k), dtype=torch.int8)


def _x(m=4, k=512):
    return torch.zeros((m, k), dtype=torch.bfloat16)


def _scale(n=32):
    return torch.full((n,), 0.01, dtype=torch.float32)


def test_valid_call_passes_the_argument_checks():
    # the only thing that can fail here is the hardware/library part, and the
    # message says which
    reason = int8.unsupported_reason(_x(), _w(), _scale())
    assert reason is None or "libxpuint8" in reason or "XPU" in reason


def test_activation_dtype_is_reported():
    assert "float16" in int8.unsupported_reason(
        _x().to(torch.float16), _w(), _scale())


def test_weight_must_be_int8_2d():
    assert "int8" in int8.unsupported_reason(_x(), _w().float(), _scale())
    assert "int8" in int8.unsupported_reason(_x(), _w()[0], _scale())


def test_column_count_must_match_the_weight():
    reason = int8.unsupported_reason(_x(k=512), _w(k=1024), _scale())
    assert "512" in reason and "1024" in reason


def test_pair_activation_wants_two_halves():
    reason = int8.unsupported_reason(_x(k=512), _w(k=512), _scale(), act="situ")
    assert "1024" in reason
    assert int8.unsupported_reason(_x(k=1024), _w(k=512), _scale(),
                                   act="situ") is not None  # device/lib only
    assert "situ" in int8.unsupported_reason(_x(), _w(), _scale(), act="silu")


def test_the_activation_set():
    # the fused FFN's activations, and which of them read a pair of halves
    assert set(int8.ACTIVATIONS) >= {"gelu_tanh", "swiglu", "situ",
                                      "geglu_erf", "gelu_erf"}
    assert "situ" in int8.PAIRED_ACTIVATIONS
    assert "geglu_erf" in int8.PAIRED_ACTIVATIONS
    assert "gelu_erf" not in int8.PAIRED_ACTIVATIONS


def test_the_new_activations_in_the_argument_checks():
    # SDXL's GEGLU is a paired activation: it wants two halves
    reason = int8.unsupported_reason(_x(k=512), _w(k=512), _scale(),
                                     act="geglu_erf")
    assert "1024" in reason
    # the erf GELU on its own is not
    reason = int8.unsupported_reason(_x(k=512), _w(k=512), _scale(),
                                     act="gelu_erf")
    assert reason is None or "libxpuint8" in reason or "XPU" in reason


def test_ffn_rejects_an_unknown_activation_before_anything_else():
    with pytest.raises(TypeError) as exc:
        int8.ffn(_x(), _w(), _scale(), _w(), _scale(), act="silu")
    assert "silu" in str(exc.value)
    with pytest.raises(TypeError) as exc:
        int8.ffn(_x(), _w(), _scale(), _w(), _scale(), act="swiglu")
    assert "swiglu" in str(exc.value)


def test_group_size_multiple():
    # the activation has to match K, or the column check fires first
    assert "256" in int8.unsupported_reason(_x(k=384), _w(k=384), _scale())


def test_output_width_multiple_of_16():
    reason = int8.unsupported_reason(_x(), _w(n=24), _scale(24))
    assert "24" in reason and "16" in reason


def test_scale_shapes():
    assert "floating point" in int8.unsupported_reason(
        _x(), _w(), torch.zeros(32, dtype=torch.int32))
    assert "dims" in int8.unsupported_reason(
        _x(), _w(), torch.zeros((2, 2, 8), dtype=torch.float32))
    assert "values" in int8.unsupported_reason(_x(), _w(), torch.zeros(7))
    assert "tensor or a python float" in int8.unsupported_reason(
        _x(), _w(), "0.01")
    # a scalar, a [N] and a [N, 1] scale are all legal
    for scale in (torch.zeros(1), torch.zeros(32), torch.zeros(32, 1), 0.01):
        reason = int8.unsupported_reason(_x(), _w(), scale)
        assert reason is None or "libxpuint8" in reason or "XPU" in reason


def test_linear_raises_the_same_message():
    reason = int8.unsupported_reason(_x(), _w(k=384), _scale())
    with pytest.raises(TypeError) as exc:
        int8.linear(_x(), _w(k=384), _scale())
    assert str(exc.value) == reason


def test_expand_scale():
    # a flat fp32 [n] tensor is returned as is: no copy on the hot path
    flat = _scale(32)
    assert int8.expand_scale(flat, 32, flat.device) is flat
    # a scalar is broadcast, which the C side does *not* do -- it indexes [0..n)
    one = torch.tensor(0.25)
    got = int8.expand_scale(one, 5, one.device)
    assert got.shape == (5,) and got.dtype == torch.float32
    assert torch.equal(got, torch.full((5,), 0.25))
    # [N, 1] (the shape checkpoints store) is flattened, not broadcast
    col = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    assert torch.equal(int8.expand_scale(col, 4, col.device), col.reshape(-1))
    # a python float needs a device and produces a materialised vector
    assert int8.expand_scale(0.5, 3, torch.device("cpu")).shape == (3,)
    with pytest.raises(TypeError):
        int8.expand_scale(0.5, 3)
    with pytest.raises(TypeError):
        int8.expand_scale("0.5", 3, torch.device("cpu"))
