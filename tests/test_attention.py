"""The Triton attention: its kernels' arithmetic against the math attention (CPU, through Triton's interpreter), and the gates' refusal
on a machine without a device.

The interpreter checks what the kernels *compute* -- the recurrence, the masks for lengths that are not a block multiple, the two
backward kernels -- in float32, where no rounding can hide a wrong formula.  Whether they compile for the card, capture, and how fast
they run is for ``debug/verify_attention_capture.py`` and ``debug/bench_attention.py`` on the hardware.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from xpu_kernels.triton import attention as A

HERE = Path(__file__).resolve().parent


def test_kernels_match_the_math_attention_in_the_interpreter():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(HERE.parent / "python"), os.environ.get("PYTHONPATH", "")]))
    done = subprocess.run([sys.executable, str(HERE / "_attention_interpreter.py")], capture_output=True, text=True, env=env, timeout=900)
    if "No module named 'triton'" in done.stderr:
        pytest.skip("triton is not installed")
    assert done.returncode == 0 and done.stdout.strip().endswith("OK"), done.stderr[-3000:]


def test_without_a_device_the_fused_op_is_refused():
    q = torch.randn(1, 2, 8, 64, dtype=torch.bfloat16)
    assert not A.usable(q)
    with pytest.raises(RuntimeError):
        A.attention(q, q, q)


def test_unsupported_shapes_are_not_usable():
    assert not A.usable(torch.randn(2, 8, 64))             # not 4-d
    assert not A.usable(torch.randn(1, 2, 8, 48))          # head_dim must be a power of two Triton tiles
    assert not A.usable(torch.randn(1, 2, 8, 64), torch.randn(1, 2, 8, 32))


def test_the_configuration_override():
    os.environ["XPU_KERNELS_ATTENTION_CONFIG"] = "64,32,4,1"
    try:
        assert A.config(64) == (64, 32, 4, 1)
        os.environ["XPU_KERNELS_ATTENTION_CONFIG"] = "64,32"
        with pytest.raises(ValueError):
            A.config(64)
    finally:
        del os.environ["XPU_KERNELS_ATTENTION_CONFIG"]
    assert A.config(64) == A.DEFAULT_CONFIG[64]
