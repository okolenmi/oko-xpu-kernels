"""Runs the attention kernels on the CPU through Triton's interpreter and compares them with autograd of the math attention.

Run by ``test_attention.py`` in a subprocess: the interpreter is chosen when ``@triton.jit`` is applied, so the environment
variable has to exist before the package is imported.  Prints ``OK`` or raises.
"""

import os

os.environ["TRITON_INTERPRET"] = "1"

import math  # noqa: E402

import torch  # noqa: E402

from xpu_kernels.triton import attention as A  # noqa: E402

assert A._kernels is not None, A._import_error

# (batch, heads, Lq, Lk, D): a block multiple, SDXL's 77-token cross-attention, and ragged both ways
CASES = ((1, 2, 64, 64, 32), (1, 2, 40, 77, 64), (2, 1, 100, 33, 16), (1, 1, 17, 130, 64))


def math_attention(q, k, v, scale):
    return torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1) @ v


for batch, heads, lq, lk, d in CASES:
    gen = torch.Generator().manual_seed(lq * 31 + lk)
    q, k, v = (torch.randn(batch, heads, n, d, generator=gen) for n in (lq, lk, lk))
    dout = torch.randn(batch, heads, lq, d, generator=gen)
    scale = 1.0 / math.sqrt(d)
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
    want_out = math_attention(*leaves, scale)
    want = torch.autograd.grad(want_out, leaves, dout)

    out, lse = A._forward(q, k, v, scale)
    assert torch.allclose(out, want_out.detach(), atol=2e-4, rtol=2e-4), ("forward", (batch, heads, lq, lk, d), (out - want_out).abs().max())
    got = A._backward(q, k, v, out, lse, dout, scale)
    for name, g, w in zip(("dq", "dk", "dv"), got, want):
        assert torch.allclose(g, w, atol=5e-4, rtol=5e-4), (name, (batch, heads, lq, lk, d), (g - w).abs().max())
print("OK")
