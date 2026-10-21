"""Fused rope backward vs torch.autograd of the eager chain: time and accuracy.

Two questions, one script:

* **How long does the backward take?** Device time, interleaved with the eager
  chain, on the production shape (M=3973, H=24, D=128, bf16), because the fused
  backward is two passes over the row and the eager one is a pile of elementwise
  kernels. Wall-clock medians of an interleaved run; the desktop must be idle.
* **How far is the gradient from the truth?** Both implementations are compared
  to a float64 evaluation of the same analytic gradient. In bf16 the eager
  chain's ``.to(bfloat16)`` makes autograd round the gradient to bf16 at that
  boundary, so autograd is ~1 ULP from the truth before the fused kernel does
  anything -- which is why the gate probes the arithmetic in fp32.

    LEVEL_ZERO_V1_SDK_PATH=~/.local/levelzero \
        PYTHONPATH=python python debug/probe_rope_backward.py
"""

import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from xpu_kernels.triton import reference, rope  # noqa: E402

DEV = "xpu"
B, S, H, D = 1, 3973, 24, 128


def make(seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = (torch.randn(B, S, H, D, generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    ang = torch.rand(B, S, 1, D // 2, generator=g, device=DEV) * 3.0
    pe = torch.stack([torch.stack([ang.cos(), -ang.sin()], -1),
                      torch.stack([ang.sin(), ang.cos()], -1)], -2)
    scale = torch.rand(D, generator=g, device=DEV) * 0.2 + 0.9
    dy = (torch.randn(B, S, H, D, generator=g, device=DEV) * 0.1).to(torch.bfloat16)
    return x, pe, scale, dy


def timed(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    events = [(torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True))
              for _ in range(iters)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.xpu.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in events)


def interleaved(a, b, iters=30, warmup=5):
    """Alternate the two implementations, so a clock change hits both."""
    for _ in range(warmup):
        a()
        b()
    torch.xpu.synchronize()
    ta, tb = [], []
    for _ in range(iters):
        ta.append(timed(a, iters=1))
        tb.append(timed(b, iters=1))
    return statistics.median(ta), statistics.median(tb)


def main():
    torch.manual_seed(0)
    print(f"device: {torch.xpu.get_device_name(0)}")
    assert rope._GATE_BWD.ok(), f"backward gate closed: {rope.describe('rms_rope_backward')}"

    x, pe, scale, dy = make()
    xr = x.detach().requires_grad_(True)
    nbytes = x.numel() * x.element_size()
    # forward: read x, write y.  backward: read x and dy, write dx.
    fwd_bytes, bwd_bytes = 2 * nbytes, 3 * nbytes

    def eager_fwd_bwd():
        y = reference.rms_rope_split_half(xr, pe, scale, 1e-6)
        torch.autograd.grad(y, xr, dy)

    def fused_fwd_bwd():
        y = rope._RmsRope.apply(xr, pe, scale, 1e-6, True)
        torch.autograd.grad(y, xr, dy)

    def eager_fwd():
        reference.rms_rope_split_half(x, pe, scale, 1e-6)

    def fused_fwd():
        rope._rows(x, pe, scale, 1e-6)

    def fused_bwd():
        rope._rows_bwd_kernel(dy, x, pe, scale, 1e-6)

    fwd_eager = timed(eager_fwd)
    fwd_fused = timed(fused_fwd)
    bwd_fused = timed(fused_bwd)
    print(f"forward   eager {fwd_eager:7.3f} ms   fused {fwd_fused:7.3f} ms "
          f"({fwd_eager / fwd_fused:5.2f}x)   [{2 * nbytes / fwd_fused / 1e6:.0f} GB/s]")
    print(f"backward              fused {bwd_fused:7.3f} ms   "
          f"                [{bwd_bytes / bwd_fused / 1e6:.0f} GB/s]")

    e, f = interleaved(eager_fwd_bwd, fused_fwd_bwd)
    print(f"fwd+bwd   eager {e:7.3f} ms   fused {f:7.3f} ms ({e / f:5.2f}x)")

    # -- accuracy: both against a float64 evaluation of the same gradient -----
    xr = x.detach().requires_grad_(True)
    reference.rms_rope_split_half(xr, pe, scale, 1e-6).backward(dy)
    auto = xr.grad
    fused = rope._rows_bwd(dy, x, pe, scale, 1e-6)
    truth = reference.rms_rope_backward(dy.double(), x.double(), pe.double(),
                                       scale.double(), 1e-6)

    def rel(a, b):
        a, b = a.double(), b.double()
        return float((a - b).norm() / b.norm())

    # the preallocated-output path must be bit-identical: it is what a captured
    # graph will use, and a capture that changes the numbers is useless
    buf = torch.empty_like(x)
    rope._rows_bwd_kernel(dy, x, pe, scale, 1e-6, out=buf)
    assert buf.view(torch.int16).equal(fused.view(torch.int16)), "out= changed the result"

    print("\nrelative L2 error against the float64 gradient")
    print(f"  torch.autograd (bf16 graph) : {rel(auto, truth):.3e}")
    print(f"  fused kernel    (bf16 out)  : {rel(fused, truth):.3e}")
    print(f"  fused vs autograd           : {rel(fused, auto):.3e}")


if __name__ == "__main__":
    main()
