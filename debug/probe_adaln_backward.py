"""adaLN forward+backward: fused vs the eager chain, and how accurate the gradient is.

The backward is one kernel for `dx` plus two torch reductions for `dscale` and
`dshift` (they sum over rows the modulation was broadcast across, which a
per-row kernel cannot do without atomics). The probe reports the total and the
pieces, because the pieces are what would be fused next if it were worth it.

    PYTHONPATH=python LEVEL_ZERO_V1_SDK_PATH=~/.local/levelzero \\
        python debug/probe_adaln_backward.py
"""

import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import xpu_kernels.triton as kt  # noqa: E402
from xpu_kernels.triton import adaln, reference  # noqa: E402

B, S, D = 1, 3973, 3072
EPS = 1e-6


def timed(fn, iters=20, warmup=3):
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


def interleaved(a, b, rounds=7, iters=15):
    ta, tb = [], []
    for _ in range(rounds):
        ta.append(timed(a, iters=iters, warmup=1))
        tb.append(timed(b, iters=iters, warmup=1))
    return statistics.median(ta), statistics.median(tb)


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def main():
    torch.manual_seed(0)
    print(f"device: {torch.xpu.get_device_name(0)}")
    print(f"  {kt.adaln.describe('rms_adaln_backward')}")

    x = (torch.randn(B, S, D, device="xpu") * 0.5).to(torch.bfloat16)
    scale = torch.randn(1, 1, D, device="xpu") * 0.1
    shift = torch.randn(1, 1, D, device="xpu") * 0.1
    dy = (torch.randn(B, S, D, device="xpu") * 0.1).to(torch.bfloat16)

    xr = x.detach().clone().requires_grad_(True)
    sr = scale.detach().clone().requires_grad_(True)
    hr = shift.detach().clone().requires_grad_(True)

    def eager_fwd_bwd():
        y = reference.rms_adaln(xr, sr, hr, EPS)
        return torch.autograd.grad(y, (xr, sr, hr), dy)

    def fused_fwd_bwd():
        y = adaln.rms_adaln(xr, sr, hr, EPS)
        return torch.autograd.grad(y, (xr, sr, hr), dy)

    def eager_fwd():
        reference.rms_adaln(x, scale, shift, EPS)

    def fused_fwd():
        adaln.rms_adaln(x, scale, shift, EPS)

    fwd_e = timed(eager_fwd)
    fwd_f = timed(fused_fwd)
    print(f"\nforward            eager {fwd_e:6.3f} ms   fused {fwd_f:6.3f} ms "
          f"({fwd_e / fwd_f:5.2f}x)")
    e, f = interleaved(eager_fwd_bwd, fused_fwd_bwd)
    print(f"forward+backward   eager {e:6.3f} ms   fused {f:6.3f} ms ({e / f:5.2f}x)")

    # where the fused backward goes
    dx = torch.empty_like(x)
    t_dx = timed(lambda: adaln._rows_bwd_kernel(dy, x, scale, EPS)[0])
    dx_k, v = adaln._rows_bwd_kernel(dy, x, scale, EPS)
    t_v = timed(lambda: adaln._rows_bwd_kernel(dy, x, scale, EPS, need_v=True)[0])
    t_ds = timed(lambda: reference._reduce_like(
        dy.to(torch.float32) * v.to(torch.float32), scale))
    t_dh = timed(lambda: reference._reduce_like(dy, shift))
    print(f"\n  dx kernel (no v)        {t_dx:6.3f} ms")
    print(f"  dx kernel (+ v for dscale) {t_v:6.3f} ms")
    print(f"  dscale reduction        {t_ds:6.3f} ms")
    print(f"  dshift reduction        {t_dh:6.3f} ms")

    # accuracy against the same gradient in float64
    gx, gs, gh = fused_fwd_bwd()
    ex, es, eh = torch.autograd.grad(
        reference.rms_adaln(xr.double(), sr.double(), hr.double(), EPS),
        (xr, sr, hr), dy.double())
    print("\nrelative L2 error against the float64 gradient")
    print(f"  dx     : {rel(gx, ex):.3e}")
    print(f"  dscale : {rel(gs, es):.3e}")
    print(f"  dshift : {rel(gh, eh):.3e}")


if __name__ == "__main__":
    main()
