"""The Triton attention on the card: does it agree with torch, does it capture and replay, and how does it compare in time.

Four parts, in the order that decides whether it is worth anything:

1. the gates (they compile the kernels and compare with torch's attention, at SDXL's shapes including the 77-token cross-attention);
2. the error against a float32 attention, forward and the three gradients, next to torch's own bf16 attention's error;
3. a device-graph capture of forward and backward, replayed on new data and compared with the eager kernels (must be identical);
4. device time of forward+backward: this kernel, torch's default (oneDNN flash; not capturable), and the math backend (what a captured
   step runs today).  The target from the trainer's brief: at least 2x faster than math, within about 1.3x of flash.

    python debug/verify_attention_capture.py
"""
import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from common import device_ms
from xpu_kernels.triton import attention as A

dev, dt = "xpu", torch.bfloat16
SHAPES = (("self 1024 (10 heads)", (4, 10, 1024, 64), (4, 10, 1024, 64)),
          ("cross 77 (10 heads)", (4, 10, 1024, 64), (4, 10, 77, 64)),
          ("self 256 (20 heads)", (4, 20, 256, 64), (4, 20, 256, 64)),
          ("cross 77 (20 heads)", (4, 20, 256, 64), (4, 20, 77, 64)))

print("== 1. gates")
q0 = torch.randn(1, 10, 64, 64, device=dev, dtype=dt)
print("usable:", A.usable(q0))
for name, gate in A.gates().items():
    print(f"  {name}: {gate.describe()}")
if not A.usable(q0):
    raise SystemExit("the gates are closed; nothing below is meaningful")


def make(qs, ks, seed):
    gen = torch.Generator(device=dev).manual_seed(seed)
    return [torch.randn(s, generator=gen, device=dev).to(dt) for s in (qs, ks, ks)] + [torch.randn(qs, generator=gen, device=dev).to(dt)]


def grads(fn, q, k, v, dout):
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
    out = fn(*leaves)
    return (out.detach(), *torch.autograd.grad(out, leaves, dout))


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


print("\n== 2. error against a float32 attention (relative L2)")
for label, qs, ks in SHAPES:
    q, k, v, dout = make(qs, ks, 1)
    truth = grads(F.scaled_dot_product_attention, q.float(), k.float(), v.float(), dout.float())
    mine = grads(A.attention, q, k, v, dout)
    torch_ = grads(F.scaled_dot_product_attention, q, k, v, dout)
    print(f"{label:22s} out/dq/dk/dv  this {[f'{rel(m, t):.1e}' for m, t in zip(mine, truth)]}  torch {[f'{rel(m, t):.1e}' for m, t in zip(torch_, truth)]}")

print("\n== 3. device graph: capture forward+backward, replay on new data, compare with eager")
CAPTURE_SHAPES = (("cross 77 (10 heads)", (4, 10, 1024, 64), (4, 10, 77, 64)),
                  ("self 1024 (10 heads)", (4, 10, 1024, 64), (4, 10, 1024, 64)))
for label, qs, ks in CAPTURE_SHAPES:
    print(f"  {label}")
    q, k, v, dout = make(qs, ks, 3)
    static = [t.clone().requires_grad_(True) for t in (q, k, v)]
    static_dout = dout.clone()

    def step():
        out = A.attention(*static)
        return out, *torch.autograd.grad(out, static, static_dout)

    step()  # warm up (compiles) outside the capture
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(graph):
            captured = step()
    except Exception as exc:  # noqa: BLE001 - this is the result the whole file exists to find out
        print(f"  CAPTURE FAILED: {type(exc).__name__}: {str(exc)[:300]}")
        captured = None
    if captured is not None:
        q2, k2, v2, dout2 = make(qs, ks, 4)
        for dst, src in zip(static, (q2, k2, v2)):
            dst.data.copy_(src)
        static_dout.copy_(dout2)
        graph.replay()
        torch.xpu.synchronize()
        eager = grads(A.attention, q2, k2, v2, dout2)
        for name, got, want in zip(("out", "dq", "dk", "dv"), captured, eager):
            print(f"  replay {name}: {'identical' if torch.equal(got, want) else f'DIFFERS (rel {rel(got, want):.1e})'}")

print("\n== 4. device time of forward+backward (ms, min of 3 interleaved rounds)")
print(f"{'shape':22s} {'this':>8s} {'flash':>8s} {'math':>8s}  this/flash  math/this")
for label, qs, ks in SHAPES:
    q, k, v, dout = make(qs, ks, 5)
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]

    def run(fn):
        def go():
            out = fn(*leaves)
            torch.autograd.grad(out, leaves, dout)
        return go

    def math_attention(a, b, c):
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(a, b, c)

    fns = {"this": run(A.attention), "flash": run(F.scaled_dot_product_attention), "math": run(math_attention)}
    for fn in fns.values():
        fn()
    torch.xpu.synchronize()
    best = {name: float("inf") for name in fns}
    for _ in range(3):
        for name, fn in fns.items():
            best[name] = min(best[name], device_ms(fn))
    t_mine, t_flash, t_math = best["this"], best["flash"], best["math"]
    print(f"{label:22s} {t_mine:8.3f} {t_flash:8.3f} {t_math:8.3f}  {t_mine / t_flash:9.2f}x {t_math / t_mine:9.2f}x")
