"""A/B the two `dL/dx` paths against an INT8 weight: bf16 dequant vs int8 GEMM.

The weight-only path (`linear_dx`) materializes `W` once and runs one bf16 GEMM.
The quantized-gradient path (`linear_dx_int8`) keeps the weight in int8 and
quantizes `dy` per row instead; it is exact in the weight and adds one
quantizer's worth of noise.

What to look at, in this order:

1. **Device time.** The int8 GEMM is faster per FLOP, but the path also pays a
   quantizer, a dequant epilogue and a final rotation (`dx` must come back to the
   unrotated basis). Measured: the GEMM is 1.85x faster and the path is 3.6x
   *slower*, because the torch quantizer costs 4.5 ms against a 1.9 ms bf16 GEMM.
   Even a perfectly fused quantizer (0.45 ms of traffic) leaves the path at
   ~2.1 ms vs 1.92 ms -- so this is a *memory* option (the weight stays int8,
   ~2x less), not a speed one, unless the final rotation can be avoided. See
   `docs/performance.md`.
2. **Gradient accuracy.** Relative L2 against the exact gradient *of the
   unquantized model*, and the cosine similarity -- a gradient can be large in
   relative error and still point the right way, and the direction is what an
   optimizer follows.

This probe cannot answer whether the noise hurts training. That needs a loss
curve (same seed, same data order) on the trainer's side; see `docs/todo.md`.

    PYTHONPATH=python python debug/probe_linear_dx_variants.py
"""

import os
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import xpu_kernels as K  # noqa: E402

DEV = "xpu"
M, N, KD = 4096, 8192, 3072


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


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
    """Alternate a and b, so a clock or desktop shift hits both."""
    ta, tb = [], []
    for _ in range(rounds):
        ta.append(timed(a, iters=iters, warmup=1))
        tb.append(timed(b, iters=iters, warmup=1))
    return statistics.median(ta), statistics.median(tb)


def main():
    torch.manual_seed(0)
    print(f"device: {torch.xpu.get_device_name(0)}, [{N}, {KD}] x batch {M}")

    w = (torch.randn(N, KD, device=DEV) * 0.02).to(torch.bfloat16)
    dy = torch.randn(M, N, device=DEV).to(torch.bfloat16)

    # the exact gradient of the *unquantized* model, for accuracy only
    x = torch.randn(M, KD, device=DEV).to(torch.bfloat16)
    xr = x.detach().clone().requires_grad_(True)
    F.linear(xr, w).backward(dy)
    exact = xr.grad

    q, ws = K.prepare_weight(w)
    w_deq = K.dequant_weight(q, ws)          # one-time, amortized

    dx_bf16 = K.linear_dx(dy, w_deq)
    dx_int8 = K.linear_dx_int8(dy, q, ws)

    print("\naccuracy against the unquantized gradient")
    for name, dx in (("bf16 dequant", dx_bf16), ("int8 GEMM  ", dx_int8)):
        print(f"  {name}: rel {rel(dx, exact):.4f}   cosine {cosine(dx, exact):.6f}")

    print("\ndevice time, interleaved medians")
    t_bf16, t_int8 = interleaved(lambda: K.linear_dx(dy, w_deq),
                                 lambda: K.linear_dx_int8(dy, q, ws))
    print(f"  bf16 dequant : {t_bf16:6.3f} ms")
    print(f"  int8 GEMM    : {t_int8:6.3f} ms  ({t_bf16 / t_int8:.2f}x)")

    # where the int8 path's time goes, so the next step is obvious
    d = (dy * ws.to(torch.bfloat16)).reshape(-1, N)
    dq, a = K.quantize_row(d)
    t_q = timed(lambda: K.quantize_row(d))
    t_acc = timed(lambda: torch._int_mm(dq, q))
    ones = torch.ones(KD, device=DEV, dtype=torch.float32)
    acc = torch._int_mm(dq, q)
    t_dq = timed(lambda: K.dequant(acc, a, ones))
    t_rot = timed(lambda: K.rotate(dx_bf16))
    print(f"    quantize_row (torch chain, 4 ops) {t_q:6.3f} ms")
    print(f"    _int_mm only                      {t_acc:6.3f} ms")
    print(f"    dequant epilogue                  {t_dq:6.3f} ms")
    print(f"    final rotate (undo ConvRot)       {t_rot:6.3f} ms")
    print(f"    -> theoretical floor with a fused quantizer: "
          f"{t_acc + t_dq + t_rot + 0.45:.2f} ms vs {t_bf16:.2f} ms for bf16")


if __name__ == "__main__":
    main()
