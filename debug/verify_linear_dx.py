"""The INT8 backward's activation gradient: convention, accuracy, one-time cost.

`dL/dx = dL/dy @ W` for `y = x @ W^T`.  With an INT8 weight, `W` is recovered by
`dequant_weight` (dequantize, then undo the ConvRot rotation), once, and the
GEMM is a plain bf16 `matmul` -- there is nothing to fuse, which is why this
script measures *accuracy* and the *one-time* dequant, not a kernel.

Three checks:

1. **Convention.** `linear_dx(dy, W)` against autograd of `F.linear(x, W)`. A
   missing transpose shows up here as a 100% error, not as a subtle one.
2. **Accuracy.** The same against the *quantized* model's weight: the gap is the
   price of INT8 weights, and it separates from the bf16-storage cost because
   the dequantized weight is bf16.
3. **What it costs once.** Dequantizing reads int8 and writes bf16 (1.5 B per
   parameter) and then rotates; for a multi-billion-parameter model that is
   seconds to a minute, which is the argument for the int8-GEMM path in
   `docs/todo.md` if a step ever needs the weight back in int8.

    PYTHONPATH=python python debug/verify_linear_dx.py
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


def main():
    torch.manual_seed(0)
    print(f"device: {torch.xpu.get_device_name(0)}, matrix [{N}, {KD}], batch {M}")

    w = (torch.randn(N, KD, device=DEV) * 0.02).to(torch.bfloat16)
    x = torch.randn(M, KD, device=DEV).to(torch.bfloat16)
    dy = torch.randn(M, N, device=DEV).to(torch.bfloat16)

    q, ws = K.prepare_weight(w)
    y_int8 = K.linear(x, q, ws)
    y_bf16 = F.linear(x, w)
    print(f"\nforward, INT8 weight vs bf16      : rel {rel(y_int8, y_bf16):.4f}")

    # -- 1. the convention ---------------------------------------------------
    xr = x.detach().clone().requires_grad_(True)
    F.linear(xr, w).backward(dy)
    grad = xr.grad                       # dy @ w, exactly
    ours = K.linear_dx(dy, w)
    print(f"linear_dx(dy, bf16 W) vs autograd : rel {rel(ours, grad):.2e} "
          f"({'exact' if rel(ours, grad) == 0 else 'ok'})")

    # -- 2. accuracy of the quantized-weight gradient ------------------------
    w_deq = K.dequant_weight(q, ws)
    print(f"\ndequantized weight vs bf16 W       : rel {rel(w_deq, w):.4f}")
    dx = K.linear_dx(dy, w_deq)
    print(f"dx with dequantized weight vs W   : rel {rel(dx, grad):.4f}"
          "   <- the INT8 model's own gradient, plus bf16 storage of W")
    # split that error into its two causes, both measured in the rotated basis
    # (q * ws is a reconstruction of the *rotated* weight, so compare there)
    w_rot = K.rotate(w).float()
    w_q = q.float() * ws.reshape(-1, 1)
    print(f"  int8 quantization alone         : rel {rel(w_q, w_rot):.4f}")
    print(f"  bf16 cast of the dequantized W  : rel {rel(w_q.bfloat16().float(), w_q):.5f}")
    print(f"  both, after the inverse rotate  : rel {rel(w_deq, w):.4f}")

    # -- 3. the one-time cost ------------------------------------------------
    t_prep = timed(lambda: K.prepare_weight(w))
    t_deq = timed(lambda: K.dequant_weight(q, ws))
    params = N * KD
    print(f"\none-time, {params / 1e6:.1f}M parameters:")
    print(f"  prepare_weight (quantize)        : {t_prep:6.2f} ms")
    print(f"  dequant_weight (dequant+rotate)  : {t_deq:6.2f} ms  "
          f"[{params * 1.5 / t_deq / 1e6:.0f} GB/s effective]")

    model_params = 42 * (9216 * 3072 + 3072 * 3072 + 8192 * 3072 + 3072 * 8192)
    est = model_params / params * t_deq / 1000
    print(f"  a 42-block model (~{model_params / 1e9:.1f}B params): "
          f"~{est:.2f} s of dequant, once")


if __name__ == "__main__":
    main()
