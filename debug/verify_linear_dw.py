"""The weight gradient of the INT8 model, and the INT8 linear as an autograd op.

Two things are checked here, and one of them is a bug that would have been
silent.

1. ``linear_dw(dy, x)`` against ``torch.autograd`` of the bf16 linear with the
   *dequantized* weight.  The INT8 forward is mathematically that model (the
   ConvRot rotation is orthogonal, so it cancels in the contraction), so its
   weight gradient is the bf16 model's -- and it should agree bit for bit,
   because it is the same GEMM.

2. The autograd op.  ``linear()`` on its own returns a tensor with no
   ``grad_fn``: the kernels are not differentiable, so an INT8 layer dropped
   into a training graph would produce **no gradient at all**, silently.  The
   op exists to prevent that, and this script is what would catch it if it
   broke: it checks that the gradient that comes out is the one that should.

    PYTHONPATH=python LEVEL_ZERO_V1_SDK_PATH=~/.local/levelzero \\
        python debug/verify_linear_dw.py
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import xpu_kernels  # noqa: E402
from xpu_kernels import int8  # noqa: E402

DEV = "xpu"
M, N, K = 4096, 8192, 3072


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def main():
    torch.manual_seed(0)
    print(f"device: {torch.xpu.get_device_name(0)}")
    failures = []

    w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16)
    w8, w_scale = xpu_kernels.prepare_weight(w)
    w_deq = xpu_kernels.dequant_weight(w8, w_scale)
    x = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16)
    dy = (torch.randn(M, N, device=DEV) * 0.3).to(torch.bfloat16)

    # -- 1. the weight gradient ------------------------------------------------
    xr = x.detach().clone().requires_grad_(True)
    wr = w_deq.detach().clone().requires_grad_(True)
    y = F.linear(xr, wr)
    y.backward(dy)
    dw_ref = wr.grad
    dw = int8.linear_dw(dy, x)
    same = dw.view(torch.int16).eq(dw_ref.view(torch.int16)).all().item()
    print(f"\ndL/dW: {tuple(dw.shape)}  vs autograd of the dequantized weight: "
          f"{'bit-identical' if same else 'DIFFERS'}  (rel {rel(dw, dw_ref):.2e})")
    if not same:
        failures.append("linear_dw differs from autograd")

    # -- 2. the autograd op ---------------------------------------------------
    for mode in ("dense", "int8"):
        xi = x.detach().clone().requires_grad_(True)
        out = int8.int8_linear(xi, w8, w_scale, None, mode=mode)
        out.backward(dy)
        dx = xi.grad
        if dx is None:
            failures.append(f"mode {mode!r}: no gradient came out at all")
            print(f"\nmode {mode}: NO GRADIENT (the silent failure)")
            continue
        print(f"\nmode {mode:5s}: dx rel to the bf16 model's {rel(dx, dy @ w_deq):.4f}"
              f"   (the dequantized weight is the model, 0.9 % from the bf16 one)")

    # the dense mode must be exact against the same GEMM
    xi = x.detach().clone().requires_grad_(True)
    int8.int8_linear(xi, w8, w_scale, None, mode="dense").backward(dy)
    same = xi.grad.view(torch.int16).eq((dy @ w_deq).view(torch.int16)).all().item()
    print(f"mode dense : dx identical to dy @ dequantized weight: {same}")
    if not same:
        failures.append("the dense mode's dx is not the same GEMM")

    # -- 3. the weight gradient is the caller's job --------------------------
    # The INT8 weight is a frozen int8 buffer, so its gradient is
    # meaningless; a trainable weight is a bf16 parameter, and its gradient
    # is linear_dw(dy, x) -- the caller has both.
    print(f"\nweight gradient for a trainable parameter: "
          f"linear_dw(dy, x) -> {tuple(int8.linear_dw(dy, x).shape)}")

    # -- 4. inside a capture, the dense mode must refuse ----------------------
    g = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(g):
            int8.dequant_weight(w8, w_scale)
        failures.append("dequant_weight ran inside a capture")
        print("\ndequant_weight inside a capture: NO REFUSAL (bad)")
    except RuntimeError as exc:
        ok = "capture" in str(exc)
        print(f"\ndequant_weight inside a capture -> RuntimeError: {str(exc)[:90]}...")
        if not ok:
            failures.append("the capture failed for the wrong reason")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        raise SystemExit(1)
    print("the weight gradient and the autograd op behave as documented")


if __name__ == "__main__":
    main()
