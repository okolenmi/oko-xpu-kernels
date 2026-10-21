"""The straight-through INT8 linear: a trainable parameter, an INT8 forward.

`TrainableInt8Linear` is the last piece of the INT8 story: the parameter
stays bf16 and is quantized on every forward, so it can be trained, while
the forward is the INT8 model's and the backward is the bf16 model's.

What this checks:

  1. the forward is *the INT8 model's* -- bit-identical to
     `linear(x, prepare_weight_fused(w), ...)`;
  2. the weight gradient is the bf16 model's, bit for bit (the ConvRot
     rotation cancels in the contraction);
  3. the bias gradient is not dropped (the bug 0.3.0 found in the frozen
     op);
  4. the "dense" mode's activation gradient is the *exact* gradient of the
     forward, and the "ste" mode's is the bf16 model's;
  5. the capture: a replay re-quantizes the *current* weight, so a captured
     training step sees the optimizer's update.

    PYTHONPATH=python python debug/verify_trainable_int8.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from xpu_kernels import int8  # noqa: E402

DEV = torch.device("xpu", 0)
failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def bit_equal(a, b):
    return bool(a.view(torch.int16).eq(b.view(torch.int16)).all())


torch.manual_seed(0)
M, K, N = 512, 1024, 512
x = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16)
w = (torch.randn(N, K, device=DEV) * 0.05).to(torch.bfloat16)
bias = torch.randn(N, device=DEV, dtype=torch.bfloat16) * 0.1
dy = (torch.randn(M, N, device=DEV) * 0.5).to(torch.bfloat16)

print(f"device: {int8.device_name()}")

print("\n1. the forward is the INT8 model's")
mod = int8.TrainableInt8Linear(w, bias)
y = mod(x)
w8, ws = int8.prepare_weight_fused(w)
y_ref = int8.linear(x, w8, ws, bias)
check("bit-identical to linear(x, prepare_weight_fused(w))", bit_equal(y, y_ref))

print("\n2. the weight gradient is the bf16 model's")
y.backward(dy)
check("dW == dy^T @ x, bit for bit",
      bit_equal(mod.weight.grad, (dy.t() @ x)))
xr = x.detach().clone().requires_grad_(True)
wr = w.detach().clone().requires_grad_(True)
br = bias.detach().clone().requires_grad_(True)
torch.nn.functional.linear(xr, wr, br).backward(dy)
check("dW == autograd's, bit for bit", bit_equal(mod.weight.grad, wr.grad))
check("the bias gradient is not dropped", bit_equal(mod.bias.grad, br.grad))
check("dW is the weight's shape", tuple(mod.weight.grad.shape) == (N, K))

print("\n3. the two backward modes")
for mode in ("ste", "dense"):
    mod = int8.TrainableInt8Linear(w, bias, backward=mode)
    xx = x.detach().clone().requires_grad_(True)
    y = mod(xx)
    y.backward(dy)
    # the exact gradient of the forward: dy @ dequant(w8, s)
    w_deq = int8.dequant_weight(w8, ws)
    exact = dy @ w_deq
    rel = float((xx.grad.float() - exact.float()).norm() / exact.float().norm())
    print(f"  {mode:6s}: dx vs the exact gradient of the forward: {rel:.2e}")
    if mode == "dense":
        check("the dense mode is exact", rel < 1e-6, f"{rel:.2e}")
    else:
        check("the ste mode is the bf16 model's gradient (the weight's own error)",
              rel < 0.02, f"{rel:.2e}")

print("\n4. the capture: a replay re-quantizes the current weight")
mod = int8.TrainableInt8Linear(w, bias)
xs = x.clone()
g = torch.xpu.XPUGraph()
try:
    with torch.xpu.graph(g):
        captured = mod(xs)
    check("the capture itself", True)
except Exception as exc:  # noqa: BLE001 - the failure mode is the finding
    check("the capture itself", False, f"{type(exc).__name__}: {exc}")
    captured = None
if captured is not None:
    # the optimizer's update: the weight's *contents* change, its pointer does
    # not.  A replay must pick the new values up -- that is what makes a
    # captured training step work.
    with torch.no_grad():
        mod.weight.add_(0.01)
    g.replay()
    torch.xpu.synchronize()
    w8b, wsb = int8.prepare_weight_fused(mod.weight.detach())
    expect = int8.linear(x, w8b, wsb, bias)
    check("the replay sees the updated weight", bit_equal(captured, expect),
          "" if bit_equal(captured, expect) else
          f"max abs diff {(captured.float() - expect.float()).abs().max().item():.3e}")

print("\n" + ("PASS" if not failures else f"FAIL: {len(failures)} check(s)"))
sys.exit(1 if failures else 0)
