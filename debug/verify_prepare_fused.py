"""The fused weight preparation against the torch chain.

`prepare_weight` is a torch chain: rotate (a C kernel) and then quantize in
torch.  `prepare_weight_fused` does both in one C pass.  They are *not* the
same function -- the C kernel keeps the rotated values in fp32 where the
chain rounds them to bf16 first -- so the question this answers is:

  1. how different are the two weights?
  2. does the INT8 path's error against the bf16 weight change?
  3. what does the speed difference amount to?

The answer decides whether a *trainable* INT8 weight is affordable: a
straight-through path re-quantizes the parameter every step.

    PYTHONPATH=python python debug/verify_prepare_fused.py
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import xpu_kernels  # noqa: E402
from xpu_kernels import int8  # noqa: E402

DEV = torch.device("xpu", 0)
failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def timed(fn, n=20, warmup=5, reps=3):
    best = float("inf")
    for _ in range(reps):
        for _ in range(warmup):
            fn()
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.xpu.synchronize()
        best = min(best, (time.perf_counter() - t0) / n * 1000)
    return best


print(f"device: {int8.device_name()}")

# ---- 1. the two preparations, and the weight they make ------------------
print("\n1. the weights")
torch.manual_seed(0)
N, K = 8192, 3072
w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16)
q_chain, s_chain = int8.prepare_weight(w)
q_fused, s_fused = int8.prepare_weight_fused(w)
ndiff = int((q_chain.view(torch.int8) != q_fused.view(torch.int8)).sum())
srel = float(((s_chain - s_fused).abs() / s_chain.abs().clamp_min(1e-30)).max())
print(f"  values differ : {ndiff} of {N * K} ({ndiff / (N * K):.2%})")
print(f"  the scale     : max relative difference {srel:.3e} "
      f"({'identical' if srel == 0 else 'one ULP apart'})")

# the INT8 path's error against the bf16 weight, for each preparation
x = (torch.randn(4096, K, device=DEV) * 0.5).to(torch.bfloat16)
w_bf = w.to(torch.float32)
ref = (x.float() @ w_bf.t())
errs = {}
for name, (q, s) in (("the chain", (q_chain, s_chain)),
                     ("the fused kernel", (q_fused, s_fused))):
    y = int8.linear(x, q, s)
    errs[name] = float((y.float() - ref).norm() / ref.norm())
    print(f"  the INT8 path's error with {name:18s}: {errs[name]:.6f}")
delta = abs(errs["the chain"] - errs["the fused kernel"])
check("the error is unchanged (within a thousandth of it)", delta < 1e-3,
      f"the difference is {delta:.2e}")

# ---- 2. the speed ------------------------------------------------------
print("\n2. the speed")
t_chain = timed(lambda: int8.prepare_weight(w))
t_fused = timed(lambda: int8.prepare_weight_fused(w))
print(f"  the chain        : {t_chain:7.3f} ms  ({t_chain * 1e6 / (N * K):.3f} ns/param)")
print(f"  the fused kernel : {t_fused:7.3f} ms  ({t_fused * 1e6 / (N * K):.3f} ns/param)"
      f"   -> {t_chain / t_fused:.1f}x")
for params, name in ((2.17e9, "SDXL's quantized base"), (3.7e9, "a 3.7 B model")):
    print(f"  {name:24s} ({params / 1e9:.2f} B): the chain "
          f"{t_chain * 1e6 / (N * K) * params / 1e3:.0f} ms, the fused kernel "
          f"{t_fused * 1e6 / (N * K) * params / 1e3:.0f} ms, per step")

# ---- 3. the fused dequantizer round-trips the fused preparation ---------
print("\n3. the round trip")
w_deq = int8.dequant_weight(q_fused, s_fused)
rel = float((w_deq.float() - w.float()).norm() / w.float().norm())
print(f"  dequant_weight(fused) vs the bf16 weight: {rel:.4f} "
      f"(the INT8 weight itself)")
check("the round trip is within the int8 weight's own error", rel < 0.02,
      f"{rel:.4f}")

print("\n" + ("PASS" if not failures else f"FAIL: {len(failures)} check(s)"))
sys.exit(1 if failures else 0)
