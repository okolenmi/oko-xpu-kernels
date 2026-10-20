import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Fair BF16 vs INT8 comparison.

The existing bench_pipeline.py measures BF16 as `linear(x, wbf, bias)` with a
pre-rotated weight, which omits the activation rotation. ConvRot needs that
rotation on *both* paths, so that comparison understates BF16.

Here both paths pay for it:

  INT8 : rotate+quantize (bf16->int8 + row scale)
         -> torch._int_mm -> dequant epilogue
  BF16 : rotate only      (bf16->bf16)
         -> torch.nn.functional.linear

Weights are identical in value: the same int8 tensor dequantized to bf16.
Correctness of the BF16 rotate is checked against float64 on CPU.
"""
import argparse
import ctypes
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.bench_pipeline import load_lib  # noqa: E402

from comfy_kitchen.tensor.int8_utils import _build_hadamard  # noqa: E402

LIB = None


def bench(fn, n=20, warmup=5, reps=3):
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


def rotate_quantize(lib, x):
    m, k = x.shape
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty(m, device=x.device, dtype=torch.float32)
    rc = lib.xpuint8_convrot_quantize(x.data_ptr(), q.data_ptr(), s.data_ptr(), int(m), int(k))
    if rc:
        raise RuntimeError(lib.xpuint8_last_error())
    return q, s


def rotate_bf16(lib, x, out):
    m, k = x.shape
    rc = lib.xpuint8_convrot_rotate_bf16(x.data_ptr(), out.data_ptr(), int(m), int(k))
    if rc:
        raise RuntimeError(lib.xpuint8_last_error())
    return out


def dequant(lib, acc, xs, ws, bias, out):
    m, n = acc.shape
    rc = lib.xpuint8_dequant_epilogue(acc.data_ptr(), out.data_ptr(), xs.data_ptr(),
                                      ws.data_ptr(), bias.data_ptr(), int(m), int(n))
    if rc:
        raise RuntimeError(lib.xpuint8_last_error())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument("--k", type=int, default=3072)
    ap.add_argument("--n", type=int, default=3072)
    ap.add_argument("--group", type=int, default=256)
    args = ap.parse_args()

    lib = load_lib()
    dev = "xpu"
    K, N, GS = args.k, args.n, args.group

    w8 = torch.randint(-127, 127, (N, K), device=dev, dtype=torch.int8)
    wt = w8.t().contiguous()                      # [K,N] for _int_mm
    wscale = torch.rand(N, device=dev, dtype=torch.float32) * 0.01 + 0.001
    bias = (torch.rand(N, device=dev, dtype=torch.bfloat16) - 0.5) * 0.1
    # ConvRot stores weights pre-rotated; dequantizing them gives the bf16
    # weight the BF16 path must use. Same numbers as the int8 path.
    wbf = (w8.float() * wscale.unsqueeze(1)).to(torch.bfloat16)

    print(f"device: {lib.xpuint8_device_name().decode()}   K=N={K}  gs={GS}")
    print()
    print(f"{'M':>6} {'rotate+quant':>13} {'rotate bf16':>12} {'_int_mm':>9} "
          f"{'epilogue':>9} {'INT8 tot':>9} {'BF16 tot':>9} {'int8/bf16':>10}")
    rows = []
    for M in args.m:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        q = torch.empty(M, K, device=dev, dtype=torch.int8)
        xs = torch.empty(M, device=dev, dtype=torch.float32)
        xr = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)

        t_rq = bench(lambda: rotate_quantize(lib, x))
        t_rb = bench(lambda: rotate_bf16(lib, x, xr))
        acc = torch._int_mm(q, wt)
        t_mm = bench(lambda: torch._int_mm(q, wt))
        t_ep = bench(lambda: dequant(lib, acc, xs, wscale, bias, out))
        t_lin = bench(lambda: torch.nn.functional.linear(xr, wbf, bias))

        int8_total = t_rq + t_mm + t_ep
        bf16_total = t_rb + t_lin
        rows.append((M, t_rq, t_rb, t_mm, t_ep, int8_total, bf16_total))
        print(f"{M:6d} {t_rq:12.3f}m {t_rb:11.3f}m {t_mm:8.3f}m {t_ep:8.3f}m "
              f"{int8_total:8.3f}m {bf16_total:8.3f}m {int8_total / bf16_total:9.2f}x")
        del x, q, xs, xr, out, acc

    # --- correctness of the bf16 rotate against float64 ---
    M = args.m[0]
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    xr = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
    rotate_bf16(lib, x, xr)
    hd = _build_hadamard(GS, dtype=torch.float64)
    xd = x.double().cpu().reshape(M, K // GS, GS)
    ref64 = (xd @ hd).reshape(M, K).to(torch.bfloat16)
    got = xr.cpu()
    rel = ((got.double() - ref64.double()).norm() / ref64.double().norm()).item()
    print(f"\nbf16 rotate vs float64 reference @ M={M}: relative L2 error {rel:.3e}")
    print("(this is the accuracy a ConvRot BF16 path would deliver)")

    print("\nSummary: INT8 pays for the rotation once and then gets a cheaper GEMM;")
    print("BF16 pays for the rotation once and then gets a free bf16 GEMM.")


if __name__ == "__main__":
    main()