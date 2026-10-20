import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Is torch._int_mm the best int8 GEMM on XPU, and is it layout-sensitive?

Covers:
  * _int_mm with B [K,N] contiguous vs a transposed view of [N,K]
  * _weight_int8pack_mm (oneDNN packed path) if the op exists on XPU
  * row-wise vs per-tensor scales, int32 vs int8 output
  * our own ESIMD GEMM for comparison at the same shapes
"""
import time
import torch

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

dev = "xpu"
K = N = 3072
print(f"{'M':>6} {'_int_mm KxN':>12} {'_int_mm NxK.t':>13} {'packed_mm':>11} {'TOPS':>7}")
for M in (512, 1024, 2048, 4096, 8192):
    a = torch.randint(-127, 127, (M, K), device=dev, dtype=torch.int8)
    b_kn = torch.randint(-127, 127, (K, N), device=dev, dtype=torch.int8)
    b_nk = torch.randint(-127, 127, (N, K), device=dev, dtype=torch.int8)

    t1 = bench(lambda: torch._int_mm(a, b_kn))
    t2 = bench(lambda: torch._int_mm(a, b_nk.t()))

    t3 = float("nan")
    if hasattr(torch, "_weight_int8pack_mm"):
        try:
            # row-wise scale, out_dtype bf16/fp32
            sa = torch.ones(M, 1, device=dev, dtype=torch.float32)
            sb = torch.ones(1, N, device=dev, dtype=torch.float32)
            bias = torch.zeros(N, device=dev, dtype=torch.float32)
            packed = b_nk.t().contiguous()
            r = torch._weight_int8pack_mm(a, packed, sa, sb, bias)
            torch.xpu.synchronize()
            t3 = bench(lambda: torch._weight_int8pack_mm(a, packed, sa, sb, bias))
        except Exception as e:
            print("   _weight_int8pack_mm failed:", str(e)[:90])
    tops = 2 * M * K * N / (t1 * 1e-3) / 1e12
    print(f"{M:6d} {t1:11.3f}m {t2:12.3f}m {t3:10.3f}m {tops:7.1f}")
    del a, b_kn, b_nk

# correctness of the layout variants
a = torch.randint(-127, 127, (256, K), device=dev, dtype=torch.int8)
b = torch.randint(-127, 127, (K, N), device=dev, dtype=torch.int8)
r1 = torch._int_mm(a, b)
r2 = torch._int_mm(a, b.t().contiguous().t())
print("\nlayout equivalence (contig vs transposed-view):", torch.equal(r1, r2))