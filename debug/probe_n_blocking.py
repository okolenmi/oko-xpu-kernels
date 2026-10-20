import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Pipeline-level experiments:

1. N-blocking: split the GEMM into column blocks so the int32 accumulator
   (50 MB monolithic) stays resident in the 18 MB L2 instead of streaming to
   DRAM. Removes ~50 MB of DRAM round-trip for the dequant epilogue.
2. L2-resident vs streaming epilogue cost.
3. Is the BF16 comparison fair? ConvRot needs the activation rotation in the
   BF16 path too -- measure rotate-only-BF16 vs rotate+quantize+int8.
"""
import ctypes, os, time
import torch

LIB = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libxpuint8.so")) \
    if os.path.exists(os.path.join(os.path.dirname(__file__), "libxpuint8.so")) else None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_lib  # noqa
LIB = load_lib()
LIB.xpuint8_init()
LIB.xpuint8_convrot_quantize.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int64]*2
LIB.xpuint8_dequant_epilogue.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int64]*2
LIB.xpuint8_dequant_epilogue.restype = ctypes.c_int
LIB.xpuint8_convrot_quantize.restype = ctypes.c_int


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


def quant(x):
    m, k = x.shape
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty(m, device=x.device, dtype=torch.float32)
    LIB.xpuint8_convrot_quantize(x.data_ptr(), q.data_ptr(), s.data_ptr(), m, k)
    return q, s


def epi(acc, xs, ws, bias):
    m, n = acc.shape
    out = torch.empty(m, n, device=acc.device, dtype=torch.bfloat16)
    LIB.xpuint8_dequant_epilogue(acc.data_ptr(), out.data_ptr(), xs.data_ptr(),
                                 ws.data_ptr(), bias.data_ptr(), m, n)
    return out


dev = "xpu"
K = N = 3072
M = 4096
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w8 = torch.randint(-127, 127, (N, K), device=dev, dtype=torch.int8)
wt = w8.t().contiguous()
ws = torch.rand(N, device=dev, dtype=torch.float32) * 0.01
bias = torch.zeros(N, device=dev, dtype=torch.bfloat16)

q, xs = quant(x)

print("--- monolithic ---")
t_gemm = bench(lambda: torch._int_mm(q, wt))
acc_full = torch._int_mm(q, wt)
t_epi = bench(lambda: epi(acc_full, xs, ws, bias))
print(f"  gemm (int32 4096x3072 = 50MB) : {t_gemm:.3f} ms")
print(f"  epilogue                       : {t_epi:.3f} ms")
print(f"  total                          : {t_gemm + t_epi:.3f} ms")

print("--- N-blocked so acc stays in the 18 MB L2 ---")
for nb in (384, 768, 1536):
    cols = [min(nb, N - s) for s in range(0, N, nb)]

    def blocked():
        for s, c in zip(range(0, N, nb), cols):
            accb = torch._int_mm(q, wt[:, s:s + c])
            epi(accb, xs, ws[s:s + c], bias[s:s + c])

    # contiguous-slice variant (requires the weight stored [N,K] and sliced on dim0)
    def blocked_ctg():
        for s, c in zip(range(0, N, nb), cols):
            wb = w8[s:s + c].t()
            accb = torch._int_mm(q, wb)
            epi(accb, xs, ws[s:s + c], bias[s:s + c])

    tb = bench(blocked)
    tc = bench(blocked_ctg)
    print(f"  block {nb:5d} ({len(cols)} blocks) strided-B {tb:.3f} ms | "
          f"row-sliced B {tc:.3f} ms")

print("--- epilogue with a small acc (L2 resident) ---")
for c in (384, 768, 1536):
    a = torch.zeros(M, c, device=dev, dtype=torch.int32)
    print(f"  acc {M}x{c} ({M*c*4/1e6:.1f} MB): "
          f"{bench(lambda: epi(a, xs, ws[:c], bias[:c])):.3f} ms")

print("--- fair BF16 comparison: does the BF16 path need the rotation too? ---")
wbf = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
print(f"  plain bf16 linear (no rotation)          : "
      f"{bench(lambda: torch.nn.functional.linear(x, wbf, bias)):.3f} ms")