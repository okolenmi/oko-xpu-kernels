import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""How expensive are the temporary allocations in the fused pipeline?

breakdown.py preallocated `out`; blocking.py allocated it inside the timed
region and the epilogue went 0.146 -> 0.275 ms. That gap is allocator cost,
not kernel cost. Measure it directly, since in ComfyUI these temporaries are
allocated on every call.
"""
import time
import torch


def bench(fn, n=50, warmup=10, reps=3):
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
M, K, N = 4096, 3072, 3072
print("torch", torch.__version__)

print("\n--- raw allocation cost ---")
for name, shape, dt in [
    ("int8  [4096,3072] 12.6MB", (M, K), torch.int8),
    ("bf16  [4096,3072] 25.2MB", (M, N), torch.bfloat16),
    ("int32 [4096,3072] 50.3MB", (M, N), torch.int32),
    ("fp32  [4096]      16KB",  (M,), torch.float32),
    ("int8  [1]         tiny",  (1,), torch.int8),
]:
    t = bench(lambda: torch.empty(shape, device=dev, dtype=dt))
    print(f"  torch.empty {name}: {t:.4f} ms")

print("\n--- does allocating then freeing the same shape reuse a block? ---")
keep = []
def alloc_free():
    y = torch.empty((M, N), device=dev, dtype=torch.bfloat16)
    return y
print(f"  alloc + immediate drop : {bench(alloc_free):.4f} ms")

buf = torch.empty((M, N), device=dev, dtype=torch.bfloat16)
print(f"  (preallocated once)    : 0.0000 ms")

print("\n--- ctypes call overhead (kernel launch, no work) ---")
import ctypes, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_lib
lib = load_lib()
lib.xpuint8_init()
lib.xpuint8_convrot_quantize.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int64]*2
lib.xpuint8_convrot_quantize.restype = ctypes.c_int
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
q = torch.empty(M, K, device=dev, dtype=torch.int8)
s = torch.empty(M, device=dev, dtype=torch.float32)
xp, qp, sp = x.data_ptr(), q.data_ptr(), s.data_ptr()

def call():
    lib.xpuint8_convrot_quantize(xp, qp, sp, M, K)
print(f"  ctypes + kernel        : {bench(call):.4f} ms")

print(f"  M=1    (launch only)   : ", end="")
x1 = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
q1 = torch.empty(1, K, device=dev, dtype=torch.int8)
s1 = torch.empty(1, device=dev, dtype=torch.float32)
print(f"{bench(lambda: lib.xpuint8_convrot_quantize(x1.data_ptr(), q1.data_ptr(), s1.data_ptr(), 1, K)):.4f} ms")

print("\n--- is a persistent workspace enough? full pipeline with preallocated ---")
w8 = torch.randint(-127, 127, (N, K), device=dev, dtype=torch.int8)
wt = w8.t().contiguous()
ws = torch.rand(N, device=dev, dtype=torch.float32) * 0.01
bias = torch.randn(N, device=dev, dtype=torch.bfloat16)
out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
lib.xpuint8_dequant_epilogue.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int64]*2
lib.xpuint8_dequant_epilogue.restype = ctypes.c_int

def prealloc():
    lib.xpuint8_convrot_quantize(xp, qp, sp, M, K)
    a = torch._int_mm(q, wt)
    lib.xpuint8_dequant_epilogue(a.data_ptr(), out.data_ptr(), sp, ws.data_ptr(),
                                 bias.data_ptr(), M, N)
    return out

def withalloc():
    qq = torch.empty(M, K, device=dev, dtype=torch.int8)
    ss = torch.empty(M, device=dev, dtype=torch.float32)
    oo = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    lib.xpuint8_convrot_quantize(xp, qq.data_ptr(), ss.data_ptr(), M, K)
    a = torch._int_mm(qq, wt)
    lib.xpuint8_dequant_epilogue(a.data_ptr(), oo.data_ptr(), ss.data_ptr(),
                                 ws.data_ptr(), bias.data_ptr(), M, N)
    return oo

print(f"  preallocated buffers   : {bench(prealloc):.4f} ms")
print(f"  allocate per call      : {bench(withalloc):.4f} ms")