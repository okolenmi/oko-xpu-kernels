import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Where does the end-to-end fused pipeline time actually go?

Stage times are min-of-3-reps of mean-of-20; see debug/README.md.
"""
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.bench_pipeline import fused_convrot_quantize, fused_dequant  # noqa
from common import bench, load_lib  # noqa  -- min-of-reps timing

lib = load_lib()
M, K, N = 4096, 3072, 3072
dev = "xpu"
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w8 = torch.randint(-127, 127, (N, K), device=dev, dtype=torch.int8)
wt = w8.t().contiguous()
ws = torch.rand(N, device=dev, dtype=torch.float32) * 0.01 + 0.001
bias = torch.randn(N, device=dev, dtype=torch.bfloat16)


# preallocated so allocation cost is excluded
q = torch.empty(M, K, device=dev, dtype=torch.int8)
xs = torch.empty(M, device=dev, dtype=torch.float32)
out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)


def cq():
    lib.xpuint8_convrot_quantize(x.data_ptr(), q.data_ptr(), xs.data_ptr(), M, K)


def mm():
    return torch._int_mm(q, wt)


acc = torch._int_mm(q, wt)


def ep():
    lib.xpuint8_dequant_epilogue(acc.data_ptr(), out.data_ptr(), xs.data_ptr(),
                                 ws.data_ptr(), bias.data_ptr(), M, N)


def all_fused():
    qa, xa = fused_convrot_quantize(lib, x)
    a = torch._int_mm(qa, wt)
    return fused_dequant(lib, a, xa, ws, bias)


t_cq = bench(cq)
t_mm = bench(mm)
t_ep = bench(ep)
t_all = bench(all_fused)
print(f"rotate+quantize (kernel only) : {t_cq:7.3f} ms")
print(f"_int_mm                       : {t_mm:7.3f} ms")
print(f"dequant epilogue (kernel only) : {t_ep:7.3f} ms")
print(f"sum of stages                 : {t_cq + t_mm + t_ep:7.3f} ms")
print(f"end-to-end (python + allocs)  : {t_all:7.3f} ms")
print(f"--> host/dispatch overhead    : {t_all - (t_cq + t_mm + t_ep):7.3f} ms"
      f"  ({100 * (t_all - (t_cq + t_mm + t_ep)) / t_all:.0f}%)")