import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time, torch
M=N=K=1024
xi = torch.randint(-127,127,(M,K),device="xpu",dtype=torch.int8)
wi = torch.randint(-127,127,(N,K),device="xpu",dtype=torch.int8)
x  = torch.randn(M,K,device="xpu",dtype=torch.bfloat16)
w  = torch.randn(N,K,device="xpu",dtype=torch.bfloat16)
def bench(f,n=20):
    for _ in range(3): f()
    torch.xpu.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.xpu.synchronize(); return (time.perf_counter()-t)/n*1000
bf  = bench(lambda: x@w.t())
i8  = bench(lambda: torch._int_mm(xi,wi.t()))
flops = 2*M*N*K
print(f"torch BF16  : {bf:8.3f} ms  {flops/bf*1e-9:7.2f} TFLOPS")
print(f"torch _int_mm: {i8:7.3f} ms  {flops/i8*1e-9:7.2f} TOPS")
