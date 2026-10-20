"""True per-step device-time budget for the INT8 path, at real MossImage shapes.

The host-side timers in comfy/xpu_int8.py measure enqueue, not execution:
torch._int_mm on XPU is fully async (0.085 ms host for a 75 GFLOP shape vs
0.542 ms synced). This measures each region with torch.xpu.synchronize() around
it and multiplies by the per-step call counts taken from the [xpu-shape] log.

Loads the ComfyUI library, not the side-project one -- the side-project build
is ~5x slower on the dequant epilogue and would misrepresent the real budget.

    python debug/probe_int8_budget.py
"""
import ctypes
import os
import time

import torch

LIB = os.environ.get(
    "XPUINT8_LIB",
    "/home/okolenmi/comfy/ComfyUI/comfy/lib/libxpuint8.so",
)

# (M, N, K) -> calls per step, from the [xpu-shape] histogram at 20 steps
# (840 calls / 20 steps = 42 per step per (N,K), split across M=3973 and M=3974)
SHAPES = [
    (3973, 3072, 3072, 42),
    (3973, 3072, 8192, 42),
    (3973, 9216, 3072, 42),
    (3973, 16384, 3072, 42),
]

REPS, WARMUP = 20, 5

lib = ctypes.CDLL(LIB)
lib.xpuint8_init.restype = ctypes.c_int
lib.xpuint8_device_name.restype = ctypes.c_char_p
lib.xpuint8_last_error.restype = ctypes.c_char_p
lib.xpuint8_convrot_quantize.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 2
lib.xpuint8_convrot_quantize.restype = ctypes.c_int
lib.xpuint8_dequant_epilogue.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 2
lib.xpuint8_dequant_epilogue.restype = ctypes.c_int

assert lib.xpuint8_init() == 0, lib.xpuint8_last_error()
print("device:", lib.xpuint8_device_name().decode())


def synced(fn, reps=REPS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
        torch.xpu.synchronize()
    return (time.perf_counter() - t0) * 1e3 / reps


tot = {"quant": 0.0, "intmm": 0.0, "dequant": 0.0, "alloc": 0.0}
print()
print(f"{'M':>5} {'N':>6} {'K':>6} {'n/step':>6} {'quant ms':>9} {'gemm ms':>9} "
      f"{'deq ms':>8} {'->q/step':>9} {'->m/step':>9} {'->d/step':>9} {'TOPS':>7}")

for M, N, K, nstep in SHAPES:
    x = torch.randn(M, K, device="xpu", dtype=torch.bfloat16)
    w8 = torch.randint(-127, 127, (N, K), device="xpu", dtype=torch.int8)
    wt = w8.t().contiguous()
    ws = torch.rand(N, device="xpu", dtype=torch.float32) * 0.01 + 1e-3
    xs = torch.empty(M, device="xpu", dtype=torch.float32)
    bias = torch.randn(N, device="xpu", dtype=torch.bfloat16)
    acc = torch.empty(M, N, device="xpu", dtype=torch.int32)
    out = torch.empty(M, N, device="xpu", dtype=torch.bfloat16)
    q = torch.empty(M, K, device="xpu", dtype=torch.int8)

    def cq():
        lib.xpuint8_convrot_quantize(x.data_ptr(), q.data_ptr(), xs.data_ptr(), M, K)

    def mm():
        torch._int_mm(q, wt)

    def dq():
        lib.xpuint8_dequant_epilogue(acc.data_ptr(), out.data_ptr(),
                                      xs.data_ptr(), ws.data_ptr(),
                                      bias.data_ptr(), M, N)

    def alloc():
        torch.empty(M, K, device="xpu", dtype=torch.int8)
        torch.empty(M, device="xpu", dtype=torch.float32)
        torch.empty(M, N, device="xpu", dtype=torch.bfloat16)

    tq, tm, td, ta = synced(cq), synced(mm), synced(dq), synced(alloc)
    for k, v in (("quant", tq), ("intmm", tm), ("dequant", td), ("alloc", ta)):
        tot[k] += v * nstep
    tops = 2.0 * M * N * K / (tm / 1e3) / 1e12
    print(f"{M:5d} {N:6d} {K:6d} {nstep:6d} {tq:9.3f} {tm:9.3f} {td:8.3f} "
          f"{tq * nstep:9.1f} {tm * nstep:9.1f} {td * nstep:9.1f} {tops:7.1f}")
    del x, w8, wt, ws, xs, bias, acc, out, q
    torch.xpu.synchronize()

print()
s = sum(tot.values())
for k, v in tot.items():
    print(f"  {k:8s} {v:7.1f} ms/step   {100 * v / s:5.1f}%")
print(f"  {'TOTAL':8s} {s:7.1f} ms/step")
