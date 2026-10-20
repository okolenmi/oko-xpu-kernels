"""Reproduce the model's q/k/v linear chain and attribute device time per kernel.

In-model the torch profiler attributes 0.43 ms/call to the dequant epilogue
(126 calls/forward).  The same library entry point on torch tensors costs
0.259 ms standalone (debug/bench_epilogue.py).  This script runs the exact
production chain -- quantize -> _int_mm -> dequant epilogue -- in a loop and
prints the per-kernel device time, so we can see whether the extra 0.17 ms is
a property of the kernel or of the data it reads.

  PYTHONPATH=/home/okolenmi/comfy/ComfyUI \
  /home/okolenmi/comfy/venv/bin/python debug/bench_qkv_chain.py
"""
import ctypes
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")
import comfy.xpu_int8 as xi  # noqa: E402

M, N, K = 3974, 3072, 3072
dev = torch.device("xpu:0")
torch.xpu.set_device(0)

lib = xi._load()
assert lib is not None and lib.xpuint8_init() == 0, lib.xpuint8_last_error()

x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w8 = torch.randint(-127, 127, (N, K), device=dev, dtype=torch.int8)
wsc = torch.rand(N, device=dev, dtype=torch.float32) * 0.01
xs = torch.empty(M, device=dev, dtype=torch.float32)
q8 = torch.empty(M, K, device=dev, dtype=torch.int8)
out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)

sq = ctypes.c_void_p(torch.xpu.current_stream(dev).sycl_queue)
P = ctypes.c_void_p


def quant():
    rc = lib.xpuint8_convrot_quantize_stream(sq, P(x.data_ptr()), P(q8.data_ptr()),
                                             P(xs.data_ptr()), M, K)
    assert rc == 0, lib.xpuint8_last_error()


def epi(acc):
    rc = lib.xpuint8_dequant_epilogue_stream(sq, P(acc.data_ptr()), P(out.data_ptr()),
                                             P(xs.data_ptr()), P(wsc.data_ptr()),
                                             ctypes.c_void_p(0), M, N)
    assert rc == 0, lib.xpuint8_last_error()


def chain():
    quant()
    acc = torch._int_mm(q8, w8.t())
    epi(acc)


# A second accumulator that no GEMM just wrote, to separate "cold read" from
# "read of a fresh GEMM output".
acc_cold = torch.empty(M, N, device=dev, dtype=torch.int32)


def show(name, fn, iters):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.XPU]) as prof:
        for _ in range(iters):
            fn()
        torch.xpu.synchronize()
    rows = []
    for ev in prof.key_averages():
        if ev.self_device_time_total > 0 and ev.count:
            rows.append((ev.self_device_time_total / ev.count / 1000.0, ev.count, ev.key))
    rows.sort(reverse=True)
    print(f"\n{name}  ({iters} iters)")
    for ms, cnt, key in rows:
        short = key if len(key) <= 70 else key[:67] + "..."
        print(f"  {ms:8.4f} ms/call  n={cnt:4d}  {short}")


show("full chain: quantize + int_mm + epilogue", chain, 10)
show("epilogue only, accumulator just written by a GEMM",
     lambda: epi(torch._int_mm(q8, w8.t())), 10)
show("epilogue only, cold accumulator", lambda: epi(acc_cold), 10)
