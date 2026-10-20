"""Measure the installed dequant epilogue at the production shape on *torch*
tensors (device USM), not on the probe's malloc_shared buffers.

The SYCL probe (debug/native/probe_epilogue) reports 0.135 ms for
M=4096/N=3072, i.e. ~560 GB/s, but the in-model torch profile shows
0.43 ms/call over 126 calls/forward.  Same kernel, 3x difference, so measure
the real library entry point on torch tensors to find out which side is right.

  PYTHONPATH=/home/okolenmi/comfy/ComfyUI \
  /home/okolenmi/comfy/venv/bin/python debug/bench_epilogue.py [M] [N]
"""
import ctypes
import os
import sys
import time

import torch

sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")
import comfy.xpu_int8 as xi  # noqa: E402

M = int(sys.argv[1]) if len(sys.argv) > 1 else 3974
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3072

lib = xi._load()
assert lib is not None, "libxpuint8 did not load"
assert lib.xpuint8_init() == 0, lib.xpuint8_last_error()

dev = torch.device("xpu:0")
torch.xpu.set_device(0)

acc = torch.randint(-(2 ** 20), 2 ** 20, (M, N), device=dev, dtype=torch.int32)
xs = torch.full((M,), 0.01, device=dev, dtype=torch.float32)
ws = torch.full((N,), 0.02, device=dev, dtype=torch.float32)
bs = torch.full((N,), 0.5, device=dev, dtype=torch.float32)
out = torch.empty((M, N), device=dev, dtype=torch.bfloat16)

q = ctypes.c_void_p(torch.xpu.current_stream(dev).sycl_queue)
fn = lib.xpuint8_dequant_epilogue_stream

bytes_moved = M * N * 6


def run(bias, iters=50):
    bp = ctypes.c_void_p(bias.data_ptr()) if bias is not None else ctypes.c_void_p(0)
    # warm
    for _ in range(5):
        rc = fn(q, ctypes.c_void_p(acc.data_ptr()), ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(xs.data_ptr()), ctypes.c_void_p(ws.data_ptr()), bp, M, N)
        assert rc == 0, lib.xpuint8_last_error()
    torch.xpu.synchronize()
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            rc = fn(q, ctypes.c_void_p(acc.data_ptr()), ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_void_p(xs.data_ptr()), ctypes.c_void_p(ws.data_ptr()), bp, M, N)
        torch.xpu.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best


for bias, name in ((bs, "with bias"), (None, "no bias")):
    ms = run(bias) * 1e3
    print(f"lib epilogue {name:<10} {ms:7.4f} ms/call  {bytes_moved / 1e9 / (ms * 1e-3):6.0f} GB/s")

# Reference: what a plain elementwise torch op of the same traffic costs.
x32 = acc.to(torch.float32)
ms = 0.0
for _ in range(3):
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        y = x32.to(torch.bfloat16)
    torch.xpu.synchronize()
    ms = min(ms or 1e9, (time.perf_counter() - t0) / 20) if ms else (time.perf_counter() - t0) / 20
print(f"torch float32->bf16 cast {M}x{N} {ms * 1e3:7.4f} ms/call  "
      f"{(M * N * 6) / 1e9 / ms:6.0f} GB/s")
