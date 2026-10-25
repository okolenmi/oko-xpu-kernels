"""The Triton weight dequantizer against the SYCL one, at SDXL's shapes, and a sweep of its launch configuration.

Per shape: time, bytes per second (3 bytes per parameter: the int8 in, the bf16 out) against the 397 GB/s copy ceiling, and how
many values differ from the SYCL kernel's output.  Then BLOCK_ROWS x BLOCK_COLS x warps at one shape.

    LEVEL_ZERO_V1_SDK_PATH=... python debug/bench_dequant_triton.py
"""
import torch

from common import bench
from xpu_kernels import dequant_weight as sycl_dequant
from xpu_kernels.triton import dequant as D

dev = "xpu"
GBPS = 397.0
SHAPES = ((1280, 1280), (1280, 2048), (1280, 5120), (10240, 1280), (5120, 1280), (2560, 2560))


def gbps(n, k, ms):
    return n * k * 3 / (ms / 1000) / 1e9


print("gate:", D.gates()["dequant_weight_triton"].describe()[:90])
gen = torch.Generator(device=dev).manual_seed(0)
print("shape            sycl ms   GB/s   triton ms   GB/s   differing values")
for n, k in SHAPES:
    w8 = torch.randint(-127, 128, (n, k), generator=gen, device=dev, dtype=torch.int8)
    s = torch.rand(n, generator=gen, device=dev) * 0.02 + 0.001
    a = sycl_dequant(w8, s)
    b = D.dequant_weight(w8, s)
    bad = int((a.view(torch.int16) != b.view(torch.int16)).sum())
    ta = bench(lambda: sycl_dequant(w8, s))
    tb = bench(lambda: D.dequant_weight(w8, s))
    print(f"{n:5d} x {k:5d} {ta:9.3f} {gbps(n, k, ta):6.0f} {tb:11.3f} {gbps(n, k, tb):6.0f}   {bad}/{n * k}")

n, k = 1280, 5120
w8 = torch.randint(-127, 128, (n, k), generator=gen, device=dev, dtype=torch.int8)
s = torch.rand(n, generator=gen, device=dev) * 0.02 + 0.001
out = torch.empty((n, k), dtype=torch.bfloat16, device=dev)
rot = D.prepare(dev)
print(f"sweep at {n} x {k} (floor {n * k * 3 / (GBPS * 1e9) * 1000:.3f} ms)")
print("rows  cols warps   ms")
for rows in (16, 32, 64, 128):
    for cols in (32, 64, 128, 256):
        for warps in (4, 8):
            grid = ((n + rows - 1) // rows, k // 256)
            try:
                t = bench(lambda: D._kernel[grid](w8, s, rot, out, n, k, BM=rows, BJ=cols, num_warps=warps), n=10)
                print(f"{rows:4d} {cols:5d} {warps:5d} {t:8.3f}")
            except Exception as exc:  # noqa: BLE001 - a configuration the backend refuses is a result too
                print(f"{rows:4d} {cols:5d} {warps:5d}   failed: {str(exc)[:60]}")
