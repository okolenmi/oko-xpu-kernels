"""The two-pass weight dequantizer (Triton scale + oneDNN product) against the SYCL one at SDXL's shapes.

Prints per shape the *device* time (what a captured graph replays) of the SYCL kernel, the Triton elementwise half alone (its
floor is 3 bytes per parameter at 397 GB/s), the product alone and the two together, then the wall time of the SYCL kernel and
of the two together (launch cost included: small shapes are launch-bound there), and how many values differ from the SYCL
kernel's output.  No launch sweep: one configuration.

    LEVEL_ZERO_V1_SDK_PATH=~/.local/levelzero python debug/bench_dequant_triton.py
"""
import torch

from common import bench, device_ms
from xpu_kernels import dequant_weight as sycl_dequant
from xpu_kernels.triton import dequant as D

dev = "xpu"
SHAPES = ((1280, 1280), (1280, 2048), (1280, 5120), (10240, 1280), (5120, 1280), (2560, 2560))
print("gate:", D.gates()["dequant_weight_triton"].describe()[:110])
gen = torch.Generator(device=dev).manual_seed(0)
print("                      ---- device ms ----                      ---- wall ms ----")
print("shape            SYCL   scale  product   both  speed-up     SYCL   both   differing values")
for n, k in SHAPES:
    w8 = torch.randint(-127, 128, (n, k), generator=gen, device=dev, dtype=torch.int8)
    s = torch.rand(n, generator=gen, device=dev) * 0.02 + 0.001
    a, b = sycl_dequant(w8, s), D.dequant_weight(w8, s)
    bad = int((a.view(torch.int16) != b.view(torch.int16)).sum())
    scaled = D.scale_to_bf16(w8, s)
    rot = D.prepare(dev)
    d_sycl, d_scale = device_ms(lambda: sycl_dequant(w8, s)), device_ms(lambda: D.scale_to_bf16(w8, s))
    d_mm, d_both = device_ms(lambda: torch.mm(scaled.view(-1, 256), rot)), device_ms(lambda: D.dequant_weight(w8, s))
    w_sycl, w_both = bench(lambda: sycl_dequant(w8, s)), bench(lambda: D.dequant_weight(w8, s))
    print(f"{n:5d} x {k:5d} {d_sycl:6.3f} {d_scale:7.3f} {d_mm:8.3f} {d_both:6.3f} {d_sycl / d_both:8.2f}x {w_sycl:8.3f} {w_both:6.3f}   {bad}/{n * k}")
