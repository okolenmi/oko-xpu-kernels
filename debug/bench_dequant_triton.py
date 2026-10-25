"""The two-pass weight dequantizer (Triton scale + oneDNN product) against the SYCL one at SDXL's shapes.

Prints per shape: the SYCL kernel, the Triton elementwise half alone (its floor is 3 bytes per parameter at 397 GB/s), the
product alone, the two together, and how many values differ from the SYCL kernel's output.  No launch sweep: one configuration.

    LEVEL_ZERO_V1_SDK_PATH=~/.local/levelzero python debug/bench_dequant_triton.py
"""
import torch

from common import bench
from xpu_kernels import dequant_weight as sycl_dequant
from xpu_kernels.triton import dequant as D

dev = "xpu"
SHAPES = ((1280, 1280), (1280, 2048), (1280, 5120), (10240, 1280), (5120, 1280), (2560, 2560))
print("gate:", D.gates()["dequant_weight_triton"].describe()[:110])
gen = torch.Generator(device=dev).manual_seed(0)
print("shape            SYCL ms   scale ms  product ms  both ms   speed-up   differing values")
for n, k in SHAPES:
    w8 = torch.randint(-127, 128, (n, k), generator=gen, device=dev, dtype=torch.int8)
    s = torch.rand(n, generator=gen, device=dev) * 0.02 + 0.001
    a, b = sycl_dequant(w8, s), D.dequant_weight(w8, s)
    bad = int((a.view(torch.int16) != b.view(torch.int16)).sum())
    scaled = D.scale_to_bf16(w8, s)
    rot = D.prepare(dev)
    t_sycl = bench(lambda: sycl_dequant(w8, s))
    t_scale = bench(lambda: D.scale_to_bf16(w8, s))
    t_mm = bench(lambda: torch.mm(scaled.view(-1, 256), rot))
    t_both = bench(lambda: D.dequant_weight(w8, s))
    print(f"{n:5d} x {k:5d} {t_sycl:9.3f} {t_scale:10.3f} {t_mm:11.3f} {t_both:9.3f} {t_sycl / t_both:9.2f}x   {bad}/{n * k}")
