import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.bench_pipeline import load_lib, fused_convrot_quantize  # noqa: E402
from comfy_kitchen.tensor.int8_utils import _build_hadamard  # noqa: E402
from comfy_kitchen.backends.eager import quantization as qz  # noqa: E402

lib = load_lib()
dev, dt = "xpu", torch.bfloat16

for K in (256, 1024, 3072, 4096, 8192):
    M = 8
    torch.manual_seed(0)
    x = torch.randn(M, K, device=dev, dtype=dt) * 0.5
    h = _build_hadamard(256, device=dev, dtype=dt)
    xr = qz._rotate_activation(x, h, 256)
    _, s_ref = qz.quantize_int8_rowwise(xr)
    _, s_got = fused_convrot_quantize(lib, x)

    # ground truth in fp64 on CPU
    xd = x.double().cpu()
    hd = h.double().cpu()
    ng = K // 256
    xr64 = (xd.reshape(M, ng, 256) @ hd).reshape(M, K)
    s_true = (xr64.abs().amax(dim=-1) / 127.0).float().to("xpu")

    rel = ((s_got - s_true).abs() / s_true)
    print(f"K={K:5d}  groups={ng:3d}  max rel vs fp64 truth = {rel.max().item():.3e}"
          f"   max rel vs eager bf16 = "
          f"{((s_ref - s_true).abs() / s_true).max().item():.3e}")