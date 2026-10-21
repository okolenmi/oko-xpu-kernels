"""SDXL's FFN: the fused GEGLU path against the unfused one.

SDXL's FFN is a GEGLU (comfy/ldm/modules/attention.py:166):

    x, gate = proj(x).chunk(2, -1)
    out = x * F.gelu(gate)

with ``proj`` a Linear(1280 -> 10240) and the second projection a
Linear(5120 -> 1280).  The INT8 path has to do the same work in int8, and
the question is what the *intermediate* costs: the unfused chain writes the
[M, 10240] bf16 tile, reads it back for the activation, writes the product
and reads that again for the quantizer.  The fused kernel reads the int32
accumulator once and writes the int8 result.

Three arms, interleaved:

    fused    : the fused kernel -- dequant, activate, rotate, quantize
    unfused  : the same INT8 path written the obvious way
    bf16     : the whole FFN in bf16 (the baseline a model would run)

    PYTHONPATH=python python debug/probe_sdxl_ffn.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import xpu_kernels  # noqa: E402
from xpu_kernels import int8  # noqa: E402

# the UNet's levels at 1024x1024 (latent 128x128): M is the token count
MS = [4096, 1024, 256, 64]
DIM, FFN = 1280, 5120


def timed(fn, n=20, warmup=5, reps=3, sync=True):
    """ms per call: the device time (sync) or the host enqueue time (no sync)."""
    best = float("inf")
    for _ in range(reps):
        for _ in range(warmup):
            fn()
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if sync:
            torch.xpu.synchronize()
        best = min(best, (time.perf_counter() - t0) / n * 1000)
    return best


def main():
    dev = "xpu"
    if not int8.available():
        print("no libxpuint8.so; build it with ./build.sh")
        return
    print(f"device: {int8.device_name()}")
    print(f"SDXL's FFN: Linear({DIM} -> {2 * FFN}) -> chunk -> x * gelu(gate) "
          f"-> Linear({FFN} -> {DIM})")
    print()
    print(f"{'M':>6} | {'fused':>9} {'unfused':>9} {'bf16':>9} | "
          f"{'fused/unfused':>15} {'fused/bf16':>11}")

    torch.manual_seed(0)
    up = (torch.randn(2 * FFN, DIM, device=dev) * 0.05).to(torch.bfloat16)
    up8, up_scale = int8.prepare_weight(up)
    down = (torch.randn(DIM, FFN, device=dev) * 0.05).to(torch.bfloat16)
    down8, down_scale = int8.prepare_weight(down)
    up_bf = int8.dequant_weight(up8, up_scale)     # the same weight, in bf16
    down_bf = int8.dequant_weight(down8, down_scale)
    del up, down

    for M in MS:
        x = (torch.randn(M, DIM, device=dev) * 0.5).to(torch.bfloat16)

        def fused():
            return int8.ffn(x, up8, up_scale, down8, down_scale, act="geglu_erf")

        def unfused():
            q, xs = int8.quantize(x)
            acc = torch._int_mm(q, up8.t())
            bf = int8.dequant(acc, xs, up_scale)
            hidden, gate = bf.chunk(2, dim=-1)
            y = hidden * torch.nn.functional.gelu(gate)
            q2, xs2 = int8.quantize(y)
            acc2 = torch._int_mm(q2, down8.t())
            return int8.dequant(acc2, xs2, down_scale)

        def bf16():
            h = torch.nn.functional.linear(x, up_bf)
            hidden, gate = h.chunk(2, dim=-1)
            y = hidden * torch.nn.functional.gelu(gate)
            return torch.nn.functional.linear(y, down_bf)

        # the two INT8 arms must agree, or the comparison is meaningless
        a, b = fused(), unfused()
        same = torch.equal(a, b)

        tf = min(timed(fused) for _ in range(3))
        tu = min(timed(unfused) for _ in range(3))
        tb = min(timed(bf16) for _ in range(3))
        print(f"{M:6d} | {tf:8.3f}m {tu:8.3f}m {tb:8.3f}m | "
              f"{tf / tu:14.2f}x {tf / tb:10.2f}x   identical: {same}")
        del x
    print()
    print("the bf16 arm does not rotate the activation, so it is the *model's*")
    print("speed, not the INT8 path's: the INT8 path pays for the rotation,")
    print("which is why the fair comparison in probe_sdxl_shapes.py rotates on")
    print("both sides.")


if __name__ == "__main__":
    main()
