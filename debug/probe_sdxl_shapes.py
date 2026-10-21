"""The INT8-vs-BF16 ratio at *SDXL's* shapes.

`bench_bf16_fair.py` measures one shape (K=N=3072) and a sweep of M.  SDXL has
neither: its linear layers are (K,N) in {640, 1280, 2048, 2560, 2816, 5120,
10240} x {320, 640, 1280, 5120}, and the token count M depends on the level
(16384 at 128x128, 4096 at 64x64, 1024 at 32x32, 256 at 16x16, 64 at 8x8).

Both paths pay for the activation rotation, because ConvRot stores the
weights pre-rotated:

    INT8 : rotate+quantize -> torch._int_mm -> the dequant epilogue
    BF16 : rotate only      -> F.linear

Two numbers are reported for each arm, because they answer different
questions and they are not the same:

  * **device** -- N iterations enqueued, then a sync: the time the *device*
    spends.  This is what a captured step pays.
  * **host** -- the same N iterations without the sync: the time the *host*
    spends enqueuing.  This is what an eager step pays, and on this card the
    eager step is host-bound.

The arms are interleaved inside one process (the card drifts 10-30 % with
thermals, so a sequential comparison invents effects).

    PYTHONPATH=python python debug/probe_sdxl_shapes.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import xpu_kernels as xk  # noqa: E402

# (K, N) of SDXL's UNet linears, with the parameter count of that shape in the
# whole UNet (measured from the diffusers config: 2.567 B parameters).
SHAPES = [
    (1280, 10240, 786.4),
    (1280, 1280, 624.2),
    (5120, 1280, 393.2),
    (2048, 1280, 314.6),
    (640, 5120, 32.8),
    (640, 640, 28.7),
]

# the token counts of the UNet's levels at 1024x1024 (latent 128x128)
MS = [4096, 1024, 256, 64]


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
    if not xk.available():
        print("no libxpuint8.so; build it with ./build.sh")
        return
    print(f"device: {xk.device_name()}")
    print()
    print(f"{'K':>6} {'N':>6} {'M':>6} | {'INT8 dev':>9} {'BF16 dev':>9} {'dev ratio':>9}"
          f" | {'INT8 host':>9} {'BF16 host':>9} | {'rotate+q':>9} {'_int_mm':>8}"
          f" {'epilogue':>8} {'rot bf16':>8} {'F.linear':>8}")
    for K, N, params in SHAPES:
        if K % 256:
            print(f"{K:6d} {N:6d}      - |  K is not a multiple of 256: the INT8 path does not apply")
            continue
        torch.manual_seed(0)
        w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
        w8, ws = xk.prepare_weight(w)
        wt = w8.t().contiguous()          # [K,N] for _int_mm
        wbf = xk.dequant_weight(w8, ws)    # the same weight, in bf16
        del w
        bias = torch.zeros(N, device=dev, dtype=torch.bfloat16)
        for M in MS:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)

            def int8():
                xk.linear(x, w8, ws, bias)

            def bf16():
                torch.nn.functional.linear(xk.rotate(x), wbf, bias)

            # device time, interleaved: one arm, then the other, alternately
            d8 = min(timed(int8) for _ in range(3))
            db = min(timed(bf16) for _ in range(3))
            h8 = min(timed(int8, sync=False) for _ in range(3))
            hb = min(timed(bf16, sync=False) for _ in range(3))

            # the parts, for the diagnosis
            q, xs = xk.quantize(x)
            acc = torch._int_mm(q, wt)
            t_rq = timed(lambda: xk.quantize(x))
            t_mm = timed(lambda: torch._int_mm(q, wt))
            t_ep = timed(lambda: xk.dequant(acc, xs, ws, bias))
            t_rb = timed(lambda: xk.rotate(x))
            t_lin = timed(lambda: torch.nn.functional.linear(x, wbf, bias))

            print(f"{K:6d} {N:6d} {M:6d} | {d8:8.3f}m {db:8.3f}m {d8/db:8.2f}x"
                  f" | {h8:8.3f}m {hb:8.3f}m | {t_rq:8.3f}m {t_mm:7.3f}m"
                  f" {t_ep:7.3f}m {t_rb:7.3f}m {t_lin:7.3f}m")
            del x, q, xs, acc
        print()


if __name__ == "__main__":
    main()
