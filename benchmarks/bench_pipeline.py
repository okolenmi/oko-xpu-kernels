"""End-to-end ConvRot INT8 linear benchmark on Intel Arc (XPU).

Compares three paths at realistic MossImage-ish shapes:
  1. eager     - comfy-kitchen's eager backend (what ComfyUI uses today)
  2. fused     - xpu-int8 fused rotate+quantize and dequant epilogue, with
                 torch._int_mm for the GEMM
  3. bf16      - dequantized BF16 weights (the current ComfyUI default)

Run:
    python benchmarks/bench_pipeline.py
"""
import argparse
import ctypes
import os
import time

import torch

from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation
from comfy_kitchen.backends.eager import quantization as qz

_LIB = None


def load_lib(path=None):
    global _LIB
    if _LIB is not None:
        return _LIB
    path = path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "build",
                                "libxpuint8.so")
    lib = ctypes.CDLL(path)
    lib.xpuint8_init.restype = ctypes.c_int
    lib.xpuint8_device_name.restype = ctypes.c_char_p
    lib.xpuint8_convrot_groupsize.restype = ctypes.c_int
    lib.xpuint8_last_error.restype = ctypes.c_char_p
    lib.xpuint8_convrot_quantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_int64,
                                             ctypes.c_int64]
    lib.xpuint8_convrot_quantize.restype = ctypes.c_int
    lib.xpuint8_convrot_rotate_bf16.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                ctypes.c_int64, ctypes.c_int64]
    lib.xpuint8_convrot_rotate_bf16.restype = ctypes.c_int
    lib.xpuint8_dequant_epilogue.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_int64,
                                             ctypes.c_int64]
    lib.xpuint8_dequant_epilogue.restype = ctypes.c_int
    rc = lib.xpuint8_init()
    if rc != 0:
        raise RuntimeError(f"xpuint8_init failed: {lib.xpuint8_last_error()}")
    _LIB = lib
    return lib


def fused_convrot_quantize(lib, x):
    """bf16 [M,K] -> int8 [M,K] + fp32 [M] scales, on device."""
    m, k = x.shape
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty(m, device=x.device, dtype=torch.float32)
    rc = lib.xpuint8_convrot_quantize(x.data_ptr(), q.data_ptr(), s.data_ptr(),
                                      int(m), int(k))
    if rc != 0:
        raise RuntimeError(f"convrot_quantize rc={rc}: {lib.xpuint8_last_error()}")
    return q, s


def fused_dequant(lib, acc, xs, ws, bias):
    m, n = acc.shape
    out = torch.empty(m, n, device=acc.device, dtype=torch.bfloat16)
    rc = lib.xpuint8_dequant_epilogue(acc.data_ptr(), out.data_ptr(), xs.data_ptr(),
                                      ws.data_ptr(),
                                      bias.data_ptr() if bias is not None else None,
                                      int(m), int(n))
    if rc != 0:
        raise RuntimeError(f"dequant_epilogue rc={rc}: {lib.xpuint8_last_error()}")
    return out


def bench(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument("--k", type=int, default=3072)
    ap.add_argument("--n", type=int, default=3072)
    ap.add_argument("--group", type=int, default=256)
    args = ap.parse_args()

    lib = load_lib()
    print("device:", lib.xpuint8_device_name().decode())
    print("convrot group size:", lib.xpuint8_convrot_groupsize())

    K, N, GS = args.k, args.n, args.group
    dev = "xpu"
    h = _build_hadamard(GS, device=dev, dtype=torch.bfloat16)
    w8 = torch.randint(-127, 127, (N, K), device=dev, dtype=torch.int8)
    wt = w8.t().contiguous()
    wscale = torch.rand(N, device=dev, dtype=torch.float32) * 0.01 + 0.001
    bias = (torch.rand(N, device=dev, dtype=torch.bfloat16) - 0.5) * 0.1
    wbf = torch.randn(N, K, device=dev, dtype=torch.bfloat16)

    print(f"\n{'M':>6} {'eager int8':>11} {'fused int8':>11} {'bf16':>9} "
          f"{'eager vs bf16':>14} {'fused vs bf16':>14}")
    for M in args.m:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)

        t_eager = bench(lambda: qz.int8_linear(x, w8, wscale, bias, convrot=True,
                                               convrot_groupsize=GS))

        def fused():
            q, xs = fused_convrot_quantize(lib, x)
            acc = torch._int_mm(q, wt)
            return fused_dequant(lib, acc, xs, wscale, bias)

        t_fused = bench(fused)
        t_bf16 = bench(lambda: torch.nn.functional.linear(x, wbf, bias))
        print(f"{M:6d} {t_eager:10.3f}m {t_fused:10.3f}m {t_bf16:8.3f}m "
              f"{t_eager / t_bf16:13.2f}x {t_fused / t_bf16:13.2f}x")
        del x

    # Correctness spot-check at the largest M.
    #
    # NOTE: comfy-kitchen's eager path rotates activations with a *bfloat16* bmm,
    # which loses enough precision that the row abs-max (and therefore the row
    # scale) can be wrong by up to ~40%. It is not a valid reference. We compare
    # against a float64 rotation on CPU instead, which is the ground truth.
    M = max(args.m)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    hd = _build_hadamard(GS, dtype=torch.float64)
    xd = x.double().cpu().reshape(M, K // GS, GS)
    xr64 = (xd @ hd).reshape(M, K)
    s64 = (xr64.abs().amax(dim=-1) / 127.0).clamp(min=1e-30)

    # same weights, dequantized in float32, fp32 matmul
    wf = w8.double().cpu() * wscale.double().cpu().unsqueeze(1)
    bf = bias.double().cpu()
    ref64 = torch.addmm(bf, xr64, wf.t())

    def rel(a):
        a = a.double().cpu()
        return ((a - ref64).norm() / ref64.norm()).item()

    q, xs = fused_convrot_quantize(lib, x)
    acc = torch._int_mm(q, wt)
    got = fused_dequant(lib, acc, xs, wscale, bias)
    eager = qz.int8_linear(x, w8, wscale, bias, convrot=True, convrot_groupsize=GS)

    print(f"\ncorrectness vs float64 ground truth @ M={M}:")
    print(f"  fused xpu-int8   relative L2 error: {rel(got):.6f}")
    print(f"  comfy eager      relative L2 error: {rel(eager):.6f}")


if __name__ == "__main__":
    main()