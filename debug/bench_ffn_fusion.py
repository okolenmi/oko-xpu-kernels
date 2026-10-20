"""Time the fused dequant+situ+quantize kernel against the pair it replaces,
at the production shape (M=3974, ffn=8192 -> accumulator [M, 16384]).

    python debug/bench_ffn_fusion.py
"""
import ctypes
import time

import torch

import comfy.xpu_int8 as X

dev = torch.device("xpu", 0)
lib = X.library()
assert lib is not None
assert hasattr(lib, "xpuint8_dequant_act_quantize_stream"), "library is stale"

M, K = 3974, 8192          # tokens, ffn width
N = 2 * K
acc = torch.randint(-1 << 20, 1 << 20, (M, N), device=dev, dtype=torch.int32)
xs = (torch.rand(M, device=dev) * 0.05 + 0.01)
ws = (torch.rand(N, device=dev) * 0.01 + 0.001)
sq = ctypes.c_void_p(torch.xpu.current_stream(dev).sycl_queue)
buf = torch.empty((M, N), device=dev, dtype=torch.bfloat16)
q = torch.empty((M, K), device=dev, dtype=torch.int8)
s = torch.empty((M,), device=dev, dtype=torch.float32)


def pair():
    rc = lib.xpuint8_dequant_epilogue_stream(
        sq, acc.data_ptr(), buf.data_ptr(), xs.data_ptr(), ws.data_ptr(), None, M, N)
    assert rc == 0, lib.xpuint8_last_error()
    rc = lib.xpuint8_convrot_quantize_act_stream(
        sq, buf.data_ptr(), q.data_ptr(), s.data_ptr(), M, K, 3)
    assert rc == 0, lib.xpuint8_last_error()


def fused():
    rc = lib.xpuint8_dequant_act_quantize_stream(
        sq, acc.data_ptr(), q.data_ptr(), s.data_ptr(), xs.data_ptr(), ws.data_ptr(),
        M, K, 3)
    assert rc == 0, lib.xpuint8_last_error()


def bench(name, fn, n=30):
    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.xpu.synchronize()
        best = min(best, (time.perf_counter() - t0) / n)
    ms = best * 1e3
    print(f"{name:<44} {ms:7.3f} ms/call  {ms * 42:7.1f} ms/forward  "
          f"{ms * 84:7.1f} ms/step")
    return ms


print(f"M={M} ffn={K} accumulator int32 [{M},{N}] "
      f"({M * N * 4 / 1e6:.0f} MB)")
a = bench("dequant -> bf16 -> quantize(act=3)", pair)
b = bench("dequant_act_quantize (fused)", fused)
print(f"\nsaved {a - b:.3f} ms/call -> {(a - b) * 84:.1f} ms/step "
      f"({100 * (a - b) / a:.0f}% of the pair)")
