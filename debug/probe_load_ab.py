"""Interleaved A/B of the 32-byte vs 64-byte load path in the rotate kernel.

The GPU drifts 10-30% with thermals over a benchmarking session, which is larger
than the effect being measured. So both variants are timed alternately in the
same process, several rounds, and only the *paired* difference is reported.

Build both first:
    ./build.sh                                     # 32-byte (current)
    cp build/libxpuint8.so /tmp/lib32.so
    # apply the load_pair variant, ./build.sh, cp to /tmp/lib64.so
    # then:  python3 debug/probe_load_ab.py
"""
import ctypes
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DECL_ROT = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]


def bind(path):
    lib = ctypes.CDLL(path)
    lib.xpuint8_init()
    lib.xpuint8_convrot_quantize.argtypes = DECL_ROT
    lib.xpuint8_convrot_quantize.restype = ctypes.c_int
    lib.xpuint8_device_name.restype = ctypes.c_char_p
    return lib


def time_once(lib, x, q, s, M, K, n=15):
    xp, qp, sp = x.data_ptr(), q.data_ptr(), s.data_ptr()
    for _ in range(3):
        lib.xpuint8_convrot_quantize(xp, qp, sp, M, K)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        lib.xpuint8_convrot_quantize(xp, qp, sp, M, K)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    a_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lib32.so"
    b_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/lib64.so"
    M, K = 4096, 3072
    dev = "xpu"

    lib32, lib64 = bind(a_path), bind(b_path)
    print("device:", lib32.xpuint8_device_name().decode())

    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    q = torch.empty(M, K, device=dev, dtype=torch.int8)
    s = torch.empty(M, device=dev, dtype=torch.float32)

    # correctness cross-check: the two builds must agree bit for bit
    a32 = torch.empty_like(q)
    a64 = torch.empty_like(q)
    lib32.xpuint8_convrot_quantize(x.data_ptr(), a32.data_ptr(), s.data_ptr(), M, K)
    lib64.xpuint8_convrot_quantize(x.data_ptr(), a64.data_ptr(), s.data_ptr(), M, K)
    same = torch.equal(a32, a64)
    print("variants agree bit-for-bit:", same)
    if not same:
        print("  (they should -- only the load path differs)")

    print(f"\n{'round':>6} {'32B loads':>11} {'64B loads':>11} {'64B / 32B':>11}")
    r32, r64 = [], []
    for i in range(7):
        t32 = time_once(lib32, x, q, s, M, K)
        t64 = time_once(lib64, x, q, s, M, K)
        r32.append(t32)
        r64.append(t64)
        print(f"{i:6d} {t32:10.3f}m {t64:10.3f}m {t64 / t32:10.3f}x")

    import statistics
    m32, m64 = statistics.median(r32), statistics.median(r64)
    print(f"\nmedian 32-byte: {m32:.3f} ms   median 64-byte: {m64:.3f} ms")
    print(f"paired median ratio (64B/32B): {m64 / m32:.3f}x  "
          f"-> {'64B is SLOWER' if m64 > m32 else '64B is faster'}")


if __name__ == "__main__":
    main()