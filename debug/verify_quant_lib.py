#!/usr/bin/env python
"""Bit-exactness check of a candidate libxpuint8.so against the installed one.

    python debug/verify_quant_lib.py [candidate.so]

Runs the plain quantize and both activation variants (GELU-tanh, SwiGLU) at
the shapes the MossImage DiT uses and compares the int8 payload and the row
scales element for element.  The candidate is loaded in the same process as
the installed library, so both run on the same torch XPU tensors.
"""
import ctypes
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

INSTALLED = "/home/okolenmi/comfy/ComfyUI/comfy/lib/libxpuint8.so"


def load(path):
    lib = ctypes.CDLL(path)
    lib.xpuint8_convrot_quantize.restype = ctypes.c_int
    lib.xpuint8_convrot_quantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_int64,
                                              ctypes.c_int64]
    lib.xpuint8_convrot_quantize_act.restype = ctypes.c_int
    lib.xpuint8_convrot_quantize_act.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                  ctypes.c_void_p, ctypes.c_int64,
                                                  ctypes.c_int64, ctypes.c_int32]
    lib.xpuint8_convrot_quantize_stream.restype = ctypes.c_int
    lib.xpuint8_convrot_quantize_stream.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 2 + \
        [ctypes.c_int64] * 2
    lib.xpuint8_convrot_quantize_act_stream.restype = ctypes.c_int
    lib.xpuint8_convrot_quantize_act_stream.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 2 + \
        [ctypes.c_int64] * 2 + [ctypes.c_int32]
    lib.xpuint8_last_error.restype = ctypes.c_char_p
    return lib


def run(lib, x, act):
    """Drive the library on torch's own queue.

    The library's non-stream entry points use its own internal SYCL queue, so
    there is no ordering against torch's queue and the kernel can read the
    input before torch has finished writing it.  The stream variants take
    torch's queue and are the ones the library is actually used through.
    """
    M, K = x.shape[0], x.shape[1] // (2 if act == 2 else 1)
    q = torch.empty((M, K), device=x.device, dtype=torch.int8)
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    stream = torch.xpu.current_stream(x.device)
    queue = ctypes.c_void_p(stream.sycl_queue)
    if act == 0:
        rc = lib.xpuint8_convrot_quantize_stream(queue, x.data_ptr(), q.data_ptr(),
                                                  s.data_ptr(), M, K)
    else:
        rc = lib.xpuint8_convrot_quantize_act_stream(queue, x.data_ptr(), q.data_ptr(),
                                                      s.data_ptr(), M, K, act)
    if rc != 0:
        raise RuntimeError(f"rc={rc}: {lib.xpuint8_last_error()}")
    torch.xpu.synchronize()
    return q, s


def main():
    cand = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/libxpuint8_test.so"
    a = load(INSTALLED)
    b = load(cand)
    print(f"installed: {INSTALLED}")
    print(f"candidate: {cand}\n")

    dev = "xpu"
    torch.manual_seed(7)
    fails = 0
    for M, K in ((3952, 3072), (3952, 8192), (512, 4096), (128, 1024), (64, 256)):
        for act in (0, 1, 2):
            width = 2 * K if act == 2 else K
            x = (torch.randn(M, width, device=dev, dtype=torch.bfloat16) * 0.7)
            qa, sa = run(a, x, act)
            qb, sb = run(b, x, act)
            dq = (qa != qb)
            nq = int(dq.sum())
            ds = (sa != sb)
            ns = int(ds.sum())
            bad = nq or ns
            fails += bool(bad)
            tag = "OK  " if not bad else "FAIL"
            print(f"  {tag} M={M:5d} K={K:5d} act={act}: q mismatches={nq:9d} "
                  f"scale mismatches={ns}")
            if nq and nq < 8:
                idx = dq.nonzero()[:8]
                for r, c in idx.tolist():
                    print(f"        q[{r},{c}] installed={int(qa[r, c])} "
                          f"candidate={int(qb[r, c])}")
    print("\nRESULT:", "all bit-identical" if not fails else f"{fails} MISMATCHES")
    return 1 if fails else 0


sys.exit(main())
