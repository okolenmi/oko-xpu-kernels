"""Bit-exactness gate for fusing the qkv dequant epilogue into rms_rope.

Old chain, per block:
    qkv GEMM -> int32 [M, 9216]
    one epilogue over the whole [M, 9216] -> bf16
    chunk -> q, k, v ; rope(q), rope(k)

New chain:
    qkv GEMM -> int32 [M, 9216]
    epilogue on the v third only (row-strided: acc[:, 2D:] with stride 9216)
    rope(q), rope(k) read the int32 accumulator and dequantize in-register

The comparison is bitwise, over the fp32 dequant rounding and the whole rope,
at the production shapes.

  LEVEL_ZERO_V1_SDK_PATH=/home/okolenmi/.local/levelzero \
  PYTHONPATH=/home/okolenmi/comfy/ComfyUI \
  /home/okolenmi/comfy/venv/bin/python debug/verify_qkv_rope_fuse.py [M ...]

(without LEVEL_ZERO_V1_SDK_PATH the Triton JIT cannot find level_zero/ze_api.h;
debug/server.sh exports it, a bare shell does not)
"""
import ctypes
import sys

import torch

sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")
import comfy.xpu_int8 as xi  # noqa: E402
import comfy.xpu_rope as xr  # noqa: E402

H, D = 24, 128
DIM = H * D
C = 3 * DIM

lib = xi._load()
assert lib is not None and lib.xpuint8_init() == 0, lib.xpuint8_last_error()
dev = torch.device("xpu:0")
torch.xpu.set_device(0)
q = ctypes.c_void_p(torch.xpu.current_stream(dev).sycl_queue)
P = ctypes.c_void_p

slice_stream = getattr(lib, "xpuint8_dequant_epilogue_slice_stream", None)
assert slice_stream is not None, "library has no row-strided epilogue: rebuild it"


def epilogue(acc, out, xs, ws, stride):
    rc = lib.xpuint8_dequant_epilogue_stream(
        q, P(acc.data_ptr()), P(out.data_ptr()), P(xs.data_ptr()), P(ws.data_ptr()),
        ctypes.c_void_p(0), acc.shape[0], out.shape[1])
    assert rc == 0, lib.xpuint8_last_error()


def epilogue_slice(acc, out, xs, ws, acc_stride):
    rc = slice_stream(q, P(acc.data_ptr()), P(out.data_ptr()), P(xs.data_ptr()),
                      P(ws.data_ptr()), ctypes.c_void_p(0), acc.shape[0],
                      out.shape[1], acc_stride)
    assert rc == 0, lib.xpuint8_last_error()


def check(M, seed=0):
    g = torch.Generator(device="xpu").manual_seed(seed)
    B, S = 1, M
    acc = torch.randint(-(2 ** 23), 2 ** 23, (M, C), generator=g, device=dev,
                        dtype=torch.int32)
    xs = torch.rand(M, generator=g, device=dev) * 0.02 + 0.001
    ws = torch.rand(C, generator=g, device=dev) * 0.02 + 0.001
    scale = torch.rand(D, generator=g, device=dev) * 0.2 + 0.9
    ang = torch.rand(B, S, 1, D // 2, generator=g, device=dev) * 3.0
    pe = torch.stack([torch.stack([ang.cos(), -ang.sin()], -1),
                      torch.stack([ang.sin(), ang.cos()], -1)], -2)

    # --- old chain -------------------------------------------------------
    full = torch.empty((M, C), device=dev, dtype=torch.bfloat16)
    epilogue(acc, full, xs, ws, C)
    ref = full.view(B, S, C)
    ref_q = xr._rows(ref[:, :, :DIM].view(B, S, H, D), pe, scale, 1e-6)
    ref_k = xr._rows(ref[:, :, DIM:2 * DIM].view(B, S, H, D), pe, scale, 1e-6)
    ref_v = ref[:, :, 2 * DIM:].reshape(M, DIM).clone()

    # --- new chain -------------------------------------------------------
    chunk = acc.view(B, S, C)
    q_acc = chunk[:, :, :DIM].view(B, S, H, D)
    k_acc = chunk[:, :, DIM:2 * DIM].view(B, S, H, D)
    got_q, got_k = xr.rms_rope_split_half_dequant(
        q_acc, k_acc, pe, ws[:DIM], ws[DIM:2 * DIM], xs.view(B, S), scale, scale, 1e-6)
    v_out = torch.empty((M, DIM), device=dev, dtype=torch.bfloat16)
    epilogue_slice(chunk[:, :, 2 * DIM:].reshape(M, DIM), v_out, xs, ws[2 * DIM:], C)

    def diff(a, b):
        return (a.reshape(-1).view(torch.int16) != b.reshape(-1).view(torch.int16)).sum().item()

    dq, dk, dv = diff(got_q, ref_q), diff(got_k, ref_k), diff(v_out, ref_v)
    n = ref_q.numel()
    print(f"M={M:5d}  q {dq:6d}/{n}  k {dk:6d}/{n}  v {dv:6d}/{ref_v.numel()}  "
          f"{'BIT-EXACT' if dq == dk == dv == 0 else 'MISMATCH'}")
    return dq + dk + dv == 0


ok = True
for M in (int(a) for a in (sys.argv[1:] or ["256", "3973", "3974"])):
    for seed in (0, 1):
        ok &= check(M, seed)
print("ALL BIT-EXACT" if ok else "FAILED")
sys.exit(0 if ok else 1)
