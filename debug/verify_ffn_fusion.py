"""Is the fused dequant+situ+rotate+quantize kernel the same function as
dequant -> bf16 -> quantize(act=3)?

That is the whole safety argument for reading the up-projection's int32
accumulator directly instead of materialising its bf16 dequant output.

    python debug/verify_ffn_fusion.py
"""
import ctypes

import torch

import comfy.xpu_int8 as X

torch.manual_seed(0)
dev = torch.device("xpu", 0)
lib = X.library()
assert lib is not None, "libxpuint8 not loaded"
assert hasattr(lib, "xpuint8_dequant_act_quantize_stream"), "library is stale"


def quantize(x2, K, act=None):
    M = x2.shape[0]
    q = torch.empty((M, K), device=dev, dtype=torch.int8)
    s = torch.empty((M,), device=dev, dtype=torch.float32)
    sq = ctypes.c_void_p(torch.xpu.current_stream(dev).sycl_queue)
    if act is None:
        rc = lib.xpuint8_convrot_quantize_stream(sq, x2.data_ptr(), q.data_ptr(),
                                                 s.data_ptr(), M, K)
    else:
        rc = lib.xpuint8_convrot_quantize_act_stream(sq, x2.data_ptr(), q.data_ptr(),
                                                     s.data_ptr(), M, K, act)
    torch.xpu.synchronize()
    assert rc == 0, lib.xpuint8_last_error()
    return q, s


print(f"{'case':<34} {'int8 mismatches':>18} {'scale':>10}")
for M, K in [(8, 256), (64, 3072), (3974, 3072), (3974, 8192)]:
    N = 2 * K  # up-projection output width = 2 * ffn
    # realistic up projection: int8 weight [N, K], random accumulator
    w = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
    qx = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
    acc = torch._int_mm(qx, w.t())
    xs = torch.rand(M, device=dev, dtype=torch.float32) * 0.05 + 0.01
    ws = torch.rand(N, device=dev, dtype=torch.float32) * 0.01 + 0.001

    # reference: dequant to bf16, then the paired-input quantizer
    ref = torch.empty((M, N), device=dev, dtype=torch.bfloat16)
    sq = ctypes.c_void_p(torch.xpu.current_stream(dev).sycl_queue)
    rc = lib.xpuint8_dequant_epilogue_stream(
        sq, acc.data_ptr(), ref.data_ptr(), xs.data_ptr(), ws.data_ptr(), None, M, N)
    torch.xpu.synchronize()
    assert rc == 0, lib.xpuint8_last_error()
    q_ref, s_ref = quantize(ref.contiguous(), K, act=3)

    # fused: int32 accumulator straight into the situ quantizer
    q_new = torch.empty((M, K), device=dev, dtype=torch.int8)
    s_new = torch.empty((M,), device=dev, dtype=torch.float32)
    rc = lib.xpuint8_dequant_act_quantize_stream(
        sq, acc.data_ptr(), q_new.data_ptr(), s_new.data_ptr(),
        xs.data_ptr(), ws.data_ptr(), M, K, 3)
    torch.xpu.synchronize()
    assert rc == 0, lib.xpuint8_last_error()

    bad = (q_ref != q_new).sum().item()
    seq = (s_ref != s_new).sum().item()
    print(f"M={M:<6} K={K:<6}                      {bad:>10d}/{q_ref.numel():<10d} "
          f"{seq:>4d}/{M} rows differ")
