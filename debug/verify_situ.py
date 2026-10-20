"""Is the fused `situ` quantizer the same function as the eager chain?

The MossImage FFN is  up_gate = ffn_up_gate(y);  out = ffn_down(situ(up_gate))
with situ(h, g) = tanh(g/4) * sigmoid(g) * tanh(h/25), h in the first half and
g in the second.  The fused path folds situ into the rotate+quantize kernel
(activation code 3), so the linear's input is never materialised.

That is only safe if the quantizer sees the same activation.  This probe
quantizes the same tensor both ways -- eagerly computed bf16 situ into the
plain quantizer, and the paired tensor into the code-3 quantizer -- and
compares the int8 output and the per-row scales.

    python debug/verify_situ.py
"""
import ctypes

import torch

import comfy.xpu_int8 as X

torch.manual_seed(0)
dev = torch.device("xpu", 0)
lib = X.library()
assert lib is not None, "libxpuint8 not loaded"

for name in ("xpuint8_convrot_quantize_stream", "xpuint8_convrot_quantize_act_stream"):
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int64, ctypes.c_int64, ctypes.c_int32] if "act_stream" in name else \
                  [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int64, ctypes.c_int64]
    fn.restype = ctypes.c_int


def quantize(x2, K, act=None, M=None):
    M = x2.shape[0] if M is None else M
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


print(f"{'case':<28} {'int8 mismatches':>16} {'max scale relerr':>18}")
for M, K in [(1, 256), (64, 3072), (3974, 3072), (3974, 8192), (4096, 8192)]:
    # scales up to 100 exercise the saturated range of both tanh arguments
    # (|g|/4 > 9 and |h|/25 > 9), where sycl::tanh has its slow path
    for scale in (1.0, 4.0, 20.0, 100.0):
        x = torch.randn(M, 2 * K, device=dev, dtype=torch.bfloat16) * scale
        hidden, gate = x.chunk(2, dim=-1)
        eager = torch.tanh(gate / 4.0) * torch.sigmoid(gate) * torch.tanh(hidden / 25.0)
        eager = eager.contiguous()

        q_ref, s_ref = quantize(eager, K)
        q_new, s_new = quantize(x.contiguous(), K, act=3)

        bad = (q_ref != q_new).sum().item()
        rel = ((s_ref - s_new).abs() / s_ref.abs().clamp_min(1e-12)).max().item()
        print(f"M={M:<6} K={K:<6} x{scale:<4}     {bad:>10d}/{q_ref.numel():<10d} {rel:>14.3e}")

# the paired halves must not be swapped: feed a tensor whose halves are
# visibly different and check the activation follows `hidden, gate` order
M, K = 8, 256
x = torch.zeros(M, 2 * K, device=dev, dtype=torch.bfloat16)
x[:, K:] = 2.0   # gate = 2 everywhere, hidden = 0
hidden, gate = x.chunk(2, dim=-1)
eager = torch.tanh(gate / 4.0) * torch.sigmoid(gate) * torch.tanh(hidden / 25.0)
q_ref, s_ref = quantize(eager.contiguous(), K)
q_new, s_new = quantize(x, K, act=3)
print(f"\nhalf order, gate=2/hidden=0: mismatches={(q_ref != q_new).sum().item()}, "
      f"scale relerr={((s_ref - s_new).abs() / s_ref.abs().clamp_min(1e-12)).max().item():.3e}")
