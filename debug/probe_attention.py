"""What is attention actually running, and how far is it from the hardware?

The device-time budget puts attention at 301 ms/step (21% of a step), which for
the FLOP count of this shape (B*H*S*S*D with S=3952, H=24, D=128) is orders of
magnitude below any plausible peak.  That smells like the math fallback:
materialise the full S x S score matrix instead of tiling over K/V.  This probe

  1. reports which SDPA backends torch-thpu offers at all,
  2. times the attention shapes of the MossImage model through every path
     that works: F.scaled_dot_product_attention under each sdpa_kernel
     selection, the explicit math path, and the current comfy wrapper,
  3. derives achieved TFLOPS and the score-matrix traffic, so the two
     hypotheses (compute bound vs materialising S x S) can be told apart.

    python debug/probe_attention.py [seq] [heads] [dim_head] [batch]
"""
import sys
import time

import torch

torch.manual_seed(0)
dev = torch.device("xpu", 0)
dt = torch.bfloat16

S = int(sys.argv[1]) if len(sys.argv) > 1 else 3952
H = int(sys.argv[2]) if len(sys.argv) > 2 else 24
D = int(sys.argv[3]) if len(sys.argv) > 3 else 128
B = int(sys.argv[4]) if len(sys.argv) > 4 else 2


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


print(f"shape B={B} H={H} S={S} D={D} dtype={dt}")
flops = 4.0 * B * H * S * S * D
print(f"attention FLOPs = {flops/1e9:.1f} GFLOP per call, "
      f"scores = {B*H*S*S*2/2**30:.2f} GiB in bf16")

q = torch.randn(B, H, S, D, device=dev, dtype=dt)
k = torch.randn(B, H, S, D, device=dev, dtype=dt)
v = torch.randn(B, H, S, D, device=dev, dtype=dt)

# ---- which backends exist -------------------------------------------------
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    names = [n for n in dir(SDPBackend) if not n.startswith("_")]
    print("SDPBackend members:", names)
except Exception as e:  # pragma: no cover
    print("no sdpa_kernel api:", e)
    sdpa_kernel = None
    SDPBackend = None

# ---- the current comfy path ----------------------------------------------
try:
    from comfy.ldm.modules.attention import optimized_attention, ComfyAttention
    from comfy.ldm.modules.attention import AttentionTensorContainer
    qq = AttentionTensorContainer(q.flatten(2))
    kk = AttentionTensorContainer(k.flatten(2))
    vv = AttentionTensorContainer(v.flatten(2))
    ms = bench(lambda: optimized_attention(qq, kk, vv, H, transformer_options={},
                                          preferred_attention=ComfyAttention()))
    print(f"  comfy optimized_attention   {ms:8.3f} ms  {flops/ms*1e-9:7.1f} TFLOPS")
except Exception as e:
    print("  comfy optimized_attention failed:", e)

# ---- explicit backend selections ------------------------------------------
if sdpa_kernel is not None:
    for name in ["FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH", "CUDNN_ATTENTION"]:
        be = getattr(SDPBackend, name, None)
        if be is None:
            continue
        try:
            with sdpa_kernel([be]):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch.xpu.synchronize()
            ms = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
            print(f"  sdpa[{name:<20}]        {ms:8.3f} ms  {flops/ms*1e-9:7.1f} TFLOPS")
        except Exception as e:
            print(f"  sdpa[{name:<20}]        unavailable: {str(e).splitlines()[0][:80]}")

# ---- default (no context) -------------------------------------------------
try:
    ms = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
    print(f"  sdpa[default]               {ms:8.3f} ms  {flops/ms*1e-9:7.1f} TFLOPS")
except Exception as e:
    print("  sdpa[default] failed:", e)

# ---- explicit math path ---------------------------------------------------
def math_path():
    s = (q @ k.transpose(-1, -2)) * (1.0 / D ** 0.5)
    p = torch.softmax(s, dim=-1)
    return p @ v


try:
    ms = bench(math_path)
    print(f"  explicit matmul+softmax     {ms:8.3f} ms  {flops/ms*1e-9:7.1f} TFLOPS")
except Exception as e:
    print("  explicit math failed:", e)

# ---- how fast is the matmul part alone? -----------------------------------
def two_gemms():
    return (q @ k.transpose(-1, -2)) @ v


try:
    ms = bench(two_gemms)
    print(f"  2x bmm, no softmax          {ms:8.3f} ms  {flops/ms*1e-9:7.1f} TFLOPS")
except Exception as e:
    print("  2x bmm failed:", e)

# ---- softmax alone --------------------------------------------------------
try:
    s = q @ k.transpose(-1, -2)
    ms = bench(lambda: torch.softmax(s, dim=-1))
    print(f"  softmax on the score matrix {ms:8.3f} ms")
except Exception as e:
    print("  softmax failed:", e)
