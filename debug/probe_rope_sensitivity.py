"""Would a different RMS reduction order change the rope output?

The Triton rms_rope kernel (comfy_kitchen, no XPU backend) sums the 128
squared activations of each head-row in one particular order.  A replacement
SYCL kernel would use a different order.  Before writing it, measure whether
that can change a single bf16 output element at the production shape.

Compares:
  * the Triton kernel that actually runs,
  * an eager fp32 reference using torch's sum,
  * the same with a deliberately different (pairwise) summation order.

If the last two agree with the first bit for bit, the reduction order is
irrelevant and a SYCL kernel can be validated byte-exactly.

    python debug/probe_rope_sensitivity.py
"""
import torch

import comfy.quant_ops

dev = torch.device("xpu", 0)
B, S, H, D = 1, 3973, 24, 128
eps = 1e-6
torch.manual_seed(0)


def make(lo=1.0, hi=3.0):
    # realistic magnitudes; the RMS is order-of-1
    return (torch.rand(B, S, H, D, device=dev) * (hi - lo) + lo).bfloat16()


q, k = make(), make()
# EmbedND output layout the kernel expects: [B, S, 1, D/2, 2, 2]
pe = torch.randn(B, S, 1, D // 2, 2, 2, device=dev, dtype=torch.float32)
pe = (pe * 0.3).bfloat16()
scale = torch.rand(D, device=dev, dtype=torch.float32).bfloat16() + 0.5

triton_q, triton_k = None, None
try:
    from comfy_kitchen.registry import registry
    registry.use_backend("triton").__enter__()
    _be = "triton"
except Exception as e:                                   # noqa: BLE001
    _be = f"eager (could not select triton: {e})"
print(f"backend: {_be}")
triton_q, triton_k = comfy.quant_ops.ck.rms_rope_split_half(
    q.clone(), k.clone(), pe, scale, scale, eps)


def eager(x, order):
    xf = x.float()
    if order == "torch":
        ss = (xf * xf).sum(dim=-1, keepdim=True)
    else:
        # pairwise tree over the 128 lanes: a different fp32 order on purpose
        t = xf * xf
        h = D
        while h > 1:
            t = t.reshape(*t.shape[:-1], 2, h // 2)
            t = t[..., 0, :] + t[..., 1, :]
            h //= 2
        ss = t
    inv = torch.rsqrt(ss / D + eps)
    n = (xf * inv * scale.float()).bfloat16().float()
    n0, n1 = n[..., :D // 2], n[..., D // 2:]
    p = pe.float()
    out0 = p[..., 0, 0] * n0 + p[..., 0, 1] * n1
    out1 = p[..., 1, 0] * n0 + p[..., 1, 1] * n1
    return torch.cat([out0, out1], dim=-1).bfloat16()


for order in ("torch", "pairwise"):
    for name, ref, x in (("q", triton_q, q), ("k", triton_k, k)):
        got = eager(x, order)
        d = (got != ref)
        n = int(d.sum())
        rel = (got.float() - ref.float()).abs()
        print(f"{order:8s} {name} vs backend: {n} / {got.numel()} differ "
              f"(max abs {float(rel.max()):.3e})")

# The decisive number: do the two summation orders agree with each other?
for name, x in (("q", q), ("k", k)):
    a = eager(x, "torch")
    b = eager(x, "pairwise")
    n = int((a != b).sum())
    print(f"torch vs pairwise {name}: {n} / {a.numel()} bf16 elements differ")
