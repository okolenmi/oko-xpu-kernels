"""Which exact arithmetic does comfy-kitchen's Triton `rms_rope_split_half` use?

The kernel replaces the eager chain

    q_norm(q) -> rope_split_half(pe)        (both in comfy/ldm/moss_image/model.py)

and `pe` is fp32 (`rope_fp32` in comfy/ldm/flux/math.py returns fp32 and nothing
casts it), so `compute_dtype` is fp32 and the norm output is the only value that
is rounded to bf16 before the rotation.  This probe runs the real Triton kernel
on the production shapes and compares its *bit pattern* against candidate
formulations, so a SYCL replacement can be written to match exactly rather than
to "within an ULP".

    python debug/probe_rope_formula.py
"""
import sys

import torch

sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")
from comfy_kitchen.backends.triton.rms_rope import (  # noqa: E402
    rms_rope_split_half as triton_rope,
)

dev = torch.device("xpu", 0)
B, S, H, D = 1, 3973, 24, 128
EPS = 1e-6
N = D // 2

torch.manual_seed(0)
# realistic magnitudes: activations are bf16, the norm weight is ~O(1)
q = (torch.randn(B, S, H, D, device=dev) * 0.3).to(torch.bfloat16)
k = (torch.randn(B, S, H, D, device=dev) * 0.3).to(torch.bfloat16)
scale = torch.rand(D, device=dev, dtype=torch.float32) * 0.2 + 0.9
# pe = [B, S, 1, D/2, 2, 2] fp32 with the [[cos,-sin],[sin,cos]] structure
ang = torch.rand(B, S, 1, N, device=dev, dtype=torch.float32) * 3.0
pe = torch.stack(
    [torch.stack([ang.cos(), -ang.sin()], -1),
     torch.stack([ang.sin(), ang.cos()], -1)], -2)

ref_q, ref_k = triton_rope(q, k, pe, scale, None, EPS)
torch.xpu.synchronize()
print(f"triton reference: q{tuple(ref_q.shape)} {ref_q.dtype}  k{tuple(ref_k.shape)}")


def bits(t):
    return t.view(torch.int16).to(torch.int32)


def report(name, out, ref):
    bad = (bits(out) != bits(ref)).sum().item()
    d = (out.float() - ref.float()).abs().max().item()
    print(f"  {name:<44} mismatches {bad:>10d}/{ref.numel():<10d}  maxabs {d:.3e}")
    return bad


def normed(x, inv, round_bf16=True):
    xf = x.float()
    x0, x1 = xf[..., :N], xf[..., N:]
    t0, t1 = x0 * inv * scale[:N], x1 * inv * scale[N:]
    if round_bf16:
        t0 = t0.to(torch.bfloat16).float()
        t1 = t1.to(torch.bfloat16).float()
    return t0, t1


def rotate(t0, t1, fma):
    f00, f01 = pe[..., 0, 0], pe[..., 0, 1]
    f10, f11 = pe[..., 1, 0], pe[..., 1, 1]
    if fma:  # round(f00*t0 + round(f01*t1)): one rounding for the mul-add
        o0 = (f00.double() * t0.double() + (f01 * t1).double()).float()
        o1 = (f10.double() * t0.double() + (f11 * t1).double()).float()
    else:    # round(round(f00*t0) + round(f01*t1))
        o0 = f00 * t0 + f01 * t1
        o1 = f10 * t0 + f11 * t1
    return torch.cat([o0, o1], -1).to(torch.bfloat16)


xf = q.float()
inv = torch.rsqrt((xf * xf).sum(-1, keepdim=True) / D + EPS)

print("\ncandidates (q):")
t0, t1 = normed(q, inv, True)
c1 = rotate(t0, t1, False)
c2 = rotate(t0, t1, True)
report("C1 norm->bf16, rotation fp32 (no contraction)", c1, ref_q)
report("C2 norm->bf16, rotation fp32 (fma contraction)", c2, ref_q)

t0n, t1n = normed(q, inv, False)
report("C3 no bf16 round of the norm output", rotate(t0n, t1n, False), ref_q)
t0b = (q.float()[..., :N] * inv * scale[:N]).to(torch.bfloat16)
t1b = (q.float()[..., N:] * inv * scale[N:]).to(torch.bfloat16)
c4 = torch.cat([(pe[..., 0, 0] * t0b + pe[..., 0, 1] * t1b).to(torch.bfloat16),
                (pe[..., 1, 0] * t0b + pe[..., 1, 1] * t1b).to(torch.bfloat16)], -1)
report("C4 rotation itself in bf16", c4, ref_q)

# how sensitive is the output to the last bit of inv_rms?  (the reduction order
# inside tl.sum is triton-internal, so this bounds the risk of a different tree)
print("\ninv_rms sensitivity (vs C1, same rounding):")
for tag, pert in (("+1 ulp", torch.nextafter(inv, torch.full_like(inv, float("inf")))),
                  ("-1 ulp", torch.nextafter(inv, torch.full_like(inv, 0.0)))):
    t0p, t1p = normed(q, pert, True)
    report(f"  C1 with inv {tag}", rotate(t0p, t1p, False), c1)

# and against a reference computed with the sum in fp64 (exact reduction)
ss64 = (q.float().double() ** 2).sum(-1, keepdim=True)
inv64 = torch.rsqrt(ss64 / D + EPS).float()
t0e, t1e = normed(q, inv64, True)
print(f"\n  inv_rms differs from the fp64-exact one in "
      f"{(inv != inv64).sum().item()}/{inv.numel()} rows")
report("  C1 with inv from an exact fp64 sum", rotate(t0e, t1e, False), c1)

# timing: what does one triton call cost at production shapes?
import time  # noqa: E402
for _ in range(3):
    triton_rope(q, k, pe, scale, None, EPS)
torch.xpu.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    triton_rope(q, k, pe, scale, None, EPS)
torch.xpu.synchronize()
ms = (time.perf_counter() - t0) / 10 * 1e3
mb = (q.numel() * 2 * 2 + q.numel() * 2) / 1e6   # read twice + write, bf16
print(f"\ntriton call (q+k together): {ms:.4f} ms  -> {mb / ms * 1e3:.0f} GB/s "
      f"({mb:.1f} MB moved)")
