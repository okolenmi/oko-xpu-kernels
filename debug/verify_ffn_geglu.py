"""Is the fused GEGLU FFN the same function as the eager chain?

SDXL's FFN is a GEGLU (comfy/ldm/modules/attention.py:166):

    x, gate = proj(x).chunk(2, -1)
    out = x * F.gelu(gate)

The library's fused path folds the activation into the rotate+quantize kernel
(activation code 4), so the [M, 2*ffn] bf16 tile never exists.  That is only
safe if the kernel's activation is bit-identical to torch's, and the
activation here is the *erf* GELU -- the one the library could not previously
reproduce (the tanh form it could).

It can: debug/native/probe_gelu_exact.cpp compares sycl::erf's GELU against
torch's F.gelu(approximate="none") over all 65536 bf16 values and finds 0
mismatches.  This script checks the consequence: the whole fused FFN, and the
fused quantizer alone, against the chains a model would write.

    PYTHONPATH=python python debug/verify_ffn_geglu.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import xpu_kernels  # noqa: E402
from xpu_kernels import int8  # noqa: E402

DEV = torch.device("xpu", 0)
failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def weights(n, k, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = (torch.randn(n, k, device=DEV, generator=g) * 0.05).to(torch.bfloat16)
    return int8.prepare_weight(w)


def bit_equal(a, b):
    return bool(a.view(torch.int16).eq(b.view(torch.int16)).all())


def unfused_geglu_ffn(x, up8, up_scale, down8, down_scale):
    """The chain a model writes, in the INT8 path."""
    q, xs = int8.quantize(x)
    acc = torch._int_mm(q, up8.t())
    bf = int8.dequant(acc, xs, up_scale)
    hidden, gate = bf.chunk(2, dim=-1)
    y = hidden * torch.nn.functional.gelu(gate)
    q2, xs2 = int8.quantize(y)
    acc2 = torch._int_mm(q2, down8.t())
    return int8.dequant(acc2, xs2, down_scale)


def unfused_gelu_ffn(x, up8, up_scale, down8, down_scale):
    q, xs = int8.quantize(x)
    acc = torch._int_mm(q, up8.t())
    bf = int8.dequant(acc, xs, up_scale)
    y = torch.nn.functional.gelu(bf)
    q2, xs2 = int8.quantize(y)
    acc2 = torch._int_mm(q2, down8.t())
    return int8.dequant(acc2, xs2, down_scale)


print(f"device: {int8.device_name()}")

print("\n1. the fused GEGLU FFN against the unfused chain")
for (m, k, f) in [(64, 256, 512), (64, 1280, 5120), (256, 1280, 5120),
                   (1024, 512, 2560)]:
    x = (torch.randn(m, k, device=DEV) * 0.5).to(torch.bfloat16)
    up8, up_scale = weights(2 * f, k, 1)
    down8, down_scale = weights(k, f, 2)
    fused = int8.ffn(x, up8, up_scale, down8, down_scale, act="geglu_erf")
    unfused = unfused_geglu_ffn(x, up8, up_scale, down8, down_scale)
    check(f"M={m} K={k} ffn={f}", bit_equal(fused, unfused),
          "" if bit_equal(fused, unfused) else
          f"max abs diff {(fused.float() - unfused.float()).abs().max().item():.3e}")

print("\n2. the fused unpaired GELU FFN (diffusers' SDXL shape)")
for (m, k, f) in [(64, 256, 512), (64, 1280, 5120)]:
    x = (torch.randn(m, k, device=DEV) * 0.5).to(torch.bfloat16)
    up8, up_scale = weights(f, k, 3)
    down8, down_scale = weights(k, f, 4)
    fused = int8.ffn(x, up8, up_scale, down8, down_scale, act="gelu_erf")
    unfused = unfused_gelu_ffn(x, up8, up_scale, down8, down_scale)
    check(f"M={m} K={k} ffn={f}", bit_equal(fused, unfused))

print("\n3. the fused quantizer alone (the activation fused into the input)")
m, k = 128, 512
# the unpaired form: one tensor in, one tensor out
x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16) * 2.0
q_ref, s_ref = int8.quantize(torch.nn.functional.gelu(x).contiguous())
q_new, s_new = int8.quantize_act(x.contiguous(), "gelu_erf")
check("gelu_erf: the int8 codes", bit_equal(q_new, q_ref),
      f"{int((q_new != q_ref).sum())} of {q_ref.numel()} differ")
check("gelu_erf: the row scales",
      bool(s_new.view(torch.int32).eq(s_ref.view(torch.int32)).all()))
# the paired form: the halves in, one tensor out
x = torch.randn(m, 2 * k, device=DEV, dtype=torch.bfloat16) * 2.0
hidden, gate = x.chunk(2, dim=-1)
q_ref, s_ref = int8.quantize((hidden * torch.nn.functional.gelu(gate)).contiguous())
q_new, s_new = int8.quantize_act(x.contiguous(), "geglu_erf")
check("geglu_erf: the int8 codes", bit_equal(q_new, q_ref),
      f"{int((q_new != q_ref).sum())} of {q_ref.numel()} differ")
check("geglu_erf: the row scales",
      bool(s_new.view(torch.int32).eq(s_ref.view(torch.int32)).all()))

print("\n4. the halves must not be swapped")
# hidden = 0, gate = 2 everywhere: the result is 0 * gelu(2) = 0.  If the
# kernel read the halves the other way round it would compute
# 2 * gelu(0) = 0 as well -- so use an asymmetric pair instead.
m, k = 8, 256
x = torch.zeros(m, 2 * k, device=DEV, dtype=torch.bfloat16)
x[:, :k] = 1.0        # hidden = 1
x[:, k:] = -2.0       # gate = -2
hidden, gate = x.chunk(2, dim=-1)
q_ref, _ = int8.quantize((hidden * torch.nn.functional.gelu(gate)).contiguous())
q_new, _ = int8.quantize_act(x, "geglu_erf")
check("hidden * gelu(gate), not gate * gelu(hidden)", bit_equal(q_new, q_ref))

print("\n5. the capture: the fused FFN replays bit-identically")
m, k, f = 64, 256, 512
x = (torch.randn(m, k, device=DEV) * 0.5).to(torch.bfloat16)
up8, up_scale = weights(2 * f, k, 5)
down8, down_scale = weights(k, f, 6)
xs = x.clone()
g = torch.xpu.XPUGraph()
try:
    with torch.xpu.graph(g):
        captured = int8.ffn(xs, up8, up_scale, down8, down_scale, act="geglu_erf")
    check("the capture itself", True)
except Exception as exc:  # noqa: BLE001 - the failure mode is the finding
    check("the capture itself", False, f"{type(exc).__name__}: {exc}")
    captured = None
if captured is not None:
    x2 = (torch.randn(m, k, device=DEV) * 0.5).to(torch.bfloat16)
    eager = int8.ffn(x2, up8, up_scale, down8, down_scale, act="geglu_erf")
    with torch.no_grad():
        xs.copy_(x2)
    g.replay()
    torch.xpu.synchronize()
    check("the replay against the eager path", bit_equal(captured, eager))

print("\n" + ("PASS" if not failures else f"FAIL: {len(failures)} check(s)"))
sys.exit(1 if failures else 0)
