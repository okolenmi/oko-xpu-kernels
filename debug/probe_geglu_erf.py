"""Where Triton's gelu differs from torch's: every bf16 gate value, one at a time.

Runs the fused forward with x = 1 on all 65 536 bf16 bit patterns, so the output is ``bf16(gelu(gate))`` and can be compared
with ``F.gelu`` value by value.  Prints how many differ, in which range of the gate, and the float32 ``erf`` of both for the
worst few -- the evidence for the statement in ``geglu.py`` about the erf tail.

    LEVEL_ZERO_V1_SDK_PATH=... python debug/probe_geglu_erf.py
"""
import torch

from xpu_kernels.triton import geglu as G

dev = "xpu"
bits = torch.arange(65536, dtype=torch.int32, device=dev).to(torch.int16)  # bit patterns, wrapping above 32767
gate = bits.view(torch.bfloat16)
gate = gate[torch.isfinite(gate.float())].reshape(1, -1)
n = gate.shape[1]
h = torch.cat([torch.ones_like(gate), gate], dim=1).contiguous()
got = G._forward(h)
want = torch.nn.functional.gelu(gate)
bad = got.view(torch.int16) != want.view(torch.int16)
print(f"{int(bad.sum())} of {n} bf16 gate values differ")
g = gate[bad].float()
if g.numel():
    for lo, hi in ((-1e30, -6), (-6, -4), (-4, -3), (-3, -2), (-2, 0), (0, 2), (2, 1e30)):
        print(f"  gate in [{lo:g}, {hi:g}): {int(((g >= lo) & (g < hi)).sum())} differ, of {int(((gate.float() >= lo) & (gate.float() < hi)).sum())}")
    order = g.abs().argsort()
    for i in order[:6].tolist():
        v = g[i]
        print(f"  gate {v.item():+.6g}: torch {want[bad][i].item():.6g}, triton {got[bad][i].item():.6g}")
