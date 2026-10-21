"""Torch's own GELU over every bf16 value, dumped for the native probe.

`debug/native/probe_gelu_exact.cpp` asks whether a SYCL kernel can reproduce
torch's GELU bit for bit.  The reference has to come from torch, so this
dumps it: the raw bf16 bit patterns of `F.gelu` over all 65536 bf16 values,
for both the erf form (SDXL's) and the tanh form (MossImage's).

    PYTHONPATH=python python debug/probe_gelu_torch.py
    ../debug/native/bin/probe_gelu_exact build/gelu_torch_erf.bin
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


def dump(x, fn, path):
    y = fn(x)
    # view(), not to(): this is a bit pattern, not a value conversion
    y.view(torch.int16).cpu().numpy().tofile(path)
    return y


def main():
    bits = torch.arange(65536, dtype=torch.int16)
    x = bits.view(torch.bfloat16)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
    os.makedirs(out, exist_ok=True)

    for dev in (["xpu", "cpu"] if torch.xpu.is_available() else ["cpu"]):
        xd = x.to(dev)
        y_erf = dump(xd, lambda t: torch.nn.functional.gelu(t, approximate="none"),
                     os.path.join(out, f"gelu_torch_erf_{dev}.bin"))
        y_tanh = dump(xd, lambda t: torch.nn.functional.gelu(t, approximate="tanh"),
                      os.path.join(out, f"gelu_torch_tanh_{dev}.bin"))
        # the float64 truth, for scale
        y64 = torch.nn.functional.gelu(xd.double(), approximate="none")
        rel = ((y_erf.double() - y64).abs() / y64.abs().clamp_min(1e-30)).nan_to_num(0)
        finite = torch.isfinite(y64) & (y64.abs() > 1e-6)
        print(f"{dev}: torch's erf-GELU vs float64 -- worst relative "
              f"{rel[finite].max().item():.3e} over {int(finite.sum())} values")
        print(f"{dev}: erf vs tanh form differ in "
              f"{int((y_erf != y_tanh).sum())} of 65536 values")
        if dev == "cpu":
            print(f"  cpu: the two forms' bit patterns are the reference for the "
                  f"erf comparison on the other device")


if __name__ == "__main__":
    main()
