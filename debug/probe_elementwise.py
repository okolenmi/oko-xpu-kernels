"""How much of the 300 ms/step of elementwise work is recoverable?

The device budget spends ~154 ms per forward (309 ms/step) in eager
pointwise kernels: tanh 48, mul 43, div 31, addcmul_ 16, sigmoid 16.  Almost
all of it is the FFN activation of the MossImage block,

    out = tanh(gate/4) * sigmoid(gate) * tanh(hidden/25)

on [tokens, 8192] bf16, which eager torch runs as seven separate kernels over
the same two tensors, plus two gated residual addcmuls on [tokens, 3072].

This probe measures, on the real shapes:

  1. the eager chain, kernel by kernel, so the traffic is attributable,
  2. the same chain under torch.compile (inductor), to see whether the XPU
     backend fuses it and what that is worth,
  3. the bandwidth floor: one read of both inputs and one write,
  4. the gated residual, eager vs compiled.

    python debug/probe_elementwise.py [tokens]
"""
import sys
import time

import torch

torch.manual_seed(0)
dev = torch.device("xpu", 0)
dt = torch.bfloat16

M = int(sys.argv[1]) if len(sys.argv) > 1 else 3974
D = 3072
F = 8192

print(f"tokens={M} dim={D} ffn={F} dtype={dt}")


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


hidden = torch.randn(M, F, device=dev, dtype=dt)
gate = torch.randn(M, F, device=dev, dtype=dt)
x = torch.randn(M, D, device=dev, dtype=dt)
y = torch.randn(M, D, device=dev, dtype=dt)
g = torch.randn(M, D, device=dev, dtype=dt)

bytes_floor = (hidden.numel() * 2 * 2 + hidden.numel() * 2)  # 2 reads + 1 write
print(f"\nFFN activation, floor = {bytes_floor/2**20:.0f} MiB -> "
      f"{bytes_floor/400e9*1e3:.3f} ms at 400 GB/s")


def eager_chain():
    return torch.tanh(gate / 4.0) * torch.sigmoid(gate) * torch.tanh(hidden / 25.0)


# individual pieces, to attribute the eager cost
print("\neager, piece by piece (per call):")
for name, fn in [
    ("gate / 4.0        ", lambda: gate / 4.0),
    ("tanh(gate/4)      ", lambda: torch.tanh(gate / 4.0)),
    ("sigmoid(gate)     ", lambda: torch.sigmoid(gate)),
    ("hidden / 25.0     ", lambda: hidden / 25.0),
    ("tanh(hidden/25)   ", lambda: torch.tanh(hidden / 25.0)),
    ("mul               ", lambda: torch.tanh(gate / 4.0) * torch.sigmoid(gate)),
]:
    ms = bench(fn)
    print(f"  {name} {ms:7.3f} ms")

ms_eager = bench(eager_chain)
print(f"  whole chain (eager)  {ms_eager:7.3f} ms")

# ---- torch.compile --------------------------------------------------------
print("\ntorch.compile (inductor, XPU):")
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = False
    compiled_chain = torch.compile(eager_chain, dynamic=False)
    out_ref = eager_chain()
    out = compiled_chain()
    torch.xpu.synchronize()
    ok = torch.allclose(out.float(), out_ref.float(), atol=0.02, rtol=0.02)
    print(f"  compiles, matches eager = {ok}")
    ms = bench(compiled_chain)
    print(f"  whole chain (compiled) {ms:7.3f} ms   speedup {ms_eager/ms:.2f}x")
except Exception as e:
    print(f"  failed: {str(e).splitlines()[0][:200]}")

# ---- gated residual -------------------------------------------------------
print("\ngated residual (x.addcmul_(y, gate)) on [M, 3072]:")
ms = bench(lambda: x.addcmul_(y, g))
print(f"  eager   {ms:7.3f} ms")


def res():
    return x.addcmul(y, g)


try:
    cres = torch.compile(res, dynamic=False)
    res()
    torch.xpu.synchronize()
    ms = bench(lambda: cres())
    print(f"  compiled {ms:7.3f} ms")
except Exception as e:
    print(f"  compiled failed: {str(e).splitlines()[0][:120]}")

# ---- what an ideal fused kernel would do ---------------------------------
print("\nreference: a single elementwise pass over [M, 8192] bf16")
a = torch.empty_like(hidden)
one = torch.randn_like(hidden)
ms = bench(lambda: torch.add(one, one, out=a))
print(f"  1 read + 1 write (torch.add) {ms:7.3f} ms  ({2*a.numel()*2/ms*1e3/1e9:.0f} GB/s)")
