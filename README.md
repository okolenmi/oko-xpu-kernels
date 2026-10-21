# xpu-kernels — fused kernels for Intel Arc (Xe2)

Two things live here:

- **`xpu_kernels.int8`** — ConvRot INT8 for XPU tensors: a SYCL library
  (`libxpuint8.so`) plus a torch-only binding. Fused rotate+quantize on the
  activation side, a fused dequant epilogue on the output side, the backward's
  `dL/dx` and `dL/dW`, and an autograd op so an INT8 layer can sit in a
  training graph instead of silently producing no gradient.
- **`xpu_kernels.triton`** — single-pass Triton kernels: `rms_norm + rope`,
  `rms_norm + adaLN`, a row-quantizer and an NF4 dequantizer. Each verifies
  itself against the implementation it replaces, once per process, before it is
  allowed to engage.

Everything in `debug/` is the evidence: the probes, the A/B harnesses and the
verifiers that produced every number in `docs/`.

## Install and use

```bash
pip install xpu-kernels     # wheel; needs a matching oneAPI runtime (see docs/api.md)
pip install -e .            # from a checkout
python -m xpu_kernels       # self-test on this machine: the library and every gate
```

```python
import xpu_kernels

w8, w_scale = xpu_kernels.prepare_weight(w)    # bf16 [N,K] -> int8 [N,K] + fp32 [N]
y = xpu_kernels.linear(x, w8, w_scale, bias)   # bf16 [..,K] -> bf16 [..,N]
y = xpu_kernels.int8_linear(x, w8, w_scale)    # the same, differentiable
```

The INT8 path applies to a linear whose contraction length is a multiple of
**256** (the ConvRot group) and whose output width is a multiple of 16;
`unsupported_reason(...)` says which of the two a shape fails.

The full API — the accumulator paths, the backward, the capture contract, the
Triton kernels, the C entry points — is in [`docs/api.md`](docs/api.md).

## What it buys

| | measured |
|---|---|
| INT8 vs BF16, **both paying the activation rotation** | parity, 15-17 % faster at M=1024, and **4× less weight memory** |
| the row-quantizer (the gradient path) | 4.17 → **0.27 ms**, 9 kernel launches become 1 |
| the NF4 dequantizer | 20.75 → **0.66 ms** (31×), bit-exact, no scratch |
| the weight dequantizer (2.17 B parameters) | 215 → **63 ms**, bit-identical |
| the fused FFN (dequant + activation + quantize in one pass) | **1.2-1.5×** over the unfused INT8 chain, bit-identical |
| `rms_norm+rope` / `rms_norm+adaLN` | **2.34×** / **1.56×**, bit-exact, differentiable |
| device-graph capture: host time per stage | 648-1282 → **40-82 µs** |

**INT8 is a memory lever on this card, not a speed lever.** At parity speed
with a quarter of the weight memory, which is why ComfyUI keeps it behind an
opt-in rather than making it the default. The bf16 path's speedups come from
fusing the norms and from capture, not from the GEMM — oneDNN already has the
last word there.

## The docs

| doc | one line |
|---|---|
| [`docs/api.md`](docs/api.md) | the API: Python, C, the backward, capture, the Triton kernels |
| [`docs/architecture.md`](docs/architecture.md) | how the rotation maps onto the hardware, and why the API is shaped this way |
| [`docs/performance.md`](docs/performance.md) | every measurement, and the device ceilings to compare against |
| [`docs/experiments.md`](docs/experiments.md) | what was tried and did not work, so it is not tried again |
| [`docs/sdxl.md`](docs/sdxl.md) | what this library can and cannot do for SDXL, and the plan for it |
| [`docs/adding-a-kernel.md`](docs/adding-a-kernel.md) | the checklist every addition follows |
| [`docs/todo.md`](docs/todo.md) | what is next, ranked, with the evidence that would close it |
| [`docs/publishing.md`](docs/publishing.md) | how a release happens (it is a deliberate, manual act) |
| [`docs/report-reduce-over-group.md`](docs/report-reduce-over-group.md) | the report sent to the vendor about a `reduce_over_group` that killed the host |
| [`docs/for-b580-diffusion-training.md`](docs/for-b580-diffusion-training.md) | what a *training* project on the same card can take |
| [`debug/README.md`](debug/README.md) | the instruments, and how to re-run them |
| [`CHANGELOG.md`](CHANGELOG.md) | what each release contains |

## Build and test

```bash
./build.sh                          # -> build/libxpuint8.so (needs icpx; oneAPI 2025.3)
PYTHONPATH=python python -m xpu_kernels       # self-test: the INT8 path and every gate
PYTHONPATH=python python -m pytest tests/ -q  # 82 CPU-only tests: the formulas and the gate rules
```

`build.sh` pins the compiler to **oneAPI 2025.3** on purpose: the `.so` links
against the SYCL runtime torch ships (`libsycl.so.8`), and a 2026.x `icpx`
produces a `.so` that cannot load — the 2025 runtime exports no
`submit_kernel*` symbol, so it is not shimmable with `-L` or an rpath. Set
`SYCL_ROOT` if your toolchain is elsewhere.

The library is optional at run time: if it is missing, the package says so
once and every call raises; ComfyUI's equivalent path falls back to a pure
torch implementation (4-6× slower, the same numbers within INT8 noise).

## Layout

```
include/xpu_kernels/convrot.h   the public C API
src/xpuint8.cpp                 the SYCL kernels
build.sh, wheel.sh              build, and build the wheel/sdist
python/xpu_kernels/int8/        the torch binding: _lib.py, ops.py, selftest.py
python/xpu_kernels/triton/      the Triton kernels, the gate, the eager reference
python/xpu_kernels/capture.py   capture readiness
reference/convrot_linear.py     a portable PyTorch ConvRot reference
kernels/                        standalone SYCL experiments (the DPAS GEMM)
debug/                          the instruments and the verification harnesses
tests/                          CPU-only unit tests
```
