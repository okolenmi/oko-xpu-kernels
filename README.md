# xpu-kernels — fused kernels for Intel Arc (Xe2), and the instruments that prove them

Kernel work for Intel Arc XPU tensors, in two halves:

- **`xpu_kernels.int8`** — a SYCL library (`libxpuint8.so`) plus a torch-only
  binding. Fused Kronecker-Hadamard rotation + INT8 quantization on the
  activation side, a fused dequant(+activation) epilogue on the output side,
  and accumulator paths that let a consumer walk an int32 GEMM result without a
  bf16 round trip. Inference only.
- **`xpu_kernels.triton`** — single-pass Triton kernels for the non-quantized
  path: `rms_norm + rope` and `rms_norm + adaLN modulation`, each bit-identical
  to the chain it replaces, each verified against a reference once per process
  before it is allowed to engage.

Everything in `debug/` is the evidence: bandwidth and epilogue ceilings, the
`tanh` slow path, attention backend equivalence, host-vs-device attribution,
and the A/B harnesses that produce the numbers quoted here. The findings are
written up in `docs/` so they do not have to be rediscovered.

## Documentation

| doc | contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | how the rotation maps onto the hardware, the register-residency constraint, DPAS layout |
| [`docs/performance.md`](docs/performance.md) | all measurements, device ceilings, stage tables, **the fair BF16 comparison** |
| [`docs/experiments.md`](docs/experiments.md) | what was tried, including the dead ends, so they are not repeated |
| [`python/xpu_kernels`](python/xpu_kernels/__init__.py) | the Python package: `xpu_kernels.int8` (ConvRot INT8) and `xpu_kernels.triton` (rope, adaLN) |
| [`docs/adding-a-kernel.md`](docs/adding-a-kernel.md) | the structure every addition follows: gate, escape hatch, reference, evidence |
| [`docs/todo.md`](docs/todo.md) | what is next, ranked, with the evidence that would close each item |
| [`docs/for-b580-diffusion-training.md`](docs/for-b580-diffusion-training.md) | what a *training* project on the same card can reuse (and what it must not) |
| [`debug/README.md`](debug/README.md) | the measurement instruments and how to re-run them |

## Headline result

Both INT8 and BF16 have to rotate the activation (ConvRot stores weights
pre-rotated), so that is what the comparison below includes:

| M | ratio INT8/BF16 (4 runs) | verdict |
|---|---|---|
| 512 | 1.00 - 1.10 | INT8 0-10% slower |
| 1024 | 0.83 - 0.85 | **INT8 15-17% faster** |
| 2048 | 0.98 - 1.01 | parity |
| 4096 | 0.94 - 0.99 | parity to 6% faster |
| 8192 | 0.90 - 0.98 | parity to 10% faster |

(Ratios are stable across runs to ~1-2%; absolute milliseconds drift 10-15%
with GPU thermals. Full single-run table in `docs/performance.md`.)

**Native INT8 is never meaningfully slower, and is 15-17% faster at M=1024**,
with 4x less weight memory. Reproduce with `debug/bench_bf16_fair.py`.

> Earlier revisions compared against a BF16 `F.linear` with no activation
> rotation and reported INT8 as 1.9-2.3x slower. That comparison was not fair;
> it omitted work the INT8 path has to do.

## Build

```bash
./build.sh                    # -> build/libxpuint8.so
```

`build.sh` needs a DPC++ (`icpx`) toolchain and **pins the version on purpose**:
the `.so` links against the SYCL runtime that torch ships (`libsycl.so.8`, oneAPI
2025.3 here). A 2026.x `icpx` produces a `.so` that cannot load — the 2025
runtime exports no `submit_kernel*` symbol, so it is not shimmable with `-L` or
an rpath. Set `SYCL_ROOT` if yours is elsewhere:

```bash
SYCL_ROOT=/opt/intel/oneapi/compiler/2025.3 ./build.sh
```

The library is optional at run time: if it is missing, the Python package says
so once and every call raises; ComfyUI's equivalent path falls back to a pure
torch implementation (4-6x slower, same numbers within INT8 noise).

## Use from Python

The binding is `python/xpu_kernels` and depends on `torch` only:

```bash
PYTHONPATH=python python -m xpu_kernels      # self-test on this machine
```

```python
import torch, xpu_kernels

w8, w_scale = xpu_kernels.prepare_weight(w)   # bf16 [N,K] -> int8 + fp32 [N], once
y = xpu_kernels.linear(x, w8, w_scale, bias)  # bf16 [..,K] -> bf16 [..,N]

# or drop-in for inference:
model.some_linear = xpu_kernels.Int8Linear.from_linear(model.some_linear)
```

The library is resolved from `$XPU_KERNELS_LIB` (or the legacy `$XPU_INT8_LIB`),
then `lib/libxpuint8.so` or `build/libxpuint8.so` next to this repo, then
`sys.prefix`.

When the consumer can walk the accumulator itself, skip the bf16 round trip:

```python
acc, x_scale = xpu_kernels.linear_acc(x, w8, w_scale)        # int32 [M,N]
y     = xpu_kernels.dequant(acc, x_scale, w_scale, bias)     # bf16 [M,N]
slice = xpu_kernels.dequant_slice(big_acc[:, 0:3072], xs, ws[:3072])   # strided, in place
q, s  = xpu_kernels.dequant_act_quantize(up_acc, xs, up_ws)  # dequant+situ+rotate+quantize
q, s  = xpu_kernels.quantize_act(x2, "situ")                 # activation fused into the quantizer
```

`dequant(linear_acc(...))` is bit-identical to `linear(...)`; the fused
activation paths are bit-identical to the unfused chain they replace
(`python -m xpu_kernels` checks the second one; the exhaustive suites are in
`debug/`). `prepare_weight` is the only supported way to make a weight: the
activation rotation is baked into the weight layout, so a plain
`w/amax*127` quantization is silently wrong.

### The backward: `dL/dx`, weight-only

For `y = x @ W^T`, the vjp is `dx = dy @ W` — no transpose, no bias. With an
INT8 weight, recover `W` once and keep it:

```python
w = xpu_kernels.dequant_weight(w8, w_scale)       # bf16 [N,K], once per weight
dx = xpu_kernels.linear_dx(dy, w)                 # bf16 [M,K]
```

or on a module, which caches it: `mod.dense_weight()`, `mod.dx(dy)`.

Why dequantize instead of an INT8 GEMM: `W[n,k] = W8[n,k] * s[n]`, so `s[n]` sits
on the *contraction* index of `dx = dy @ W` and cannot fold into an epilogue
(the forward gets away with it because its weight scale is per output column).
Dequantizing reads 1.5 bytes per parameter and writes bf16 — 2.6 ms per 25 M
parameters here, ~0.4 s once for a 3.7 B-parameter model.

Against the same GEMM in autograd the result is **exact** (it is the same
oneDNN bf16 GEMM, so `linear_dx(dy, W)` == the autograd gradient bit for bit).
Against the *unquantized* weight the gradient is 0.9 % off, which is the INT8
weight itself (0.90 % quantization + 0.17 % for storing the dequantized weight
in bf16); `python -m xpu_kernels` and `debug/verify_linear_dx.py` both print it.
There is also `linear_dx_int8(dy, w8, w_scale)`, which keeps the weight in int8
and quantizes `dy` instead (exact in the weight, since `s[n]` becomes a column
scale on `dy`). Measured at M=4096: **it is 3.6x slower** than the bf16 path
(6.9 vs 1.9 ms -- the quantizer, epilogue and final rotation cost more than the
faster GEMM saves) and 40 % noisier, so treat it as a *memory* option
(2x less weight memory), not a speed one.

Not here yet: `dW = dy^T @ x` (only needed for full fine-tuning) -- see
[`docs/todo.md`](docs/todo.md), which also has the graph-capture checklist.

For a worked example of wiring this into a real model — weight loading, the
per-layer fallback when the device is offloaded or the library is missing, and
the fused FFN path — read `comfy/xpu_int8.py` in the ComfyUI tree; it is the
same library and the same C API, just with ComfyUI's dispatch around it.

## Use the Triton kernels

The other half, for the bf16 path. Both kernels fuse a norm with the op that
reads it, and both refuse to run until they have proved themselves against the
implementation you would otherwise fall back to:

```python
import xpu_kernels.triton as k

k.set_reference(k.rope, my_rms_rope)        # once, before first use
q, kk = k.rope.rms_rope_split_half(q, kk, freqs_cis, q_scale, k_scale)
x = k.adaln.rms_adaln(x, scale, shift)

print(k.status())   # rms_rope          : active: verified against the reference (1572864 values), num_warps=4
                    # rms_rope_dequant  : active: verified against the reference (1572864 values), num_warps=4
                    # rms_rope_backward : active: verified within 0.0001 (worst relative error 7.1e-08, 786432 values), num_warps=4
                    # rms_adaln         : active: verified against the reference (786432 values), num_warps=4
```

If `comfy_kitchen` is importable, its kernels are the default reference and no
`set_reference` is needed. Without a reference the kernels stay inactive and
`status()` says why — a kernel whose numerics nobody has pinned down is not
worth the speed. `XPU_KERNELS_NO_ROPE=1` / `XPU_KERNELS_NO_ADALN=1` disable them
outright (the legacy `MOSS_NO_*` names still work).

The gate itself is shared and tested on CPU (`tests/test_gate.py`): it coalesces
"verified once per process", "any mismatch disables the kernel for the rest of
the process", and "never pass a probe that could not run". `docs/adding-a-kernel.md`
is the checklist for a new kernel, including the backward-pass rules.

`rms_adaln` is differentiable too: one kernel for `dx` plus a two-stage
reduction for `dscale`/`dshift` (sums over the rows the modulation was broadcast
across, both in the same pass, no atomics), 6.3× faster than the eager chain
forward+backward and verified against `torch.autograd` at 1.5e-7.

`rms_rope_split_half` is differentiable: with a grad-requiring input it routes
through an `autograd.Function` whose backward is a single fused kernel (the
per-head `dot` product never leaves registers, so there is no scratch buffer and
nothing extra to launch). Measured on this card at the production shape, one
tensor: **~0.62 ms forward+backward vs ~8.9 ms** for the eager chain, 14.3×,
and the gradient is *closer* to the float64 truth than `torch.autograd`'s
(1.7e-3 vs 2.3e-3) because the fused path never rounds the gradient to bf16 at
the forward's cast. `freqs_cis` and `scale` are treated as constants; a trainer
that learns them needs a differentiable chain instead
(`xpu_kernels.triton.reference.rms_rope_split_half`).

Measured on this card, 42-block DiT at M≈3974: rope 2.34× (35.7 → 15.3 ms per
forward), adaLN 1.56× (12.7 → 8 ms), both bit-exact. A different `num_warps`
changes the reduction order and the output; do not tune it without re-running
the gate.

## Use from C

```c
xpuint8_convrot_quantize(x, q, scale, M, K);   // bf16[M,K] -> int8[M,K] + fp32[M]
acc = torch._int_mm(q, wt);                     // int8 GEMM, oneDNN
xpuint8_dequant_epilogue(acc, out, scale, ws, bias, M, N);  // -> bf16[M,N]
```

The BF16 path needs only the rotation:

```c
xpuint8_convrot_rotate_bf16(x, xr, M, K);      // bf16[M,K] -> bf16[M,K]
y = torch.nn.functional.linear(xr, w_bf16, bias);
```

Every entry point has a `_stream` variant taking the queue pointer from
`torch.xpu.current_stream(dev).sycl_queue`, which is what the Python binding
uses; the non-stream form waits. See `include/xpu_kernels/convrot.h` for the full
C API and `docs/architecture.md` for the algorithm.

---

## Status

| component | state |
|---|---|
| ESIMD `dpas` layout/probe | resolved, documented below |
| Correct ESIMD INT8 GEMM | working, CPU-verified, **43 TOPS** (was 0.7); `torch._int_mm` (oneDNN) is still faster and is what ships |
| Fused ConvRot rotate+quantize kernel | working, fp64-verified, **1.4x faster than eager** |
| DPAS `RepeatCount` + accumulating form (kernels/int8_gemm_v4.cpp) | working, CPU-verified, 2.5x fewer instructions but **slower** (28 vs 43 TOPS) |
| Fused INT8 dequant epilogue | working, fp64-verified, **9x faster than eager, at bandwidth limit** |
| End-to-end ConvRot INT8 linear | working, **1.7-2.1x faster than eager** |
| Native INT8 vs dequantized BF16 *including* the rotation | **parity, 15-17% faster at M=1024** — see the headline table. INT8's real win is 4x less weight memory |
| Fused dequant+situ+rotate+quantize | working, **bit-identical to the unfused chain**, 7 launches -> 1 |
| Row-strided dequant epilogue | working, bit-identical; lets a q|k|v GEMM drop the q/k epilogue entirely |
| Triton `rms_norm + rope` | working, **2.34×**, bit-identical to the shipped kernel (`num_warps=4` pinned) |
| Triton rope **backward** | working, **14.3×** on fwd+bwd vs the eager chain, differentiable wrapper |
| Triton `rms_norm + adaLN` | working, **1.56×**, bit-identical, **differentiable** (fwd+bwd **6.3×** vs eager) |
| INT8 `dL/dx` (weight-only) | working, **exact** vs the same GEMM in autograd, 0.9 % from the unquantized weight (that is the INT8 weight); dequant is one-time |
| INT8 `dW`, quantized-`dy` gradient | **not implemented** — design and the scale-layout argument in [`docs/todo.md`](docs/todo.md) |
| Device-graph capture | rope fwd+bwd captured in one `XPUGraph`, replay bit-identical, **host time 694 → 81 µs** per stage |

## Layout

```
include/xpu_kernels/convrot.h   public C API (callable on torch XPU data_ptr())
src/xpuint8.cpp                 both SYCL kernels + C API
build.sh                        builds build/libxpuint8.so with icpx (version-pinned)
python/xpu_kernels/int8/        torch-only binding: _lib.py loader, ops.py, selftest.py
python/xpu_kernels/triton/      kernels + _gate.py; reference.py: eager chain and analytic backward
pyproject.toml                  `pip install .` for another project
benchmarks/bench_pipeline.py    end-to-end: eager vs fused vs BF16
benchmarks/bench_int8.py        torch _int_mm vs BF16 microbenchmark
reference/convrot_linear.py     portable PyTorch ConvRot reference
kernels/                        standalone experiments (DPAS probes, GEMM, per-kernel tests)
debug/                          measurement instruments and the verification harnesses
docs/                           architecture, performance, experiments, findings, how to add a kernel, todo
tests/                          CPU-only unit tests (the gate's contract)
```

## Build and run

```bash
./build.sh                        # -> build/libxpuint8.so
PYTHONPATH=python python -m xpu_kernels      # self-test: int8 + every gate
PYTHONPATH=python python -m pytest tests/ -q # CPU-only: the gate's contract
python benchmarks/bench_pipeline.py
python debug/bench_bf16_fair.py   # the honest INT8-vs-BF16 number
```

Triton's XPU backend compiles the kernels against the Level Zero headers on
first use. If they are not on the default include path, point
`LEVEL_ZERO_V1_SDK_PATH` at the SDK that came with the driver install — that is
a first-run cost only, the compiled kernels are cached.

Per-kernel tests:

```bash
icpx -O3 -fsycl kernels/convrot_quant.cpp   -o /tmp/cq   && /tmp/cq 4096 3072
icpx -O3 -fsycl kernels/dequant_epilogue.cpp -o /tmp/epi  && /tmp/epi 4096 3072
icpx -O3 -fsycl kernels/gemm_tiled.cpp      -o /tmp/gt   && /tmp/gt 1024 1024 1024 16
```

## API

```c
int  xpuint8_init(void);
const char *xpuint8_device_name(void);
int  xpuint8_convrot_groupsize(void);              // 256

// bf16 [M,K] -> int8 [M,K] + fp32 [M] row scales
int  xpuint8_convrot_quantize(const void *x, int8_t *q, float *scale, int64_t M, int64_t K);

// int32 [M,N] -> bf16 [M,N];   out = acc * x_scale[m] * w_scale[n] + bias[n]
int  xpuint8_dequant_epilogue(const int32_t *acc, void *out, const float *x_scale,
                              const float *w_scale, const float *bias, int64_t M, int64_t N);
const char *xpuint8_last_error(void);
```

## Algorithm: fused ConvRot rotation

ConvRot's transform matrix is `H = kron(H4,...,H4) / sqrt(gs)` with

```
H4 = [[ 1,  1,  1, -1],
      [ 1,  1, -1,  1],
      [ 1, -1,  1,  1],
      [-1,  1,  1,  1]]
```

A Kronecker product of *identical* factors applies `H4` along each axis of the 4D
tensor view, and — because every factor is the same matrix — the axis-to-stride
mapping is arbitrary. Indexing a `gs=256` group as `i = 16*w + q` gives

```
i0 = q & 3      i1 = (q >> 2) & 3      i2 = w & 3      i3 = w >> 2
```

so every axis collapses to a plain 4-point butterfly:

| axis | where | how |
|---|---|---|
| 0 | inside a register | 4 broadcast-swizzles + `H4` sign vectors |
| 1 | inside a register | 4 broadcast-swizzles + `H4` sign vectors |
| 2 | across registers `{4g+0..4g+3}` | full-width elementwise butterfly |
| 3 | across registers `{c, c+4, c+8, c+12}` | full-width elementwise butterfly |

Each butterfly is `H4/2`; four of them give `(H4/2)^4 = 1/16 = 1/sqrt(256)`.
This costs ~2 vector ops per element instead of the 256 MACs per element that a
`bmm(M, ng, 256, 256)` rotation needs.

A group lives in 16 `vec<float,16>` registers (16 zmm, 1 KB) with fully unrolled
compile-time indices, so it never spills. The grid is one sub-group (16 lanes)
per row; lane `l` owns ConvRot groups `l, l+16, l+32, ...`, so the rotation needs
no cross-lane communication at all. The only sub-group collective is the row
abs-max, done with a 4-step XOR butterfly over `select_from_group`.

`H4` does **not** factor into signed `H2` butterflies (verified exhaustively over
all 192 signed-`H2` level matrices, depths 3 and 4), so the 4-point form is the
cheap one.

## Verification

`kernels/convrot_quant.cpp` checks against a CPU Kronecker reference;
`benchmarks/bench_pipeline.py` checks end-to-end output against a **float64**
rotation on CPU:

```
correctness vs float64 ground truth @ M=8192:
  fused xpu-int8   relative L2 error: 0.008787
  comfy eager      relative L2 error: 0.009353
```

The fused path is slightly *more* accurate than comfy-kitchen's eager backend,
because it accumulates the rotation in fp32.

**comfy-kitchen's eager path is not a valid reference.** It rotates activations
with a bfloat16 `bmm`, and the resulting row abs-max — hence the row scale — can
be wrong by up to ~40%:

```
K=  256  groups=  1  our scale vs fp64: 0.0e+00   eager scale vs fp64: 4.08e-01
K= 3072  groups= 12  our scale vs fp64: 6.4e-08   eager scale vs fp64: 2.39e-01
```

Consequently ~8% of quantized activations differ from eager by +/-1. That is
eager's error, not ours.

## Performance

Precise stage breakdown at M=4096, K=N=3072 (measured with preallocated buffers):

| stage | time | share | floor | headroom |
|---|---|---|---|---|
| rotate + row-quantize | 0.55 ms | 43% | 0.40 ms | 0.15 ms |
| `torch._int_mm` | 0.53 ms | 41% | 0.41 ms (190 TOPS peak) | 0.11 ms |
| dequant epilogue | 0.15 ms | 12% | 0.15 ms (bandwidth limit) | none |
| python alloc/dispatch | 0.28 ms | 22% | ~0 | benchmark artifact |
| **total** | **1.53 ms** | | | **~0.26 ms** |

\* benchmark artifact: the loop re-allocates the 12.6 MB int8 and 25 MB bf16
output buffers every iteration. Reused buffers do not pay this.

End-to-end sweep (K=N=3072). **The `bf16` column here omits the activation
rotation, so it is not the fair comparison** — use the headline table at the
top or `debug/bench_bf16_fair.py`; this table is kept only as history of the
eager-vs-fused INT8 gap:

```
     M  eager int8  fused int8      bf16  eager vs bf16  fused vs bf16
   512      0.782m      0.275m    0.140m          5.58x          1.96x
  1024      0.854m      0.536m    0.252m          3.39x          2.13x
  2048      1.664m      1.021m    0.405m          4.11x          2.52x
  4096      3.288m      1.877m    0.832m          3.95x          2.26x
  8192      6.344m      3.322m    1.631m          3.89x          2.04x
```

## Optimization log

All numbers measured on the B580 at M=4096, K=N=3072.

### Device ceilings (measured, not estimated)

These were wrong in an earlier revision of this file, so they are re-measured
here. The old "788 GB/s streaming" figure came from a buffer that fit in the
18 MB L2 and was re-read in a loop; it is not a DRAM number.

| buffer | read | write | copy (r+w) |
|---|---|---|---|
| 12.6 MB | 291 GB/s | 577 GB/s | 776 GB/s |
| 25.2 MB | 310 GB/s | 329 GB/s | 606 GB/s |
| 50.3 MB | 322 GB/s | 346 GB/s | 640 GB/s |
| 400 MB  | 468 GB/s | 397 GB/s | 885 GB/s |

So the pipeline's DRAM floor is roughly: rotate 37.8 MB -> 0.12 ms,
epilogue 75 MB -> 0.12 ms.

### Where the rotate+quantize time actually goes

The first ablation of this kernel was wrong -- it wrote the output **twice**,
which inflated the "store" attribution. With a single store pass:

| variant | time | delta |
|---|---|---|
| full kernel | 0.407 ms | |
| minus transform | 0.273 ms | transform = 0.134 ms (33%) |
| minus int8 store | 0.276 ms | store = 0.131 ms (32%) |
| minus abs-max + sub-group reduce | 0.381 ms | 0.025 ms (6%) |
| minus sub-group reduce only | 0.407 ms | 0.000 ms |

The load+store skeleton alone is 0.273 ms for 37.8 MB = 138 GB/s, i.e. **4x
below the streaming floor**. That gap is register residency: 256 floats (16
zmm) must stay live across load -> row abs-max -> quantize, so all 16 loads
complete before the first store and there is nothing to overlap with. An
earlier micro-benchmark isolated it exactly -- same copy loop, 0.402 ms when
all 16 registers are held vs 0.128 ms when load and store are interleaved.

The row-scale reduce itself is free (0.000 ms). It is the *serialization it
forces* that costs.

### Tried and rejected, each with data

Everything below was implemented and measured on this kernel:

| idea | result vs 0.407 ms baseline |
|---|---|
| fp16 butterfly (8 zmm instead of 16) | **1.14 ms** -- no native 512-bit fp16 ops here |
| shared-memory staging of the strided axis | **1.33 ms** -- barriers cost more than the pressure |
| 3-pass (rotate+stage bf16, reduce, quantize) | **0.75 ms** -- pass 1 alone is 0.73 ms; the 16 registers are still live |
| forced full unrolling of the group loop | **1.27 ms** -- spills |
| work-group size 16 / 32 / 64 / 128 / 256 | 0.60 / 0.60 / 0.57 / 0.57 / 0.59 ms -- not a lever |
| 64-byte stores instead of 16-byte | no change (0.114 vs 0.111 ms) |
| `vec<int32_t>` pack instead of scalar float packs | no change (0.411 vs 0.407 ms) |
| sub-group-scalar bf16 conversion via rounding-bit trick | **0.24-0.26 ms** vs 0.135 -- the scalar loop is already vectorised |
| 2 or 4 rows per work-group | **0.76 / 0.42 ms** -- 1.6-3x loss |
| AOT `-fsycl-targets=spir64_gen12` | unavailable: this compiler only accepts generic `spir64` |

**Idle lanes are free.** At K=3072 only 12 of 16 lanes have work. Measured per
element, K=1024 (4/16 lanes busy) is the *fastest* case at 33.4 Melem/s vs
26.5 at K=3072 and 19-20 at K=4096 -- that ordering is cache size, not lane
occupancy, so there is nothing to reclaim there.

### GEMM: our implementation is L2-bandwidth bound, not DPAS bound

v3 reaches 43 TOPS = 12.5% of the 344 TOPS DPAS peak. A shape sweep shows why:

| M x N x K | ms | TOPS | B+A traffic | effective |
|---|---|---|---|---|
| 4096x3072x3072 | 1.78 | 43.4 | 7.25 GB | 4.07 TB/s |
| 4096x256x3072 | 0.156 | 41.2 | 0.60 GB | 3.86 TB/s |
| 512x3072x3072 | 0.237 | 40.9 | 0.91 GB | 3.83 TB/s |
| 8192x3072x3072 | 3.50 | 44.2 | 14.50 GB | 4.14 TB/s |
| 4096x3072x12288 | 13.32 | 23.2 | 28.99 GB | 2.18 TB/s |

TOPS is flat at 41-44 for every cacheable shape and collapses only when the
working set leaves L2. Effective traffic sits at **~4 TB/s in all of them**.
The kernel is saturating L2, not DPAS. (B is re-read once per row tile,
A once per column tile: 4.8 GB + 2.4 GB.)

`kernels/int8_gemm_v4.cpp` implements the two DPAS forms v3 was not using --
`dpas<8,R,T,CT,BT,AT,...,N,BN,AN>(C,B,A) == C + A x B`, which folds the
accumulate into the instruction, and `RepeatCount=R`, which amortises the B
load over R rows (2.5x fewer instructions per unit of work). CPU-verified
correct at R=1,2,4,8. It is nonetheless **slower**: 28 TOPS at R=8 vs 43 for
v3, because the 8-zmm accumulator operand costs more than the instructions
saved. Registers, again.

The honest conclusion is that reaching `_int_mm`'s 151 TOPS needs a fully
blocked, double-buffered GEMM with ROWS/NT tiling and shared-memory staging --
worth maybe 2x on B traffic, i.e. ~0.9 ms, still 1.8x behind `_int_mm`. So
the fused epilogue (which would only save 0.15 ms) cannot pay for it.
**Keep `torch._int_mm`.**

### Things checked that turned out to be non-issues

* `_int_mm` layout sensitivity: `[K,N]` contiguous vs an `[N,K]` transposed
  view differ by less than run-to-run noise. `_weight_int8pack_mm` exists but
  requires a **float** A, so it is not an int8xint8 path.
* N-blocking the GEMM so the int32 accumulator stays in the 18 MB L2: the
  blocked `_int_mm` costs 0.99 ms vs 0.50 monolithic, versus a 0.13 ms saving
  on the epilogue. Net loss.
* Temporary allocation: `torch.empty` costs 0.004 ms regardless of size. The
  0.146 vs 0.275 ms difference seen for the epilogue between benchmarks is an
  artefact of whether the output buffer happens to be L2-warm, not allocator
  cost. Real models rotate buffers, so **0.27 ms is the honest figure**.

### Current stage table

| stage | time | floor | at floor? |
|---|---|---|---|
| rotate + quantize | 0.41 ms (0.56 via the library) | 0.12 ms DRAM / 0.27 ms register-resident | 40% of the residency floor |
| `torch._int_mm` | 0.50 ms | oneDNN is the best available | yes, use it |
| dequant epilogue | 0.135-0.275 ms | 0.12 ms | 88% (0.135, warm) |
| total kernel time | 1.05-1.19 ms | | |

## Why the first comparison said INT8 loses, and what the fair one says

The early measurement compared the INT8 pipeline against `linear(x, wbf, bias)`
with a **pre-rotated** weight and a **pre-rotated activation**, i.e. it handed
BF16 the activation rotation for free. ConvRot stores the weights rotated, so
both paths have to rotate the activation: BF16 with the rotate-only kernel,
INT8 with rotate+quantize. The old comparison therefore understated the BF16
side by exactly that kernel, and it is the reason the first revision of this
file reported INT8 as 1.9-2.3× slower.

With that kernel counted on both sides (`debug/bench_bf16_fair.py`), the ratio
is 0.90-1.10 across M=512..8192 — parity, with INT8 15-17% ahead at M=1024
(see the headline table).

The rotation is what makes them close: it is 82% of the eager INT8 layer, and
even at its memory floor it costs about what the GEMM difference is worth.
Driving it below that floor would need the 256 in-register floats (load → row
abs-max → quantize) to stop being live at once; every attempt (fp16
butterflies, shared-memory staging, 3-pass, unrolling, wider stores, 2-4 rows
per work-group) measured slower — see the optimization log above.

The honest summary: **INT8 is a memory lever on this card, not a speed lever.**
4× less weight memory and 2× for cached activations, at parity speed, which is
why ComfyUI keeps it behind the `supports_int8_compute()` opt-in instead of
making it the default.

## ESIMD `dpas` findings (resolved)

The earlier blocker — `dpas` returning 16 instead of 32 for all-ones — was a
template-size mistake, not a hardware quirk. `kernels/probe512b.cpp` establishes:

```cpp
dpas<8, 1, int, signed char, signed char,
     dpas_argument_type::s8, dpas_argument_type::s8, 512, 32>(B, A)
```

computes **16 output columns**, each a 32-element int8 dot product. All 32 lanes of
`A` are consumed in order: `A[i]` pairs with `B[ACTIVE_B[i]]` where

```
ACTIVE_B = 0..3, 64..67, 128..131, 192..195, 256..259, 320..323, 384..387, 448..451
```

i.e. only the *even* VNNI groups (group stride 128 bytes, 4 bytes per group).
The `BN=256, AN=32` variant only exposes 8 columns and therefore only 16 active
MAC lanes — that was the source of the "16 instead of 32" result.
`kernels/int8_gemm_v2.cpp` uses the 512-byte layout with weights pre-packed
offline and passes a CPU reference check.