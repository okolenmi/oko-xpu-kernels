# Performance

Intel Arc B580, `torch 2.12.1+xpu`, DPC++ 2026.0. Unless stated otherwise,
M=4096, K=N=3072, ConvRot group size 256. All timings are **min over 3 reps of
mean over 20 iterations** — a single mean over 20 on this GPU is not
trustworthy (the same binary has measured 0.52 and 0.63 ms for one call).

Reproduce with the probes in `debug/` (see `debug/README.md`).

---

## The headline: does INT8 beat BF16?

`debug/bench_bf16_fair.py`. **Both paths pay for the activation rotation**,
because ConvRot stores the weights pre-rotated so the BF16 path needs it too.

Across repeated runs the **ratios are stable to ~1-2%** while absolute
milliseconds drift 10-15% with GPU thermals, so the ratio is the meaningful
quantity:

| M | ratio INT8/BF16 (range over 4 runs) | verdict |
|---|---|---|
| 512 | 1.00 - 1.10 | INT8 0-10% slower |
| 1024 | 0.83 - 0.85 | **INT8 15-17% faster** |
| 2048 | 0.98 - 1.01 | parity |
| 4096 | 0.94 - 0.99 | parity to 6% faster |
| 8192 | 0.90 - 0.98 | parity to 10% faster |

A representative single run (cold GPU), for the shape of the numbers:

```
     M  rotate+quant  rotate bf16   _int_mm  epilogue  INT8 tot  BF16 tot  int8/bf16
   512        0.093m       0.064m    0.075m    0.025m    0.193m    0.180m      1.07x
  1024        0.123m       0.126m    0.135m    0.041m    0.300m    0.351m      0.85x
  2048        0.286m       0.300m    0.246m    0.142m    0.674m    0.680m      0.99x
  4096        0.520m       0.550m    0.479m    0.270m    1.269m    1.334m      0.95x
  8192        0.948m       1.048m    0.940m    0.523m    2.411m    2.550m      0.95x
```

**INT8 is never meaningfully slower and is clearly faster at M=1024.** It wins
by 15-17% there, is at parity to 10% faster for M>=2048, and is 0-10% slower at
M=512 (where neither path has enough work to amortise the rotation). With 4x
less weight memory, that is a usable trade — and a different conclusion from
what the unfair comparison gave.

The spread above is thermal, not noise between runs: the warm-GPU runs cluster
closer to 1.0 and the cold-GPU runs spread wider. Read the range, not a point.

**INT8 is at parity to 15% faster for M>=1024 and ~7% slower at M=512**, with
4x less weight memory. That is a usable trade, and it is a different conclusion
from what an unfair comparison gives.

> An earlier version of `benchmarks/bench_pipeline.py` compared INT8 against
> `F.linear(x, wbf, bias)` — a BF16 path with **no activation rotation** — and
> reported INT8 as 1.9-2.3x slower. That number is wrong; it omits work the
> INT8 path has to do. Use `debug/bench_bf16_fair.py`.

Accuracy, versus a float64 reference:

| path | relative L2 error |
|---|---|
| this library, full INT8 pipeline | 0.008791 |
| comfy-kitchen eager INT8 | 0.009351 |
| this library, BF16 rotation only | 3.1e-09 (exact; both round to bf16) |

comfy-kitchen's eager backend rotates activations with a **bfloat16 bmm**, which
loses enough precision that the row abs-max — and therefore the row scale — is
wrong by up to **40%**. It is not a valid reference. See
`debug/verify_rotate_scales.py`.

---

## Device ceilings

`debug/native/probe_bandwidth.cpp`. These supersede an earlier claim of
"788 GB/s streaming" in the README, which came from a buffer that fit in the
18 MB L2 and was re-read in a loop. It was not a DRAM number.

| buffer | read | write | copy (r+w) |
|---|---|---|---|
| 12.6 MB | 291 GB/s | 577 GB/s | 776 GB/s |
| 25.2 MB | 310 GB/s | 329 GB/s | 606 GB/s |
| 50.3 MB | 322 GB/s | 346 GB/s | 640 GB/s |
| 400 MB | 468 GB/s | 397 GB/s | 885 GB/s |

L2 is **18 MB**. That number matters more than it looks: it is why per-element
throughput *rises* as K falls, and it is why N-blocking the GEMM looked
attractive.

Pipeline DRAM floors: rotate 37.8 MB → 0.12 ms; epilogue 75 MB → 0.12 ms.

---

## Stage breakdown

`debug/bench_stages.py` at M=4096, K=N=3072:

| stage | time | floor | at floor? |
|---|---|---|---|
| rotate + quantize | 0.52 ms | 0.27 ms register-resident | 52% |
| `torch._int_mm` | 0.50 ms | oneDNN is the best available | — |
| dequant epilogue | 0.145-0.275 ms | 0.12 ms | 88% (warm) |
| **kernel total** | **1.17-1.30 ms** | | |
| apparent overhead | 0.21 ms | | see below |

**The apparent overhead is mostly buffer cache-coldness, not CPU dispatch.**
Measured piece by piece at M=4096: rotate+quantize is 0.643 ms with a
preallocated output buffer and 0.838 ms when the buffer is freshly allocated
each call; `_int_mm` 0.606 ms; epilogue 0.318 ms. Those sum to the full fused
linear (~1.67 ms), so there is no 15-20% CPU tax hiding anywhere. `torch.empty`
costs 0.006 ms and `_check()` 0.005 ms. What the difference actually is: a
reused output buffer stays in the 18 MB L2, a rotating one does not.

The epilogue range is a buffer-warmth artefact: 0.135-0.145 ms writing the
same output buffer repeatedly, ~0.27 ms writing a fresh one. Real models rotate
buffers, so **0.27 ms is the honest number**.

---

## rope and adaLN: the Triton kernels

Upstream's kernel is one program per (batch, head, seq): it loads the head row to
compute the sum of squares, loads it again to rotate, and re-loads the fp32
`freqs_cis` table once per head. At S=3973, H=24, D=128 that is 95 352 programs
per tensor, each touching 256 B of a 6 KB-strided row.

This package's version keeps the arithmetic identical and changes the schedule:
one program per (batch, seq), heads looped inside. Measured with
`debug/probe_rope_variants.py` and `debug/probe_adaln_variants.py`, bit-exact
against the shipped kernels (0/12.2M and 0/12.2M values differ):

| kernel | upstream | fused | ratio |
|---|---|---|---|
| rope (q + k) | 0.587 ms | 0.251 ms | 2.34× |
| adaLN (per call) | 0.206 ms | 0.132 ms | 1.56× |

In the model these are 44 ms and 16 ms per forward (42 blocks), from a marginal
step budget of ~1094 ms.

## rope backward (the differentiable wrapper)

`debug/probe_rope_backward.py`, one tensor of the production shape (S=3973,
H=24, D=128, bf16 = 24.4 MB), device-time medians, interleaved:

| | time | traffic | effective |
|---|---|---|---|
| eager chain forward | 3.4-3.6 ms | | |
| fused forward | 0.25-0.44 ms | 73 MB | 170-290 GB/s |
| fused backward | 0.22-0.34 ms | 73 MB | 215-340 GB/s |
| eager forward + backward | 8.9-9.3 ms | | |
| fused forward + backward | 0.62-0.64 ms | | **14.3-14.6×** |

Ranges, not single numbers: these were taken with the desktop in use, and the
absolute times move by ~1.7x between runs on this card (the ratio is the stable
quantity — that is the whole reason the repo measures things as pairs). The
bottom of each range is an idle desktop.

The first version of the backward was two kernels and a scratch buffer (a
per-row `dot` pass, then a write pass): 1.13 ms. Both passes are unnecessary —
`dot` is per *head row*, and one program already holds all the heads of one
(batch, seq), so the whole backward fits in one program with nothing spilled.
The 2-D head tile (instead of a loop over 24 heads) and the single pass brought
it to ~0.22 ms, which is at the bandwidth floor for this card: ~340 GB/s for a
read-read-write stream of this size, the same rate the forward reaches at 290
GB/s with its extra `pe` loads. There is nothing left to win in this kernel.

**Accuracy.** Against a float64 evaluation of the same gradient (same probe):

| | relative L2 error |
|---|---|
| `torch.autograd` of the bf16 eager chain | 2.34e-3 |
| fused backward (bf16 output) | 1.66e-3 |

The fused gradient is *closer* to the truth, because PyTorch rounds the gradient
to bf16 at the forward's `.to(bfloat16)` boundary and the fused kernel keeps
fp32 throughout. The two differ from each other by 2.8e-3, which is why the
device gate probes the arithmetic in fp32 (worst relative error 7e-8) and the
formula is pinned separately by `tests/test_rope_backward.py` in float64 against
`torch.autograd.gradcheck`.

---

## INT8 backward: `dL/dx = dy @ W`

There is no kernel here, and that is the finding. `linear_dx` is one bf16 GEMM,
so the only new costs are the one-time weight dequant and the fact that the
weight geometry no longer matches the forward's.

The weight scale is on the contraction index for the backward (see
`docs/todo.md`), so `W` is materialized (`dequant_weight`: dequantize, then undo
the ConvRot rotation — the rotation is its own inverse) and the GEMM is
`torch.matmul`. `debug/verify_linear_dx.py`, N=8192, K=3072, M=4096:

| | |
|---|---|
| `linear_dx(dy, W)` vs autograd of `F.linear`, same `W` | **0** — the same oneDNN GEMM |
| `dequant_weight` vs the bf16 weight | 0.94 % (int8 0.90 %, bf16 cast 0.17 %) |
| `dx` with the quantized weight vs the unquantized model | 0.97 % |
| dequant, 25.2 M params | 2.6 ms (~15 GB/s effective: cast, scale, cast, rotate) |

0.9 % for per-row symmetric int8 is what the arithmetic gives on Gaussian data
(`absmax ~ 3.5 sigma`, so `scale/2/sqrt(3) / sigma ~ 0.008`), and the same 0.9 %
is in the forward — 1.27 % there for this synthetic matrix. The gradient error is
not an extra layer of noise on top of the forward's, it *is* the forward's.

`prepare_weight` costs 3.0 ms for the same matrix, so quantizing a 3.7 B model is
~0.4 s each way. That is a one-time startup cost, not a step cost; it is the
argument for the int8-GEMM variant (weight stays int8, no dequant, 2x less
memory) if someone needs the weight back in int8 — at the price of quantizing
`dy`, which is the thing to A/B.

---

## rotate + quantize: where the time is

`debug/native/probe_rotate_ablation.cpp`. Each mode removes exactly one piece.

| variant | time | delta |
|---|---|---|
| full | 0.407 ms | |
| − transform | 0.273 | transform = 0.134 ms (33%) |
| − int8 store | 0.276 | store = 0.131 ms (32%) |
| − abs-max + sub-group reduce | 0.381 | 0.025 ms (6%) |
| − sub-group reduce only | 0.407 | **0.000 ms** |

> The first version of this ablation wrote the output **twice**, which
> inflated the store attribution to 0.215 ms and sent the optimisation effort
> after the wrong thing. The probe in the repo is the corrected version and its
> header says so.

The load+store skeleton is 0.273 ms for 37.8 MB = 138 GB/s, **4x below the
streaming floor**. That gap is register residency (see
`docs/architecture.md`); the row-scale dependency is what forces it.

Progression: **0.765 → 0.52 ms**, from
- explicit accumulation in `inreg()` instead of one expression (one live
  product temporary instead of four),
- loading and transforming one axis-2 group at a time so load latency overlaps
  the butterfly (0.63 → 0.50 in isolation),
- compile-time `W` indices so the array stays in registers.

---

## The hand-written GEMM is L2-bound, not DPAS-bound

`debug/probe_gemm_shapes.py`. v3 reaches 43 TOPS = 12.5% of the 344 TOPS DPAS
peak. The shape sweep says why:

| M x N x K | ms | TOPS | B+A traffic | effective |
|---|---|---|---|---|
| 4096x3072x3072 | 1.78 | 43.4 | 7.25 GB | 4.07 TB/s |
| 4096x256x3072 | 0.156 | 41.2 | 0.60 GB | 3.86 TB/s |
| 512x3072x3072 | 0.237 | 40.9 | 0.91 GB | 3.83 TB/s |
| 8192x3072x3072 | 3.50 | 44.2 | 14.50 GB | 4.14 TB/s |
| 4096x3072x12288 | 13.32 | 23.2 | 28.99 GB | 2.18 TB/s |

TOPS is flat for every cacheable shape and traffic sits at ~4 TB/s in all of
them. It saturates L2. It only falls off when the working set leaves L2.

`kernels/int8_gemm_v4.cpp` uses the two DPAS forms v3 ignores (accumulator
operand, `RepeatCount=R`), which cut instructions per unit of work by 2.5x. It
is **slower**: 28 TOPS at R=8. See `docs/architecture.md`.

**Decision: keep `torch._int_mm`.** 151 TOPS, fully blocked and double-buffered.

---

## Things that turned out not to be problems

Recorded because each looked like headroom.

| hypothesis | measurement |
|---|---|
| idle lanes at K=3072 (4/16 busy) cost 25% | no — K=1024 (4/16 busy) is the *fastest* per element; it's a cache-size effect |
| several rows per work-group | 1.6-3x **loss** |
| N-blocking the GEMM to keep `acc` in L2 | blocked `_int_mm` 0.99 ms vs 0.50, vs 0.13 saved |
| 64-byte stores | no measurable difference |
| 64-byte loads | no measurable difference -- paired A/B, 0.995x, bit-identical (`debug/probe_load_ab.py`) |
| `vec<int32_t>` pack instead of scalar float packs | no measurable difference |
| rounding-bit trick for float→bf16 | **2x slower** (0.24 vs 0.135) — the scalar loop is already vectorised |
| AOT `-fsycl-targets=spir64_gen12` | unavailable: this compiler accepts only generic `spir64` |
| `_int_mm` layout / packing sensitivity | within noise; `_weight_int8pack_mm` needs a **float** A, so it is not an int8xint8 path |
| `torch.empty` in the hot path | 0.004 ms regardless of size |
| `vec<float,16>` silently 256-bit | no — ~10 T float-ops/s, near peak |

All three SYCL runtimes present on this machine (level_zero, NEO OpenCL,
rusticl/Mesa) give identical performance, so JIT choice is not a factor.

---

## What is left

One bottleneck, and it is a hardware-shape constraint rather than a missing
optimisation:

> The rotate must hold 256 floats (16 zmm) live across load → row abs-max →
> quantize. That costs 3.1x versus an interleaved access pattern, and it is why
> the rotate runs at 138 GB/s instead of 600.

Closing it needs one group split across 4 lanes with a single cross-lane axis.
A 512-bit cross-lane exchange needs 16 `select_from_group` calls, so the same
verdict as the shared-memory version is expected — but it is the only shape
that has not actually been implemented.

**If the rotate reached its 0.27 ms structural floor, INT8 would be ~35%
faster than BF16 instead of at parity.**
---

## Timing discipline

The GPU drifts 10-30% over a benchmarking session with thermals, which is
**larger than most of the effects being measured**. Two consequences:

1. Everything reported as a ratio (INT8/BF16, old/new) is taken from
   *interleaved* rounds in one process, not from separate runs.
   `debug/probe_load_ab.py` is the template: load two builds, alternate
   between them 7 times, compare medians.

2. A separate-run comparison will invent effects that are not there. An early
   64-byte-load A/B was run that way and appeared to show a 10% regression;
   the paired comparison shows 0.995x. The earlier conclusion was wrong and
   was retracted.

A single mean over 20 iterations is not trustworthy on this device — the same
binary has measured 0.52 and 0.73 ms for one call.
