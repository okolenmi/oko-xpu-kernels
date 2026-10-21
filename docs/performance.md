# Performance

Intel Arc B580, `torch 2.12.1+xpu`. The shipped library is built with
**oneAPI 2025.3** (the pin in `build.sh`); the standalone probes were also run
under 2026.0, and the A/B conclusions were identical under both. Unless stated
otherwise, M=4096, K=N=3072, ConvRot group size 256. All timings are **min over
3 reps of mean over 20 iterations** — a single mean over 20 on this GPU is not
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

## adaLN backward

`debug/probe_adaln_backward.py`, B=1, S=3973, D=3072, bf16, modulation `[1, 1, D]`,
medians:

| | time | |
|---|---|---|
| eager chain forward | 1.67 ms | |
| fused forward | 0.222 ms | **7.5× vs eager** (1.56× vs the shipped Triton kernel, see above) |
| eager forward+backward | 5.14 ms | |
| fused forward+backward | 0.812 ms | **6.3×** |
| — `dx` kernel (+ `rstd`) | 0.25-0.27 ms | |
| — modulation kernel, both sums | 0.35 ms | |
| — reshape/launch glue | ~0.13 ms | |

`dx` is the norm's vjp, `rstd * (dv - (v . dv)/D * v)`, and it is per row, so it
is one kernel. The modulation gradients are sums *across* rows:

    dscale = sum(dy * x * rstd)    dshift = sum(dy)

over the rows each modulation vector was broadcast across. A per-row kernel
cannot reduce across rows, and atomics would cost more than the writes and make
the result depend on execution order, so it is two kernels: one program per
(group, run of 16 rows, column block) writes a partial, and a tiny second kernel
adds the partials of each group. Deterministic, no atomics.

The `dx` kernel hands back `rstd` (one fp32 per row -- 16 KB for the whole model)
rather than `v`, which is what makes the reduction one pass: it reads `x` and `dy`
once, and no `v` tensor is ever materialized. That is also why the gradient got
*more* accurate than the torch version it replaced:

| vs the same gradient in float64 | before (torch reduction) | now |
|---|---|---|
| `dx` | 1.7e-05 | 1.7e-05 |
| `dscale` | 1.7e-03 | **3.1e-07** |
| `dshift` | 1.6e-03 | **5.8e-08** |

The torch version multiplied two bf16 tensors in fp32 into a 48.8 MB temporary,
so it inherited bf16's storage error and then summed; the kernel keeps `dy * x *
rstd` in fp32 registers and never rounds the product. The device gate probes all
three outputs (in fp32, against `torch.autograd` of the eager chain) including a
`[B, 1, D]` modulation, which is the case where the group is not the whole
tensor -- worst relative error 1.5e-7 over 1.59 M values.

Chunk size was swept, 8 to 128 rows per program: 0.35-0.89 ms, best at 16 with a
128-column block (the shipped `MOD_CHUNK`/`MOD_BLOCK`), and all sizes agree with
float64 to 2-4e-7, so the constant is a tuning knob, not a correctness one.

---

## Graph capture: verified, and it is a host-time win

`debug/verify_graph_capture.py` captures two stages, replays them on new data,
and compares against the eager path:

* the fused rope forward **and** its backward for q and k, in one graph -- all
  four tensors (q/k outputs, q/k gradients) bit-identical on replay;
* the adaLN forward and backward, including both modulation gradients -- all
  four tensors bit-identical. That one matters because its backward allocates a
  6 MB partials buffer *inside* the captured region (the two-stage reduction),
  which is exactly the allocation pattern the graph pool has to absorb.

Two things this settles:

- **No workspace plumbing is needed for capture.** Allocations made inside a
  capture are carved from the graph's private pool and replayed at the same
  addresses, so `torch.empty` inside the kernel launchers is not a problem. The
  requirement is that nothing inside the captured region *waits on the device*
  or moves data to the host -- and nothing does.
- **The first call must be outside the capture.** The gate probes the kernel
  against the reference on first use, which synchronises and reads values back.
  `status()` (or one warmup iteration) must run before capture. That is now a
  rule in `docs/adding-a-kernel.md`.

Device time is unchanged by the graph (four kernels either way; at the
production shape 1.15 ms eager vs 1.15 ms of graph work plus the static-input
copies). The win is host time, which is what a launch-bound step pays:

| at S=256, where the host is the limit | rope | adaLN |
|---|---|---|
| eager: autograd engine + the torch chain | 648-1282 µs | 632 µs |
| graph replay | 40-82 µs | 40 µs |

**8-16x less host time per stage.** The ranges are run-to-run variance on an
interactive desktop (the device time in those runs was 0.33-0.68 ms, so the
measurement is not device-limited); the eager number is the one that grows with
the number of kernels, and it is the one a launch-bound step pays. Across 42
blocks that is ~25-50 ms/step of host budget back, which matters when the step
is ~22 300 launches at ~48 µs each.

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

`prepare_weight` costs 3.0 ms for the same matrix (0.172 ns/param), so
quantizing a 3.7 B model is ~0.4 s each way. That is a one-time startup cost
*if* the base is frozen — but it is a **per-step** cost for a trainable int8
weight, and there the C kernel does the same work 16.7x faster (38 ms for
3.7 B). The two are not the same function, so that is a decision, not a swap
(`docs/todo.md` item 1).

### The quantized-`dy` variant: a memory option, not a speed one

`linear_dx_int8` keeps the weight in int8 and quantizes `dy` instead, which is
exact in the weight: `s[n]` is a row scale on `W`, hence a *column* scale on `dy`,
so `dx = (dy * s) @ W8` and then the rotation is undone. `_int_mm` beats the
bf16 GEMM, as it should. But the path as a whole does not:

| `debug/probe_linear_dx_variants.py`, M=4096, N=8192, K=3072, interleaved | time | rel vs unquantized | cosine |
|---|---|---|---|
| bf16 dequant (`linear_dx`) | **2.08 ms** | 0.93 % | 0.999957 |
| int8 GEMM (`linear_dx_int8`) | 2.75 ms | 1.35 % | 0.999910 |
| — quantize `dy` (fused Triton kernel) | 0.27 ms | | |
| — `_int_mm` | 1.05 ms | | |
| — dequant epilogue | 0.26 ms | | |
| — final rotation | 0.55 ms | | |

The GEMM is 1.85x faster (not 3-4x: at M=4096 this GEMM is bandwidth-bound on the
weight, the same story as the headline INT8-vs-BF16 result). The quantizer used
to be a torch chain and cost 4.5 ms; it is now a fused Triton kernel at 0.27 ms
(**16x faster, and 9 kernel launches become 1**), which is what moved the path
from 6.63 ms -- 3.4x the bf16 path -- to 2.75 ms, or 1.33x.

So the honest framing: **keep the weight int8 for the 2x memory saving, not for
speed**, and know that it costs ~30 % more time and ~45 % more gradient error
(0.0135 vs 0.0093 relative; both cosines are 0.9999+). The only way this becomes
a speed win is to not rotate back -- i.e. keep the residual stream in the rotated
basis, which is a whole-model decision, not a kernel one.

### Why the fused quantizer is not bit-exact, and what it costs to be

The fused quantizer (`xpu_kernels.triton.quantize`) is the one kernel in this
package that is *not* bit-exact against its reference, and the reason is worth
recording because it is a property of the hardware, not of the kernel.

The reference is a torch chain that divides `x` by the row scale in fp32. On
this backend, torch's two division forms are **not** the same function:

| form | correctly rounded? | measured |
|---|---|---|
| `x / s` (tensor ÷ tensor) | yes | matches an fp64 division 1 048 576/1 048 576 |
| `x / s` (tensor ÷ python scalar) | no -- a reciprocal multiply | matches an fp64 division 907 674/1 048 576 |

and Triton's own options are different again: `libdevice.div_rn` is correctly
rounded but costs **6.3 ms** for a 3974x8192 tensor (a software division, 15
GB/s), against 0.35 ms for the fast form. So a *bit-exact* fused quantizer
would be **slower than the torch chain it replaces** (6.3 ms vs 4.6 ms), and
the fusion would buy nothing.

The fast form is 16x faster and moves 2-3e-4 of the quantized values by one
unit: **9.1e-4 relative on the gradient**, against the int8 path's own 1.35 %
error -- fifteen times smaller. The row *scale* is exact (it is a division by a
constant, which the fast path does correctly), so the quantization's accuracy
is unchanged: 0.0135 relative against the unquantized gradient, before and
after.

The gate therefore compares with a relative budget (2e-3, the measured
deviation is 3.3e-4) rather than bitwise -- the same mechanism the backward
kernels use -- and the formula itself is pinned in float64 by
`tests/test_quantize.py`. A wrong formula (a missing clamp, the wrong scale, a
rotation that should not be there) is off by ~1e-1 and cannot hide inside that
budget. `XPU_KERNELS_NO_QUANTIZE=1` switches the kernel off, and the chain
returns.

The gate caught the whole story by itself: it first reported "73 of 721 984
values differ" on the very first run of the kernel, which is what led to the
division measurements above.

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

## The FFN epilogue's row max: 16 % of the kernel, not taken

The situ/quantize epilogue has to know the row max before it can store a single
byte, and the row spans a whole work-group (`K/256` sub-groups x 16 lanes).
Three ways to get it, all in `debug/native/probe_ffn_situ.cpp`
(M=3973, K=8192, 292.9 MB of traffic):

| row-max form | time | notes |
|---|---|---|
| write sub-group maxes to local memory, barrier, serial loop (production) | 0.925 ms | |
| ... the same kernel with the row max removed entirely (mode 2, wrong result) | 0.720 ms | the reduction costs **0.223 ms = 24 %** |
| `sycl::reduce_over_group(it.get_group(), vm, sycl::maximum<>())` (mode 15) | 0.774 ms | 0.052 ms above the no-row-max floor |

Interleaved A/B (mode 0 and mode 15 alternating inside the same second, 15
pairs x 20 launches, best-of-nothing): **mode15 / mode0 = 0.836**, range
0.831-0.845, identical under the 2025.3 toolchain (0.833, 0.830-0.840). The
int8 output is **bit-identical**, 0 of 32 546 816 bytes differ -- `fmax` over
finite values does not care about order, and the row max is exactly the same
value, so the resulting scale is too.

**Not shipped, and the reason is not the 0.6 %.** 42 calls per step x 0.925 ms
is ~39 ms/step, so the win is ~6.4 ms of ~1050 (0.6 %), below a step-level
slope's noise floor. Against that: this construct used to kill the process
(`SIGSEGV`/`SIGBUS`, no SYCL exception, twice, which is why it was removed from
the probe). A gate cannot catch that -- the failure is the process dying, not a
wrong number. So the production kernels keep the butterfly + local-memory form.

It no longer crashes (driver `1.17.39758`, both toolchains, ~35 000
reductions of real work-group widths), which is a statement about this driver
and this machine, **not a fix claim** -- no changelog entry was found and the
crash was never filed upstream before the report in
`docs/report-reduce-over-group.md`, which was sent to the vendor. The probe
keeps the variant behind `PROBE_REDUCE=1` with a bit-exactness check against
mode 0, so the next driver bump can re-test it in one command:

    PROBE_REDUCE=1 ./debug/native/bin/probe_ffn_situ 3973 8192 device

The activation quantizer (`kernels/xpuint8.cpp`, second `lmax` block) has the
same row max and would presumably behave the same; it was not measured, so do
not add the ~2 ms it might be worth to the 6.4 above.

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

**Decision: keep `torch._int_mm`.** 151 TOPS, fully blocked and
double-buffered. The reasoning is in `docs/architecture.md`; the two rejected
DPAS forms are in `docs/experiments.md`.

---

## Things that turned out not to be problems

Recorded because each looked like headroom. The full list, with the probes,
is in `docs/experiments.md`; the two that are easiest to mistake for a
result:

* **idle lanes at K=3072** (4 of 16 busy) cost 25 % — no: K=1024, with the
  same 4/16 busy, is the *fastest* per element. It is a cache-size effect.
* **several rows per work-group** — a 1.6-3x *loss*.

All three SYCL runtimes present on this machine (level_zero, NEO OpenCL,
rusticl/Mesa) give identical performance, so JIT choice is not a factor.

---

## The headline, at other shapes

The headline number — INT8 at parity with BF16 — was measured at K=N=3072.
**SDXL has no such shape**, and at SDXL's own shapes the ratio is not parity:
2.5x faster on the FFN's down-projection, 1.5x slower on its up-projection
(the int32 accumulator), and host-bound below M≈1024. That measurement, and
what it means for a model, is in `docs/sdxl.md`
(`debug/probe_sdxl_shapes.py`).

---

## The weight dequantizer

`int8.dequant_weight` -- the backward's weight-only path -- was a torch chain:
`w8.to(float32) * scale` (a full fp32 copy of the weight), then
`.to(bfloat16)` (another copy), then the rotation (a third). Fifteen bytes of
traffic per parameter, and for the 2.17 B quantized parameters of an SDXL base
that is **215 ms per call** -- which is the number the training project says
decides whether an INT8 base is trainable at all, because it rebuilds the
dense weight for every layer of every backward.

The fused kernel reads the int8 and writes the bf16, undoing the rotation in
the same pass: **3 bytes per parameter**, and the arithmetic is the chain's,
in the chain's order (the fp32 multiply, the bf16 rounding, *then* the
rotation -- the rounding has to happen before the butterfly, which is what
makes it bit-identical).

| 100 M parameters, B580 | time | vs the chain |
|---|---|---|
| torch chain | 9.92 ms | -- |
| **fused kernel** | **2.93 ms** | **3.4x**, bit-identical |

For the training project's 2.17 B parameters: **215 ms -> 63 ms** per backward.
That is still 4 % of a 1.5 s step, and the int8 backward (`mode="int8"`) needs
no dequant at all -- but the *exact* backward is now affordable, where before
it was not. The fused kernel runs at 103 GB/s against a 14 ms bandwidth floor,
so there is headroom left in it (the int8 unpack is a scalar loop); the shape
of that work is the same as the rotate's, and the rotate's is closed.

## The NF4 dequantizer

`xpu_kernels.triton.nf4.dequant_nf4` -- the frozen-base storage format
(QLoRA's NF4: 16 normal quantiles, a per-block absmax, two 4-bit indices per
byte) dequantized in one kernel instead of a torch chain.

The chain is the obvious one -- unpack, gather the code, multiply by the
block's scale -- and it is expensive, because it materializes the unpacked
indices, casts them to int64 for the gather, and holds the gathered codes and
the scaled product at the same time.

| 100 M parameters, B580 | time | traffic | scratch |
|---|---|---|---|
| torch chain | 20.75 ms | ~40 bytes/parameter (~4 GB) | ~16 bytes/parameter |
| **fused kernel** | **0.66 ms** (31.5x) | 2.6 bytes/parameter (256 MB) | none |

389 GB/s for the fused version -- a gather-heavy kernel, so not at the card's
468 GB/s read floor, but four times the chain's effective rate with sixteen
times less traffic.

**Bit-exact**, which the int8 row-quantizer could not be: the arithmetic is a
table lookup, a multiply and a cast, with no division and no rounding mode to
get wrong. The gate compares the kernel against the chain bit for bit, in fp32
and at the production bf16 (`tests/test_nf4.py` pins the format itself: the
codebook's definition, the nibble order, the block layout -- so a weight
stored by one implementation is readable by the other).

The scratch matters as much as the time: the chain's peak can exceed the
bf16 weight the format was meant to replace, and it allocates inside a forward
pass, which inside a device-graph capture means the allocation is replayed on
every step. The fused kernel's only allocation is its output.

The trainer's side of this is in `docs/nf4-storage-vision.md` in the
*trainer's* tree (`B580-diffusion-training/docs/`), not here.

## The bf16 path: what is left, measured

The question "can bf16 be made faster?" has an answer for each candidate, and
most of them are closed:

| candidate | verdict |
|---|---|
| fuse the activation into the GEMM (oneDNN post-op) | **closed**: `mkldnn::_linear_pointwise` exists on this backend but accepts only `relu` -- `gelu_tanh` and `gelu_erf` are refused by name ("not supported for conv/linear post unary fusion") -- and the `relu` path does not actually fuse: 2.234 ms against 2.252 ms for the unfused `F.linear` alone, i.e. noise. The GELU pass itself costs +0.366 ms at M=4096, K=3072, N=8192 (a read and a write of 67 MB), so there was something to win and oneDNN does not win it here. |
| the norms | **done**: rms_norm+rope and rms_norm+adaLN are fused Triton kernels, 1.5-2x on those stages, and their backwards are fused too (14x on rope fwd+bwd). |
| the activation's transcendental | **done**: `sycl::tanh` has a data-dependent slow path on this card, and the `expm1` form is bf16-identical for every finite bf16 input. |
| the GEMMs | **closed**: oneDNN is at the card's int8 rate; a hand-written ESIMD GEMM was 3-4x slower. |
| the attention | **closed**: every XPU SDPA backend is the same oneDNN kernel. |
| the rest | at the measured memory floor (read 468 GB/s, write 397 GB/s on a 400 MB buffer). |

The one shape still open in the bf16 path is the **residual add** of a
pre-norm block: folding it into the norm's epilogue would save one launch and
one read-write pass per add (~84 launches and ~6 ms of device time per step
in a 42-block model, in a step that is host-bound). It is training-safe --
the add's backward is the identity -- but the payoff is small next to the
capture, which removes the launches altogether. It is item 4 in
`docs/todo.md`.

## What is left: the rotate's floor (closed)

**The rotate's structural floor is not reachable, and the last open shape
cannot be built at the shapes that matter.** This took a measurement to
establish (`debug/native/probe_cross_lane.cpp`), and the answer is not the one
the notes above predicted.

The constraint is not really "16 registers live": it is that **the row's
abs-max is over the row**, and the row's values only exist after the
rotation. So every element of a row waits for every element of that row to be
transformed -- the load, the transform, the scale and the store are serialised
per row, whatever the register pressure is. The 3.1x in the copy-loop
experiment is that serialisation, not the spilling.

The shape that was left to try is a group split across 4 lanes with a single
cross-lane axis (the axis-3 butterfly moves across the lanes, the axis-2 one
stays inside). The measurements:

| | measured |
|---|---|
| one cross-lane exchange, 512 bits, vector form | **84 ns** (a scalar `select_from_group` is 17.3 ns, so the 512-bit move costs about five of them, not sixteen) |
| exchanges the split needs, per lane per row | 12 at K=1024, **36 at K=3072**, 96 at K=8192 |
| what that costs | ~1 us at K=1024, ~3 us at K=3072 -- **negligible** against a prize of 250 us |

So the exchange is *not* the obstacle, and the "16 `select_from_group` calls
per exchange" figure overstates it by 3x. The obstacle is the **reduction
domain**:

| K | groups per row | lane-slots if each group is split 4 ways | sub-groups of 16 lanes |
|---|---|---|---|
| 1024 | 4 | 16 | **1** -- feasible |
| 3072 (the model's) | 12 | 48 | 3 -- the row's abs-max would have to cross them |
| 8192 (the FFN's) | 32 | 128 | 8 -- likewise |

A row's scale is reduced inside one sub-group today. Split the groups and the
row's scale would have to cross sub-groups, which means shared memory and a
barrier (measured at 1.33 ms for the staging variant) or a second pass
(0.75 ms) -- both far worse than the 0.402 ms the rotate costs now. And if the
16-lane sub-group is kept and each lane takes three quarter-groups instead,
the pressure only falls from 16 registers to 12, which is not what the
interleaving needed.

At K=1024 the split *is* buildable (one sub-group, 12 exchanges, ~1 us). It
would be worth a try there -- but K=1024 is not a shape this model has, and
the model is what the library is for.

**The only route to the 0.27 ms floor left is to change the format**: a scale
per block instead of per row would let each block's stores proceed
independently, but it changes the numbers (the scale is part of the
quantization), so it is a design decision with its own gate, not an
optimisation of this one. The practical consequence for the headline stands:
**INT8 stays at parity with BF16, not 35 % faster**, and the gap is a
property of the format rather than a missing kernel.

---

## Timing discipline

The GPU drifts 10-30% over a benchmarking session with thermals, which is
**larger than most of the effects being measured**. Everything reported as a
ratio is therefore taken from *interleaved* rounds in one process, never from
separate runs — `debug/probe_load_ab.py` is the template, and
`debug/probe_sdxl_shapes.py` is the same idea applied to SDXL's shapes. A
single mean over 20 iterations is not trustworthy on this device: the same
binary has measured 0.52 and 0.73 ms for one call. The rules are written out
in `debug/README.md`.
