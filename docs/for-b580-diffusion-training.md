# What the B580-diffusion-training project can take from this one

Written 2026-10-21, for `/home/okolenmi/Desktop/B580-diffusion-training`
(`next/` tree). Both projects run on the same machine — i5-8600K, Arc B580
12.2 GB, torch 2.12.1+xpu, oneAPI runtime 2025.3.2 — so the numbers below are
*this* card's, but they were measured on a 42-block Cosmos-style DiT
(MossImage, M≈3974 per forward), not on SDXL at batch 3. **Ratios transfer
better than absolute milliseconds.**

The trainer's own `next/manual/speed.md` already has the launch-level story
right (~22 300 launches/step at ~48 µs, graphs to halve the step). Nothing here
contradicts it. What follows is the layer below it — per-kernel device time,
and three kernel fusions that are bit-exact and already proven on this card.

---

## 1. Directly reusable: `libxpuint8` + `python/xpu_kernels`

`xpu_kernels.int8` is a torch-only binding (no ComfyUI import) over the same
`libxpuint8.so` that runs ComfyUI's INT8 path. Build: `./build.sh`;
check: `python -m xpu_kernels`.  The two Triton kernels from §2 are vendored in
the same package as `xpu_kernels.triton`, so a port does not have to copy them
out of ComfyUI.

```python
import xpu_kernels
w8, ws = xpu_kernels.prepare_weight(w)          # bf16 [N,K] -> int8 [N,K] + fp32 [N]
y = xpu_kernels.linear(x, w8, ws, bias)         # bf16 [..,K] -> bf16 [..,N]
model.proj = xpu_kernels.Int8Linear.from_linear(model.proj)   # drop-in, inference
```

Where it fits their project:

| use | fits? | why |
|---|---|---|
| `step/probe.py` (frozen base, no adapter) | **yes** | forward only, and the probe compares base vs adapter *within the same numerics*, so an INT8 base is self-consistent |
| validation/sample renders inside training | **yes**, with a caveat | the INT8 base is a different function from the bf16 base; a loss curve or an image compared against a bf16 reference is not comparable |
| the training graph itself | **no** | there is no backward. The GEMM is `torch._int_mm` and every kernel is forward-only |
| frozen base + LoRA | **partly** | the forward could be INT8, but `dL/dx` needs `W` in bf16, and with checkpointing `W` is live anyway — so the memory win mostly evaporates |

Measured on this card, so nobody has to re-derive it: native INT8 ConvRot is
**parity with a fair bf16 linear at M≈4096** (0.94–0.99×) and 15–17% faster at
M=1024. Its real win is **4× less weight memory**, not flops. If the goal were
speed in the training graph, this is not the lever.

## 2. Forkable: two Triton kernels that are exactly what a Cosmos DiT needs

`next/families/anima/fingerprint.py` says Anima is a Cosmos transformer with an
LLM adapter and a Wan VAE — the same family as the model this branch optimises.
When its DiT lands it will contain `rms_norm + adaLN modulation`, `rope` inside
attention, and a SwiGLU/situ FFN. Three of those already have fused Triton
kernels here, each verified bit-exact against the eager chain and each with a
one-line escape hatch:

| file | fused | measured here | note for training |
|---|---|---|---|
| `xpu_kernels/triton/adaln.py` | RMSNorm + adaLN scale/shift in one pass over the row | 25 → 16 ms/step (**1.56×**), bit-exact | **differentiable now**: one kernel for `dx` + a two-stage reduction for `dscale`/`dshift`; fwd+bwd **6.3×** vs the eager chain, 1.5e-7 against autograd. Works with every modulation shape a trainer passes, and the modulation gradients are more accurate than the eager chain's |
| `xpu_kernels/triton/rope.py` | RMSNorm + rope in one pass; also reads the INT8 GEMM accumulator directly | rope alone **2.34×** (35.7 → 15.3 ms/forward), bit-exact | **differentiable now**: fused backward, fwd+bwd **14.3×** vs the eager chain (0.62 vs 8.88 ms per tensor), gradient closer to float64 than autograd's |
| `src/xpuint8.cpp` `situ_reg` | situ with `expm1` tanh | −0.41 ms/call, −34.7 ms/step | see §3 — this one is a *finding*, not just code |
| `xpu_kernels/int8/ops.py` | INT8 weight → the backward's `dx = dy @ W` | exact vs the same GEMM in autograd, 0.9% from the unquantized weight (that *is* the INT8 weight), dequant is one-time (~0.4 s for 3.7B) | `dequant_weight` + `linear_dx`, or `Int8Linear.dx(dy)`. `dW` and a quantized-`dy` variant are still open — `docs/todo.md` |

Two design rules in those files worth copying even if the kernels are rewritten:

1. **Self-test once per process, then fall back.** Each module runs its kernel
   against the shipped chain on first use and disables itself on any mismatch.
   A custom kernel in a training loop that silently changes numerics is worse
   than a slow one.
2. **One env flag per kernel** (`XPU_KERNELS_NO_ROPE`, `XPU_KERNELS_NO_ADALN`;
   the legacy `MOSS_NO_*` names are accepted) so a suspect result can be
   bisected without a code change. The gate is shared
   (`triton/_gate.py`, CPU-tested in `tests/test_gate.py`), so a new kernel only
   has to supply a probe, not a verification mechanism.

Caveat: both kernels are written for MossImage's exact dtypes and shapes
(bf16, `dim=3072`, 24 heads × 128, `num_warps=4` because a different warp count
changes the reduction order and the output). For Anima, treat them as a
template and re-derive the constants — do not assume the tuning carries over.

## 3. The `tanh` finding (costs nothing to check, applies broadly)

`sycl::tanh` on this card has a **data-dependent slow path**: on
saturated/denormal inputs it costs 3.5× more. Replacing it with
`expm1(2x) / (expm1(2x) + 2)` is bf16-identical for **every finite bf16 input**
(exhaustively checked over all 65 536 encodings) and has no slow path. In the
FFN activation alone that was 34.7 ms/step, 2.9% of the step.

Any activation in the training path that calls `tanh` on inputs that saturate is
a candidate: GELU-tanh (SDXL's QuickGELU is sigmoid, so no), the VAE, and any
gated activation. The probe that found it is `debug/native/probe_tanh_exact.cpp`
(reference `expm1` form included); the same A/B shape works for any
transcendental (`next/manual/speed.md` has no row for this because it is a
kernel-internal effect, not a launch).

## 4. Card facts that save a measurement cycle

| fact | evidence | consequence |
|---|---|---|
| Attention is a single oneDNN kernel whichever SDPA backend is selected | `docs/xpu_int8_perf_findings.md` §attention | 275 ms/step here (24%); no backend choice to make, and no reason to write one. The mem-efficient backend (the only capturable one) is not a compromise |
| `torch._int_mm` (oneDNN) beats a hand-written ESIMD GEMM | 145–190 TOPS vs 43 TOPS | do not write a GEMM. 459 ms/step here (40%) and near the card's int8 rate |
| A 2026.x `icpx` produces a `.so` that cannot load into this torch | undefined `submit_kernel_direct_with_event_impl` | any compiled extension must be built with oneAPI **2025.3** (`libsycl.so.8`). `build.sh` pins it and fails loudly; copy that check |
| Streaming read ≈ 470–575 GB/s; a copy or read→write mix ≈ 300–430 GB/s **per direction** (600–885 GB/s counting both) | `debug/native/probe_bandwidth.cpp`, `docs/performance.md` | the floor to compare a kernel against. A kernel "faster" than this is a measurement error — the old 788 GB/s "read" figure came from a buffer that fit in the 18 MB L2 |
| `use_deterministic_algorithms(True)` switches on `fill_uninitialized_memory` | 2174 fill kernels, 92 ms/forward here | if reproducibility is ever turned on for a training run, clear `fill_uninitialized_memory` around the step: the fills write memory that is overwritten before it is read, and the output stays byte-identical |
| `total_memory - reserved` counts *other processes'* VRAM as free | 12 216 MB reported vs 3469 MB actual | ComfyUI's OOM cause. Their `next/engine/driver_memory.py` already uses `mem_get_info`, so nothing to fix — just do not reintroduce the proxy in planning |
| the L0 submission thread burns a full core in `sched_yield` | `perf` profile, 88.8% of samples | harmless when launch-bound (the device is idle anyway) and irrelevant once graphed; it is not a bug to chase |
| `sycl::reduce_over_group` used to segfault the host | twice (SIGSEGV/SIGBUS, no exception, then nothing in ~35 000 launches on driver 1.17.39758, both toolchains) | re-tested 2026-10-21 with `debug/native/reproduce_reduce_over_group.cpp`; it is **16 % faster** than the butterfly row max and bit-identical, but still do not ship it -- the failure mode is a killed process, which no gate can catch. Re-test after every driver/compiler bump (`PROBE_REDUCE=1`, `docs/performance.md`) |

## 5. Measurement doctrine that saved this project from itself

Their `manual/speed.md` is a table of levers; this is the methodology that made
those numbers trustworthy, and it transfers as-is:

- **Device time, not wall time.** Host timers here measure enqueue (they made a
  kernel look 3× better than it was). Their `trainer.timing=true` synchronises,
  which is honest but slow; a profiler-attributed per-kernel table for one step
  is the same instrument ComfyUI uses (`COMFY_PROFILE_FWD`).
- **Interleaved paired A/B, never sequential.** Same process, alternating arms;
  anything below the noise floor is not a result.
- **An idle desktop.** Any window activity moves a step by up to +20% here,
  more than any remaining optimisation. Numbers taken while the machine is in
  use are not comparable.
- **Reproducibility is per session.** We proved the same build produces a
  different output hash across restarts (`--deterministic` fixed it). The
  training analogue: loss at step *n* is only comparable within one process,
  one graph set, one kernel compilation cache. Re-baseline when any of those
  change.

## 6. What does not transfer

- **How to consume it:** this is not hypothetical any more. ComfyUI installs
  the package editable (`pip install -e ../xpu-kernels`) and imports
  `xpu_kernels.triton.{rope,adaln}` directly, with its own old copies reduced to
  re-export shims; the switch was verified by a seed-1234 render whose PNG md5
  did not change. Do the same from the training project: one editable install,
  no vendored copies.
- INT8 in the training graph: `dL/dx` **is** available now —
  `dequant_weight(w8, w_scale)` once, then `linear_dx(dy, w)` (or
  `Int8Linear.dx(dy)`). It is exact against autograd of the same GEMM and 0.9 %
  from the unquantized weight, which is the INT8 weight, not the backward. Still
  missing: `dW` (full fine-tuning only) and the faster int8-GEMM variant that
  keeps the weight quantized; `docs/todo.md` has both plus the graph-capture
  checklist.
- The graph-capture and `addcmul_` findings — MossImage-specific.
- Prompt/text encoding: closed here as "fast enough", no result worth porting.
- The ComfyUI memory-accounting patch: their code is already correct.

## 7. If you want one thing ported first

The Anima DiT is the same family as the model these kernels were tuned on, so
the highest-value action is to port `xpu_kernels/triton/rope.py` — it is already
differentiable, so it needs no wrapper work — and `adaln.py` **with** an autograd
wrapper (its backward is `docs/todo.md` item 3). Expected: roughly 1.5–2× on the
norm/rope portion of the forward, and 14× on the rope forward+backward
(0.62 vs 8.88 ms per tensor here), bit-exact forward, and a template for the FFN
fusion later. Everything needed to do it — the kernels, the
self-test gate, the escape hatch, and the verification harness pattern in
`debug/verify_*.py` — is in this repo.

Second, if they ever add a compiled extension of their own: copy
`build.sh`'s toolchain pin, its fail-loudly version check, and the
`_check()`-style device guard pattern (a partially offloaded model must fall
back, not raise `mat2 is on cpu`).
