# What the B580-diffusion-training project can take from this one

Written for `/home/okolenmi/Desktop/B580-diffusion-training`. Both projects
run on the same machine — i5-8600K, Arc B580 12.2 GB, torch 2.12.1+xpu,
oneAPI runtime 2025.3.2 — so the numbers below are *this* card's, but they
were measured on a 42-block Cosmos-style DiT (MossImage, M≈3974 per forward),
not on SDXL at batch 3. **Ratios transfer better than absolute
milliseconds.**

The trainer's own `next/manual/speed.md` already has the launch-level story
right (~22 300 launches/step at ~48 µs, graphs to halve the step). What
follows is the layer below it.

## 1. Directly reusable: the package

`xpu_kernels.int8` is a torch-only binding (no ComfyUI import) over the same
`libxpuint8.so` that runs ComfyUI's INT8 path. The two Triton kernels live in
the same package as `xpu_kernels.triton`, so a port does not have to copy them
out of ComfyUI.

```python
import xpu_kernels
w8, ws = xpu_kernels.prepare_weight(w)          # bf16 [N,K] -> int8 [N,K] + fp32 [N]
y = xpu_kernels.linear(x, w8, ws, bias)         # bf16 [..,K] -> bf16 [..,N]
y = xpu_kernels.ffn_situ(x, up8, up_s, down8, down_s)   # down(situ(up(x))), 2 GEMMs
reason = xpu_kernels.unsupported_reason(x, w8, ws)       # None, or why this shape cannot
```

What it is for, in one line: **4× less weight memory at parity speed.** Native
INT8 ConvRot is at parity with a fair bf16 linear at M≈4096 (0.94-0.99×) and
15-17 % faster at M=1024. If the goal were speed in the training graph, this
is not the lever.

The backward is available now: `dequant_weight(w8, w_scale)` once, then
`linear_dx(dy, w)` (or `Int8Linear.dx(dy)`). It is exact against autograd of
the same GEMM and 0.9 % from the unquantized weight — that 0.9 % *is* the INT8
weight, not the backward. The weight gradient is `linear_dw(dy, x) = dy^T @ x`
and nothing else: the rotation cancels in the contraction, so the INT8
model's weight gradient is the bf16 model's, bit for bit
(`debug/verify_linear_dw.py`).

**The memory arithmetic decides whether an INT8 base fits at all.** The
weight-only backward needs the weight in bf16, so an INT8 base costs 1 + 2 =
**3 bytes per parameter** (3.7 GB + 7.4 GB for a 3.7 B model) — more than a
bf16 base, and on a 12.2 GB card, more than fits. Two ways out:

* **rebuild and drop**: the dense weight is rebuilt from int8 for that one
  layer's backward and dropped again, so the base stays at **1 byte per
  parameter** and only one layer's copy exists at a time. The fused
  dequantizer makes this affordable: 215 ms → **63 ms** for 2.17 B parameters;
* **never dequantize** (`int8_linear(..., mode="int8")`): the weight stays
  int8 and `dy` is quantized instead. One byte per parameter, no copy at all,
  and 1.33× the bf16 backward path's time (2.75 ms against 2.08 ms at
  M=4096, N=8192) instead of 3.4×. Its accuracy is 1.05 % relative against the
  bf16 gradient (the bf16 path: 0.93 %). It needs a loss-curve A/B — same
  seed, same data order — before it becomes a default.

Without the autograd op an INT8 layer in a training graph produces **no
gradient at all, silently**: `linear()` returns a tensor with no `grad_fn`.
The library now ships the op itself (`int8.int8_linear`, `Int8Linear`).

### Three pieces the SDXL path can use now

The trainer's SDXL path is a GEGLU (`nodes/model/attention.py:146`:
`x * F.gelu(gate)`), and three things in the library were built for it:

| what | why it matters there |
|---|---|
| `int8.ffn(..., act="geglu_erf")` | the whole GEGLU FFN in two GEMMs and two kernels, with the `[M, 10240]` bf16 tile never written: **1.2-1.5× faster** than the unfused INT8 chain, and bit-identical to it. The erf-GELU was proved over every bf16 value (`debug/native/probe_gelu_exact.cpp`), so this is not an approximation |
| `int8.prepare_weight_fused(w)` | the weight preparation in one C pass: **15.5×** the torch chain (22.6 ms against 350 ms for 2.17 B, per step). It is not the same function as `prepare_weight` (5.7 % of the values differ by one unit), and the INT8 path's error is unchanged |
| `int8.TrainableInt8Linear` | a **trainable** INT8 weight: the parameter stays bf16 and is quantized every forward, the forward is the INT8 model's and the backward is the bf16 model's (`dW` bit for bit). It is capture-safe: a captured step re-quantizes the *current* weight on every replay |

`docs/sdxl.md` has the measurements and the arithmetic behind them.

## 2. Forkable: two Triton kernels that are exactly what a Cosmos DiT needs
When the Anima DiT lands it will contain `rms_norm + adaLN modulation`, `rope`
inside attention, and a SwiGLU/situ FFN. Three of those already have fused
Triton kernels here, each verified bit-exact against the eager chain:

| file | fused | measured here |
|---|---|---|
| `xpu_kernels/triton/adaln.py` | RMSNorm + adaLN scale/shift in one pass | **1.56×**, bit-exact, **differentiable** (fwd+bwd **6.3×** vs the eager chain, 1.5e-7 against autograd) |
| `xpu_kernels/triton/rope.py` | RMSNorm + rope in one pass | **2.34×**, bit-exact, **differentiable** (fwd+bwd **14.3×**, 0.62 vs 8.88 ms per tensor) |
| `xpu_kernels/int8/ops.py` `ffn_situ` | `down(situ(up(x)))` without the bf16 `[M, 2*ffn]` intermediate | 2 GEMMs instead of 2 GEMMs + 3 memory-bound passes |

The adaLN's modulation gradients are *more* accurate than the eager chain's
(3.1e-7 against 1.7e-3), because the sums stay in fp32 instead of going
through a bf16 temporary.

Two design rules worth copying even if the kernels are rewritten:

1. **Self-test once per process, then fall back.** Each module runs its kernel
   against the shipped chain on first use and disables itself on any
   mismatch. A custom kernel in a training loop that silently changes numerics
   is worse than a slow one.
2. **One env flag per kernel** (`XPU_KERNELS_NO_ROPE`, `XPU_KERNELS_NO_ADALN`),
   so a suspect result can be bisected without a code change. The gate is
   shared (`triton/_gate.py`, CPU-tested in `tests/test_gate.py`), so a new
   kernel only has to supply a probe, not a verification mechanism.

Caveat: both kernels are written for MossImage's exact dtypes and shapes
(bf16, `dim=3072`, 24 heads × 128, `num_warps=4` because a different warp
count changes the reduction order and the output). For Anima, treat them as a
template and re-derive the constants.

## 3. The `tanh` finding (costs nothing to check, applies broadly)

`sycl::tanh` on this card has a **data-dependent slow path**: on
saturated/denormal inputs it costs 3.5× more. Replacing it with
`expm1(2x) / (expm1(2x) + 2)` is bf16-identical for **every finite bf16 input**
(exhaustively checked over all 65 536 encodings) and has no slow path. In the
FFN activation alone that was 34.7 ms/step, 2.9 % of the step.

Any activation in the training path that calls `tanh` on inputs that saturate
is a candidate: the VAE, and any gated activation. (SDXL's QuickGELU is
sigmoid, so it is not one.) The probe is
`debug/native/probe_tanh_exact.cpp`; the same A/B shape works for any
transcendental.

## 4. Card facts that save a measurement cycle

| fact | evidence | consequence |
|---|---|---|
| Attention is a single oneDNN kernel whichever SDPA backend is selected | `docs/performance.md` | no backend choice to make, and no reason to write one. The mem-efficient backend (the only capturable one) is not a compromise |
| `torch._int_mm` (oneDNN) beats a hand-written ESIMD GEMM | 145-190 TOPS vs 43 | do not write a GEMM |
| A 2026.x `icpx` produces a `.so` that cannot load into this torch | undefined `submit_kernel_direct_with_event_impl` | any compiled extension must be built with oneAPI **2025.3** (`libsycl.so.8`). `build.sh` pins it and fails loudly; copy that check |
| Streaming read ≈ 470-575 GB/s; a copy or read→write mix ≈ 300-430 GB/s **per direction** | `debug/native/probe_bandwidth.cpp`, `docs/performance.md` | the floor to compare a kernel against. A kernel "faster" than this is a measurement error |
| `use_deterministic_algorithms(True)` switches on `fill_uninitialized_memory` | 2174 fill kernels, 92 ms/forward here | if reproducibility is turned on for a training run, clear it around the step |
| `total_memory - reserved` counts *other processes'* VRAM as free | 12 216 MB reported vs 3469 MB actual | the ComfyUI OOM cause; do not reintroduce the proxy in planning |
| `sycl::reduce_over_group` used to segfault the host | twice, then nothing in ~35 000 launches on driver 1.17.39758 | re-test after every driver/compiler bump (`PROBE_REDUCE=1`); the report to the vendor is in `docs/report-reduce-over-group.md` |

## 5. Measurement doctrine that saved this project from itself

- **Device time, not wall time.** Host timers here measure enqueue (they made a
  kernel look 3× better than it was). A profiler-attributed per-kernel table
  for one step is the same instrument ComfyUI uses (`COMFY_PROFILE_FWD`).
- **Interleaved paired A/B, never sequential.** Same process, alternating arms;
  anything below the noise floor is not a result.
- **An idle desktop.** Any window activity moves a step by up to +20 % here,
  more than any remaining optimisation.
- **Reproducibility is per session.** The same build produces a different
  output hash across restarts (`--deterministic` fixed it). Loss at step *n* is
  only comparable within one process, one graph set, one kernel compilation
  cache. Re-baseline when any of those change.

## 6. Two lines to put in the trainer's init

**One line, because a captured step that silently fell back to the eager chain
is slower than an eager step and looks like a regression nobody can explain:**

```python
import xpu_kernels
xpu_kernels.prepare_for_capture(model, need=("rms_rope", "rms_adaln"))
xpu_kernels.require_capture_ready(model, need=("rms_rope", "rms_adaln"))
```

`prepare_for_capture` materializes each `Int8Linear`'s dense (bf16) weight
(~0.4 s for a 3.7 B model) and latches the fused kernels' verdicts, so
neither happens inside the capture. The package *refuses* both: `dense_weight()`
raises if the stream is capturing, and the gate refuses to run its self-test
there (the probe synchronizes). `capture_status()` returns the same
information as a list.

**One more line, or the fused kernels will refuse to run.** The gate compares
a kernel against *the implementation it replaces*, and its default reference
is `comfy_kitchen` — which the trainer's environment does not have. Since the
entry points now ask their own gate, `rms_rope_split_half` raises instead of
running unverified. Give it the fallback to compare against:

```python
import xpu_kernels.triton as kt
kt.set_reference("rope", your_rms_rope)     # your eager chain
kt.set_reference("adaln", your_rms_adaln)
```

Then the gate proves the kernel against *your* fallback, which is the
comparison that means something. `kt.status()` says which kernel is closed and
why.

## 7. What does not transfer

- **How to consume it:** ComfyUI installs the package editable
  (`pip install -e ../xpu-kernels`) and imports `xpu_kernels.triton.{rope,adaln}`
  directly, with its own old copies reduced to re-export shims; the switch was
  verified by a seed-1234 render whose PNG md5 did not change. Do the same from
  the training project: one editable install, no vendored copies.
- **A trainable INT8 weight** (the parameter would be bf16 and re-quantized
  per step). The chain costs 638 ms for a 3.7 B model per step; the C kernel
  would do it in 38 ms, but it is not the same function and needs its own
  decision — `docs/todo.md` item 1.
- The graph-capture and `addcmul_` findings — MossImage-specific.
- Prompt/text encoding: closed here as "fast enough".

## 8. If you want one thing ported first

Port `xpu_kernels/triton/rope.py` — it is already differentiable, so it needs
no wrapper work — and `adaln.py`. Expected: roughly 1.5-2× on the norm/rope
portion of the forward, and 14× on the rope forward+backward (0.62 vs 8.88 ms
per tensor here), bit-exact forward, and a template for the FFN fusion later.
Everything needed to do it — the kernels, the self-test gate, the escape
hatch, and the verification harness pattern in `debug/verify_*.py` — is in this
repo.
