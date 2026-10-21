# SDXL: what this library can do for it, and the plan

Everything below is measured on this card (Arc B580, torch 2.12.1+xpu) at
SDXL's own shapes, or derived from a measured figure — the status of each
number is marked. The probe is `debug/probe_sdxl_shapes.py`; the parameter
map comes from the diffusers config of `stabilityai/stable-diffusion-xl-base-1.0`.

## 1. The parameter map (measured)

| part | parameters | in linears | INT8-eligible | convolutions |
|---|---|---|---|---|
| UNet | 2567.5 M | 2232.6 M | **2170.8 M** (84.6 %) | 334.8 M |
| VAE | 83.7 M | 2.1 M | 2.1 M | 81.6 M |
| CLIP-L | 123.1 M | 122.9 M | 122.9 M | 0.1 M |
| OpenCLIP bigG | 694.7 M | 694.1 M | 694.0 M | 0.5 M |
| **total** | **3469 M** | 3051 M | **2989.8 M (86.2 %)** | 417.0 M |

"INT8-eligible" means the contraction length is a multiple of 256 (the ConvRot
group) and the output width a multiple of 16. The rule excludes exactly two
things in SDXL:

| excluded | parameters | why |
|---|---|---|
| the 3×3 convolutions (the UNet and the VAE) | 416.4 M (12.0 %) | the ConvRot is a *linear* format: the contraction is spatial, so the activation's scale would be per channel, and that cannot fold into one epilogue scale |
| the 640- and 320-dim linears | 61.8 M (1.8 %) | 640 = 2.5 × 256 and 320 = 1.25 × 256, so the ConvRot group does not divide them |

**Correction to an earlier estimate**: the 640-dim linears were thought to hold
~0.4 B parameters. They hold 62 M. A 64-element ConvRot group would fit them
and reach 100 % of the linears — and would be worth **1.8 % of the model**, for
a change of the stored weight format. That idea is closed by measurement, not
by taste (`docs/todo.md`).

## 2. The speed, at SDXL's shapes (measured)

The library's headline number — INT8 at parity with BF16 — was measured at
K=N=3072. SDXL has no such shape. `debug/probe_sdxl_shapes.py` measures both
arms (both pay for the activation rotation, because ConvRot stores the weights
pre-rotated) at the UNet's shapes and token counts, interleaved.

Device time per call, batch 1 at 1024×1024 (M = the token count of the level):

| K | N | M | INT8 | BF16 | ratio |
|---|---|---|---|---|---|
| 5120 | 1280 | 4096 | 0.647 ms | 1.608 ms | **0.40×** |
| 2048 | 1280 | 4096 | 0.366 ms | 0.557 ms | **0.66×** |
| 1280 | 1280 | 4096 | 0.289 ms | 0.296 ms | 0.97× |
| 1280 | 10240 | 4096 | 1.980 ms | 1.318 ms | **1.50×** |

Three things come out of this, and they are the whole SDXL story:

**(a) INT8 is 2.5× faster where the weight dominates.** The FFN's
down-projection (K=5120, N=1280) reads 6.5 MB as int8 against 13 MB as bf16,
and the int8 GEMM is faster on top of that. This is the shape with 393 M
parameters.

**(b) INT8 is 1.5× slower on the FFN's up-projection (K=1280, N=10240)** —
the shape with 786 M parameters, the single biggest one. The reason is the
dequant epilogue: it reads the int32 accumulator (4 bytes per output) and
writes bf16 (2), so it moves 6 bytes per output element — at M=4096 that is
252 MB, 0.93 ms, which is the measured epilogue time. The int8 GEMM at that
shape is only 19 % faster than the bf16 one (0.94 vs 1.17 ms) because it is
bandwidth-bound, not compute-bound. The epilogue's 0.93 ms is bought back at
0.23 ms. **The int32 accumulator is the price of the format at large N and
small K.**

**(c) Below M≈1024 the device is idle and the *host* is the cost.** The INT8
call costs 0.25 ms of host time against the bf16 call's 0.10 ms — 2.4×, and
independent of the shape, because the INT8 path is three launches plus the
package's Python glue and the bf16 path is two launches plus torch's. A
device-graph capture removes it entirely (the library's contract, verified).

So the honest summary for SDXL: **the INT8 path is a memory lever, and at
SDXL's shapes it is not free.** It buys 4× less weight memory (3.0 GB on the
whole pipeline) at a price that is paid in the one big-N projection and in the
host time — and the capture pays the host part back.

## 3. The plan, ranked

**1. The fused FFN, with SDXL's activation — done.** The library now has
SDXL's GEGLU (`geglu_erf`) and the plain erf GELU (`gelu_erf`), both fused
into the rotate+quantize kernel, so the FFN is two GEMMs and two kernels
instead of eight, and the `[M, 10240]` bf16 tile never exists.

The obstacle turned out not to be an obstacle. The plan said the erf-GELU
could not be reproduced bit-exactly against torch (the tanh form could), and
that this needed a decision with a measurement. The measurement said
otherwise: `debug/native/probe_gelu_exact.cpp` compares `sycl::erf`'s GELU
against torch's `F.gelu(approximate="none")` over all 65 536 bf16 values
and finds **0 mismatches** — oneDNN's erf and SVML's agree on every bf16
input, so the fused path is *bit-identical* to the chain a model writes
(`debug/verify_ffn_geglu.py`).

| M | fused | unfused INT8 | bf16 | fused / unfused |
|---|---|---|---|---|
| 4096 | 2.178 ms | 3.281 ms | 2.348 ms | **0.66×** |
| 1024 | 0.567 ms | 0.794 ms | 0.576 ms | **0.71×** |
| 256 | 0.376 ms | 0.467 ms | 0.186 ms | 0.81× |
| 64 | 0.385 ms | 0.472 ms | 0.138 ms | 0.82× |

**1.2-1.5× faster than the unfused INT8 chain**, and at parity with bf16 at
the big M (0.93× at M=4096, 0.98× at M=1024). Below M≈1024 the bf16 path
wins because the INT8 path is host-bound there, and the capture is what
removes that.

What it does *not* fix: the int32 accumulator's traffic on the up-projection
(finding (b)). The fusion removes the bf16 round trip, not the accumulator.

**2. The capture.** Not library work — the trainer's. The library's part is
done and verified: the INT8 forward, both backwards and the fused FFN
capture and replay bit-identically, the gate refuses to probe inside a
capture, and `prepare_for_capture` materializes the dense weights outside
it. Finding (c) says this is where the INT8 path's host cost goes: 0.25 ms
per call × ~70 attention blocks × 2 linears is seconds of host time per step.

**3. The fused weight preparation, and the straight-through path — done.**
For full fine-tuning of SDXL the parameter is bf16 and the forward quantizes
it every step. The torch chain costs **350 ms for 2.17 B parameters, per
step**; the C kernel does the same work in **22.6 ms** (15.5×) ✓. The two are
not the same function — the C kernel keeps the rotated values in fp32 where
the chain rounds them to bf16, so the scale differs by up to one bf16 ULP and
5.7 % of the values differ by one unit — so it is a *separate* function
(`prepare_weight_fused`), and the INT8 path's error against the bf16 weight
is unchanged (0.0123 against 0.0122) ✓.

`TrainableInt8Linear` is the module: a bf16 parameter, quantized every
forward, with the bf16 model's `dW` in the backward (bit for bit) and two
backward modes — `"ste"` (no copy of the weight, the weight's own 0.9 %
error) and `"dense"` (the exact gradient, at the price of a transient 2
bytes/param copy: 4.3 GB for a 2.17 B base, which does not fit beside the
activations on a 12.2 GB card). It is capture-safe: a captured step
re-quantizes the *current* weight on every replay ✓.

**4. The int8 convolution path.** 416 M parameters, 12 % of the model, and
the ConvRot does not reach them: the contraction is spatial, so the
activation's scale is per *channel*, and the product of a per-channel scale
with the weight's per-output-channel scale cannot fold into one epilogue
scale. It needs a new epilogue, not a new kernel. A project, and only worth
it if the memory becomes the binding constraint.

**5. Closed by measurement: the 64-element ConvRot group.** Worth 1.8 % of the
model, for a format break. Not done.

## 4. The memory arithmetic (derived from the measured figures)

For a LoRA training run on a 12.2 GB card, with the UNet frozen:

| what | bf16 base | int8 base (the eligible 86 %) |
|---|---|---|
| the whole pipeline's weights | 6.94 GB | 3.95 GB (2.99 GB int8 + 0.96 GB bf16) |
| the UNet alone | 5.14 GB | 2.97 GB |
| the backward | the weight is already there | `mode="int8"`: nothing to rebuild. `mode="dense"`: rebuild-and-drop, **63 ms per backward** for the 2.17 B (the fused dequantizer; it was 215 ms as a torch chain) |
| what does *not* fit | — | `mode="dense"` with a *materialized* weight: 3 bytes/param = 8.9 GB for the UNet, which does not fit beside the activations |

The NF4 storage is the next step down (1.73 GB for the eligible weights, 0.58
bytes/param) and it is worth it only when the memory binds and the speed does
not: the forward becomes a dequant plus a bf16 GEMM, and the gradient carries
NF4's error (~1-2 %) instead of int8's (0.9 %). The trainer's own analysis of
that trade is in `docs/nf4-storage-vision.md` in the trainer's tree.

**The baseline has to be re-taken with the storage.** An int8 base is a
different function from a bf16 base (0.9 %), and an SDXL checkpoint is *not*
pre-rotated: adopting the INT8 path means storing the base pre-rotated, so the
bf16 comparison must use the rotated weights, or it is not a comparison at all.

## 5. The whole step, in the training project's own UNet (measured)

The numbers above are per layer. The step is the question that matters, and
it was measured in the training project's own SDXL UNet (its `next/` tree,
borrowed read-only; the instrument is `debug/probe_sdxl_trainer_step.py`,
the report is in the trainer's `docs/sdxl-int8-measurements.md`).

768×768, batch 1, the base frozen (the LoRA-shaped step), with the
trainer's own checkpointing and its own capture pattern:

| arm | eager step | **captured step** | host time | aten calls | weights | peak |
|---|---|---|---|---|---|---|
| the UNet as it is (bf16) | 400.7 ms | **565.6 ms** | 1058 ms | 44 277 | 4.89 GB | 5.23 GB |
| the library's INT8 linears | 986.0 ms | **628.1 ms** | 2735 ms | 85 822 | 2.84 GB | 3.21 GB |
| + the library's fused GEGLU FFN | 1038.6 ms | **624.4 ms** | 2741 ms | 85 822 | 2.85 GB | 3.21 GB |

Three things fall out of it:

* **The INT8 path doubles the host cost** (four extra kernels per linear), so
  it is 2.5-2.7× slower *eagerly*. It is only viable under the capture.
* **Under the capture it costs 10-12 % of the step** and saves 42 % of the
  weight memory and 39 % of the peak. On a 12 GB card that is the difference
  between fitting and not.
* **The fused GEGLU does not move the whole step**: 1.2-1.5× of the FFN is
  invisible in the step, because the FFN is a small part of it. It is free
  and bit-identical, but it is not a lever on the step.

And where the step's device time actually goes (torch's profiler with the XPU
activity, one 512×512 step: **0.263 s over 12 171 kernel calls**):

| | calls | device time | share |
|---|---|---|---|
| the GEMMs (`gemm_kernel`, `mm`, `addmm`) | 4 398 | 118.8 ms | **45 %** |
| the convolutions (forward, backward, `conv_reorder`) | 555 | 57.2 ms | **22 %** |
| the attention (the flash backward) | 280 | 14.2 ms | 5 % |
| the elementwise, norm, cat and the rest | ~6 900 | 65 ms | 25 % |

So the library's INT8 path works on the 45 %, and **the convolutions — 22 %
of the step, and 334.8 M of the UNet's parameters — are the largest piece it
cannot reach** (item 4).

## 6. What to measure next

1. The int8 convolution epilogue, if item 4 is ever taken up: whether a
   per-(channel, output-channel) scale can be folded into an epilogue at all.
2. The *per-layer* choice: with the fusion in place, the INT8 path is at
   parity with bf16 at the big M and slower at the small one, so which layers
   should be INT8 at all is a question with a measurement behind it, not a
   blanket rule.
3. The step-level trade for the *trainable* base (the straight-through path):
   the 22.6 ms of re-quantization per step, against the memory the int8
   base saves and the accuracy it costs. The frozen base was measured (above);
   the trainable one has not.
