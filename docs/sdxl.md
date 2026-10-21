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

**1. The fused FFN, with SDXL's activation.** This is the item that makes
finding (b) disappear. The library already fuses the FFN for MossImage
(`ffn_situ` = `down(situ(up(x)))` in two GEMMs, with the epilogue and the
second quantizer fused into one pass: dequant → activation → rotate →
quantize). SDXL's FFN is the same shape — ComfyUI's `GEGLU` is
`Linear(1280→10240)` → `chunk` → `x * F.gelu(gate)` → `Linear(5120→1280)` —
so the *structure* fits exactly (the C kernel's "hidden, gate" pair layout is
the chunk's layout).

What it saves: the unfused path writes the [M, 10240] bf16 tile, reads it for
the activation, writes the product, reads it again for the quantizer. The
fused path reads the int32 accumulator once and writes the int8 result.

The obstacle, stated plainly: **the erf-GELU cannot be reproduced bit-exactly
against torch.** The tanh form could (MossImage's `gelu_tanh` is bit-identical
to `F.gelu(approximate="tanh")` because the C code reproduces torch's steps);
the erf form goes through oneDNN's `erf`, and there is no way to know that
`sycl::erf` rounds the same way. The library's rule is that a fused kernel may
not change the arithmetic: a one-step difference in the activation moves the
row's abs-max, which moves a whole row's int8 values, which changes the
image. So this item is a **decision with a measurement**, not a task: either
the erf-GELU is proven bit-exact (unlikely) or the fused FFN ships with a
relative budget like the row-quantizer's, and the PNG gate moves. Measure
first: how far does the output move, and what does it cost to re-take the
gate?

**2. The capture.** Not library work — the trainer's. The library's part is
done and verified: the INT8 forward and both backwards capture and replay
bit-identically, the gate refuses to probe inside a capture, and
`prepare_for_capture` materializes the dense weights outside it. Finding (c)
says this is where the INT8 path's host cost goes: 0.25 ms per call × ~70
attention blocks × 2 linears is seconds of host time per step.

**3. The fused weight preparation, and the straight-through path.** For full
fine-tuning of SDXL the parameter is bf16 and the forward quantizes it every
step. The torch chain costs **638 ms for 2.17 B parameters, per step**; the C
kernel does the same work in **38 ms** (16.7×). The two are not the same
function — the C kernel keeps the rotated values in fp32 where the chain
rounds them to bf16, so the scale differs by one ULP and 5.7 % of the values
differ by one unit — so it needs a decision and a gate of its own
(`docs/todo.md` items 1 and 2).

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

## 5. What to measure next

1. The fused GEGLU: how far does the output move if the erf-GELU is not
   bit-exact, and what does the PNG gate do? (A probe that computes the
   activation in the C kernel and compares it against torch's, element by
   element, over the bf16 encodings — the same shape as
   `debug/native/probe_tanh_exact.cpp`.)
2. The fused FFN's traffic saving, measured: the same probe, comparing the
   unfused (dequant → gelu → quantize) and the fused chain at M=4096,
   N=10240.
3. The int8 convolution epilogue, if item 4 is ever taken up: whether a
   per-(channel, output-channel) scale can be folded into an epilogue at all.
4. A whole-step SDXL measurement with the capture, to see whether the INT8
   path's device cost (finding b) is what the step actually pays.
