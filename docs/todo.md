# TODO

Ranked by what unblocks the most. Every item names the evidence that would
close it; "done" means measured, not implemented.

The items marked *done* or *closed* are kept for the record — the numbers in
them are the ones the decisions rest on. **The open ones are 4, 5 and 7.**
The closed items are also in the table at the bottom, one line each.

## 1. A fused weight preparation — done

`prepare_weight` is a torch chain: **4.07 ms per 25 M parameters**, i.e.
**350 ms for 2.17 B, per step**. The C kernel that does the same work (rotate
+ quantize, per row) takes **0.26 ms** — 15.5x, or **22.6 ms** for 2.17 B.

It is **not the same function**, and that is why it is a *separate* function
(`prepare_weight_fused`) rather than a quiet swap: the chain rounds the
rotated values to bf16 before the abs-max, the C kernel keeps them in fp32,
so the scale differs by up to one bf16 ULP and 5.7 % of the values differ by
one unit. The INT8 path's error against the bf16 weight is unchanged
(0.0123 against 0.0122 — the fused one is marginally better), and the round
trip through the fused dequantizer is exact.

Decided: a separate function, its own name, no change to `prepare_weight`.
Verified in `debug/verify_prepare_fused.py`.

## 2. The straight-through training path — done

`TrainableInt8Linear`: the parameter stays bf16, the forward quantizes it
every step, and the backward returns the bf16 model's `dW = dy^T @ x` (bit
for bit — the rotation cancels). The forward is the INT8 model's, so the
baseline has to be re-taken with that storage.

Two backward modes: `"ste"` (the default) computes `dx = dy @ W` with the
bf16 parameter — no copy, the weight's own 0.9 % error; `"dense"` computes
the exact `dy @ dequant(w8, s)` at the price of a transient full-precision
copy (2 bytes/param, 4.3 GB for a 2.17 B base, which does not fit beside the
activations on a 12.2 GB card).

The forward is capture-safe: the quantization is a device-side kernel with no
host synchronization, so a captured step re-quantizes the *current* weight on
every replay — verified in `debug/verify_trainable_int8.py`.

## 3. A 64-element ConvRot group — closed by measurement

The group is 256 elements, so a linear's contraction length must be a multiple
of 256. SDXL's 640- and 320-dim layers are not, and cannot use the format at
all. A 64-element group (three H4 axes instead of four) would fit both and
reach 100 % of the linear layers.

**Measured, and it is not worth it.** The 640- and 320-dim linears of SDXL
hold **61.8 M parameters — 1.8 % of the model** (the estimate of ~0.4 B was
wrong; `debug/probe_sdxl_shapes.py` and the diffusers config). The price
would be a change of the stored weight format, every user re-quantizing once,
and a slightly worse int8 weight (the rotation would spread the outliers over
64 values instead of 256). 1.8 % does not pay for that.

It stays a decision the *format* can still make: the same question comes back
for a model whose narrow linears are the majority. See `docs/sdxl.md`.

## 4. The residual add, folded into the norm's epilogue

The last open shape in the bf16 path: one launch and one read-write pass per
add — ~84 launches and ~6 ms of device time per step in a 42-block model, in a
step that is host-bound. Training-safe (the add's backward is the identity),
and small next to the capture, which removes the launches altogether.

## 5. An int8 conv path (a project, not an improvement)

SDXL's 3×3 convs are 334.8 M of the UNet's 2567 M (13 %), and the VAE adds
81.6 M more. The ConvRot does not reach them: the contraction is spatial, so
the activation's scale is per *channel*, and the product of that with the
weight's per-output-channel scale cannot fold into a single epilogue scale. It
needs a new epilogue, not a new kernel.

The number that decides it is in the trainer's step: **the convolutions are
22 % of the SDXL step's device time** (the GEMMs are 45 %, the attention 5 %)
-- `docs/sdxl.md`. So this is the largest remaining piece of an SDXL model
that the library cannot quantize, and the largest single lever it does not
have.

## 6. SDXL: the fused FFN with an erf-GELU — done

SDXL's FFN is `Linear(1280→10240)` → `chunk` → `x * F.gelu(gate)` →
`Linear(5120→1280)`, which is the same shape as the library's fused
`ffn_situ`. The library's version was MossImage's activation; SDXL's is the
erf-GELU, and the plan said it could not be reproduced bit-exactly against
torch.

**It can.** `debug/native/probe_gelu_exact.cpp` compares `sycl::erf`'s GELU
against torch's `F.gelu(approximate="none")` over all 65 536 bf16 values and
finds 0 mismatches — oneDNN's erf and SVML's agree on every bf16 input, so
the fused path is bit-identical to the chain a model writes
(`debug/verify_ffn_geglu.py`).

The library now has `geglu_erf` (SDXL's GEGLU) and `gelu_erf` (diffusers'
shape), both fused into the rotate+quantize kernel, and `int8.ffn(...)` takes
the activation as an argument (`ffn_situ` is the old name for one of them).
The fused FFN is **1.2-1.5× faster than the unfused INT8 chain** at SDXL's
shapes, and at parity with bf16 at the big M.

## 7. Small things

- [ ] the NF4 **double quantization** (the absmax values quantized a second
      time) — a separate small pass, on the trainer's side;
- [ ] `xpu-kernels bench` — one CLI over the probes in `debug/` (they all
      take their shape as a constant today). Low value until someone else
      needs to reproduce a number;
- [ ] the activation quantizer's row max: the same construct the FFN epilogue
      uses, not measured, presumably worth ~2 ms/step. Not taken because the
      construct once killed the host process (see the closed list).

## Closed

| item | the evidence |
|---|---|
| the INT8 `dL/dx` (weight-only dequant) | exact against the same GEMM in autograd; 0.9 % from the unquantized weight, and that is the weight — `debug/verify_linear_dx.py` |
| the quantized-`dy` variant | measured: a memory option, not a speed one (1.33× the bf16 path) — `debug/probe_linear_dx_variants.py` |
| the INT8 `dW` | `dy^T @ x`, bit-identical to autograd — `debug/verify_linear_dw.py` |
| the INT8 linear as an autograd op | `int8.int8_linear`, `Int8Linear`; the bias-gradient bug found on the way |
| capture readiness | `xpu_kernels.capture`; the dense weight and the gate both refuse to run inside a capture — `debug/verify_capture_guards.py`, `debug/verify_int8_capture.py` |
| the fused row-quantizer | 4.17 → 0.27 ms, 9 launches → 1; the one kernel that is not bit-exact, and why (hardware division) — `docs/performance.md` |
| the fused NF4 dequantizer | 20.75 → 0.66 ms (31×), bit-exact, no scratch |
| the fused weight dequantizer | 215 → 63 ms for 2.17 B parameters, bit-identical |
| the fused weight *preparation* | `prepare_weight_fused`: 15.5× the chain (22.6 ms against 350 ms for 2.17 B, per step), the error unchanged — `debug/verify_prepare_fused.py` |
| the straight-through INT8 linear | `TrainableInt8Linear`: a bf16 parameter quantized every forward, `dW` the bf16 model's bit for bit, capture-safe — `debug/verify_trainable_int8.py` |
| the SDXL step, measured in the trainer's own UNet | the INT8 path is 2.5-2.7x slower eagerly (it doubles the host cost) and 10-12 % slower under the capture, where it saves 42 % of the weight memory and 39 % of the peak — `debug/probe_sdxl_trainer_step.py`; and the *chain* dequantizer refuses to run inside a capture, so an INT8 base on a library without the fused one cannot be captured at all (0.2) |
| the SDXL parameter map and the speed at SDXL's shapes | 86 % of the parameters are INT8-eligible; the INT8 path is 2.5× faster on the FFN's down-projection and 1.5× slower on its up-projection (the int32 accumulator) — `debug/probe_sdxl_shapes.py`, `docs/sdxl.md` |
| SDXL's activation in the fused FFN | `geglu_erf` and `gelu_erf`, bit-identical to torch's chains (the erf-GELU proved over every bf16 value); 1.2-1.5× faster than the unfused INT8 chain — `debug/verify_ffn_geglu.py` |
| a 64-element ConvRot group | closed by measurement: the layers it would reach are 1.8 % of SDXL, not 15 % |
| the rope and adaLN backwards | fused, differentiable, verified against autograd; the adaLN's modulation gradients are *more* accurate than the torch chain's |
| the rotate's structural floor | closed by measurement: the floor is a property of the format (the row's abs-max is over the rotated row), not a missing kernel — `debug/native/probe_cross_lane.cpp` |
| the hand-written int8 GEMM | L2-bandwidth bound at 43 TOPS; `torch._int_mm` reaches 151 — keep oneDNN |
| the FFN epilogue's row max | `reduce_over_group` is 16 % faster and bit-identical, but it once killed the host process; a gate cannot catch that — `docs/report-reduce-over-group.md` (sent upstream) |
| ComfyUI on the package | the Triton half and the INT8 wrapper both live in the package; the PNG gate is unchanged |
| packaging and CI | 0.2.0 and 0.3.0 on PyPI, sdist + manylinux wheel, attestations; the published wheel is verified against the PNG gate — `debug/verify_published_wheel.sh` |
