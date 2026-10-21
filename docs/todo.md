# TODO

Ranked by what unblocks the most. Every item names the evidence that would
close it; "done" means measured, not implemented.

The closed items are at the bottom, one line each, with where the evidence
lives. The long history is in `CHANGELOG.md` and `docs/performance.md`.

## 1. A fused weight preparation (the straight-through path's blocker)

`prepare_weight` is a torch chain: **4.34 ms per 25 M parameters**, i.e.
**638 ms for a 3.7 B model, per step**. The C kernel that does the same work
(rotate + quantize, per row) takes **0.26 ms** — 16.7×, or **38 ms** for
3.7 B.

It is **not the same function**, and that is the whole difficulty: the chain
rounds the rotated values to bf16 before the abs-max, the C kernel keeps them
in fp32, so the scale differs by one ULP and 5.7 % of the values differ by one
unit. Switching would change every model's weights, so it needs a decision and
a gate of its own, not a quiet swap.

What it buys: the **straight-through** training path, where the parameter is
bf16 and the forward quantizes it per step. At 638 ms that is a third of a
step; at 38 ms it is noise.

- [ ] decide: a separate `prepare_weight_fused`, or a switch, or a change to
      the C kernel so the two agree;
- [ ] measure the INT8 path's error against bf16 with the fused preparation
      (the expected change is ~1e-4, far below the path's own 0.94 %);
- [ ] gate it: the row scale and the values are pinned against the chain in
      float64, as `tests/test_quantize.py` does for the quantizer.

## 2. The straight-through training path

The weight gradient already exists and is exact: `linear_dw(dy, x) = dy^T @ x`
— the rotation cancels, so the INT8 model's weight gradient is the bf16
model's, bit for bit (`debug/verify_linear_dw.py`).

What is missing is the module: a parameter that stays bf16, a forward that
quantizes it per step, and a backward that returns `dW`. It is the last piece
of the INT8 story and it depends on item 1.

- [ ] the module, with the capture rules of `Int8Linear` (no dequant inside a
      graph; the per-step quantization *is* capturable, but its cost is
      per-step);
- [ ] the memory arithmetic: 2 bytes/param for the parameter, 1 for the
      quantized copy, and the optimizer state on top.

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
needs a new epilogue, not a new kernel. Only worth it if the memory saving
becomes the binding constraint.

## 6. SDXL: the fused FFN with an erf-GELU

SDXL's FFN is `Linear(1280→10240)` → `chunk` → `x * F.gelu(gate)` →
`Linear(5120→1280)`, which is the same shape as the library's fused
`ffn_situ`. The library's version is MossImage's activation; SDXL's is the
erf-GELU, and **the erf-GELU cannot be reproduced bit-exactly against torch**
(the tanh form could: the C code reproduces torch's steps). A fused kernel
that changes the arithmetic moves the row's abs-max, which moves a whole row's
int8 values, which changes the image — so this is a decision with a
measurement, not a task. See `docs/sdxl.md`.

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
| the SDXL parameter map and the speed at SDXL's shapes | 86 % of the parameters are INT8-eligible; the INT8 path is 2.5× faster on the FFN's down-projection and 1.5× slower on its up-projection (the int32 accumulator) — `debug/probe_sdxl_shapes.py`, `docs/sdxl.md` |
| a 64-element ConvRot group | closed by measurement: the layers it would reach are 1.8 % of SDXL, not 15 % |
| the rope and adaLN backwards | fused, differentiable, verified against autograd; the adaLN's modulation gradients are *more* accurate than the torch chain's |
| the rotate's structural floor | closed by measurement: the floor is a property of the format (the row's abs-max is over the rotated row), not a missing kernel — `debug/native/probe_cross_lane.cpp` |
| the hand-written int8 GEMM | L2-bandwidth bound at 43 TOPS; `torch._int_mm` reaches 151 — keep oneDNN |
| the FFN epilogue's row max | `reduce_over_group` is 16 % faster and bit-identical, but it once killed the host process; a gate cannot catch that — `docs/report-reduce-over-group.md` (sent upstream) |
| ComfyUI on the package | the Triton half and the INT8 wrapper both live in the package; the PNG gate is unchanged |
| packaging and CI | 0.2.0 and 0.3.0 on PyPI, sdist + manylinux wheel, attestations; the published wheel is verified against the PNG gate — `debug/verify_published_wheel.sh` |
