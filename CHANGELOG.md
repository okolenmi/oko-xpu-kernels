# Changelog

## Unreleased

*Documentation and measurement only — no behaviour change.*

* **The docs were audited and rewritten.** The README was a front door with an
  optimization log, an ESIMD layout description, a status table and three
  copies of the performance tables inside it; it is now a front door
  (**4 899 → 779 words**) and the long material lives where it belongs: the
  API in `docs/api.md`, the hardware in `docs/architecture.md`, the
  measurements in `docs/performance.md`, the dead ends in
  `docs/experiments.md`. The TODO is now a ranked list of *open* items with a
  one-line record of the closed ones (**3 905 → 1 291 words**). Three
  documents took over what the README used to carry — the API, the SDXL plan
  and the vendor report — so the total is 21 100 words against 24 700, and
  what came out of it was duplication, not information.
* **The broken references were fixed.** `docs/todo.md` pointed at a report file
  that was not in the repository (`docs/report-reduce-over-group.md`, now
  written), and the trainer's document pointed at a document that has never
  existed here (`docs/xpu_int8_perf_findings.md` → `docs/performance.md`).
  The stale claims were corrected too: the wheel's platform tag in
  `docs/publishing.md` was still `linux_x86_64`, the release procedure did not
  say that a release is **two** dispatches (a rehearsal, then the upload), and
  the version numbers were still 0.2.0.
* **SDXL was measured, and there is a plan.** `docs/sdxl.md` and
  `debug/probe_sdxl_shapes.py`. The headline findings:
  * 86 % of SDXL's parameters (2990 M of 3469 M) are eligible for the INT8
    path; the excluded are the 3×3 convolutions (416 M) and the 640/320-dim
    linears (62 M);
  * at SDXL's own shapes the INT8 path is **2.5× faster** on the FFN's
    down-projection, at parity on the attention projections, and **1.5×
    slower** on the FFN's up-projection — the int32 accumulator (6 bytes per
    output element) costs more there than the int8 GEMM saves;
  * below M≈1024 the device is idle and the *host* is the cost: 0.25 ms per
    INT8 call against 0.10 ms for bf16, which the device-graph capture
    removes;
  * the 64-element ConvRot group, which was going to reach SDXL's narrow
    linears, is worth **1.8 %** of the model, not 15 % — the earlier estimate
    was wrong, and the idea is closed by measurement.
* `debug/verify_published_wheel.sh` is now part of the repo: it downloads the
  *published* wheel, checks that it carries its own `.so`, runs every gate
  from it, and renders the reference image with it (md5
  `a2aacfda45409493a490b0402b61f850`). It is the check that a release is good,
  and it is a script now, not a habit.

* **SDXL's activation is fused, and it is bit-exact.** The library now has
  `geglu_erf` (SDXL's `hidden * gelu(gate)`, the GEGLU of ComfyUI and of the
  training project) and `gelu_erf` (diffusers' plain GELU MLP), both fused
  into the rotate+quantize kernel, so an SDXL FFN is two GEMMs and two
  kernels instead of eight, and the `[M, 10240]` bf16 tile never exists.

  The plan said this could not be done: the erf-GELU was expected to be
  irreproducible against torch, because the tanh form could be reproduced
  (`sycl::tanh` and `sycl::expm1` agree on every bf16) and oneDNN's erf was
  not assumed to round like SVML's. **The measurement said otherwise**:
  `debug/native/probe_gelu_exact.cpp` compares the two over all 65 536 bf16
  values and finds **0 mismatches**. So the fused path is bit-identical to the
  chain a model writes (`debug/verify_ffn_geglu.py` checks the whole FFN, the
  half order and the capture).

* **The FFN takes the activation as an argument.** `int8.ffn(x, up, up_s,
  down, down_s, act=...)`, with `ffn_situ` kept as the name for MossImage's
  one. The activations are `gelu_tanh`, `swiglu`, `situ`, `geglu_erf` and
  `gelu_erf`; `xpu_kernels.ACTIVATIONS` is the map and
  `xpu_kernels.PAIRED_ACTIVATIONS` says which of them read a pair of halves.

* **Measured:** the fused FFN is **1.2-1.5× faster than the unfused INT8
  chain** at SDXL's shapes (2.178 ms against 3.281 ms at M=4096), and at
  parity with bf16 at the big M (0.93×).

* **SDXL's step was measured in the training project's own UNet** (its `next/`
  tree, borrowed read-only; `debug/probe_sdxl_trainer_step.py`, the report in
  the trainer's `docs/sdxl-int8-measurements.md`). The trade, at 768×768 with
  the trainer's own checkpointing and capture pattern:

  | arm | eager step | captured step | host time | aten calls | weights | peak |
  |---|---|---|---|---|---|---|
  | the UNet as it is (bf16) | 400.7 ms | **565.6 ms** | 1058 ms | 44 277 | 4.89 GB | 5.23 GB |
  | the INT8 linears | 986.0 ms | **628.1 ms** | 2735 ms | 85 822 | 2.84 GB | 3.21 GB |
  | + the fused GEGLU FFN | 1038.6 ms | **624.4 ms** | 2741 ms | 85 822 | 2.85 GB | 3.21 GB |

  The INT8 path **doubles the host cost** (four extra kernels per linear), so
  it is 2.5-2.7× slower eagerly and only **10-12 % slower under the
  capture**, where it saves **42 % of the weight memory and 39 % of the
  peak**. The fused GEGLU, 1.2-1.5× on the FFN in isolation, is invisible in
  the whole step.

  The device-time table for one step (torch's profiler with the XPU activity:
  0.263 s over 12 171 kernel calls) says where the work is: **the GEMMs are
  45 %, the convolutions 22 %, the attention 5 %**. The convolutions are the
  largest piece of an SDXL model the library cannot quantize.

  A correction, for the record: the XPU profiler **works**. It registers PTI
  callbacks with the Level Zero loader, prints a registration warning, and
  produces a complete device-time table. What is not survivable is an
  out-of-memory during the profiled step: the device is lost, the profiler's
  cleanup then fails, and the exception escapes a destructor, so the process
  dies and the table is lost with it. It looks like the profiler crashing; it
  is the device being gone.

* The PNG gate is unchanged (`a2aacfda45409493a490b0402b61f850`) with the
  rebuilt library, and `python -m xpu_kernels` now checks the fused GEGLU
  against the unfused chain as part of the self-test.

* **A fused weight preparation** (`int8.prepare_weight_fused`): the rotate and
  the quantization in one C pass instead of a torch chain. **15.5× faster** —
  0.26 ms against 4.07 ms for 25 M parameters, which is 22.6 ms against 350 ms
  for SDXL's 2.17 B quantized parameters, *per step*.

  It is a separate function on purpose, because it is **not the same
  function**: the C kernel keeps the rotated values in fp32 where the chain
  rounds them to bf16 first, so the row's scale differs by up to one bf16 ULP
  and 5.7 % of the values differ by one unit. The INT8 path's error against
  the bf16 weight is unchanged (0.0122 against 0.0123), and the round trip
  through the fused dequantizer is exact. Swapping it into `prepare_weight`
  would change every stored weight in the wild, so it did not happen.

* **The straight-through INT8 linear** (`int8.TrainableInt8Linear`): the
  parameter stays bf16 and is quantized on every forward, so it can be
  trained. The forward is the INT8 model's; the backward is the bf16 model's
  — `dW = dy^T @ x`, bit for bit, because the ConvRot rotation cancels in the
  contraction. Two backward modes: `"ste"` (the default) computes
  `dx = dy @ W` with the bf16 parameter, so no copy of the weight is made and
  the error is the weight's own (0.9 %); `"dense"` computes the exact
  `dy @ dequant(w8, s)` at the price of a transient full-precision copy.

  The forward is **capture-safe**: the quantization is a device-side kernel
  with no host synchronization, so a captured training step re-quantizes the
  *current* weight on every replay — which is the property that makes it
  usable inside a graph at all (`debug/verify_trainable_int8.py`).

## 0.3.0

*Released after 0.2.0, which shipped the INT8 path, the two Triton kernels and
the packaging. This one is the training half: a differentiable INT8 linear, a
capture contract, two fused kernels, and the measurements that say what the
remaining headroom is.*

### Added

* **The INT8 linear is an autograd op** (`int8.int8_linear`, and
  `Int8Linear` now uses it). It was not: `linear()` returns a tensor with no
  `grad_fn`, so an INT8 layer dropped into a training graph produced **no
  gradient at all, silently** -- the one thing a training graph must not do.
  The backward is `linear_dx` (the weight is dequantized for that one layer's
  backward and dropped again, so the memory the quantization saved stays
  saved) or `linear_dx_int8` (the weight stays int8: no copy at all, one byte
  per parameter, 1.05 % from the bf16 gradient against 0.93 % for the dense
  path). The dense mode refuses inside a device-graph capture, because a
  captured dequant would be replayed on every step.
* **The weight gradient, `linear_dw(dy, x)`** -- and the surprise is how
  little there is to it. The INT8 forward is *mathematically* the bf16 forward
  with the dequantized weight (the ConvRot rotation is orthogonal, so it
  cancels in the contraction, and the activation's quantization is exact for
  the same reason), so the weight gradient of the INT8 model is the bf16
  model's: `dy^T @ x`, with no scale, no rotation, and, unlike `dx`, no
  dequantized weight at all. Verified bit-identical to `torch.autograd` of the
  dequantized weight (`debug/verify_linear_dw.py`, rel 0.00e+00).
* **Capture readiness** (`xpu_kernels.capture`): `prepare_for_capture(model,
  need=...)` materializes each `Int8Linear`'s dense weight and latches the
  fused kernels' verdicts *outside* a device-graph capture;
  `require_capture_ready(...)` is the same check as an exception, and
  `capture_status(...)` returns it as a list. The reason it matters: the
  dequantized weight is a full-precision copy of every quantized weight
  (~0.4 s for a 3.7 B model), and a captured kernel is replayed on every
  step, so materializing it inside the capture would cost that much *per
  step*.
* **The INT8 forward and both backward variants are capture-safe**, verified
  on the card (`debug/verify_int8_capture.py`): captured, replayed on new
  data, bit-identically to eager -- including the check that the graph
  really contains the C kernels, because a graph that did not would replay
  the first input's answer without any error.
* **A fused row-quantizer** (`xpu_kernels.triton.quantize`): the per-row int8
  quantizer of the gradient path, as a Triton kernel. 4.17 ms -> 0.27 ms at
  3974x8192, and 9 kernel launches become 1 -- which is what moved the
  quantized-gradient backward from 3.4x the bf16 path to 1.33x, while keeping
  the weight in int8 (1 byte per parameter instead of 3).
* **A fused NF4 dequantizer** (`xpu_kernels.triton.nf4`): the QLoRA storage
  format dequantized in one kernel. 100 M parameters: 20.75 ms -> 0.66 ms
  (31.5x), 2.6 bytes of traffic per parameter against the chain's ~40, and
  **no scratch at all** -- the chain holds ~16 bytes per parameter live (its
  int64 index tensor alone is 8), so its peak can exceed the bf16 weight the
  format was meant to replace.
* **A fused weight dequantizer** (a new kernel in `libxpuint8.so`): the
  backward's weight-only path. The torch chain moved fifteen bytes per
  parameter -- an fp32 copy of the weight, a bf16 copy, then the rotated one
  -- which is 215 ms for the 2.17 B quantized parameters of an SDXL base, the
  number the training project says decides whether an INT8 base is trainable
  at all. The kernel moves three, and is bit-identical: 215 ms -> 63 ms.
* Two refusals, because both failure modes are silent or expensive rather
  than fatal: `Int8Linear.dense_weight()` raises if the stream is
  capturing, and the gate refuses to run its self-test inside a capture (the
  probe synchronizes, which is illegal there, and the resulting error names
  a synchronisation instead of the kernel that needed it).

### Changed

* **The fused entry points ask their own gate.** `usable()` is where the
  gate is asked and the caller was expected to ask it; a caller that skipped
  the question got an *unverified* kernel, silently. The entry points now ask
  the gate themselves (`Gate.require()`) and raise. It raises rather than
  falling back because the fallback belongs to the caller: the package's own
  reference is not bit-identical to ComfyUI's `comfy_kitchen` (173 of 12.2 M
  values differ), so substituting it would change the caller's numbers in
  exactly the case where the gate is closed. The backward keeps failing
  soft -- the torch gradient *is* the eager chain's gradient.

  **This is the one change a consumer can feel.** The gate's default
  reference is `comfy_kitchen`, so in an environment without it (CI, and the
  training project) the gate is closed and the fused kernels now refuse to
  run until a reference is injected:
  `xpu_kernels.triton.set_reference("rope", your_fallback)`.
* **The row-quantizer is the one kernel that is not bit-exact**, and the
  reason is hardware, not sloppiness: on this backend torch's tensor÷tensor
  division is correctly rounded but its tensor÷scalar division is a
  reciprocal multiply, the two disagree on 13 % of values, and Triton's
  correctly rounded division costs 18x more than the fast one -- so a
  bit-exact fused quantizer would be *slower* than the torch chain it
  replaces. The kernel moves 2-3e-4 of the quantized values by one unit
  (9.1e-4 relative on the gradient, fifteen times below the int8 path's own
  1.35 % error); the row scale is exact, and the gate compares with a
  relative budget while the formula is pinned in float64 by the tests.
  `XPU_KERNELS_NO_QUANTIZE=1` switches it off.
* **Releasing is a deliberate act.** The release workflow is dispatched by
  hand (the tag to release, and an explicit `publish` yes/no); a push
  publishes nothing.

### Measured, and not taken

* **The rotate's structural floor is not reachable.** The constraint is the
  format, not the implementation: the row's abs-max is over the *rotated*
  row, so the load, the transform, the scale and the store are serialised
  per row whatever the register pressure is. The last open shape (a group
  split across 4 lanes) was measured rather than guessed: the exchange is
  cheap (84 ns per 512-bit move) but splitting the groups makes the row's
  scale cross sub-groups -- 3 at K=3072, 8 at K=8192 -- which means a
  barrier (1.33 ms) or a second pass (0.75 ms), both far worse than the
  0.402 ms the rotate costs now. **INT8 stays at parity with BF16**, and the
  only route left is a scale per *block* instead of per row, which changes
  the numbers and is therefore a design decision, not an optimisation.
* **The bf16 path has almost nothing left**: oneDNN's post-op fusion is not
  available here (`gelu_tanh`/`gelu_erf` are refused by name, and the
  `relu` path does not actually fuse), the norms are fused, the activation's
  transcendental is fixed, and the GEMMs and the attention are oneDNN's.

### Fixed

* The autograd op dropped the bias's gradient. `Int8Linear`'s own bias has
  `requires_grad=False`, so it never showed -- but a caller who passes a
  trainable bias would have had it silently never learn.
* `capture_status` demanded a materialized dense weight from every
  `Int8Linear`, including the ones in `int8` mode, where the weight never
  leaves int8 and no dense copy is ever needed.
* The fused quantizer returned its row scales flattened where the torch
  chain returns them per row of `x`.

### Packaging

* PyPI does not accept the platform tag `linux_x86_64` ("400 Bad Request:
  Binary wheel ... has an unsupported platform tag"), so the wheel is tagged
  `manylinux_2_35_x86_64` and the CI builds it on ubuntu-22.04 (glibc 2.35)
  to match, checking the `.so` with `objdump -T` so the tag cannot claim a
  glibc the payload does not need. The tag is not a promise of
  self-containment: the `.so` links `libsycl.so.8` unbundled, like Intel's
  own runtime wheels.
