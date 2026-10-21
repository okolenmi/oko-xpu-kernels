# Changelog

The published **0.2.0** is the INT8 path, the two Triton kernels and the
packaging. Everything below has been verified but *not* released: the policy
is that a version is for something worth versioning, not for every function,
so the changes wait here until they are worth a number.

## Unreleased

### Added

* **The INT8 linear is an autograd op** (`int8.int8_linear`, and
  `Int8Linear` now uses it). It was not: `linear()` returns a tensor with no
  `grad_fn`, so an INT8 layer dropped into a training graph produced **no
  gradient at all, silently** -- the one thing a training graph must not do.
  The backward is `linear_dx` (the weight is dequantized for that one
  layer's backward and dropped again, so the memory the quantization saved
  stays saved) or `linear_dx_int8` (the weight stays int8: no copy at all,
  one byte per parameter, 1.05 % from the bf16 gradient against 0.93 % for
  the dense path). The dense mode refuses inside a device-graph capture,
  because a captured dequant would be replayed on every step.
* **The weight gradient, `linear_dw(dy, x)`** -- and the surprise is how
  little there is to it. The INT8 forward is *mathematically* the bf16
  forward with the dequantized weight (the rotation is orthogonal, so it
  cancels in the contraction), so the weight gradient of the INT8 model is
  the bf16 model's: `dy^T @ x`, with no scale, no rotation, and no
  dequantized weight at all. Verified bit-identical to `torch.autograd` of
  the dequantized weight (`debug/verify_linear_dw.py`, rel 0.00e+00).
* **A fused row-quantizer** (`xpu_kernels.triton.quantize`): the per-row int8
  quantizer of the gradient path, as a Triton kernel. 4.17 ms -> 0.27 ms at
  3974x8192, and 9 kernel launches become 1 -- which is what moved the
  quantized-gradient backward from 3.4x the bf16 path to 1.33x, while keeping
  the weight in int8 (1 byte per parameter instead of 3).
* **The INT8 forward and both backward variants are capture-safe**, verified on
  the card (`debug/verify_int8_capture.py`): captured, replayed on new data,
  bit-identically to eager. A graph that did not contain the C kernels would
  have replayed the *first* input's answer -- the script checks for that
  explicitly.
* **Capture readiness** (`xpu_kernels.capture`): `prepare_for_capture(model,
  need=...)` materializes each `Int8Linear`'s dense weight and latches the
  fused kernels' verdicts *outside* a device-graph capture;
  `require_capture_ready(...)` is the same check as an exception, and
  `capture_status(...)` returns it as a list. The reason it matters: the
  dequantized weight is a full-precision copy of every quantized weight
  (~0.4 s for a 3.7 B model), and a captured kernel is replayed on every
  step, so materializing it inside the capture would cost that much *per
  step*.
* Two refusals, because both failure modes are silent or expensive rather
  than fatal: `Int8Linear.dense_weight()` raises if the stream is
  capturing, and the gate refuses to run its self-test inside a capture (the
  probe synchronizes, which is illegal there, and the resulting error names
  a synchronisation instead of the kernel that needed it).
* `debug/verify_capture_guards.py`: on the card, both refusals fire inside a
  real capture, and `Int8Linear.dx` captures and replays **bit-identically
  to eager**.

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

  A consequence outside ComfyUI: the gate's default reference is
  `comfy_kitchen`, so in an environment without it (CI, and the training
  project) the gate is closed and the fused kernels refuse to run until a
  reference is injected -- `xpu_kernels.triton.set_reference("rope", yours)`.
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

### Packaging

* PyPI does not accept the platform tag `linux_x86_64` ("400 Bad Request:
  Binary wheel ... has an unsupported platform tag"), so the wheel is tagged
  `manylinux_2_35_x86_64` and the CI builds it on ubuntu-22.04 (glibc 2.35)
  to match, checking the `.so` with `objdump -T` so the tag cannot claim a
  glibc the payload does not need. The tag is not a promise of
  self-containment: the `.so` links `libsycl.so.8` unbundled, like Intel's
  own runtime wheels.
* Releasing is now a deliberate act: the release workflow is dispatched by
  hand with the tag and an explicit `publish` yes/no, and a push publishes
  nothing. See `docs/publishing.md`.

### Verified

* 63 CPU tests (13 new), the device self-test (`python -m xpu_kernels`:
  `backward dx vs autograd: rel 0.00e+00`), and the ComfyUI PNG gate
  unchanged at `a2aacfda45409493a490b0402b61f850` with no fallbacks in the
  log.
