# Changelog

The published **0.2.0** is the INT8 path, the two Triton kernels and the
packaging. Everything below has been verified but *not* released: the policy
is that a version is for something worth versioning, not for every function,
so the changes wait here until they are worth a number.

## Unreleased

### Added

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
