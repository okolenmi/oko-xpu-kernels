# TODO

Ranked by what unblocks the most. Every item names the evidence that would close
it; "done" means measured, not implemented.

## 1. INT8 backward for training (`dL/dx = dL/dy @ W`)

The library was inference-only: `linear` computes `y = x @ W^T` with an int8
weight and an int8 activation. Training needs the vjp, `dx = dy @ W`, and the
weight gradient (`dW = dy^T @ x`) if the weights are not frozen.

**dx: done, weight-only dequant.** The reason it is not an INT8 GEMM is the
scale layout: `W[n, k] = W8[n, k] * s[n]`, so

    dx[m, k] = sum_n dy[m, n] * W8[n, k] * s[n]

and `s[n]` sits on the *contraction* index. Unlike the forward, where the weight
scale is per output column and folds into the epilogue, it cannot be applied
after the GEMM.

What shipped: `dequant_weight(w8, w_scale)` — dequantize, then undo the ConvRot
rotation, which works because the rotation is its own inverse (that is what makes
`(x R) @ (W R)^T == x @ W^T` true in the forward) — and
`linear_dx(dy, w)` = `dy @ w`. `Int8Linear.dense_weight()` caches it,
`Int8Linear.dx(dy)` uses it.

Evidence (`debug/verify_linear_dx.py`, N=8192, K=3072, M=4096, bf16):

| | |
|---|---|
| `linear_dx(dy, W)` vs autograd of `F.linear`, same `W` | **0** (same oneDNN GEMM) |
| `dequant_weight` vs the bf16 weight | 0.94 % (int8 0.90 %, bf16 cast 0.17 %) |
| `dx` with the dequantized weight vs the unquantized model | 0.97 % |
| one-time dequant | 2.6 ms / 25 M params → ~0.4 s for a 3.7 B model |

So the gradient a trainer gets is the gradient of the INT8 model it is training,
within bf16 storage of the weight. Nothing accumulates: adapters, gradients and
optimizer state stay fp32.

**Quantized gradients: not shipped, and they need a decision.** The fast variant
is exact in the weight. Since `s[n]` is a *row* scale on `W`, it is a *column*
scale on `dy`:

    dx = dy @ diag(s) @ W8 = (dy * s) @ W8

so `dy * s` can be quantized per row (`a[m]`), the GEMM is int8×int8 against the
weight as stored, and `dx = a[m] * (dy8 @ W8)`, rotated at the end. That keeps
the weight int8 (no 0.4 s dequant, 2x less memory than the bf16 copy) and adds
one quantizer's worth of noise to the gradient — exactly the tradeoff to A/B on
a real loss curve, same seed and data order, against the weight-only path. Do
not make it the default: a 1 % per-step gradient error is invisible in one step
and decisive over ten thousand, and the error that matters is the divergence of
the loss curve, not the per-element relative error.

- [x] `dequant_weight` + `linear_dx`, verified and measured (above).
- [x] the int8-GEMM variant, measured: **a memory option, not a speed one.**
      `debug/probe_linear_dx_variants.py`, M=4096, N=8192, K=3072: the path
      went from 6.63 ms (3.4x the bf16 path) to **2.75 ms (1.33x)** once the
      quantizer became a fused Triton kernel (`xpu_kernels.triton.quantize`:
      4.17 ms -> 0.27 ms, and 9 kernel launches become 1). The accuracy is
      unchanged: 1.35 % vs 0.93 % relative, cosine 0.999910 vs 0.999957.
      Keep it for the 2x weight-memory saving.
      Two things learned the hard way, both pinned by the selftest now:
      `quantize()` must not be reused (it rotates -- the rotation is already in
      the weight, so reusing it would silently rotate the gradient), and
      `dequant`'s `w_scale` is indexed by the *accumulator's columns* (K here,
      not N); passing the longer of the two only works because it reads the
      buffer's first K entries.
- [x] the quantizer is fused now, and the finding is worth keeping: **a
      bit-exact fused quantizer is not attainable at a useful speed on this
      card.** The reference chain divides in fp32, and on this backend torch's
      tensor÷tensor division is correctly rounded while its tensor÷scalar
      division is a reciprocal multiply -- the two disagree on 13 % of values.
      Triton's correctly rounded division (`libdevice.div_rn`) is 18x slower
      (6.3 ms vs 0.35 ms for 3974x8192), i.e. slower than the chain it would
      replace. So the kernel uses the fast division, which moves 2-3e-4 of the
      quantized values by one unit -- **9.1e-4 relative on the gradient**,
      against the path's own 1.35 % error -- and the gate compares with a
      relative budget (2e-3) while the formula is pinned in float64 by
      `tests/test_quantize.py`. The row *scale* is exact. The gate found the
      whole thing on the kernel's first run ("73 of 721 984 values differ").
      `docs/performance.md` has the measurements.
- [x] `dW`: **`linear_dw(dy, x)` -- and the surprise is how little there is
      to it.** The INT8 forward is *mathematically* the bf16 forward with the
      dequantized weight: the ConvRot rotation is orthogonal, so it cancels in
      the contraction, and the activation's quantization is exact for the same
      reason. So the weight gradient of the INT8 model is the bf16 model's:

          dL/dW = dy^T @ x

      No scale, no rotation, and -- unlike `linear_dx` -- no dequantized weight
      at all: the weight does not appear in its own gradient. Verified
      bit-identical to `torch.autograd` of the dequantized weight
      (`debug/verify_linear_dw.py`, rel 0.00e+00).
      What the quantization *does* change is the gradient's value (the forward
      used the dequantized weight), which is the straight-through convention
      the dx path already uses: 0.9 % from the unquantized model.
- [x] the INT8 linear is now **an autograd op** (`int8.int8_linear`,
      `Int8Linear`), which it was not: `linear()` returns a tensor with no
      `grad_fn`, so an INT8 layer dropped into a training graph produced **no
      gradient at all, silently**. The op's backward is `linear_dx` (the
      weight is dequantized for that one layer's backward and dropped again --
      the memory the quantization saved stays saved) or `linear_dx_int8` (the
      weight stays int8: no copy at all, one byte per parameter, 1.05 % from
      the bf16 gradient against 0.93 % for the dense path). The dense mode
      refuses inside a device-graph capture, because a captured dequant would
      be replayed on every step.
- [ ] for full fine-tuning the *parameter* is bf16 and the forward quantizes
      it per step, and that is where the remaining cost is:
      `prepare_weight` is a torch chain (3.0 ms per 25 M parameters, ~0.4 s
      for a 3.7 B model, per step). The C kernel would do it in one pass, but
      it is **not the same function**: it keeps the rotated values in fp32
      instead of rounding them to bf16 first, so the scale and 5.7 % of the
      values differ -- switching would change every model's weights (and the
      recorded PNG hashes), so it needs its own decision and its own gate
      rather than a quiet swap.

## 2. Device-graph capture of the step

The trainer needs the whole step capturable: no host sync, no fresh allocations
during the step, no shapes that depend on values.

**Verified for rope and adaLN** (`debug/verify_graph_capture.py`): forward and
backward captured in one `torch.xpu.XPUGraph` for each, replayed on new data,
bit-identical to eager in all eight tensors -- including adaLN's modulation
gradients, whose two-stage reduction allocates a partials buffer inside the
capture. Host time per stage 648-1282 us -> 40-82 us (8-16x).

What that changed in the plan:

- **Workspace threading is not needed.** Allocations inside a capture come from
  the graph's private pool and replay at the same addresses, so `torch.empty` in
  a launcher is fine as long as nothing inside the region waits on the device.
  `out=` remains useful for a *non-captured* step that wants stable addresses,
  not for capture.
- **The first call must be outside the capture**, because the gate probes on
  first use (it syncs and reads values back). Warm up or call `status()` first.
  Now a rule in `docs/adding-a-kernel.md`.

Still open:

- [x] the same capture verification for `adaln`, including both modulation
      gradients and the partials buffer its backward allocates inside the
      capture: bit-identical replay. Host cost 632 -> 40 us per stage.
- [x] the INT8 side, verified on the card
      (`debug/verify_capture_guards.py`): `Int8Linear.dx` captures and replays
      **bit-identically to eager** (one `torch.matmul` against the
      dequantized weight), and `linear_dx_int8` is `_int_mm` plus elementwise
      torch, which is torch's business inside a graph.
- [x] `dequant_weight` must not run inside a capture, and now it *cannot*:
      `Int8Linear.dense_weight()` raises if the current stream is capturing.
      The reason is money, not correctness -- the dequant is a full-precision
      copy of the weight (~0.4 s for a 3.7 B model), and a captured kernel is
      replayed on every step.
- [x] the fallback policy, decided and implemented: a captured step that
      silently fell back to the eager chain is slower than an eager step and
      looks like a regression nobody can explain, so the package now says so.
      `xpu_kernels.prepare_for_capture(model, need=...)` materializes the
      dense weights and latches the gates' verdicts;
      `require_capture_ready(...)` is the one-line check for the trainer's
      init and raises with the reasons. The gate itself refuses to probe
      inside a capture (the probe synchronizes), so the failure is a sentence
      about the gate rather than a synchronisation error from deep inside a
      capture. All of it is in `xpu_kernels/capture.py`, CPU-tested
      (`tests/test_capture.py`) and verified on the card.
- [x] **the fused entry points now consult their own gate.** `usable()` is
      where the gate is asked, and the caller is expected to call it; a caller
      that skipped the question got an *unverified* kernel, silently, with no
      error at all (found while verifying the guards:
      `debug/verify_capture_guards.py`). The entry points now ask the gate
      themselves (`Gate.require()`), so the promise holds whatever the caller
      does. It *raises* rather than falling back, and that is the deliberate
      choice: the fallback belongs to the caller, and the package's own
      reference is not necessarily bit-identical to it (ComfyUI's is
      `comfy_kitchen`'s, and the two differ in 173 of 12.2 M values), so
      quietly substituting it would change ComfyUI's numbers in exactly the
      case where the gate is closed. The backward keeps failing soft (the
      torch gradient *is* the eager chain's gradient), so a closed backward
      gate costs speed and not correctness. Pinned in `tests/test_gate.py`;
      the PNG gate is unchanged (`a2aacfda45409493a490b0402b61f850`).
- [ ] a captured *whole step* is the trainer's integration work, not this
      library's; the library's part is the per-kernel guarantee above.

## 3. Backward for `adaln.rms_adaln`

**Done**, including the fused reduction (`adaln._GATE_BWD`,
`debug/probe_adaln_backward.py`): `dx`, `dscale` and `dshift`, verified against
`torch.autograd` of the eager chain in fp32 (1.5e-7 worst relative over 1.59 M
values, both a single shared modulation vector and one per batch element) and the
formula pinned in float64 by `tests/test_adaln_backward.py` (autograd,
`gradcheck`, the norm's orthogonality property, and the reduction shapes for
every broadcast a trainer passes: `[1,1,D]`, `[D]`, `[B,S,D]`, `[1,S,D]`,
`[B,1,D]`).

Forward+backward 0.81 ms vs 5.14 ms eager (6.3×). The order was: 3.9× with two
torch reductions, 6.3× once they became two Triton kernels. It also made
`dscale`/`dshift` three orders of magnitude more accurate (1.7e-3 → 3.1e-7)
because the sums now stay in fp32 instead of going through a bf16 temporary --
worth remembering: the "fast path" was also the precise one.

Nothing open here. The next lever is not in this stage.

## 4. Switch ComfyUI to import this package

**Done for the Triton half** (ComfyUI `1fa5478b`). `comfy/xpu_rope.py` and
`comfy/xpu_adaln.py` are now re-export shims over `xpu_kernels.triton.*`; every
name is mirrored, private ones included, so `model.py` did not change. The
package is installed editable into that venv, so which code runs is decided by
the install and not by the working directory.

Verified end to end, because a shim is exactly the kind of change that can leave
a stale copy silently in place: seed 1234, 3 steps, `--deterministic`, fresh
server -> `moss-image_00001_.png` md5 `a2aacfda45409493a490b0402b61f850`,
identical to the pre-switch reference. The server log shows the three forward
gates verifying against `comfy_kitchen` on first use and the qkv dequant fusion
engaged; the two backward gates correctly stay unprobed under `no_grad`.

**The INT8 wrapper: also done.** `comfy/xpu_int8.py` was not a duplicate -- it
was a 717-line superset the package could not express. The missing entry points
now live in `xpu_kernels.int8`, and the ComfyUI module is a thin adapter over it
(layout unwrapping, the `None`-means-fallback convention, the per-prompt scale
memo). The two that turned out to be ComfyUI glue, not library code:

| entry point | where it went |
|---|---|
| `convrot_int8_linear` | `int8.linear(..., act=, residual=, residual_scale=, return_acc=)` -- one entry point for a projection |
| `convrot_ffn_situ` | `int8.ffn_situ` -- `down(situ(up(x)))` in two GEMMs, ~40 lines of composition |
| `convrot_int8_acc` | **stays in ComfyUI**: it only unwraps `TensorWiseINT8Layout` and passes `return_acc=True` |
| `dequant_epilogue_slice` | was already `int8.dequant_slice` (same C kernel, same strided walk) |

The four API questions, answered (`docs/architecture.md` has the reasoning):

* **`None` vs raise.** `unsupported_reason(...)` returns the reason string,
  `linear` raises it as `TypeError`, ComfyUI branches on the predicate. A caller
  can now tell "this card cannot do it" from "you passed the wrong dtype".
* **`_SCALE_CACHE`.** The library caches nothing; `expand_scale` is pure and
  takes the fast path (returns its input) when the scale is already a flat fp32
  `[n]` tensor. The prompt-scoped memo stayed in the ComfyUI adapter, next to
  the `reset_caches()` that `execution.py` calls. `prepare_weight` now returns
  an fp32 scale instead of a bf16 one -- the kernel reads fp32, and an 8-bit
  mantissa on the scale was quantizing the quantizer.
* **`XPU_INT8_TIMING`/`STATS`.** Removed. The per-region breakdown (rotate /
  `_int_mm` / epilogue) was scaffolding for an optimisation question that is
  closed; the adapter keeps a whole-call host timer and the `[xpu-mem]` line
  under the same env var, and `debug/` probes answer the rest.
* **Naming.** The module is `xpu_kernels.int8`, so the operations are bare:
  `linear`, `linear_acc`, `ffn_situ`, `dequant`, `quantize`. `convrot_*` stays
  only on the ComfyUI-side names that `comfy/ops.py` calls.

Verified: the package's own tests (`pytest`, 50 CPU-only), `python -m
xpu_kernels` on the card, ComfyUI's `tests-unit/xpu_int8/test_xpu_int8_linear.py`
(the fused path vs comfy_kitchen vs a float64 reference, plus the fallback and
residual cases), and the PNG gate -- seed 1234, 3 steps, `--deterministic`,
fresh server: md5 `a2aacfda45409493a490b0402b61f850`, unchanged. The server log
shows the whole model on the fast path with no fallbacks.

Two notes for the next person:

* The `.so` ComfyUI now loads is `xpu-kernels/build/libxpuint8.so`, not
  `ComfyUI/comfy/lib/libxpuint8.so`: the package's loader prefers the repo it is
  installed from. The two sources differ only in the include path, but they are
  separate builds -- the PNG gate above ran against the package's. `build.sh` in
  this repo is now the build that matters.
* `execution.py` reads `comfy.xpu_int8._LIB` and calls `reset_caches()` at prompt
  end. Both are part of the adapter's contract; removing either deletes the
  per-prompt cache clear silently. There is a `try`-less attribute read, so it
  fails loudly (it did, during this change).

## 5. Packaging and CI

- [x] `.github/workflows/ci.yml`: a CPU job (ubuntu-latest, python 3.10 and
      3.12) that installs torch from the CPU index, `pip install -e . --no-deps`,
      checks that the package imports **without** `libxpuint8.so`
      (`available()` must be False, not an exception), and runs `pytest`. All 40
      tests are CPU-only by design, so this job is the contract for the reference
      formulas and the gate rules.
      A second job runs `./build.sh` + `python -m xpu_kernels` on a self-hosted
      `[self-hosted, xpu]` runner, gated by the repository variable
      `XPU_RUNNER=yes` so a clone does not sit with a permanently queued job.
      Needed on that runner: oneAPI 2025.3, `LEVEL_ZERO_V1_SDK_PATH`, and the
      INT8 self-test tolerance (it is not bit-exact by nature).
- [x] PyPI name: **`xpu-kernels` is free** (the earlier check returned 503; the
      JSON API answers 404 now, which means no project by that name). The GitHub
      repo is `okolenmi/oko-xpu-kernels`, so the dist name and the repo name
      differ -- decide which one to publish under before the first upload.
- [x] `./wheel.sh` builds a wheel that *carries* `libxpuint8.so` (there is no
      way to install it without a DPC++ toolchain -- the point is that the
      consumer does not need one), plus `--sdist` for a source-only archive.
      Staging matters: the `.so` goes into `xpu_kernels/int8/` because that is
      where the loader looks first, and `wheel.sh` removes it again on exit
      because a staged copy shadows `build/libxpuint8.so` for every later
      in-tree run. `setup.py` exists only to stop bdist_wheel tagging the result
      `py3-none-any`: it is `py3-none-linux_x86_64` (an ELF, but no CPython
      extension -- the ABI is torch's), and it is *not* manylinux because
      `libsycl.so.8` is linked, not bundled.
      `MANIFEST.in` puts `src/`, `include/` and `build.sh` in the sdist (it
      carried only Python before, so the documented "run ./build.sh" was
      impossible for an sdist consumer), and keeps a `.so` out of it.
      Verified end to end: extract the sdist, `./build.sh`, `python -m
      xpu_kernels` on the card (every gate active); `pip install` from the sdist
      reports `available() == False` until a `.so` is built. Install the wheel
      into a scratch prefix: import, `available()` True, `linear` runs, and the
      loader picks the bundled `.so` rather than the repo's. That check is now a CI step on the XPU runner
      (which also gets `pip install -e .` -- it was missing, so the device job
      could not have imported the package on a clean runner).
- [x] Publish path settled: PyPI **trusted publishing** (no token anywhere) from
      `.github/workflows/release.yml`, environment `pypi`, dist name
      `xpu-kernels` under repo `okolenmi/oko-xpu-kernels`. Both artifacts go to
      PyPI -- the sdist alone would leave `pip install xpu-kernels` with nothing
      to compute with -- and the same bytes are attached to the GitHub Release.
      **The version stays plain** (`0.2.0`): PEP 440 says PyPI *must not* accept
      local version identifiers, so the earlier `0.2.0+oneapi2025.3` plan is
      impossible; the toolchain moved to the tag (`v0.2.0-oneapi2025.3`) and the
      release notes. Checklist in `docs/publishing.md`; metadata gained PEP 639
      `license`/`license-files`, `[project.urls]` and classifiers, and
      `twine check` passes on both artifacts.
      The wheel is built **in CI on a GitHub-hosted runner**, not on a
      self-hosted one: the `.so` needs a DPC++ compiler, oneAPI 2025.3.2 is an
      apt package, and no runner is registered (the job would have queued
      forever). The compiler version is *pinned* by an apt preference and
      asserted, because the `.so` links `libsvml`/`libirng`/`libimf` and a
      consumer with a different oneAPI patch release may lack the symbols --
      fail closed rather than publish something that cannot load. Rehearse with
      `workflow_dispatch` (builds, publishes nothing), then tag.
      Remaining: move the tag onto the current commit (the wheel's platform
      tag was wrong, and a re-run of the failed job would use the workflow at
      the tag's commit), then re-run the PNG gate against the *published*
      wheel, not only the local build.
      **PyPI does not accept the platform tag `linux_x86_64`** -- it answered
      "400 Bad Request: Binary wheel ... has an unsupported platform tag" --
      so the wheel is tagged `manylinux_2_35_x86_64` and the CI job builds on
      ubuntu-22.04 (glibc 2.35) to match, checking the `.so` with `objdump -T`
      so the tag cannot claim a glibc the payload does not need. The tag is
      not a promise of self-containment: the `.so` links `libsycl.so.8`
      unbundled, like Intel's own runtime wheels. Accepted tags are in
      `warehouse/utils/wheel.py`.
- [ ] `xpu-kernels bench` -- one CLI over the probes in `debug/` (they all take
      their shape as a constant today). Low value until someone else needs to
      reproduce a number.

## 6. Opportunities measured, not taken

- [x] **the rotate's register residency -- closed, and not buildable.** The
      rotate is 3.1x off its floor because the row's abs-max is over the
      *rotated* row, so every element of a row waits for every other element
      of that row to be transformed: the load, the transform, the scale and
      the store are serialised per row, whatever the register pressure is.
      The last open shape (a group split across 4 lanes, one cross-lane axis)
      was measured rather than guessed
      (`debug/native/probe_cross_lane.cpp`): the exchange is *cheap* -- 84 ns
      per 512-bit move, 36 per lane per row at K=3072, ~3 us against a 250 us
      prize, and the "16 `select_from_group` calls" figure overstates it by
      3x -- but splitting the groups makes the row's scale cross sub-groups
      (3 at K=3072, 8 at K=8192), which means a barrier (1.33 ms) or a second
      pass (0.75 ms), both worse than the 0.402 ms the rotate costs now.
      Keeping the 16-lane sub-group and giving each lane three quarter-groups
      only drops the pressure from 16 registers to 12, which is not what the
      interleaving needed. So **INT8 stays at parity with BF16**, and the gap
      is a property of the format, not a missing kernel. The only route left
      is a scale per *block* instead of per row -- which changes the numbers,
      so it is a design decision with its own gate, not an optimisation.
- [x] **a fused NF4 dequantizer** (trainer-facing). `xpu_kernels.triton.nf4`:
      the QLoRA storage format (16 normal quantiles, a per-block absmax, two
      4-bit indices per byte) dequantized in one kernel. 100 M parameters:
      **20.75 ms -> 0.66 ms (31.5x)**, 2.6 bytes of traffic per parameter
      against the chain's ~40, and **no scratch at all** -- the chain holds
      ~16 bytes per parameter live (the int64 index tensor alone is 8), so its
      peak can exceed the bf16 weight the format was meant to replace. It is
      **bit-exact** against the chain, in fp32 and at the production bf16 --
      unlike the int8 row-quantizer, this arithmetic has no division and no
      rounding mode to get wrong. It is also capture-safe (one allocation, the
      output). The vision document, with the memory arithmetic and the
      comparison against the int8 base, is in the trainer's tree
      (`docs/nf4-storage-vision.md`).
      Still open there: the *double quantization* (the absmax values quantized
      a second time) is a separate small pass, and the trainer's own
      `materialize()` is the reference the kernel is compared against.
- [ ] **the residual add of a pre-norm block, folded into the norm's
      epilogue.** The last open shape in the bf16 path: one launch and one
      read-write pass per add (~84 launches and ~6 ms of device time per step
      in a 42-block model, in a step that is host-bound). Training-safe --
      the add's backward is the identity.
- [ ] **`reduce_over_group` row max in the FFN epilogue: 16 % of the kernel,
      ~6.4 ms/step, not shipped.** The construct used to kill the host process
      (twice); a re-test on 2026-10-21 no longer reproduces it (~35 000 launches,
      both toolchains) and the variant is bit-identical and 0.836x the
      production time interleaved. It stays out because a gate cannot catch a
      `SIGSEGV`: the process dies instead of returning a wrong number, and 0.6 %
      of a step does not buy that risk. The probe keeps it behind
      `PROBE_REDUCE=1`, so the decision can be revisited in one command after a
      driver bump: `PROBE_REDUCE=1 ./debug/native/bin/probe_ffn_situ 3973 8192 device`.
      Revisit if the driver that "fixed" it gets a changelog or a filed issue
      (this is worth reporting: a compiler/runtime bug that silently vanished).
      **The report is written**: `docs/report-reduce-over-group.md` has the
      text ready for the tracker, with the reproducer's build and run commands
      and the three questions worth asking (was it known, is the fix in the
      current driver or just dormant, and can the runtime throw instead of
      killing the process). It needs sending -- the tracker is on the user's
      side of the wall.
      The activation quantizer has the same row max and was not measured.

## Dead ends, recorded so they are not retried

- Attention: every XPU SDPA backend is the same oneDNN kernel.
- oneDNN GEMM: the card's int8 rate is the ceiling.
- A dequant epilogue as a *kernel*: the win is less traffic, not more math.
- Rope with a different `num_warps`: 1.6× faster, 36/12.2M values differ.
- Sampler-loop stages: none of the remaining ones are bandwidth-bound.
