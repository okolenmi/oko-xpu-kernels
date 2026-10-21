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
      `debug/probe_linear_dx_variants.py`, M=4096, N=8192, K=3072: 6.94 ms vs
      1.92 ms for the bf16 path. The GEMM is 1.85x faster; the torch quantizer
      costs 4.57 ms on top, and even a perfectly fused one (~0.45 ms) leaves the
      path at ~2.1 ms because the dequant epilogue and the final rotation eat the
      rest. Accuracy: 1.37 % vs 0.97 % relative, cosine 0.999906 vs 0.999953.
      Keep it for the 2x weight-memory saving only.
      Two things learned the hard way, both pinned by the selftest now:
      `quantize()` must not be reused (it rotates -- the rotation is already in
      the weight, so reusing it would silently rotate the gradient), and
      `dequant`'s `w_scale` is indexed by the *accumulator's columns* (K here,
      not N); passing the longer of the two only works because it reads the
      buffer's first K entries.
- [ ] if the memory saving is wanted, do not bother fusing the quantizer for
      speed; only a design that avoids rotating `dx` back can make int8 faster
      (i.e. keep the residual stream in the rotated basis -- a whole-model
      decision).
- [ ] `dW` (only if full fine-tuning is on the table): `dy^T @ x` with the
      *activation* scale on the other index; same analysis, opposite side.

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
- [ ] the INT8 side needs no capture test of its own: `linear_dx` is one
      `torch.matmul` and `linear_dx_int8` is `_int_mm` plus elementwise torch,
      all of which are torch's business inside a graph. What *is* ours is
      `dequant_weight`, which runs once before capture and must not be inside
      one -- it allocates a second copy of the weight.
- [ ] decide the fallback policy: `reference.rms_rope_backward` allocates ~10
      tensors and is not what anyone wants inside a capture. The gate is latched
      per process, so the practical answer is "check `status()` at startup, fail
      loudly if a kernel you intend to capture is inactive" -- a one-line check
      the trainer can put in its init.
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
      Remaining: create the `pypi` environment in repo settings (optional --
      GitHub creates it implicitly), push the workflow, tag `v0.2.0`, and
      re-run the PNG gate against the *published* wheel, not only the local
      build.
- [ ] `xpu-kernels bench` -- one CLI over the probes in `debug/` (they all take
      their shape as a constant today). Low value until someone else needs to
      reproduce a number.

## 6. Opportunities measured, not taken

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
      The activation quantizer has the same row max and was not measured.

## Dead ends, recorded so they are not retried

- Attention: every XPU SDPA backend is the same oneDNN kernel.
- oneDNN GEMM: the card's int8 rate is the ceiling.
- A dequant epilogue as a *kernel*: the win is less traffic, not more math.
- Rope with a different `num_warps`: 1.6× faster, 36/12.2M values differ.
- Sampler-loop stages: none of the remaining ones are bandwidth-bound.
