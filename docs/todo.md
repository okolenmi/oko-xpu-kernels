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

**Not done: the INT8 wrapper.** `comfy/xpu_int8.py` is not a duplicate and cannot
be a shim -- it is a superset (717 lines) with model-shaped entry points the
package does not have:

| ComfyUI entry point | what it is |
|---|---|
| `convrot_int8_linear` | ConvRot linear with residual / input-act / return-acc options, falls back by returning None |
| `convrot_ffn_situ` | `situ(up_gate(x)) @ down.T` in one pass, consuming the up-projection's int32 accumulator |
| `convrot_int8_acc` | the GEMM without an epilogue |
| `dequant_epilogue_slice` | bf16 dequant of a column slice of a wider accumulator (the v third of fused qkv) |

- [ ] **move those four into `xpu_kernels.int8`**, plus the timing/stat
      scaffolding (`STATS`, `timing_stats`, `reset_caches`) if it is wanted
      outside ComfyUI. Only then can `comfy/xpu_int8.py` become a shim. They are
      the reusable API — a trainer swapping in an INT8 linear wants
      `convrot_int8_linear`'s signature, not `linear` plus five decisions. This
      is a real refactor, not a re-export: the C ABI is already shared, so it is
      only wrapper code that is duplicated today.

## 5. Packaging and CI

- [ ] a GitHub Actions job that builds `build/libxpuint8.so` with oneAPI and runs
      `pytest tests/` (the gate tests need no GPU) plus `python -m xpu_kernels`
      on a self-hosted XPU runner when one exists.
- [ ] PyPI: confirm `xpu-kernels` is free (the earlier check returned 503), then
      `pip wheel` in CI.
- [ ] `xpu-kernels bench` — one CLI over the probes in `debug/` (they all take
      their shape as a constant today). Low value until someone else needs to
      reproduce a number.

## Dead ends, recorded so they are not retried

- Attention: every XPU SDPA backend is the same oneDNN kernel.
- oneDNN GEMM: the card's int8 rate is the ceiling.
- A dequant epilogue as a *kernel*: the win is less traffic, not more math.
- Rope with a different `num_warps`: 1.6× faster, 36/12.2M values differ.
- Sampler-loop stages: none of the remaining ones are bandwidth-bound.
