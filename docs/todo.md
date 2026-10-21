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

**Verified for the rope stage** (`debug/verify_graph_capture.py`): forward and
backward for q and k captured in one `torch.xpu.XPUGraph`, replayed on new data,
bit-identical to eager in all four tensors. Host time per stage 648-694 us ->
73-81 us (~9x).

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

- [ ] the same capture verification for `adaln` (forward; backward once item 3
      lands) and for the INT8 `linear_dx` GEMM.
- [ ] decide the fallback policy: `reference.rms_rope_backward` allocates ~10
      tensors and is not what anyone wants inside a capture. The gate is latched
      per process, so the practical answer is "check `status()` at startup, fail
      loudly if a kernel you intend to capture is inactive" -- a one-line check
      the trainer can put in its init.
- [ ] a captured *whole step* is the trainer's integration work, not this
      library's; the library's part is the per-kernel guarantee above.

## 3. Backward for `adaln.rms_adaln`

Same shape of work as the rope backward: `dx`, `dscale` and `dshift` (a trainer
that learns the modulation needs all three; adaLN modulation params are
per-block, so their gradients are a column reduction). Gate: `torch.autograd` of
the eager chain in fp32, plus float64 gradcheck; the formula is much simpler than
rope's (no reduction over the row beyond the norm).

## 4. Switch ComfyUI to import this package

`comfy/xpu_rope.py` and `comfy/xpu_adaln.py` are copies. Replacing them with
imports needs the gate to be exercised in the server and the reference PNGs
re-verified (`seed 1234: a2aacfda…`, `seed 2345: 2cc08eae…`). No perf change
expected; the point is one copy.

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
