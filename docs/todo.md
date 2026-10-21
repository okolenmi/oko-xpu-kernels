# TODO

Ranked by what unblocks the most. Every item names the evidence that would close
it; "done" means measured, not implemented.

## 1. INT8 backward for training (`dL/dx = dL/dy @ W`)

The library is inference-only today: `linear` computes `y = x @ W^T` with an int8
weight and an int8 activation. Training needs the vjp, `dx = dy @ W`, and the
weight gradient (`dW = dy^T @ x`) if the weights are not frozen.

**dx first, weight-only dequant.** The reason is the scale layout, and it is not
a detail: `W[n, k] = W8[n, k] * s[n]`, so

    dx[m, k] = sum_n dy[m, n] * W8[n, k] * s[n]

and `s[n]` sits on the *contraction* index. Unlike the forward (where the
weight scale is per output column and folds into the epilogue), it cannot be
applied after the GEMM. So the first version dequantizes the weight
(`dequant(w8, w_scale)`, already in the library) and runs a bf16/oneDNN GEMM;
a quantized-`dy` version is a *second* step, and only if it is A/B-able against
this one.

Why this is affordable in a LoRA trainer: `W` is frozen, so the dequantized copy
is paid once, not per step, and the LoRA path `dy @ (W0 + B A)` adds one small
GEMM. For full fine-tuning the bf16 copy is the memory question, not the kernel.

**Then, optionally, quantized gradients.** Quantizing `dL/dy` adds noise the
optimizer sees directly. Do not ship it as a default; A/B it on a real loss
curve (same seed, same data order) against the weight-only version, and report
the divergence — a per-step gradient error of 1% is invisible in one step and
decisive over ten thousand.

- [ ] `dequant` a weight once, keep it, `dx = dy @ w_bf16` — correctness against
      autograd of the bf16 linear (float64 gradcheck), device time vs bf16
      `matmul`.
- [ ] the same with `dy` quantized per row; A/B the training curve.
- [ ] `dW` (only if full fine-tuning is on the table): `dy^T @ x` in int8 needs
      the *activation* scale on the other index; same analysis, opposite side.

## 2. Device-graph capture of the step

The trainer needs the whole step capturable: no host sync, no fresh allocations
during the step, no shapes that depend on values.

What the Triton kernels need:

- [x] `rope._rows_bwd_kernel(..., out=)` allocates nothing when `out` is given.
- [ ] thread a preallocated `dx` (and eventually `out`) through
      `_RmsRope.apply`, so the autograd path allocates nothing either. An
      `autograd.Function` can take a non-differentiable workspace tensor and
      return `None` for its grad; the wrapper then keeps one buffer per shape.
- [ ] make the *fallback* explicit: `reference.rms_rope_backward` allocates
      ~10 tensors. If the gate is closed, the step cannot be captured — decide
      before capture, and fail loudly rather than mid-capture.
- [ ] audit for hidden syncs: no `.item()`, no `.cpu()`, no `.tolist()` in any
      launcher (the Python gates do not sync, but they do compile on first use —
      run `status()` before capture).
- [ ] the same audit for the INT8 entry points: they enqueue on the caller's
      stream and never read back today; keep it that way.

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
