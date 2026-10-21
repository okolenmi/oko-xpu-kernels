# Adding a kernel

The package is small on purpose, and the shape of every addition is the same.
This is the checklist; the *why* is in the docstrings it points at.

## Where things go

```
python/xpu_kernels/int8/        C entry points: _lib.py (loader), ops.py (API), selftest.py
python/xpu_kernels/triton/      Triton kernels: one module per kernel + _gate.py
tests/                          CPU-only unit tests (no XPU needed, so CI can run them)
debug/                          the evidence for every claim: probes, A/B harnesses, verifiers
docs/                           measurements, dead ends, and this file
```

A kernel that needs a C entry point goes in `src/xpuint8.cpp` (the file is named
for the ABI, which is stable) plus `python/xpu_kernels/int8/ops.py`; a kernel
that can be written in Triton goes in `python/xpu_kernels/triton/`. Do not put a
reference model implementation here — the repo is kernels and instruments.

## The four contracts

Every fused kernel in this package has all four. A kernel without them is a
patch in somebody else's tree, not a library.

1. **A bit-exact gate.** Compare against the implementation being replaced, on
   synthetic inputs at the production shape, once per process, and latch the
   verdict. Use `triton._gate.Gate`; the contract is in its docstring and the
   failure modes are pinned in `tests/test_gate.py`. Numeric tolerance is not
   allowed: these pipelines are chaotic in their intermediates (one ULP in a
   norm scale flips a bf16 rounding and the image with it), and a training run
   accumulates the error into every weight.
2. **An escape hatch.** One environment variable per kernel,
   `XPU_KERNELS_NO_<KERNEL>=1`, checked at import, so an A/B or a
   "is this the thing that broke" bisect needs no code change. The old
   `MOSS_NO_*` names stay accepted for the ComfyUI copies.
3. **A reference injection point.** `set_reference(fn)` must let the caller
   supply the path they fall back to, because the gate is only meaningful
   against *their* implementation. `comfy_kitchen`'s kernel is the default when
   importable; with no reference at all the kernel stays inactive.
4. **A fallback that is visible.** `describe()` says `active`/`inactive` and
   why; `python -m xpu_kernels` prints every gate. A kernel that silently
   disables itself is worse than no kernel, because nobody finds out.

## Checklist

- [ ] Implement the kernel; keep the *arithmetic* of the replaced chain, change
      only the schedule. Comment anything that looks redundant but is
      load-bearing (a re-load, a rounding step, `num_warps`).
- [ ] Pin the launch configuration in a module constant (`NUM_WARPS`) and say why
      it cannot be tuned. Triton's reduction order follows the block layout, so a
      different warp count changes the output — that is a code change the gate
      must re-approve, not a knob.
- [ ] Write the probe: production shape, real dtype, strided layout if the
      consumer passes one, its own `torch.Generator` so a caller's noise stream
      is untouched. Return one pair per output (including a forward *and* a
      backward of the same op) — the gate fails if any pair differs.
- [ ] Register the gate: `gates()` in the module returns `{name: Gate}`;
      `triton/__init__.py` collects them and `status()` prints them.
- [ ] Add the evidence. If a claim lands in a docstring (an x factor, a
      bandwidth floor), the probe or A/B that produced it goes in `debug/` with
      the shape and the conditions.
- [ ] Run `python -m xpu_kernels` (everything verified) and `pytest tests/`.
- [ ] If the kernel is deployed in ComfyUI, re-run its paired image gate before
      believing the docstring.

## Backward passes

A backward is a kernel like any other, with two extra rules:

* **Verify the pair together.** The gate's probe returns the forward and the
  backward in one mapping, so a forward that matches and a backward that does
  not cannot pass as half-right.
* **Reference is `torch.autograd` of the eager chain**, not a hand-written
  derivative. The eager chain is the thing being replaced and it is
  differentiable by construction; a hand-written reference can be wrong in the
  same way the kernel is, and then the comparison proves nothing. Where an
  algebraic identity is involved, add a float64 gradcheck as well — it catches
  errors that a bit-exact match against a wrong formula would hide.

* **Prefer a reduction to an atomic.** A gradient that sums across rows cannot
  be a per-row kernel, and atomics make the result depend on execution order.
  The adaLN backward does it in two stages -- one program per (group, run of
  rows, column block) writes a partial, a second kernel adds the partials --
  which is deterministic and, measured, faster than the fp32 temporary the
  torch version needed (`docs/performance.md`).
* **Return the cheapest thing the next pass needs.** The same kernel first
  returned `v` (a whole tensor) so torch could multiply and reduce it; returning
  `rstd` (one scalar per row) instead removed a 24 MB write, a 24 MB read and a
  48.8 MB fp32 temporary, and made the sums more accurate. When a backward needs
  a cross-row quantity, check whether one number per row will do.

## Device graphs

A training step wants to be captured, so a kernel must not do any of these
*inside* the captured region:

* synchronise (`.item()`, `.cpu()`, `.tolist()`, an explicit `synchronize()`),
* read a device value to decide control flow,
* depend on a shape that comes from data.

Allocating is fine: allocations made during a capture are carved from the graph's
private pool and replayed at the same addresses (`torch.empty` in a launcher is
not a problem — `debug/verify_graph_capture.py` captures the rope forward and
backward and replays bit-identically).

The one thing that must happen **before** capture is the first call, because the
gate probes on first use: it runs the kernel against the reference, synchronises
and reads values back. Run `status()` (or one warmup iteration) first. A capture
that starts with an unprobed gate fails, and a gate that closes mid-capture
cannot fall back — so a trainer that captures should assert
`describe("name").startswith("active")` at startup rather than discover it later.

## What not to add

* A fused attention or GEMM. On this card every SDPA backend is one oneDNN
  kernel and `torch._int_mm` beats any hand-written GEMM (`docs/performance.md`);
  a fused op there buys nothing and costs a gate.
* A kernel whose win is below the noise floor. Measure with the harnesses in
  `debug/` (device time, interleaved A/B, idle desktop) before writing the gate.
* A second copy of a kernel already in ComfyUI. Move it here and import it, or
  leave it there; two copies drift.
