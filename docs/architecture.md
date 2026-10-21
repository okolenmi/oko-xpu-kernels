# Architecture

## What ConvRot is

ConvRot (as used by the MossImage INT8 checkpoints) stores weights that have
been **pre-rotated** by a Kronecker-Hadamard transform before quantization. At
inference the activation has to be rotated the same way, then quantized per row
to int8, then multiplied by the int8 weights.

So a ConvRot INT8 linear is three things:

```
x [M,K] bf16
  -> rotate along K in blocks of 256          (H = kron(H4,H4,H4,H4)/16)
  -> per-row abs-max -> row scale -> int8     (per-row symmetric, 7 bits)
  -> int8 GEMM with the pre-rotated int8 weights
  -> dequantise + bias
```

The BF16 path is the *same thing minus the quantization*: the dequantized
weights are already rotated, so only the activation needs rotating.

That symmetry is why the two paths are directly comparable, and why comparing
against a plain `F.linear` on pre-rotated weights (which is what an early
version of `bench_pipeline.py` did) is not a fair comparison.

## The rotation, and how it maps onto the hardware

```
H      = kron(H4, H4, H4, H4) / 16
H4     = [[ 1, 1, 1,-1],
          [ 1, 1,-1, 1],
          [ 1,-1, 1, 1],
          [-1, 1, 1, 1]]
```

Each butterfly applies `H4/2`; four of them give `(H4/2)^4 = 1/16`, which is the
`1/sqrt(256)` normalisation. So the whole transform is **adds, subtracts and
sign flips — no multiplies by anything but 0.5**.

A 256-element group is held in 16 `vec<float,16>` registers (16 zmm) with
element `i = 16w + q`, which gives four base-4 digits:

| digit | source | shape of the butterfly |
|---|---|---|
| `i0 = q & 3` | lane index, low 2 bits | in-register, stride 1 |
| `i1 = (q>>2) & 3` | lane index, high 2 bits | in-register, stride 4 |
| `i2 = w & 3` | register index, low 2 bits | across registers `{4g..4g+3}` |
| `i3 = w >> 2` | register index, high 2 bits | across registers `{c, c+4, c+8, c+12}` |

Axes 0 and 1 are therefore per-register and need no cross-register movement:
4 broadcast-swizzles plus 4 sign vectors each, and because the arithmetic is
`+/-` of sign-flipped inputs, `0.5 * (s0*a + s1*b + s2*c + s3*d)` folds the
0.5 into the sign vectors.

**Why this assignment matters.** The Kronecker factors are all the same matrix,
so the digit-to-(register, lane) assignment is free — any permutation of the
four digits gives the same transform. That freedom is what makes 2 of the 4
axes free (cross-register, elementwise) and the other 2 in-register. What it
does *not* let you do is make all four free: exactly two digits must live in
the register index and two in the lane index, and one of the two register-index
axes is always strided. The strided axis-3 butterfly is why every scheme below
needs all 16 registers live at once.

`H4` provably does not factor into signed `H2` butterflies (exhaustive search,
depths 3 and 4), so the butterfly form above is the cheap one.

### Butterfly forms

4-point `H4/2` butterfly:

```
a = u0+u1, b = u0-u1, c = u2+u3, d = u2-u3
v0 = (a+d)/2, v1 = (a-d)/2, v2 = (b+c)/2, v3 = (c-b)/2
```

In-register axis, written as an accumulation chain (the one-expression form
needs four live product temporaries and spills):

```
t = s0 * swz(v) ;  t += s1 * swz(v) ;  t += s2 * swz(v) ;  t += s3 * swz(v)
```

Total: ~1.75 vector ops per element, i.e. 0.034 ms of issue time for the whole
M=4096 rotation. **The transform is not the cost.**

## The one structural constraint

The row abs-max forces all 256 values of a group to stay live:

```
load 16 zmm  ->  4-axis transform  ->  row abs-max  ->  sub-group reduce  ->  quantize -> store
```

Nothing may be stored before the scale is known, so all 16 loads must complete
before the first store, and there is nothing to overlap them with. Measured
cost of that constraint, from `probe_rotate_floor_analysis`:

```
same copy loop, 16 registers held live   0.402 ms
same copy loop, load/store interleaved    0.128 ms
```

3.1x, from register residency alone. The sub-group reduce itself measures
**0.000 ms** — it is the serialisation it forces, not the reduce.

The bf16 rotate kernel has no such dependency, and indeed it reaches 160 GB/s
on 50.4 MB versus 74 GB/s for rotate+quantize on 37.8 MB.

## Kernels

| kernel | file | job |
|---|---|---|
| 1 | `src/xpuint8.cpp` `launch_convrot_quant` | bf16 [M,K] -> int8 [M,K] + fp32 [M] scales |
| 2 | `src/xpuint8.cpp` `launch_rotate_bf16` | bf16 [M,K] -> bf16 [M,K], rotation only |
| 3 | `src/xpuint8.cpp` `launch_dequant_epilogue` | int32 [M,N] -> bf16 [M,N], scales + bias |

One work-item per 16-lane sub-group; one sub-group owns one row; lane `l` owns
ConvRot group `l`. Kernels 1 and 2 are templated on how many groups fit in a
sub-group (`G = ceil(K/256/16)`), so the register indices stay compile-time —
a runtime-indexed `W[g][w]` gets demoted to scratch memory and costs ~35%.

**kernel 2 stores in a flat loop, deliberately.** Interleaving the axis-3
butterflies with the stores is 1.44x slower *and* produced wrong results
(3.3e-02 error vs float64 instead of 3.1e-09). Four butterflies, then a
straight 16-register convert+store loop.

## DPAS layout (for the hand-written GEMM)

```
dpas<8, 1, int, s8, s8, 512, 32>(B, A)
```
computes 16 output columns, each a 32-element int8 dot product. All 32 lanes of
A are consumed in order: `A[i]` pairs with `B[ACTIVE_B[i]]` where `ACTIVE_B` are
the *even* VNNI groups (group stride 128 bytes, 4 bytes per group):

```
ACTIVE_B = {0,1,2,3, 64..67, 128..131, 192..195,
            256..259, 320..323, 384..387, 448..451}
```

An early version used `BN=256`, which silently computed only half the columns.

ESIMD also exposes two forms `int8_gemm_v3.cpp` does not use:

```
dpas<8, R, T, CT, BT, AT, BP, AP, N, BN, AN>(C, B, A)  ==  C + A x B
```
with `AN == R*32` bytes of A holding R rows, and
`ResultN == R * ExecutionSize`. That is `kernels/int8_gemm_v4.cpp`. It cuts
instructions per unit of work by 2.5x and is nonetheless **slower** (28 vs 43
TOPS at R=8) because the 8-zmm accumulator operand costs more than it saves.
It is kept because it is the right place to start if the GEMM is ever revisited.

This ESIMD build has **no arithmetic operators on `simd<>`** (only `dpas`), so
accumulators are `vec<int32_t,16>` and operands/results are bridged with
`memcpy`, which the compiler turns into register moves.

## Why the GEMM is not used in the pipeline

Our GEMM is **L2-bandwidth bound at ~4 TB/s**, not DPAS bound — TOPS is flat at
41-44 across every cacheable shape and collapses only when the working set
leaves L2. B is re-read once per row tile and A once per column tile: 4.8 GB +
2.4 GB at M=N=K=3072.

`torch._int_mm` reaches 151 TOPS (oneDNN, fully blocked and double-buffered).
Matching it needs exactly the tiling that is absent here, and even a generous
2x on B traffic lands at ~0.9 ms versus `_int_mm`'s 0.50 ms — not enough to pay
for the 0.15 ms epilogue the fusion would delete. So the pipeline uses
`torch._int_mm` and the GEMM stays a standalone, CPU-verified reference
implementation.
## The Python API, and why it is shaped this way

`python/xpu_kernels/int8/` has one file for the ctypes surface (`_lib.py`) and one
for the operations (`ops.py`). Four decisions that a library has to make, and
that a module inside an application does not:

**Unsupported is a reason, not a `None`.** `unsupported_reason(x, w8, w_scale,
act)` returns a specific string ("K=384 is not a multiple of the ConvRot group
size (256)") or `None`; `linear` raises `TypeError` with exactly that string.
ComfyUI keeps its per-layer fallback by branching on the predicate, and a
library user gets an error that says what to fix. One source of truth, so the
predicate and the exception cannot disagree.

**Only one entry point per projection.** `linear(x, w8, w_scale, bias=None,
act=None, residual=None, residual_scale=None, return_acc=False)` covers what the
kernels fuse: an activation inside the input quantizer, the pre-norm addcmul
after the epilogue, and the accumulator path that skips the epilogue entirely.
`ffn_situ` is the same idea one level up (`down(situ(up(x)))` in two GEMMs).
`N % 16` is part of the contract because the epilogue reads 16 columns at a time
and would otherwise drop the tail in silence.

**No state.** The library caches nothing: `expand_scale` is pure (and returns the
input unchanged when it is already a flat fp32 `[n]` tensor), so a hot loop sees
no hidden allocation the first time and a different one later. ComfyUI's
per-prompt scale memo lives in `comfy/xpu_int8.py`, which is also where
`reset_caches()` is called from `execution.py`.

**No instrumentation.** `XPU_INT8_TIMING` and the per-region `STATS` counters
were development scaffolding: they measured the caller's own host time and their
job is done. An optimisation question now goes to a probe in `debug/`.

## Verification: what makes these kernels safe to drop in

Every kernel here is a *schedule* change over arithmetic that already exists in
the caller. That is deliberate — a fused kernel that also changes the arithmetic
cannot be proven, only measured — and it gives one acceptance test: the kernel
must reproduce the bits of the chain it replaces, on this machine and in this
process.

* The INT8 side is verified by the suites in `debug/` (`verify_epilogue.py`,
  `verify_situ_fusion.py`, `verify_acc_rope.py`, ...) against the unfused chain,
  plus the pipeline-level PNG hashes recorded in `docs/performance.md`.
* The Triton side verifies itself at first use through `triton/_gate.py`: it runs
  the kernel and the caller's fallback on synthetic inputs at the production
  shape, compares bit patterns, and latches the verdict. A mismatch or an
  un-runnable probe disables the kernel for the rest of the process.
* `tests/test_gate.py` pins the gate's failure modes on CPU, so the verification
  machinery itself has a test that does not need a GPU.

The two rules that follow from this, and that are easy to get wrong:

1. **Do not tune a launch parameter that changes a reduction order.** For the
   Triton kernels that is `num_warps`; at 1 it is 1.6x faster and 36/12.2M values
   differ. The gate catches it, which is why it is a code change and not a knob.
2. **A backward pass is verified against `torch.autograd` of the eager chain**,
   as a pair with its forward (one probe, both outputs), with a float64
   gradcheck where an algebraic identity is involved. See
   `docs/adding-a-kernel.md`.

## Capturing a step

A device-graph capture freezes a region of the step and replays it. The
package's contribution to that is narrow and mostly about *not* being in the
way: the kernels are plain launches into the current stream, so they capture
like anything else.

Three things can go wrong inside a capture, and each of them is silent or
expensive rather than fatal. The package now refuses all three, because a
trainer that captures a step gets no second chance to notice.

* **The self-test.** Each fused kernel verifies itself once per process
  against a reference, and that probe synchronizes and reads values back --
  illegal during a capture. The gate therefore refuses to probe inside one
  (`triton/_gate.py`), and the fix is to let the verdict latch before the
  capture starts.
* **The fallback.** If the gate is closed, the eager chain runs instead.
  That is correct, and it is what nobody wants inside a graph: the rope
  backward allocates about ten tensors, and in a capture those come from the
  graph's private pool and are replayed with the graph.
* **The dequantized weight.** `Int8Linear.dx` needs the weight in bf16, which
  is a second, full-precision copy of every quantized weight (~0.4 s for a
  3.7 B model). Materializing it inside the capture is not a correctness bug --
  the copy is deterministic -- but it is re-computed on *every replay*.
  `Int8Linear.dense_weight()` refuses to do it there.

`xpu_kernels/capture.py` is the interface: `prepare_for_capture(model, need=...)`
does the work outside the capture, `require_capture_ready(...)` is the check to
call before capturing, and `capture_status(...)` returns the same information
as a list. The contract is verified on CPU (`tests/test_capture.py`) and on the
card (`debug/verify_capture_guards.py`), which also captures `Int8Linear.dx`
and replays it bit-identically to eager.

One hole is left open, and it is recorded in `docs/todo.md`: the fused entry
points do not consult their own gate. `usable()` is where the gate is asked,
and the *caller* is expected to ask it -- a caller that skips the question gets
an unverified kernel, silently.
