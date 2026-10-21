# The API

Four surfaces: the INT8 path (Python), the backward and the capture contract
(Python), the Triton kernels (Python), and the C library.

## The INT8 path

```python
import xpu_kernels

w8, w_scale = xpu_kernels.prepare_weight(w)     # bf16 [N,K] -> int8 [N,K] + fp32 [N]
y = xpu_kernels.linear(x, w8, w_scale, bias)     # bf16 [..,K] -> bf16 [..,N]
```

`prepare_weight` is the **only** supported way to make a weight: the rotation
is baked into the weight layout, so a plain `w/amax*127` quantization is
silently wrong.

When the consumer can walk the accumulator itself, skip the bf16 round trip:

```python
acc, x_scale = xpu_kernels.linear_acc(x, w8, w_scale)                    # int32 [M,N]
y     = xpu_kernels.dequant(acc, x_scale, w_scale, bias)                 # bf16 [M,N]
slice = xpu_kernels.dequant_slice(big_acc[:, 0:3072], xs, ws[:3072])     # strided, in place
q, s  = xpu_kernels.dequant_act_quantize(up_acc, xs, up_ws)              # dequant+situ+rotate+quantize
q, s  = xpu_kernels.quantize_act(x2, "situ")                             # activation fused in
```

`dequant(linear_acc(...))` is bit-identical to `linear(...)`; the fused
activation paths are bit-identical to the unfused chains they replace.
`python -m xpu_kernels` checks them; the exhaustive suites are in `debug/`.

Composed paths and per-layer fallbacks:

```python
y = xpu_kernels.ffn(x, up8, up_s, down8, down_s, act="geglu_erf")  # the fused FFN
reason = xpu_kernels.unsupported_reason(x, w8, w_scale)  # None, or why this shape cannot
model.some_linear = xpu_kernels.Int8Linear.from_linear(model.some_linear)
```

### The activations

| name | id | form | used by |
|---|---|---|---|
| `gelu_tanh` | 1 | the tanh approximation, on `[.., K]` | MossImage |
| `swiglu` | 2 | `silu(gate) * up`, on `[.., 2K]` as `[gate, up]` | — |
| `situ` | 3 | `tanh(g/4)·sigmoid(g)·tanh(h/25)`, on `[.., 2K]` as `[hidden, gate]` | MossImage |
| `geglu_erf` | 4 | `hidden * gelu(gate)`, on `[.., 2K]` as `[hidden, gate]` | **SDXL** |
| `gelu_erf` | 5 | the erf GELU, on `[.., K]` | **SDXL** (diffusers' shape) |

`xpu_kernels.ACTIVATIONS` is the map; `xpu_kernels.PAIRED_ACTIVATIONS` says
which of them read a pair of halves. Every one of them is **bit-identical to
the chain it replaces** — the erf GELU included, which was proved over all
65 536 bf16 values (`debug/native/probe_gelu_exact.cpp`).

### The fused FFN

```python
y = xpu_kernels.ffn(x, up_w8, up_scale, down_w8, down_scale, act="geglu_erf")
```

`up_w8` is `[2*ffn, K]` holding `[hidden | gate]` for a paired activation and
`[ffn, K]` for an unpaired one; `down_w8` is `[K, ffn]`. The up-projection's
int32 accumulator goes straight into the fused dequant+activation+rotate+
quantize kernel, so the `[M, 2*ffn]` bf16 tile never exists — the FFN is two
GEMMs and two kernels, not eight. The down-projection's epilogue is the
ordinary one, so `down_bias` is free, but the up-projection must be unbiased
(its bias would have to be added before the activation, inside the kernel).

Measured at SDXL's shapes, the fused path is **1.2-1.5× faster than the
unfused INT8 chain** (`debug/probe_sdxl_ffn.py`).

`unsupported_reason` returns a *reason*, not a `None`: "K=384 is not a multiple
of the ConvRot group size (256)". `linear` raises it as a `TypeError` with that
exact string, so a caller can tell "this card cannot do it" from "you passed
the wrong dtype".

The library is resolved from `$XPU_KERNELS_LIB` (or the legacy
`$XPU_INT8_LIB`), then `lib/libxpuint8.so` or `build/libxpuint8.so` next to
the repo, then `sys.prefix`.

## The backward

For `y = x @ W^T` the vjp is `dx = dy @ W` — no transpose, no bias. With an
INT8 weight, recover `W` once and keep it:

```python
w  = xpu_kernels.dequant_weight(w8, w_scale)   # bf16 [N,K], once per weight
dx = xpu_kernels.linear_dx(dy, w)              # bf16 [M,K]
```

or on a module, which caches it: `mod.dense_weight()`, `mod.dx(dy)`.

Why dequantize instead of an INT8 GEMM: `W[n,k] = W8[n,k] * s[n]`, so `s[n]`
sits on the *contraction* index of `dx = dy @ W` and cannot fold into an
epilogue (the forward gets away with it because its weight scale is per output
column). The fused dequantizer moves 3 bytes per parameter and is
bit-identical to the torch chain.

`linear_dx(dy, W)` is **exact** against autograd of the same GEMM (it is the
same oneDNN bf16 GEMM) and 0.9 % from the unquantized weight — that 0.9 % *is*
the INT8 weight, not the backward. There is also `linear_dx_int8(dy, w8,
w_scale)`, which keeps the weight in int8 and quantizes `dy` instead: 1.05 %
from the bf16 gradient, 1.33× the bf16 path's time, and 2× less weight
memory. It is a memory option, not a speed one.

The weight gradient is `linear_dw(dy, x) = dy^T @ x`, and nothing else: the
ConvRot rotation is orthogonal, so it cancels in the contraction and the INT8
model's weight gradient is the bf16 model's, bit for bit
(`debug/verify_linear_dw.py`).

### The INT8 linear, differentiable

`linear()` returns a tensor with no `grad_fn`, so an INT8 layer in a training
graph produces **no gradient at all, silently**. `int8_linear` is the autograd
op that prevents it:

```python
y = xpu_kernels.int8_linear(x, w8, w_scale, mode="dense")   # or mode="int8"
```

| mode | the backward | memory | gradient vs the bf16 model |
|---|---|---|---|
| `dense` | `dy @ dequant(w8, s)`, rebuilt for that one layer and dropped | 1 byte/param + one layer's copy | 0.93 % |
| `int8` | `dy` quantized, the weight never leaves int8 | 1 byte/param, no copy | 1.05 % |

The dense mode refuses inside a device-graph capture (a captured dequant would
be replayed on every step).

## Capturing a step

A device graph replays whatever was inside it, every step, forever. Two things
in this package must not happen inside one, and both refuse to:

```python
xpu_kernels.prepare_for_capture(model, need=("rms_rope", "rms_adaln"))
xpu_kernels.require_capture_ready(model, need=("rms_rope", "rms_adaln"))
xpu_kernels.capture_status(model)              # the same, as a list
```

`prepare_for_capture` materializes each `Int8Linear`'s dense (bf16) weight —
a full-precision copy of every quantized weight, ~0.4 s for a 3.7 B model — and
latches the fused kernels' verdicts, so that neither happens inside the
capture. `Int8Linear.dense_weight()` raises if the stream is capturing, and
the fused kernels' self-test refuses to run there (it synchronizes, which is
illegal during a capture).

`Int8Linear.dx` captures and replays **bit-identically to eager**
(`debug/verify_capture_guards.py`, `debug/verify_int8_capture.py`).

## The Triton kernels

Both kernels fuse a norm with the op that reads it, and both refuse to run
until they have proved themselves against the implementation you would
otherwise fall back to:

```python
import xpu_kernels.triton as kt

kt.set_reference("rope", my_rms_rope)          # once, before first use
q, kk = kt.rope.rms_rope_split_half(q, kk, freqs_cis, q_scale, k_scale)
x = kt.adaln.rms_adaln(x, scale, shift)

print(kt.status())   # rms_rope: active: verified against the reference (…), num_warps=4
```

If `comfy_kitchen` is importable, its kernels are the default reference and no
`set_reference` is needed. Without a reference the kernels stay inactive and
`status()` says why — a kernel whose numerics nobody has pinned down is not
worth the speed. `XPU_KERNELS_NO_ROPE=1` / `XPU_KERNELS_NO_ADALN=1` disable
them outright (the legacy `MOSS_NO_*` names still work).

`rms_adaln` is differentiable: one kernel for `dx` plus a two-stage reduction
for `dscale`/`dshift` (sums over the rows the modulation was broadcast across,
in the same pass, no atomics), 6.3× faster than the eager chain and verified
against `torch.autograd` at 1.5e-7.

`rms_rope_split_half` is differentiable too: with a grad-requiring input it
routes through an `autograd.Function` whose backward is a single fused kernel
(the per-head `dot` never leaves registers, so there is no scratch buffer and
nothing extra to launch). 0.62 ms forward+backward against 8.9 ms for the
eager chain, 14.3×, and the gradient is *closer* to the float64 truth than
`torch.autograd`'s (1.7e-3 vs 2.3e-3). `freqs_cis` and `scale` are treated as
constants; a trainer that learns them needs the differentiable chain in
`xpu_kernels.triton.reference`.

The two newer kernels have no ComfyUI counterpart, so they are called
directly:

```python
q, s = kt.quantize.quantize(x)          # the int8 row-quantizer (not bit-exact: see performance.md)
w    = kt.nf4.dequant_nf4(codes, absmax) # the NF4 storage format, bit-exact
```

A different `num_warps` changes the reduction order and the output; do not
tune it without re-running the gate.

## The C library

```c
xpuint8_convrot_quantize(x, q, scale, M, K);   // bf16[M,K] -> int8[M,K] + fp32[M]
acc = torch._int_mm(q, wt);                     // int8 GEMM, oneDNN
xpuint8_dequant_epilogue(acc, out, scale, ws, bias, M, N);  // -> bf16[M,N]

xpuint8_convrot_rotate_bf16(x, xr, M, K);      // bf16[M,K] -> bf16[M,K]
xpuint8_dequant_weight(w8, scale, out, N, K);  // int8[N,K] + fp32[N] -> bf16[N,K]
```

Every entry point has a `_stream` variant taking the queue pointer from
`torch.xpu.current_stream(dev).sycl_queue`, which is what the Python binding
uses; the non-stream form waits. `include/xpu_kernels/convrot.h` has the full
C API.
