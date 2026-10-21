"""Fused ConvRot INT8 primitives for Intel Arc, on torch XPU tensors.

A thin, dependency-free (torch only) binding over ``libxpuint8.so``.  Nothing
here imports ComfyUI; the same library also drives ComfyUI's XPU INT8 path, and
this package is the supported way to use it from another project.

The point of the library is that the *activation* side of an INT8 GEMM is
fused into one kernel, and the *output* side can be consumed without a round
trip through bf16:

    bf16 [M,K] --rotate+quantize-------------------------> int8 [M,K] + fp32 [M]
    int8 [M,K] x int8 [N,K].T ------------------int_mm----> int32 [M,N]
    int32 [M,N] --dequant(+bias)--------------------------> bf16 [M,N]
                 --or dequant+activation+rotate+quantize--> int8 [M,K2] + fp32 [M]

The weights are stored pre-rotated and quantized per output channel, so an
inference path costs one GEMM plus two memory-bound kernels instead of the
eager bf16 rotation (bmm + abs + amax + div + round + clamp + cast) that torch
would otherwise run.

    import xpu_kernels, torch
    w8, ws = xpu_kernels.prepare_weight(w)              # once, offline
    y = xpu_kernels.linear(x, w8, ws)                   # bf16 [M,K] -> bf16 [M,N]
    y = xpu_kernels.linear(x, w8, ws, act="gelu_tanh")  # activation fused into
                                                       # the input quantizer
    y = xpu_kernels.ffn_situ(x, up8, up_s, down8, down_s)  # situ FFN, two GEMMs

``unsupported_reason`` is the non-raising form of the argument checks, for a
caller that falls back per layer instead of raising.

Bit-exactness contract
----------------------
``dequant(linear_acc(x, w8, ws)[0], ...)`` is bit-identical to ``linear(x, ...)``:
the fast path never rounds an intermediate that the slow path keeps in fp32,
and the dequant arithmetic is ``float(acc) * (w_scale[n] * x_scale[m])``
rounded to bf16, in that order.  ``linear`` is *not* bit-identical to a bf16
matmul -- it is an INT8 approximation by construction.  The fused
activation paths keep bf16 rounding where the unfused chain had it.

Constraints worth knowing before you rely on this
------------------------------------------------
* ``K`` must be a multiple of the ConvRot group size (256 for the specialised
  kernels).  ``N`` must be a multiple of 16 for the dequant epilogue, which
  reads the output 16 columns at a time -- both are checked, because the C side
  would otherwise silently drop the tail.
* Weights must be prepared with ``prepare_weight`` (or an equivalent
  ComfyUI-produced ``TensorWiseINT8Layout`` with ``convrot``); a plain
  ``w / amax * 127`` quantization is incompatible with the activation
  rotation.
* ``tanh`` inside the situ activation uses ``expm1(2x)/(expm1(2x)+2)`` because
  ``sycl::tanh`` has a data-dependent slow path on this card; it is
  bf16-identical to ``sycl::tanh`` for every finite bf16 input.
* Inference only.  There is no backward pass; the GEMM itself comes from
  ``torch._int_mm`` and the kernels are forward primitives.

Measured on an Arc B580 (M=3974, K=8192, N=3072), for sizing expectations:
stream read ~520-575 GB/s, read+write mixes ~280-330 GB/s, the dequant
epilogue ~283 GB/s of its 6 bytes/element, and the fused FFN kernel ~270 GB/s
of its 7 bytes/element.  Those are the floors the kernels sit on; a kernel
"faster" than them is a measurement error.
"""

from __future__ import annotations

from ._lib import available, device_name, load
from .ops import (
    Int8Linear,
    dequant,
    dequant_weight,
    dequant_weight_torch,
    dequant_act_quantize,
    dequant_slice,
    expand_scale,
    ffn_situ,
    int8_linear,
    linear,
    linear_acc,
    linear_dw,
    linear_dx,
    linear_dx_int8,
    prepare_weight,
    quantize,
    quantize_act,
    quantize_row,
    quantize_row_torch,
    rotate,
    unsupported_reason,
)
from .selftest import selftest

__all__ = [
    "Int8Linear", "available", "dequant", "dequant_act_quantize",
    "dequant_slice", "dequant_weight", "dequant_weight_torch", "device_name", "expand_scale",
    "ffn_situ", "linear", "linear_acc", "linear_dx", "linear_dx_int8", "load",
    "prepare_weight", "quantize", "quantize_act", "quantize_row",
    "quantize_row_torch", "rotate",
    "selftest", "unsupported_reason",
]
