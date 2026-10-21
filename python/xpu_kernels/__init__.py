"""xpu-kernels — fused kernels for Intel Arc (Xe2) XPU tensors, plus the
instruments used to prove them.

Two halves:

``xpu_kernels.int8``
    A SYCL library (``libxpuint8.so``) and a torch-only binding for ConvRot
    INT8: fused rotate+quantize on the activation side, a fused
    dequant+activation epilogue on the output side, and accumulator paths that
    let a consumer walk an int32 GEMM result without a bf16 round trip.
    Inference only.

``xpu_kernels.triton``
    Single-pass Triton kernels for the non-quantized path: ``rms_norm + rope``
    and ``rms_norm + adaLN modulation``.  Each is bit-identical to the multi-kernel
    chain it replaces, and each verifies that once per process against a
    reference implementation before it is used.

The flat namespace below is the INT8 API (the same names are available as
``xpu_kernels.int8.<name>``).  Run ``python -m xpu_kernels`` to self-test the
installed library on this machine.
"""

from __future__ import annotations

from .int8 import (
    Int8Linear,
    available,
    dequant,
    dequant_act_quantize,
    dequant_slice,
    dequant_weight,
    device_name,
    linear,
    linear_acc,
    linear_dx,
    load,
    prepare_weight,
    quantize,
    quantize_act,
    rotate,
    selftest,
)
from . import int8, triton

__version__ = "0.2.0"

__all__ = [
    "Int8Linear", "available", "dequant", "dequant_act_quantize",
    "dequant_slice", "dequant_weight", "device_name", "linear", "linear_acc",
    "linear_dx", "load", "prepare_weight", "quantize", "quantize_act", "rotate",
    "selftest",
    "int8", "triton",
]
