"""The INT8 weight dequantizer as a Triton elementwise pass and one oneDNN product: ``int8 [N, K]`` + ``fp32 [N]`` -> the unrotated ``bf16 [N, K]``.

``xpu_kernels.int8.dequant_weight`` multiplies by the row scale, rounds to bf16 and undoes the ConvRot rotation of every
256-element group in one SYCL kernel, at 57-150 GB/s of a 397 GB/s ceiling (a lane holds a whole group in 16 registers and
the int8 unpack is scalar).  The rotation is a multiplication of each group by a fixed 256x256 matrix, and ``torch.mm`` does
that at ~40 TFLOPS on this card (0.08 ms for a 1280x5120 weight); the dequantization before it is bandwidth-bound and a
Triton elementwise kernel runs it at the copy ceiling.  Two passes (3 + 4 bytes per parameter) instead of one, each at its
floor.

The matrix is the library's own rotation applied to the identity, so whatever the C kernel rotates by, this does; the rotation
is an involution, which makes the matrix its own transpose.  The arithmetic differs from the butterfly's in the order of the
additions, so the result is not bit-identical: a rotated value within an fp32 rounding error of a bf16 tie rounds the other
way.  The gate compares with a relative budget.

``XPU_KERNELS_NO_DEQUANT_TRITON=1`` switches it off.
"""

from __future__ import annotations

import logging
import os

import torch

from .._capture import is_capturing
from ._gate import ACCELERATORS, Gate, probe_device

log = logging.getLogger(__name__)

__all__ = ["dequant_weight", "gates", "prepare", "scale_to_bf16", "set_reference", "usable"]

GROUP = 256
BLOCK = 1024
NUM_WARPS = 4

_kernel = None
_import_error = None
_matrices: dict[torch.device, torch.Tensor] = {}

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _scale_kernel(w8_ptr, scale_ptr, out_ptr, k, blocks, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        row = (pid // blocks).to(tl.int64)
        col = (pid % blocks) * BLOCK + tl.arange(0, BLOCK)
        mask = col < k
        w = tl.load(w8_ptr + row * k + col, mask=mask, other=0)
        s = tl.load(scale_ptr + row)
        # the chain rounds the scaled value to bf16 before the rotation, so the rotation's input is this rounded value
        tl.store(out_ptr + row * k + col, (w.to(tl.float32) * s).to(tl.bfloat16), mask=mask)

    _kernel = _scale_kernel

except Exception as exc:  # noqa: BLE001 - no triton means no kernel
    _import_error = exc


def enabled() -> bool:
    return _kernel is not None and os.environ.get("XPU_KERNELS_NO_DEQUANT_TRITON", "") in ("", "0")


def prepare(device) -> torch.Tensor:
    """The rotation matrix on ``device``: the library's rotation applied to the identity. Run it before a capture."""
    device = torch.device(device)
    matrix = _matrices.get(device)
    if matrix is None:
        if is_capturing():
            raise RuntimeError("dequant: the rotation matrix is built on first use, which cannot happen inside a capture; "
                               "call xpu_kernels.triton.dequant.prepare(device) first")
        from ..int8 import ops
        matrix = ops.rotate(torch.eye(GROUP, dtype=torch.bfloat16, device=device)).contiguous()
        _matrices[device] = matrix
    return matrix


def scale_to_bf16(w8, scale):
    """``(w8 * scale[:, None])`` rounded to bf16, rotation not applied: the first half of the dequantizer."""
    n, k = w8.shape
    out = torch.empty((n, k), dtype=torch.bfloat16, device=w8.device)
    blocks = (k + BLOCK - 1) // BLOCK
    _kernel[(n * blocks,)](w8, scale, out, k, blocks, BLOCK=BLOCK, num_warps=NUM_WARPS)
    return out


def _dequant(w8, scale):
    n, k = w8.shape
    return torch.mm(scale_to_bf16(w8, scale).view(-1, GROUP), prepare(w8.device)).view(n, k)


def dequant_weight(w8, w_scale):
    """The unrotated bf16 weight behind an INT8 ConvRot weight; the same function as ``xpu_kernels.dequant_weight``."""
    _GATE.require()
    if w8.dtype != torch.int8 or w8.dim() != 2 or w8.shape[1] % GROUP or w_scale.numel() != w8.shape[0]:
        raise ValueError("dequant: expected int8 [N, K] with K a multiple of 256 and one scale per row")
    return _dequant(w8.contiguous(), w_scale.detach().to(torch.float32).contiguous())


def usable(w8) -> bool:
    if not enabled() or w8.dtype != torch.int8 or w8.dim() != 2 or w8.shape[1] % GROUP:
        return False
    if w8.device.type not in ACCELERATORS:
        return False
    return _GATE.ok()


def _reference(w8, w_scale):
    from ..int8 import ops
    return ops.dequant_weight_torch(w8, w_scale)


def _probe(ref):
    if _kernel is None:
        raise RuntimeError("the triton dequantizer is not available here")
    device = probe_device()
    gen = torch.Generator(device=device).manual_seed(0)
    pairs = {}
    for rows, k in ((64, 1280), (48, 5120), (16, 256)):
        w8 = torch.randint(-127, 128, (rows, k), generator=gen, device=device, dtype=torch.int8)
        scale = torch.rand(rows, generator=gen, device=device) * 0.02 + 0.001
        want = ref(w8, scale)
        pairs[f"{rows}x{k}"] = (_dequant(w8, scale), want)
        # the elementwise half alone is exact: it is a multiply and a rounding, no reduction
        pairs[f"scaled {rows}x{k}"] = (scale_to_bf16(w8, scale), (w8.float() * scale[:, None]).to(torch.bfloat16))
    return pairs


#: One bf16 unit flipped in a few values is ~1e-4 of the norm; a wrong matrix or a missing rounding is ~1e-1.
REL = 1e-3

_GATE = Gate("dequant_weight_triton", _probe, default_reference=lambda: _reference,
             env="XPU_KERNELS_NO_DEQUANT_TRITON", num_warps=NUM_WARPS, rel=REL)


def set_reference(fn=None):
    """Set (or clear) the dequantizer the gate compares against: ``fn(w8, scale) -> bf16 [N, K]``."""
    _GATE.set_reference(fn)


def gates():
    return {"dequant_weight_triton": _GATE}
