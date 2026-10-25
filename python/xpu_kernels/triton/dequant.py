"""The INT8 weight dequantizer as a Triton kernel: ``int8 [N, K]`` + ``fp32 [N]`` -> the unrotated ``bf16 [N, K]``.

``xpu_kernels.int8.dequant_weight`` is a SYCL kernel that multiplies by the row scale, rounds to bf16 and undoes the ConvRot
rotation in registers.  It runs at about a fifth of the card's copy bandwidth: the rotation of a 256-element group needs all
256 values live in one lane, and that is the same register-residency floor the activation rotate has (``docs/experiments.md``).
The activation rotate cannot escape it because the row's abs-max has to be taken over the rotated row.  Here there is no
abs-max, so the rotation can be written as what it is, a multiplication of each group by a fixed 256x256 matrix, and handed to
the matrix unit: the data is read once as int8 and written once as bf16.

The matrix is not written down here: it is the library's own rotation applied to the identity, so whatever the C kernel
rotates by, this does.  Rotation is an involution (``rotate(rotate(x)) == x``), which is also what makes the matrix its own
transpose and the product below ``a @ R`` with no transpose.

The arithmetic differs from the C kernel's butterfly in the order of the additions (the matrix unit accumulates in fp32), so the
result is not bit-identical: a rotated value that lands within an fp32 rounding error of a bf16 tie rounds the other way.
The gate compares with a relative budget, and ``debug/verify_dequant_triton.py`` counts the values that differ.

``XPU_KERNELS_NO_DEQUANT_TRITON=1`` switches it off.
"""

from __future__ import annotations

import logging
import os

import torch

from .._capture import is_capturing
from ._gate import ACCELERATORS, Gate, probe_device

log = logging.getLogger(__name__)

__all__ = ["dequant_weight", "gates", "prepare", "set_reference", "usable"]

GROUP = 256
# Rows per program, output columns per matrix-unit product, warps.  Nothing here changes the numbers except through the
# accumulation order inside the matrix unit, which the gate re-verifies.
BLOCK_ROWS = 32
BLOCK_COLS = 64
NUM_WARPS = 4

_kernel = None
_import_error = None
_matrices: dict[torch.device, torch.Tensor] = {}

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _dequant_rotate_kernel(w8_ptr, scale_ptr, rot_ptr, out_ptr, rows_total, K,
                               BM: tl.constexpr, BJ: tl.constexpr):
        pid = tl.program_id(0)
        group = tl.program_id(1)
        rows = pid * BM + tl.arange(0, BM)
        in_rows = rows < rows_total
        cols = tl.arange(0, 256)
        base = rows.to(tl.int64)[:, None] * K + group * 256
        w = tl.load(w8_ptr + base + cols[None, :], mask=in_rows[:, None], other=0)
        scale = tl.load(scale_ptr + rows, mask=in_rows, other=0.0)
        # the chain rounds the scaled value to bf16 *before* the rotation, so this does too
        a = (w.to(tl.float32) * scale[:, None]).to(tl.bfloat16)
        for j in tl.static_range(0, 256, BJ):
            out_cols = j + tl.arange(0, BJ)
            r = tl.load(rot_ptr + cols[:, None] * 256 + out_cols[None, :])
            acc = tl.dot(a, r)
            tl.store(out_ptr + base + out_cols[None, :], acc.to(tl.bfloat16), mask=in_rows[:, None])

    _kernel = _dequant_rotate_kernel

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


def _launch(w8, scale, out):
    n, k = w8.shape
    grid = ((n + BLOCK_ROWS - 1) // BLOCK_ROWS, k // GROUP)
    _kernel[grid](w8, scale, prepare(w8.device), out, n, k, BM=BLOCK_ROWS, BJ=BLOCK_COLS, num_warps=NUM_WARPS)
    return out


def dequant_weight(w8, w_scale):
    """The unrotated bf16 weight behind an INT8 ConvRot weight; the same function as ``xpu_kernels.dequant_weight``."""
    _GATE.require()
    if w8.dtype != torch.int8 or w8.dim() != 2 or w8.shape[1] % GROUP or w_scale.numel() != w8.shape[0]:
        raise ValueError("dequant: expected int8 [N, K] with K a multiple of 256 and one scale per row")
    w8 = w8.contiguous()
    scale = w_scale.detach().to(torch.float32).contiguous()
    return _launch(w8, scale, torch.empty(w8.shape, dtype=torch.bfloat16, device=w8.device))


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
    for rows, k in ((64, 1280), (48, 5120), (16, 256)):  # 48 is not a multiple of the row block
        w8 = torch.randint(-127, 128, (rows, k), generator=gen, device=device, dtype=torch.int8)
        scale = (torch.rand(rows, generator=gen, device=device) * 0.02 + 0.001)
        pairs[f"{rows}x{k}"] = (_launch(w8, scale, torch.empty((rows, k), dtype=torch.bfloat16, device=device)), ref(w8, scale))
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
