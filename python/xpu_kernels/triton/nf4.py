"""A fused NF4 dequantizer: packed nibbles + per-block absmax -> bf16.

NF4 is the 4-bit storage format QLoRA uses for a frozen base: every 64
elements of the flattened weight share one absmax scale, and each element is
stored as an index into a 16-entry codebook of normal-distribution quantiles
(two indices per byte).

The point of this kernel is that the dequantization has to happen *every
forward*, for every layer, and the obvious torch chain is expensive: it
materializes the unpacked indices, casts them to int64 for the gather, and
holds the gathered codes and the scaled product at the same time -- about 40
bytes of traffic and 16 bytes of scratch per parameter, which is more memory
than the bf16 weight the format was meant to replace.

This kernel does it in one pass: read the packed byte, extract the nibble,
look the code up, multiply by the block's scale, store bf16.  That is 2.6
bytes of traffic per parameter and **no temporaries at all** -- the output is
the only allocation -- which also makes it safe to put inside a device-graph
capture (an allocation inside a capture comes from the graph's private pool
and is replayed on every step).

Bit-exactness is attainable here, unlike in the int8 row-quantizer: the
arithmetic is a table lookup, a multiply and a cast, with no division and no
rounding mode to get wrong.  The gate compares against a torch chain bit for
bit, on the production shape.

The codebook is the standard one (bitsandbytes' ``create_normal_map()``:
the normal quantiles, computed with ``torch.special.ndtri`` in float64).  A
caller with its own table -- the trainer's, for instance -- can pass it in,
and then the comparison is against *that* table.
"""

from __future__ import annotations

import logging

import torch

from ._gate import ACCELERATORS, Gate, probe_device

log = logging.getLogger(__name__)

__all__ = ["codebook", "dequant_nf4", "gates", "set_reference", "usable"]

_kernel = None
_triton = None
_import_error = None
_logged = False

#: Elements per absmax block (bitsandbytes' own default for 4-bit types).
BLOCKSIZE = 64

#: One program per this many elements.  A multiple of the block size, so a
#: tile never straddles a scale change.
TILE = 1024

NUM_WARPS = 4

try:
    import triton
    import triton.language as tl

    _triton = triton

    @triton.jit
    def _nf4_dequant_kernel(
        packed_ptr, absmax_ptr, codebook_ptr, out_ptr,
        n_elements,
        BLOCK_SHIFT: tl.constexpr,     # log2 of the block size
        OUT_FP32: tl.constexpr,
        TILE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * TILE + tl.arange(0, TILE)
        mask = offs < n_elements

        # two elements per byte: the even index is the low nibble
        packed = tl.load(packed_ptr + (offs >> 1), mask=mask, other=0)
        nib = (packed >> (4 * (offs & 1))) & 0x0F

        # the codebook lookup and the block's scale: both are gathers, but the
        # codebook is 16 values and the scale is one per block, so both stay
        # in L1
        code = tl.load(codebook_ptr + nib, mask=mask, other=0.0)
        scale = tl.load(absmax_ptr + (offs >> BLOCK_SHIFT), mask=mask, other=0.0)

        val = code * scale
        if OUT_FP32:
            tl.store(out_ptr + offs, val, mask=mask)
        else:
            tl.store(out_ptr + offs, val.to(tl.bfloat16), mask=mask)

    _kernel = _nf4_dequant_kernel

except Exception as exc:  # noqa: BLE001 - no triton means no kernel
    _import_error = exc


def codebook(device=None, dtype=torch.float32) -> torch.Tensor:
    """The 16 NF4 levels, in bitsandbytes' definition.

    ``create_normal_map()`` with its own defaults (offset 0.9677083, the extra
    zero): the normal quantiles, computed with the inverse normal CDF in
    float64 and then scaled so the largest is 1.  A caller with its own table
    (the trainer computes the same thing) can pass it to :func:`dequant_nf4`
    instead, and then the kernel uses *that* table.
    """
    offset = 0.9677083
    v1 = torch.special.ndtri(
        torch.linspace(offset, 0.5, 9, dtype=torch.float64)[:-1])
    v3 = -torch.special.ndtri(
        torch.linspace(offset, 0.5, 8, dtype=torch.float64)[:-1])
    values = torch.cat([v1, torch.zeros(1, dtype=torch.float64), v3])
    values = values.sort().values
    values = values / values.max()
    return values.to(device=device, dtype=dtype)


def _reference_chain(packed, absmax, codebook_, blocksize=BLOCKSIZE):
    """The torch chain this kernel must match, bit for bit.

    It is the obvious one -- unpack, gather, scale -- and it is what the
    trainer's ``NF4WeightStore.materialize()`` does.  It is the reference
    because it is what exists, not because it is good: it moves about 40
    bytes per parameter and holds 16 bytes of scratch per parameter.
    """
    n = packed.numel() * 2
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    idx = torch.stack([low, high], dim=1).view(-1)[:n]
    codes = codebook_[idx.long()]
    blocks = torch.arange(n, device=packed.device) // blocksize
    return codes * absmax[blocks]


def enabled() -> bool:
    """True if the kernel is there and has not been switched off."""
    if _kernel is not None:
        return True
    global _import_error, _logged
    if _import_error is not None:
        if not _logged:
            _logged = True
            log.info("triton nf4 dequantizer unavailable: %s", _import_error)
        return False
    import os
    if os.environ.get("XPU_KERNELS_NO_NF4", "") not in ("", "0"):
        return False
    return True


def _launch(packed, absmax, codebook_, out, n, blocksize, dtype):
    """The raw kernel launch -- no gate, no checks.

    The probe calls this, not :func:`dequant_nf4`: the public entry asks the
    gate, and the gate's own probe would then be asking the gate.
    """
    _kernel[(_triton.cdiv(n, TILE),)](
        packed, absmax, codebook_, out,
        n,
        BLOCK_SHIFT=blocksize.bit_length() - 1,
        OUT_FP32=(dtype == torch.float32),
        TILE=TILE, num_warps=NUM_WARPS)
    return out


def dequant_nf4(packed, absmax, codebook_=None, *, dtype=torch.bfloat16,
                 blocksize=BLOCKSIZE, out=None):
    """Dequantize an NF4 weight: packed nibbles + per-block absmax -> ``dtype``.

    ``packed`` is uint8 with two 4-bit indices per byte (the even index in the
    low nibble), ``absmax`` is fp32 with one scale per ``blocksize`` elements of
    the *flattened* weight, and the result is **flat** -- one element per weight
    element, in the order they were packed.  The caller reshapes it (the
    weight's shape is the caller's business, and a reshape is a view).

    The result is bit-identical to the torch chain, and the chain's cost is
    what this replaces: ~40 bytes of traffic and 16 bytes of scratch per
    parameter against 2.6 bytes and no scratch here.
    """
    # The argument checks come *before* the gate: a caller's mistake is the
    # caller's error, and it should be reported as such whether or not the
    # kernel is verified on this machine.
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed must be uint8, got {packed.dtype}")
    if absmax.dtype != torch.float32:
        raise TypeError(f"absmax must be float32, got {absmax.dtype}")
    if blocksize & (blocksize - 1):
        raise ValueError(f"blocksize must be a power of two, got {blocksize}")
    n = packed.numel() * 2
    if absmax.numel() * blocksize < n:
        raise ValueError(f"absmax has {absmax.numel()} blocks but the weight "
                         f"needs {n // blocksize} at {blocksize} per block")
    _GATE.require()
    if codebook_ is None:
        codebook_ = codebook(packed.device)
    if out is None:
        out = torch.empty(n, dtype=dtype, device=packed.device)
    elif out.numel() != n:
        raise ValueError(f"out has {out.numel()} elements, the weight has {n}")
    return _launch(packed, absmax, codebook_, out, n, blocksize, dtype)


def usable(packed) -> bool:
    """May the fused dequantizer be used for this input?"""
    if not enabled():
        return False
    if packed.dtype != torch.uint8:
        return False
    if packed.device.type not in ACCELERATORS:
        return False
    return _GATE.ok()


def _probe(ref):
    """The fused kernel against the torch chain, on a production-shaped case.

    Random nibbles (so every codebook entry is hit) and random scales, at the
    block size and the block counts a real weight has.
    """
    if _kernel is None:
        # No Triton here: say so, rather than failing with a NameError from
        # inside the launch (which is what the first CI run of this module
        # reported).
        raise RuntimeError("the triton nf4 dequantizer is not available here")
    device = probe_device()
    gen = torch.Generator(device=device).manual_seed(0)
    n_blocks = 4096
    packed = torch.randint(0, 256, (n_blocks * BLOCKSIZE // 2,), generator=gen,
                           device=device, dtype=torch.uint8)
    absmax = (torch.rand(n_blocks, generator=gen, device=device) * 0.02 + 1e-4)
    cb = codebook(device)
    n = packed.numel() * 2
    want = ref(packed, absmax, cb)
    got = _launch(packed, absmax, cb,
                  torch.empty(n, dtype=want.dtype, device=packed.device),
                  n, BLOCKSIZE, want.dtype)
    # and the production dtype: the chain casts to bf16 at the end, the kernel
    # rounds on the store, and both must land on the same bits
    want_bf16 = want.to(torch.bfloat16)
    got_bf16 = _launch(packed, absmax, cb,
                        torch.empty(n, dtype=torch.bfloat16, device=packed.device),
                        n, BLOCKSIZE, torch.bfloat16)
    return {"nf4 fp32": (got, want), "nf4 bf16": (got_bf16, want_bf16)}


def _default_reference():
    return _reference_chain


_GATE = Gate(
    "nf4_dequant", _probe,
    default_reference=_default_reference,
    env="XPU_KERNELS_NO_NF4",
    num_warps=NUM_WARPS,
)


def set_reference(fn=None):
    """Set (or clear) the reference the gate compares against."""
    _GATE.set_reference(fn)


def gates():
    """The gates of this module, by name."""
    return {"nf4_dequant": _GATE}
