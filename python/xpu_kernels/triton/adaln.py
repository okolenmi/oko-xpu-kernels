"""Single-pass RMS adaptive layer norm (`rms_adaln`) for the MossImage blocks.

`comfy_kitchen.backends.triton.adaln` walks every row twice: pass 1 accumulates
the sum of squares, pass 2 reloads `x` to normalise and modulate.  For the
MossImage shape (d = 3072, `block_d = min(next_power_of_2(d), 4096)`) the loop is
a single iteration, so `x` is read twice and written once -- 72 MB per call
instead of 48.  The mean is pinned to 0 for the RMS variant, so pass 1 needs
*nothing* but the sum of squares: the reload is a pure dependency, not a
computation.

Keeping the row in registers is bit-identical (0/12,205,056) and measures 1.56x
(0.206 -> 0.132 ms per call at 3973 x 3072).  Same caveat as `xpu_rope.py`:
`num_warps=4` is load-bearing (the shipped kernel's default; at 1, 2 or 8 the
reduction layout changes and hundreds of values move), and `_self_test()`
re-checks bit-equality per process so a Triton upgrade degrades to
correct-but-slower rather than changing the image.

`subtract_mean=True` (real LayerNorm) is not covered and is not on this path.

MOSS_NO_ADALN_ROWS=1 disables it (A/B and escape hatch).
"""

import os

import torch

_ENABLED = os.environ.get("MOSS_NO_ADALN_ROWS", "") != "1"

_kernel = None
_triton = None
_verified = None
_WHY = None          # why the last self-test said what it said
_REFERENCE = None    # injected reference implementation, see set_reference()

try:
    import triton
    import triton.language as tl

    _triton = triton

    @triton.jit
    def _rms_adaln_rows_kernel(
        x_ptr, scale_ptr, shift_ptr, y_ptr,
        d, scale_group, shift_group,
        stride_xr, stride_sr, stride_shr, stride_yr,
        eps,
        block_d: tl.constexpr, dtype: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_d)
        mask = cols < d

        # one load of the row: the sum of squares and the modulate pass share it
        x = tl.load(x_ptr + row * stride_xr + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.math.rsqrt(tl.sum(x * x) / d + eps)

        sc = tl.load(scale_ptr + (row // scale_group) * stride_sr + cols,
                     mask=mask, other=0.0).to(tl.float32)
        sh = tl.load(shift_ptr + (row // shift_group) * stride_shr + cols,
                     mask=mask, other=0.0).to(tl.float32)

        # same expression as the shipped kernel with mean == 0.0
        out = (x - 0.0) * rstd * (1.0 + sc) + sh
        tl.store(y_ptr + row * stride_yr + cols, out.to(dtype), mask=mask)

    _kernel = _rms_adaln_rows_kernel
except Exception:  # pragma: no cover - no triton on this box
    _kernel = None


def enabled():
    return _ENABLED and _kernel is not None


def _shipped_path_is_triton():
    """Only substitute for the backend the production op would actually pick.

    `comfy/quant_ops.py` disables the Triton backend unless the server runs with
    `--enable-triton-backend`; without it the shipped op uses its eager
    implementation, which is a different function, so we must not replace it.
    """
    try:
        import comfy_kitchen
        return bool(comfy_kitchen.registry.is_available("triton"))
    except Exception:
        return False


def _flat(x, scale, shift):
    """Mirror the shipped wrapper: flatten to (rows, d) and prep the broadcast."""
    orig_shape = x.shape
    d = x.shape[-1]
    n = x.numel() // d
    x_flat = x.reshape(n, d)
    if not x_flat.is_contiguous():
        x_flat = x_flat.contiguous()
    from comfy_kitchen.backends._modulation import adaln_prep_modulation

    scale_flat, scale_group = adaln_prep_modulation(scale, x, n, d)
    shift_flat, shift_group = adaln_prep_modulation(shift, x, n, d)
    return orig_shape, x_flat, n, d, scale_flat, scale_group, shift_flat, shift_group


def usable(x, scale, shift):
    if not enabled() or not _shipped_path_is_triton():
        return False
    if x.device.type != "xpu" or x.dtype not in (torch.bfloat16, torch.float32):
        return False
    if x.ndim < 2 or x.shape[-1] == 0 or x.shape[-1] > 4096:
        return False  # the shipped kernel loops over d in that case
    if x.shape[-1] % 2:
        return False
    return _self_test()


def _run(x, scale, shift, eps):
    orig_shape, x_flat, n, d, s_flat, s_group, h_flat, h_group = _flat(x, scale, shift)
    out = torch.empty_like(x_flat)
    _kernel[(n,)](
        x_flat, s_flat, h_flat, out,
        d, s_group, h_group,
        x_flat.stride(0), s_flat.stride(0), h_flat.stride(0), out.stride(0),
        eps,
        block_d=min(_triton.next_power_of_2(d), 4096),
        dtype=tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float32,
        num_warps=4)  # num_warps=4 is load-bearing
    return out.reshape(orig_shape)


def _self_test():
    global _verified
    if _verified is not None:
        return _verified
    _verified = False
    try:
        shipped = _REFERENCE if _REFERENCE is not None else _default_reference()

        gen = torch.Generator(device="xpu").manual_seed(0)
        x = (torch.randn(256, 3072, generator=gen, device="xpu") * 0.5).to(torch.bfloat16)
        scale = torch.randn(1, 3072, generator=gen, device="xpu") * 0.1
        shift = torch.randn(1, 3072, generator=gen, device="xpu") * 0.1
        ref = shipped(x, scale, shift, 1e-6)
        got = _run(x, scale, shift, 1e-6)
        bad = (got.view(torch.int16) != ref.view(torch.int16)).sum().item()
        import logging
        if bad == 0:
            logging.info("xpu_adaln: single-pass rms_adaln active, bit-identical "
                         "to the shipped kernel on the self-test")
            _verified = True
        else:
            logging.warning("xpu_adaln: self-test found %d differing values, "
                            "falling back to the shipped rms_adaln", bad)
    except Exception as exc:
        global _WHY
        _WHY = f"unverified: {exc}"
        import logging
        logging.warning("xpu_adaln: self-test failed (%s), using the shipped "
                        "rms_adaln", exc)
    return _verified


def _default_reference():
    """The reference this branch was verified against, when it is importable."""
    from comfy_kitchen.backends.triton.adaln import rms_adaln
    return rms_adaln


def set_reference(fn=None):
    """Point the self-test at a reference implementation, or clear it.

    ``fn`` must have the signature ``fn(x, scale, shift, eps) -> x'`` and be the
    implementation the caller falls back to; the gate only proves anything
    against that.  Without it, ComfyUI's kernel is used when importable.
    """
    global _REFERENCE, _verified, _WHY
    _REFERENCE, _verified, _WHY = fn, None, None


def describe():
    """One line: is this kernel in use, and if not, why."""
    if not _ENABLED:
        return "disabled: MOSS_NO_ADALN_ROWS=1"
    ok = _self_test()
    why = _WHY or ("verified against the reference" if ok else "self-test mismatch")
    return f"{'active' if ok else 'inactive'}: {why}"


def rms_adaln(x, scale, shift, eps=1e-6):
    """Drop-in for `comfy_kitchen.rms_adaln` (mean pinned to 0)."""
    return _run(x, scale, shift, eps)
