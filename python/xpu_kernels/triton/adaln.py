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
reduction layout changes and hundreds of values move), and the gate (`_GATE`)
re-checks bit-equality per process so a Triton upgrade degrades to
correct-but-slower rather than changing the image.

`subtract_mean=True` (real LayerNorm) is not covered and is not on this path.

XPU_KERNELS_NO_ADALN=1 disables it (the legacy MOSS_NO_ADALN_ROWS is
accepted); that flag is both the A/B switch and the escape hatch.
"""

import logging
import os

import torch

from . import reference as _reference
from ._gate import ACCELERATORS, Gate, probe_device

log = logging.getLogger(__name__)

# Load-bearing, same reason as the rope kernel: the shipped kernel's default, and
# at 1, 2 or 8 the reduction layout changes and hundreds of values move.
NUM_WARPS = 4

_kernel = None
_kernel_bwd = None
_kernel_mod = None
_triton = None

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

    @triton.jit
    def _rms_adaln_bwd_rows_kernel(
        x_ptr, dy_ptr, scale_ptr, dx_ptr, rstd_ptr,
        d, scale_group,
        stride_xr, stride_sr, stride_dyr, stride_dxr,
        eps,
        block_d: tl.constexpr, write_rstd: tl.constexpr, out_dtype: tl.constexpr,
    ):
        """One program per row: ``dx``, plus ``rstd`` for the modulation sums.

        ``y = v * (1 + scale) + shift`` with ``v = x * rstd``, so with
        ``dv = dy * (1 + scale)``:

            dx = rstd * (dv - (v . dv) / D * v)      (the norm's vjp)
            dscale = sum(dy * v),  dshift = sum(dy)  over the broadcast axes

        Only the first line is per-row, and it is the whole kernel.  The two
        modulation gradients are sums over rows, so the kernel hands back
        ``rstd`` (a scalar per row, 16 KB for the whole model) instead of ``v``:
        the reduction then reads ``x`` and ``dy`` once and never needs a
        materialised ``v``.  ``write_rstd`` is False when neither ``scale`` nor
        ``shift`` needs a gradient, so an inference-only backward does not pay
        for it.

        ``shift`` is not read: ``dx`` and ``dscale`` do not depend on it.
        """
        row = tl.program_id(0)
        cols = tl.arange(0, block_d)
        mask = cols < d

        x = tl.load(x_ptr + row * stride_xr + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.math.rsqrt(tl.sum(x * x) / d + eps)
        v = x * rstd

        sc = tl.load(scale_ptr + (row // scale_group) * stride_sr + cols,
                     mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row * stride_dyr + cols, mask=mask, other=0.0).to(tl.float32)

        dv = dy * (1.0 + sc)
        dx = rstd * (dv - (tl.sum(v * dv) / d) * v)

        tl.store(dx_ptr + row * stride_dxr + cols, dx.to(out_dtype), mask=mask)
        if write_rstd:
            tl.store(rstd_ptr + row, rstd)

    @triton.jit
    def _rms_adaln_mod_partials_kernel(
        x_ptr, dy_ptr, rstd_ptr, part_ptr,
        d, group, n_chunks, n_rows,
        stride_xr, stride_dyr,
        d_block: tl.constexpr, chunk: tl.constexpr,
    ):
        """Stage 1 of the modulation sums: one program per (group, chunk, block).

            dscale = sum(dy * x * rstd),  dshift = sum(dy)

        summed over the rows of one broadcast group.  A per-row program cannot
        reduce across rows, and atomics would both cost more than the writes and
        make the result order-dependent, so each program sums a run of ``chunk``
        rows of one group into a partial and stage 2 adds the partials.

        ``rstd`` comes from the ``dx`` kernel: without it this pass would have to
        read each row twice (once for the sum of squares, once for its block).
        """
        pid = tl.program_id(0)
        g = pid // n_chunks
        c = pid % n_chunks
        cols = tl.program_id(1) * d_block + tl.arange(0, d_block)
        mask = cols < d

        base = g * group + c * chunk
        acc_s = tl.zeros((d_block,), tl.float32)
        acc_v = tl.zeros((d_block,), tl.float32)
        for i in tl.static_range(chunk):
            r = base + i
            live = (c * chunk + i) < group and r < n_rows
            # clamp so a masked-out tail never reads out of bounds
            r = tl.minimum(r, n_rows - 1)
            x = tl.load(x_ptr + r * stride_xr + cols, mask=mask & live, other=0.0).to(tl.float32)
            dy = tl.load(dy_ptr + r * stride_dyr + cols, mask=mask & live, other=0.0).to(tl.float32)
            rs = tl.load(rstd_ptr + r)
            rs = tl.where(live, rs, 0.0)
            acc_s += dy
            acc_v += dy * x * rs

        n_part = tl.num_programs(0)
        tl.store(part_ptr + pid * d + cols, acc_s, mask=mask)
        tl.store(part_ptr + (n_part + pid) * d + cols, acc_v, mask=mask)

    @triton.jit
    def _rms_adaln_mod_reduce_kernel(
        part_ptr, dscale_ptr, dshift_ptr,
        d, n_chunks, n_groups,
        d_block: tl.constexpr,
    ):
        """Stage 2: add the chunk partials of one group. No atomics, deterministic."""
        g = tl.program_id(0)
        cols = tl.program_id(1) * d_block + tl.arange(0, d_block)
        mask = cols < d
        n_part = n_chunks * n_groups
        acc_s = tl.zeros((d_block,), tl.float32)
        acc_v = tl.zeros((d_block,), tl.float32)
        for c in range(0, n_chunks):
            pid = g * n_chunks + c
            acc_s += tl.load(part_ptr + pid * d + cols, mask=mask, other=0.0)
            acc_v += tl.load(part_ptr + (n_part + pid) * d + cols, mask=mask, other=0.0)
        tl.store(dshift_ptr + g * d + cols, acc_s, mask=mask)
        tl.store(dscale_ptr + g * d + cols, acc_v, mask=mask)

    _kernel = _rms_adaln_rows_kernel
    _kernel_bwd = _rms_adaln_bwd_rows_kernel
    _kernel_mod = (_rms_adaln_mod_partials_kernel, _rms_adaln_mod_reduce_kernel)
except Exception:  # pragma: no cover - no triton on this box
    _kernel = None
    _kernel_bwd = None
    _kernel_mod = None


def enabled():
    return _kernel is not None and _GATE.disabled_by is None


def _shipped_path_is_triton():
    """Only substitute for the backend the production op would actually pick.

    `comfy/quant_ops.py` disables the Triton backend unless the server runs with
    `--enable-triton-backend`; without it the shipped op uses its eager
    implementation, which is a different function, so we must not replace it.
    """
    try:
        import comfy_kitchen
    except Exception:
        return True  # not running inside ComfyUI: no dispatch to contradict
    try:
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
    scale_flat, scale_group = _prep_modulation(scale, x, n, d)
    shift_flat, shift_group = _prep_modulation(shift, x, n, d)
    return orig_shape, x_flat, n, d, scale_flat, scale_group, shift_flat, shift_group


def usable(x, scale, shift):
    if not enabled() or not _shipped_path_is_triton():
        return False
    if x.device.type not in ACCELERATORS or x.dtype not in (torch.bfloat16, torch.float32):
        return False
    if x.ndim < 2 or x.shape[-1] == 0 or x.shape[-1] > 4096:
        return False  # the shipped kernel loops over d in that case
    if x.shape[-1] % 2:
        return False
    return _GATE.ok()


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
        num_warps=NUM_WARPS)
    return out.reshape(orig_shape)


def _prep_modulation(t, x, n, d):
    """Flatten a scale/shift tensor to (rows, d) + a broadcast group.

    comfy_kitchen's helper keeps a broadcast over inner dims as one row per
    distinct vector; without it we materialise the broadcast instead.  Both copy
    the same values -- the grouping only avoids redundant read traffic -- so the
    choice cannot change a result.
    """
    try:
        from comfy_kitchen.backends._modulation import adaln_prep_modulation
    except Exception:
        flat = t.expand_as(x).reshape(n, d)
        return (flat if flat.is_contiguous() else flat.contiguous()), 1
    return adaln_prep_modulation(t, x, n, d)


#: Rows per program in stage 1 of the modulation sums.  Large enough that the
#: partial buffer is small (group/chunk rows x d), small enough to keep the
#: grid wide: at S=3973, D=3072 this is ~250 x 24 programs.
MOD_CHUNK = 16

#: Columns per program in both stages of the modulation sums.
MOD_BLOCK = 128


def _rows_bwd_kernel(dy, x, scale, eps, need_rstd=True):
    """Launch the backward kernel. No gate, no fallback: the probe calls this.

    Returns ``(dx, rstd)`` with ``rstd`` None when ``need_rstd`` is false.
    """
    d = x.shape[-1]
    n = x.numel() // d
    x_flat = x.reshape(n, d)
    if not x_flat.is_contiguous():
        x_flat = x_flat.contiguous()
    dy_flat = dy.reshape(n, d)
    if not dy_flat.is_contiguous():
        dy_flat = dy_flat.contiguous()
    s_flat, s_group = _prep_modulation(scale, x, n, d)

    dx = torch.empty_like(x_flat)
    rstd = torch.empty((n,), device=x.device, dtype=torch.float32)
    _kernel_bwd[(n,)](
        x_flat, dy_flat, s_flat, dx, rstd,
        d, s_group,
        x_flat.stride(0), s_flat.stride(0), dy_flat.stride(0), dx.stride(0),
        eps,
        block_d=min(_triton.next_power_of_2(d), 4096),
        write_rstd=need_rstd,
        out_dtype=tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float32,
        num_warps=NUM_WARPS)
    return dx.reshape(x.shape), (rstd if need_rstd else None)


def _mod_reduction_kernel(dy, x, rstd, group, n_groups):
    """Launch the two-stage modulation sums. Returns ``(dscale, dshift)`` fp32.

    Both are ``[n_groups, d]``: one row per distinct modulation vector.  The
    caller maps that back to the shape the modulation was passed in.
    """
    d = x.shape[-1]
    n = x.numel() // d
    x_flat = x.reshape(n, d)
    dy_flat = dy.reshape(n, d)
    n_chunks = (group + MOD_CHUNK - 1) // MOD_CHUNK
    part = torch.empty((2 * n_groups * n_chunks, d), device=x.device,
                       dtype=torch.float32)
    dscale = torch.empty((n_groups, d), device=x.device, dtype=torch.float32)
    dshift = torch.empty((n_groups, d), device=x.device, dtype=torch.float32)

    part_kernel, reduce_kernel = _kernel_mod
    d_block = MOD_BLOCK
    part_kernel[(n_groups * n_chunks, (d + d_block - 1) // d_block)](
        x_flat, dy_flat, rstd, part,
        d, group, n_chunks, n,
        x_flat.stride(0), dy_flat.stride(0),
        d_block=d_block, chunk=MOD_CHUNK, num_warps=NUM_WARPS)
    reduce_kernel[(n_groups, (d + d_block - 1) // d_block)](
        part, dscale, dshift,
        d, n_chunks, n_groups,
        d_block=d_block, num_warps=NUM_WARPS)
    return dscale, dshift


def _rows_bwd(dy, x, scale, shift, eps, need_mod=True):
    """The fused backward, or the torch one when the kernel has not been proved.

    Returns ``(dx, dscale, dshift)``.  The fallback is the analytic gradient in
    `reference`, which is the same arithmetic with torch's reduction order -- it
    is slower, not wrong.

    The modulation sums run as one kernel *per group structure*: `dscale` follows
    `scale`'s groups and `dshift` follows `shift`'s, and when the two differ (a
    per-dim `scale` with a per-block `shift`, say) the pair is not fused and both
    come from torch.  Same numbers either way.
    """
    if _kernel_bwd is None or not _GATE_BWD.ok():
        return _reference.rms_adaln_backward(dy, x, scale, shift, eps)

    if not need_mod:
        dx, _ = _rows_bwd_kernel(dy, x, scale, eps, need_rstd=False)
        return dx, None, None

    d = x.shape[-1]
    n = x.numel() // d
    _, s_group = _prep_modulation(scale, x, n, d)
    _, h_group = _prep_modulation(shift, x, n, d)
    if s_group != h_group:
        return _reference.rms_adaln_backward(dy, x, scale, shift, eps)

    dx, rstd = _rows_bwd_kernel(dy, x, scale, eps, need_rstd=True)
    dscale, dshift = _mod_reduction_kernel(dy, x, rstd, s_group, n // s_group)
    return (dx,
            _reference._reduce_like(dscale, scale),
            _reference._reduce_like(dshift, shift))


def _probe_forward(ref):
    """One row-block, bf16, with a scale/shift that is not all ones."""
    device = probe_device()
    gen = torch.Generator(device=device).manual_seed(0)
    x = (torch.randn(256, 3072, generator=gen, device=device) * 0.5).to(torch.bfloat16)
    scale = torch.randn(1, 3072, generator=gen, device=device) * 0.1
    shift = torch.randn(1, 3072, generator=gen, device=device) * 0.1
    return {"x": (_run(x, scale, shift, 1e-6), ref(x, scale, shift, 1e-6))}


def _default_reference():
    """The reference this branch was verified against, when it is importable."""
    from comfy_kitchen.backends.triton.adaln import rms_adaln
    return rms_adaln


def _probe_backward(ref):
    """Fused backward vs ``torch.autograd`` of the eager chain, in fp32.

    fp32 for the same reason as the rope backward: in bf16 the forward's cast
    makes autograd round the gradient, which would put the reference ~1 ULP away
    from the smooth gradient and force a budget loose enough to hide a real
    error (``debug/probe_rope_backward.py`` measures that effect).

    All three outputs are checked, because the kernel produces ``dx`` and ``v``
    and ``v`` is what ``dscale`` is built from -- a right ``dx`` with a wrong
    ``v`` would still train, wrongly, and only show up as a drifting modulation.
    """
    device = probe_device()
    gen = torch.Generator(device=device).manual_seed(3)
    B, S, D = 2, 128, 3072
    x = torch.randn(B, S, D, generator=gen, device=device) * 0.3
    scale = torch.randn(1, 1, D, generator=gen, device=device) * 0.1
    shift = torch.randn(1, 1, D, generator=gen, device=device) * 0.1
    dy = torch.randn(B, S, D, generator=gen, device=device) * 0.1

    n = x.numel() // D
    out = {}
    # two modulation shapes: a single vector shared by every row, and one per
    # batch element.  The second has n_groups > 1, which is what exercises the
    # chunk-to-group mapping in the reduction kernel; the first has one group of
    # n rows, where an off-by-one in `row // group` would show up as a wrong sum.
    for tag, mod in (("", scale), ("_grouped", scale.repeat(B, 1, 1))):
        xx = x.detach().clone().requires_grad_(True)
        sx = mod.detach().clone().requires_grad_(True)
        hx = mod.detach().clone().requires_grad_(True)
        gx, gs, gh = torch.autograd.grad(ref(xx, sx, hx, 1e-6), (xx, sx, hx), dy)

        _, group = _prep_modulation(mod, x, n, D)
        dx, rstd = _rows_bwd_kernel(dy, x, mod, 1e-6)
        dscale, dshift = _mod_reduction_kernel(dy, x, rstd, group, n // group)
        out["dx" + tag] = (dx, gx)
        out["dscale" + tag] = (_reference._reduce_like(dscale, mod), gs)
        out["dshift" + tag] = (_reference._reduce_like(dshift, mod), gh)
    return out


#: Budget for the backward: same arithmetic, different reduction order (see
#: ``rope.BWD_REL``).  Measured worst case on the fp32 probe is ~1e-6.
BWD_REL = 1e-4

_GATE = Gate(
    "rms_adaln", _probe_forward,
    default_reference=_default_reference,
    env="XPU_KERNELS_NO_ADALN", legacy_env="MOSS_NO_ADALN_ROWS",
    num_warps=NUM_WARPS,
)


_GATE_BWD = Gate(
    "rms_adaln_backward", _probe_backward,
    default_reference=lambda: _reference.rms_adaln,
    env="XPU_KERNELS_NO_ADALN_BWD",
    num_warps=NUM_WARPS, rel=BWD_REL,
)


def set_reference(fn=None):
    """Point the gates at the implementation they must match, or clear them.

    ``fn`` must have the signature ``fn(x, scale, shift, eps) -> x'`` and be the
    path the caller falls back to.  Without it, ``comfy_kitchen``'s kernel is
    used when importable; otherwise the kernels stay inactive.  The backward
    gate wants the *differentiable forward* to differentiate; its default is
    ``reference.rms_adaln``, which takes the same signature.
    """
    _GATE.set_reference(fn)
    _GATE_BWD.set_reference(fn)


def gates():
    """This module's permission slips, by name (for the status report)."""
    return {"rms_adaln": _GATE, "rms_adaln_backward": _GATE_BWD}


def describe(which="rms_adaln"):
    """One line: is this kernel in use, and if not, why."""
    return gates()[which].describe()


def _self_test():
    """Kept for callers that predate the gate object."""
    return _GATE.ok()


class _RmsAdaLN(torch.autograd.Function):
    """The fused forward with the fused backward attached."""

    @staticmethod
    def forward(ctx, x, scale, shift, eps):
        out = _run(x, scale, shift, eps)
        # the modulation gradients are sums over rows, so `v` is only worth
        # materializing when one of them is actually being learned
        need_v = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        ctx.save_for_backward(x, scale, shift)
        ctx.eps = eps
        ctx.need_v = need_v
        return out

    @staticmethod
    def backward(ctx, dy):
        x, scale, shift = ctx.saved_tensors
        dx, dscale, dshift = _rows_bwd(
            dy, x, scale, shift, ctx.eps,
            ctx.need_v and (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]))
        if not ctx.needs_input_grad[1]:
            dscale = None
        if not ctx.needs_input_grad[2]:
            dshift = None
        return dx, dscale, dshift, None


def rms_adaln(x, scale, shift, eps=1e-6):
    """Drop-in for `comfy_kitchen.rms_adaln` (mean pinned to 0).

    Differentiable: with a grad-requiring input this routes through
    :class:`_RmsAdaLN`, whose backward is a single fused kernel (or the torch
    gradient in `reference` if the backward gate is closed).  Under
    ``torch.no_grad`` it is the plain forward, unchanged and bit-exact.
    """
    if torch.is_grad_enabled() and (x.requires_grad or scale.requires_grad
                                   or shift.requires_grad):
        return _RmsAdaLN.apply(x, scale, shift, eps)
    return _run(x, scale, shift, eps)
