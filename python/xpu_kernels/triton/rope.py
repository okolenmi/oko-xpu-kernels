"""Single-pass RMSNorm + split-half RoPE for the MossImage attention path.

The shipped op is `comfy_kitchen.backends.triton.rms_rope.rms_rope_split_half`, a
Triton kernel with grid (batch, head, seq).  Per program, per head, it

  * loads the whole head row to compute the sum of squares,
  * loads the two halves again to rotate them,
  * loads the four `pe` planes again -- `pe` is fp32 (see `rope_fp32` in
    `comfy/ldm/flux/math.py`, which returns fp32 and nothing casts it), so that is
    1 KB per program, re-read once per head for a table a few MB in size.

That is 95 352 tiny programs for q (and as many for k) at the production shape,
each touching 256 B of a 6 KB-strided row.  This module keeps the arithmetic
identical and changes only the schedule: one program per (batch, seq) with the
head loop inside, so the row is read as one contiguous 6 KB run and `pe` is
loaded once per sequence instead of once per head.  Measured 2.34x on q+k
(0.587 ms vs 1.373 ms) at S=3973, H=24, D=128 with the model's strided q/k.

Bit-exactness, and why the code looks the way it does
----------------------------------------------------
`tl.sum` over the 128-element row is *not* always exact: squares of bf16 values
are exact, but summing 128 of them is only almost always exact in fp32, and where
it is not, the reduction order decides the last bit of `inv_rms`.  One ULP there
moves a bf16 rounding, and ~36 moved q values out of 12.2M change the whole image
(PSNR 23 dB) -- the model is chaotic in q/k.

Triton's reduction order follows the block layout, so bit-exactness pins two
things that must not be "optimised" back:

  * `num_warps=4` (the shipped kernel's default).  At num_warps=1 the same code
    is 1.6x faster again and produces 36/12.2M mismatches.  Do not change it.
  * the two half-rows are re-loaded, not carved out of the register block.
    `tl.reshape` + `tl.trans` + `tl.split` on the loaded row is 0/12.2M too, but
    it forces a layout conversion that makes the kernel *slower* than the
    shipped one (1.43 ms vs 1.37 ms), so the schedule win would be thrown away.

`debug/probe_rope_variants.py` is the evidence for both statements, and the
gate (`_GATE`, see `_gate.py`) re-checks the whole thing once per process against
the shipped kernel on synthetic strided tensors, falling back to it if anything
differs -- so a Triton upgrade that changes its layouts degrades to
correct-but-slower instead of changing the image.

XPU_KERNELS_NO_ROPE=1 disables it entirely (the legacy MOSS_NO_ROPE_ROWS
is accepted); the flag is both the A/B switch and the escape hatch.
"""

import logging
import os

import torch

from ._gate import ACCELERATORS, Gate, probe_device

log = logging.getLogger(__name__)

# Load-bearing: Triton's reduction order follows the block layout, so a different
# warp count moves the last bit of `inv_rms` (num_warps=1 is 1.6x faster and
# produces 36/12.2M mismatches).  Changing it is a code change the gate must
# re-approve -- see debug/probe_rope_variants.py.
NUM_WARPS = 4

_kernel = None
_kernel_dq = None
_triton = None
_import_error = None
_logged = False

try:
    import triton
    import triton.language as tl

    _triton = triton

    @triton.jit
    def _rms_rope_rows_kernel(
        x_ptr, freqs_ptr, scale_ptr, out_ptr,
        head_dim, n_heads, freqs_batch, freqs_seq,
        stride_x_b, stride_x_s, stride_x_h, stride_x_d,
        stride_o_b, stride_o_s, stride_o_h, stride_o_d,
        stride_f_b, stride_f_s, stride_f_dim, stride_f_rot, stride_f_pair,
        epsilon,
        block_dim: tl.constexpr, n_pairs: tl.constexpr, split_half: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        batch_idx = tl.program_id(1)

        offs = tl.arange(0, n_pairs)
        full = tl.arange(0, block_dim)

        # pe is loaded once per (batch, seq) and shared by every head
        freqs_b = tl.where(freqs_batch == 1, 0, batch_idx)
        freqs_s = tl.where(freqs_seq == 1, 0, seq_idx)
        fbase = freqs_ptr + freqs_b * stride_f_b + freqs_s * stride_f_s
        freqs_00 = tl.load(fbase + offs * stride_f_dim)
        freqs_01 = tl.load(fbase + offs * stride_f_dim + stride_f_pair)
        freqs_10 = tl.load(fbase + offs * stride_f_dim + stride_f_rot)
        freqs_11 = tl.load(fbase + offs * stride_f_dim + stride_f_rot + stride_f_pair)

        if split_half:
            dim_idx_0 = offs
            dim_idx_1 = offs + n_pairs
        else:
            dim_idx_0 = offs * 2
            dim_idx_1 = offs * 2 + 1

        scale_0 = tl.load(scale_ptr + dim_idx_0).to(tl.float32)
        scale_1 = tl.load(scale_ptr + dim_idx_1).to(tl.float32)

        x_row = x_ptr + batch_idx * stride_x_b + seq_idx * stride_x_s
        o_row = out_ptr + batch_idx * stride_o_b + seq_idx * stride_o_s

        for h in range(n_heads):
            x_base = x_row + h * stride_x_h
            x_full = tl.load(x_base + full * stride_x_d).to(tl.float32)
            inv_rms = tl.math.rsqrt(tl.sum(x_full * x_full, axis=0) / head_dim + epsilon)

            # re-loaded on purpose: see the note above
            x_0 = tl.load(x_base + dim_idx_0 * stride_x_d).to(tl.float32)
            x_1 = tl.load(x_base + dim_idx_1 * stride_x_d).to(tl.float32)
            x_0 = (x_0 * inv_rms * scale_0).to(tl.bfloat16).to(tl.float32)
            x_1 = (x_1 * inv_rms * scale_1).to(tl.bfloat16).to(tl.float32)

            out_0 = freqs_00 * x_0 + freqs_01 * x_1
            out_1 = freqs_10 * x_0 + freqs_11 * x_1

            o_base = o_row + h * stride_o_h
            tl.store(o_base + dim_idx_0 * stride_o_d, out_0)
            tl.store(o_base + dim_idx_1 * stride_o_d, out_1)

    @triton.jit
    def _rms_rope_rows_dq_kernel(
        acc_ptr, xs_ptr, w_ptr, freqs_ptr, scale_ptr, out_ptr,
        head_dim, n_heads, freqs_batch, freqs_seq,
        stride_a_b, stride_a_s, stride_a_h, stride_a_d,
        stride_xs_b, stride_xs_s,
        stride_o_b, stride_o_s, stride_o_h, stride_o_d,
        stride_f_b, stride_f_s, stride_f_dim, stride_f_rot, stride_f_pair,
        epsilon,
        block_dim: tl.constexpr, n_pairs: tl.constexpr, split_half: tl.constexpr,
    ):
        """`_rms_rope_rows_kernel` reading the INT8 GEMM accumulator directly.

        `acc` is the int32 output of the quantized projection, still in the
        accumulator: the dequant the separate epilogue used to do happens here,
        in fp32, with the same expression and the same bf16 rounding:
        `float(acc) * (w_scale[n] * x_scale[m])` rounded to bf16 (the epilogue's
        `F16(a.convert<float>()) * (w * x_scale[m])` then `convert<BF>()`).
        Everything after that point is the deployed kernel unchanged, so the
        reduction order and the rope arithmetic are identical.
        """
        seq_idx = tl.program_id(0)
        batch_idx = tl.program_id(1)

        offs = tl.arange(0, n_pairs)
        full = tl.arange(0, block_dim)

        x_row_scale = tl.load(xs_ptr + batch_idx * stride_xs_b + seq_idx * stride_xs_s)

        freqs_b = tl.where(freqs_batch == 1, 0, batch_idx)
        freqs_s = tl.where(freqs_seq == 1, 0, seq_idx)
        fbase = freqs_ptr + freqs_b * stride_f_b + freqs_s * stride_f_s
        freqs_00 = tl.load(fbase + offs * stride_f_dim)
        freqs_01 = tl.load(fbase + offs * stride_f_dim + stride_f_pair)
        freqs_10 = tl.load(fbase + offs * stride_f_dim + stride_f_rot)
        freqs_11 = tl.load(fbase + offs * stride_f_dim + stride_f_rot + stride_f_pair)

        if split_half:
            dim_idx_0 = offs
            dim_idx_1 = offs + n_pairs
        else:
            dim_idx_0 = offs * 2
            dim_idx_1 = offs * 2 + 1

        scale_0 = tl.load(scale_ptr + dim_idx_0).to(tl.float32)
        scale_1 = tl.load(scale_ptr + dim_idx_1).to(tl.float32)

        a_row = acc_ptr + batch_idx * stride_a_b + seq_idx * stride_a_s
        o_row = out_ptr + batch_idx * stride_o_b + seq_idx * stride_o_s

        for h in range(n_heads):
            a_base = a_row + h * stride_a_h
            w_base = w_ptr + h * head_dim

            w_full = tl.load(w_base + full)
            a_full = tl.load(a_base + full * stride_a_d)
            x_full = (a_full.to(tl.float32) * (w_full * x_row_scale)).to(tl.bfloat16).to(tl.float32)
            inv_rms = tl.math.rsqrt(tl.sum(x_full * x_full, axis=0) / head_dim + epsilon)

            # re-loaded per half, exactly like the bf16 kernel: same L1 hits
            w_0 = tl.load(w_base + dim_idx_0)
            w_1 = tl.load(w_base + dim_idx_1)
            a_0 = tl.load(a_base + dim_idx_0 * stride_a_d)
            a_1 = tl.load(a_base + dim_idx_1 * stride_a_d)
            x_0 = (a_0.to(tl.float32) * (w_0 * x_row_scale)).to(tl.bfloat16).to(tl.float32)
            x_1 = (a_1.to(tl.float32) * (w_1 * x_row_scale)).to(tl.bfloat16).to(tl.float32)
            x_0 = (x_0 * inv_rms * scale_0).to(tl.bfloat16).to(tl.float32)
            x_1 = (x_1 * inv_rms * scale_1).to(tl.bfloat16).to(tl.float32)

            out_0 = freqs_00 * x_0 + freqs_01 * x_1
            out_1 = freqs_10 * x_0 + freqs_11 * x_1

            o_base = o_row + h * stride_o_h
            tl.store(o_base + dim_idx_0 * stride_o_d, out_0)
            tl.store(o_base + dim_idx_1 * stride_o_d, out_1)

    _kernel = _rms_rope_rows_kernel
    _kernel_dq = _rms_rope_rows_dq_kernel
except Exception as exc:  # pragma: no cover - no triton on this box
    _import_error = exc
    _kernel_dq = None


def _bucket(x, freqs_cis):
    """True for the [B, S, H, D] layout with freqs broadcast per head.

    Mirrors `comfy_kitchen._rope_utils.detect_rms_rope_bnhd`, which is what the
    shipped kernel uses to choose its strides.
    """
    if x.ndim != 4 or freqs_cis.ndim != 6:
        return False
    head_dim = x.shape[3]
    if head_dim == 0 or head_dim % 2 != 0:
        return False
    if tuple(freqs_cis.shape[3:]) != (head_dim // 2, 2, 2):
        return False
    if freqs_cis.shape[0] not in (1, x.shape[0]):
        return False
    return freqs_cis.shape[1] in (1, x.shape[1]) and freqs_cis.shape[2] == 1


def enabled():
    return _kernel is not None and _GATE.disabled_by is None


def _shipped_path_is_triton():
    """Only substitute for the backend the production op would actually pick.

    `comfy/quant_ops.py` disables the Triton backend unless the server is run
    with `--enable-triton-backend`; in that case the shipped op uses its eager
    implementation, which is a *different function* (173/12.2M values), so we
    must not silently replace it.
    """
    try:
        import comfy_kitchen
    except Exception:
        return True  # not running inside ComfyUI: no dispatch to contradict
    try:
        return bool(comfy_kitchen.registry.is_available("triton"))
    except Exception:
        return False


def _rows(x, freqs_cis, scale, epsilon, split_half=True):
    B, S, H, D = x.shape
    if not scale.is_contiguous():
        scale = scale.contiguous()
    out = torch.empty_like(x)
    _kernel[(S, B)](
        x, freqs_cis, scale, out,
        D, H, freqs_cis.shape[0], freqs_cis.shape[1],
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        freqs_cis.stride(0), freqs_cis.stride(1), freqs_cis.stride(3),
        freqs_cis.stride(4), freqs_cis.stride(5),
        epsilon,
        block_dim=_triton.next_power_of_2(D), n_pairs=D // 2,
        split_half=split_half, num_warps=NUM_WARPS)
    return out


def _rows_dq(acc, freqs_cis, scale, epsilon, x_scale, w_scale, split_half=True):
    """As `_rows`, but `acc` is the int32 GEMM accumulator of the projection.

    `x_scale` is the quantizer's per-row scale, [B, S]; `w_scale` is the
    projection's per-output-column scale for this slice, [H * D].
    """
    B, S, H, D = acc.shape
    if not scale.is_contiguous():
        scale = scale.contiguous()
    if not x_scale.is_contiguous():
        x_scale = x_scale.contiguous()
    if not w_scale.is_contiguous():
        w_scale = w_scale.contiguous()
    if x_scale.shape != (B, S):
        x_scale = x_scale.reshape(B, S)
    if w_scale.numel() != H * D:
        raise ValueError("w_scale must cover H * D columns")
    out = torch.empty((B, S, H, D), dtype=torch.bfloat16, device=acc.device)
    _kernel_dq[(S, B)](
        acc, x_scale, w_scale, freqs_cis, scale, out,
        D, H, freqs_cis.shape[0], freqs_cis.shape[1],
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        x_scale.stride(0), x_scale.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        freqs_cis.stride(0), freqs_cis.stride(1), freqs_cis.stride(3),
        freqs_cis.stride(4), freqs_cis.stride(5),
        epsilon,
        block_dim=_triton.next_power_of_2(D), n_pairs=D // 2,
        split_half=split_half, num_warps=NUM_WARPS)
    return out


def _probe_forward(ref):
    """Synthetic strided q/k, the layout the model actually passes.

    Its own generator, so the sampler's noise stream is untouched.  Both q and k
    are returned: a failure should say which one moved.
    """
    device = probe_device()
    gen = torch.Generator(device=device).manual_seed(0)
    B, S, H, D = 1, 256, 24, 128
    qkv = (torch.randn(B, S, 3 * H * D, generator=gen,
                       device=device) * 0.3).to(torch.bfloat16)
    q, k, _ = qkv.chunk(3, dim=-1)
    q = q.view(B, S, H, D)
    k = k.view(B, S, H, D)
    scale = torch.rand(D, generator=gen, device=device) * 0.2 + 0.9
    ang = torch.rand(B, S, 1, D // 2, generator=gen, device=device) * 3.0
    pe = torch.stack([torch.stack([ang.cos(), -ang.sin()], -1),
                      torch.stack([ang.sin(), ang.cos()], -1)], -2)
    ref_q, ref_k = ref(q, k, pe, scale, None, 1e-6)
    return {"q": (_rows(q, pe, scale, 1e-6), ref_q),
            "k": (_rows(k, pe, scale, 1e-6), ref_k)}


def _default_reference():
    """The reference this branch was verified against, when it is importable."""
    from comfy_kitchen.backends.triton.rms_rope import rms_rope_split_half
    return rms_rope_split_half


def _probe_dequant(ref):
    """Does the fused dequant+rope match dequant -> rope, bit for bit?

    The reference dequant is the same fp32 expression in torch, which is what
    `xpuint8_dequant_epilogue` computes (two fp32 multiplies and a
    round-to-nearest-even bf16 store; `debug/bench_epilogue.py` cross-checks the
    two).  The rope half is then the reference kernel, so a mismatch means
    either the dequant rounding or the reduction moved.
    """
    device = probe_device()
    gen = torch.Generator(device=device).manual_seed(1)
    B, S, H, D = 1, 256, 24, 128
    M, C = B * S, 3 * H * D
    acc = torch.randint(-(2 ** 23), 2 ** 23, (M, C), generator=gen,
                        device=device, dtype=torch.int32)
    xs = torch.rand(M, generator=gen, device=device) * 0.02 + 0.001
    ws = torch.rand(C, generator=gen, device=device) * 0.02 + 0.001
    scale = torch.rand(D, generator=gen, device=device) * 0.2 + 0.9
    ang = torch.rand(B, S, 1, D // 2, generator=gen, device=device) * 3.0
    pe = torch.stack([torch.stack([ang.cos(), -ang.sin()], -1),
                      torch.stack([ang.sin(), ang.cos()], -1)], -2)

    achunk = acc.view(B, S, 3 * H * D)
    pairs = {}
    for name, lo in (("q", 0), ("k", H * D)):
        sl = achunk[:, :, lo:lo + H * D].view(B, S, H, D)
        w = ws[lo:lo + H * D]
        ref_x = (acc[:, lo:lo + H * D].to(torch.float32)
                 * (w * xs[:, None])).to(torch.bfloat16).view(B, S, H, D)
        want, _ = ref(ref_x, ref_x, pe, scale, None, 1e-6)
        pairs[name] = (_rows_dq(sl, pe, scale, 1e-6, xs.view(B, S), w), want)
    return pairs


_GATE = Gate(
    "rms_rope", _probe_forward,
    default_reference=_default_reference,
    env="XPU_KERNELS_NO_ROPE", legacy_env="MOSS_NO_ROPE_ROWS",
    num_warps=NUM_WARPS,
)
# The fused dequant variant is a separate permission: it consumes an int32
# accumulator, so it can be wrong in ways the bf16 kernel is not.
_GATE_DQ = Gate(
    "rms_rope_dequant", _probe_dequant,
    default_reference=_default_reference,
    env="XPU_KERNELS_NO_ROPE", legacy_env="MOSS_NO_ROPE_ROWS",
    num_warps=NUM_WARPS,
)


def set_reference(fn=None):
    """Point both gates at the implementation they must match, or clear them.

    ``fn`` must have the signature ``fn(q, k, freqs_cis, scale, k_scale, eps)``
    and be the path the caller falls back to.  Without it, ``comfy_kitchen``'s
    kernel is used when importable; otherwise both kernels stay inactive.
    """
    _GATE.set_reference(fn)
    _GATE_DQ.set_reference(fn)


def gates():
    """This module's permission slips, by name (for the status report)."""
    return {"rms_rope": _GATE, "rms_rope_dequant": _GATE_DQ}


def describe(which="rms_rope"):
    """One line: is this kernel in use, and if not, why."""
    return {"rms_rope": _GATE, "rms_rope_dequant": _GATE_DQ}[which].describe()


def _self_test():
    """Kept for callers that predate the gate object."""
    return _GATE.ok()


def dequant_ready():
    """True when the fused dequant+rope kernel is available on this process."""
    if not enabled() or _kernel_dq is None:
        return False
    if not _shipped_path_is_triton():
        return False
    return _GATE_DQ.ok()


def dequant_usable(x_acc, freqs_cis):
    """True when the fused dequant+rope kernel can take this INT8 accumulator."""
    if x_acc.dtype != torch.int32 or x_acc.device.type not in ACCELERATORS:
        return False
    if not _bucket(x_acc, freqs_cis):
        return False
    return dequant_ready()


def rms_rope_split_half_dequant(q_acc, k_acc, freqs_cis, q_w, k_w, x_scale,
                                q_scale, k_scale=None, epsilon=1e-6, split_half=True):
    """rms_rope_split_half with the INT8 dequant folded into the load."""
    if k_scale is None:
        k_scale = q_scale
    if k_w is None:
        k_w = q_w
    return (_rows_dq(q_acc, freqs_cis, q_scale, epsilon, x_scale, q_w, split_half),
            _rows_dq(k_acc, freqs_cis, k_scale, epsilon, x_scale, k_w, split_half))


def usable(x, freqs_cis, rot_dim=0):
    if not enabled() or not _shipped_path_is_triton():
        return False
    if x.dtype != torch.bfloat16 or x.device.type not in ACCELERATORS:
        return False
    if rot_dim and rot_dim != x.shape[3]:
        return False  # partial rotary: not fused in the shipped kernel either
    if not _bucket(x, freqs_cis):
        return False
    return _GATE.ok()


def rms_rope_split_half(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0):
    """Drop-in for `comfy_kitchen.rms_rope_split_half` on the fused shapes."""
    if k_scale is None:
        k_scale = q_scale
    return (_rows(q, freqs_cis, q_scale, epsilon),
            _rows(k, freqs_cis, k_scale, epsilon))
