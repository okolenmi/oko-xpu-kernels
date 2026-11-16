"""A flash attention in Triton whose device graph can be captured: ``softmax(q k^T / sqrt(d)) v``, forward and backward.

Why it exists.  The attention torch runs on XPU (oneDNN's flash kernel) takes its work-group scratch size at launch time, which the
SYCL Graph extension cannot record, so a captured training step falls back to the math attention and pays for the scores it
materializes (+36 % device time on SDXL's UNet).  A Triton kernel allocates its on-chip memory statically and has no such limit;
``debug/verify_attention_capture.py`` is the check that it captures and replays.

What it is.  The standard FlashAttention-2 recurrence: the scores never leave the chip, the forward keeps a running max and sum
(base 2) and stores the row's log-sum-exp for the backward.  The backward is two kernels, one that owns a block of queries (``dq``)
and one that owns a block of keys (``dk``, ``dv``): no atomics, so the result does not depend on scheduling.  Inputs are
``[batch, heads, length, head_dim]``; no mask, no dropout, not causal; the key and query lengths need not match or be a multiple of
a block (cross-attention to 77 text tokens is the point).

What it costs.  It runs near torch's flash path and several times faster than the math fallback that a captured step has to use
(``docs/experiments.md`` 5c).  That needs Triton to see the card's matrix units: its Intel backend asks an OpenCL platform and is
told there are none, so ``tl.dot`` lowers to FMA and the same kernel crawls.  Set ``TRITON_INTEL_DEVICE_EXTENSIONS`` before the first
compile -- the warning below prints the three extensions -- or ``usable()`` declines.  The block and warp tables are measurements;
``XPU_KERNELS_ATTENTION_CONFIG`` (``BLOCK_M,BLOCK_N,num_warps,num_stages``) overrides all three kernels, which is a probe, not a
tuning.

Both gates compare against torch's ``scaled_dot_product_attention`` on a relative budget: the reduction order differs and bf16
rounds the probabilities before the second product, as every flash kernel does.  ``XPU_KERNELS_NO_ATTENTION=1`` switches it off.
"""

from __future__ import annotations

import logging
import math
import os

import torch

from ._gate import ACCELERATORS, Gate, probe_device

log = logging.getLogger(__name__)

__all__ = ["attention", "gates", "set_reference", "usable"]

_triton = None
_import_error = None
_kernels = None

DTYPES = (torch.bfloat16, torch.float16, torch.float32)
HEAD_DIMS = (16, 32, 64, 128)
LOG2E = 1.4426950408889634

#: (BLOCK_M, BLOCK_N, num_warps, num_stages) by head dimension for the forward, measured on the B580 (``docs/experiments.md`` 5c).
#: Head dimension 64 is SDXL's and is the only one measured; the others keep the tutorial's starting point.
DEFAULT_CONFIG = {16: (64, 64, 4, 2), 32: (128, 64, 8, 2), 64: (128, 32, 8, 2), 128: (128, 32, 8, 2)}

#: The backward's two kernels do different work from the forward, so they want different blocks: dQ owns BLOCK_M queries and
#: loops over keys in BLOCK_N; dK/dV owns BLOCK_N keys and loops over queries in BLOCK_M.  Sharing the forward's block sizes
#: left the dK/dV kernel with 64 keys per program and a 128-wide query tile, which spilled and cost an order of magnitude
#: (experiments.md 5c).
BWD_DQ_CONFIG = {16: (64, 32, 4, 2), 32: (128, 32, 8, 2), 64: (128, 16, 8, 2), 128: (128, 32, 8, 2)}
BWD_DKDV_CONFIG = {16: (16, 128, 8, 2), 32: (16, 128, 8, 2), 64: (16, 64, 8, 2), 128: (16, 128, 8, 2)}

try:
    import triton
    import triton.language as tl

    _triton = triton

    @triton.jit
    def _fwd_kernel(Q, K, V, O, LSE, sm_scale, Lq, Lk,
                    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        bh = tl.program_id(1).to(tl.int64)
        q_base, k_base = bh * Lq * D, bh * Lk * D
        # One descriptor per head rather than over the whole [batch*heads*length, D] view: the rows past Lq then clip at the
        # head's end instead of writing into the next head, which is what a masked pointer store did before.
        desc_q = tl.make_tensor_descriptor(Q + q_base, shape=[Lq, D], strides=[D, 1], block_shape=[BLOCK_M, D])
        desc_k = tl.make_tensor_descriptor(K + k_base, shape=[Lk, D], strides=[D, 1], block_shape=[BLOCK_N, D])
        desc_v = tl.make_tensor_descriptor(V + k_base, shape=[Lk, D], strides=[D, 1], block_shape=[BLOCK_N, D])
        desc_o = tl.make_tensor_descriptor(O + q_base, shape=[Lq, D], strides=[D, 1], block_shape=[BLOCK_M, D])
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row = offs_m < Lq
        q = desc_q.load([pid_m * BLOCK_M, 0])
        qk_scale = sm_scale * 1.4426950408889634
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        for start_n in range(0, Lk, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            k_t = desc_k.load([start_n, 0]).T
            s = tl.dot(q, k_t) * qk_scale
            s = tl.where(start_n + tl.arange(0, BLOCK_N)[None, :] < Lk, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp2(s - m_new[:, None])
            alpha = tl.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = desc_v.load([start_n, 0])
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(v.dtype), v, acc)
            m_i = m_new
        acc = acc / l_i[:, None]
        desc_o.store([pid_m * BLOCK_M, 0], acc.to(O.dtype.element_ty))
        tl.store(LSE + bh * Lq + offs_m, m_i + tl.log2(l_i), mask=row)

    @triton.jit
    def _delta_kernel(O, DO, DELTA, Lq, D: tl.constexpr, BLOCK_M: tl.constexpr):
        pid_m = tl.program_id(0)
        bh = tl.program_id(1).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        mask = offs_m[:, None] < Lq
        base = bh * Lq * D + offs_m[:, None] * D + offs_d[None, :]
        o = tl.load(O + base, mask=mask, other=0.0).to(tl.float32)
        do = tl.load(DO + base, mask=mask, other=0.0).to(tl.float32)
        tl.store(DELTA + bh * Lq + offs_m, tl.sum(o * do, 1), mask=offs_m < Lq)

    @triton.jit
    def _bwd_dq_kernel(Q, K, V, DO, LSE, DELTA, DQ, sm_scale, Lq, Lk,
                       D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        bh = tl.program_id(1).to(tl.int64)
        q_base, k_base = bh * Lq * D, bh * Lk * D
        desc_q = tl.make_tensor_descriptor(Q + q_base, shape=[Lq, D], strides=[D, 1], block_shape=[BLOCK_M, D])
        desc_k = tl.make_tensor_descriptor(K + k_base, shape=[Lk, D], strides=[D, 1], block_shape=[BLOCK_N, D])
        desc_v = tl.make_tensor_descriptor(V + k_base, shape=[Lk, D], strides=[D, 1], block_shape=[BLOCK_N, D])
        desc_do = tl.make_tensor_descriptor(DO + q_base, shape=[Lq, D], strides=[D, 1], block_shape=[BLOCK_M, D])
        desc_dq = tl.make_tensor_descriptor(DQ + q_base, shape=[Lq, D], strides=[D, 1], block_shape=[BLOCK_M, D])
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row = offs_m < Lq
        q = desc_q.load([pid_m * BLOCK_M, 0])
        do = desc_do.load([pid_m * BLOCK_M, 0])
        lse = tl.load(LSE + bh * Lq + offs_m, mask=row, other=0.0)
        delta = tl.load(DELTA + bh * Lq + offs_m, mask=row, other=0.0)
        qk_scale = sm_scale * 1.4426950408889634
        dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        for start_n in range(0, Lk, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            k = desc_k.load([start_n, 0])
            v = desc_v.load([start_n, 0])
            s = tl.dot(q, k.T) * qk_scale
            p = tl.where(start_n + tl.arange(0, BLOCK_N)[None, :] < Lk, tl.exp2(s - lse[:, None]), 0.0)
            dp = tl.dot(do, v.T)
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k)
        desc_dq.store([pid_m * BLOCK_M, 0], (dq * sm_scale).to(DQ.dtype.element_ty))

    @triton.jit
    def _bwd_dkdv_kernel(Q, K, V, DO, LSE, DELTA, DK, DV, sm_scale, Lq, Lk,
                         D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_n = tl.program_id(0)
        bh = tl.program_id(1).to(tl.int64)
        q_base, k_base = bh * Lq * D, bh * Lk * D
        desc_q = tl.make_tensor_descriptor(Q + q_base, shape=[Lq, D], strides=[D, 1], block_shape=[BLOCK_M, D])
        desc_k = tl.make_tensor_descriptor(K + k_base, shape=[Lk, D], strides=[D, 1], block_shape=[BLOCK_N, D])
        desc_v = tl.make_tensor_descriptor(V + k_base, shape=[Lk, D], strides=[D, 1], block_shape=[BLOCK_N, D])
        desc_do = tl.make_tensor_descriptor(DO + q_base, shape=[Lq, D], strides=[D, 1], block_shape=[BLOCK_M, D])
        desc_dk = tl.make_tensor_descriptor(DK + k_base, shape=[Lk, D], strides=[D, 1], block_shape=[BLOCK_N, D])
        desc_dv = tl.make_tensor_descriptor(DV + k_base, shape=[Lk, D], strides=[D, 1], block_shape=[BLOCK_N, D])
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        col = offs_n < Lk
        k = desc_k.load([pid_n * BLOCK_N, 0])
        v = desc_v.load([pid_n * BLOCK_N, 0])
        qk_scale = sm_scale * 1.4426950408889634
        dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
        for start_m in range(0, Lq, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m = start_m + tl.arange(0, BLOCK_M)
            row = offs_m < Lq
            q = desc_q.load([start_m, 0])
            do = desc_do.load([start_m, 0])
            lse = tl.load(LSE + bh * Lq + offs_m, mask=row, other=0.0)
            delta = tl.load(DELTA + bh * Lq + offs_m, mask=row, other=0.0)
            s_t = tl.dot(k, q.T) * qk_scale
            p_t = tl.where(row[None, :], tl.exp2(s_t - lse[None, :]), 0.0)
            dv += tl.dot(p_t.to(do.dtype), do)
            dp_t = tl.dot(v, do.T)
            ds_t = p_t * (dp_t - delta[None, :])
            dk += tl.dot(ds_t.to(q.dtype), q)
        desc_dk.store([pid_n * BLOCK_N, 0], (dk * sm_scale).to(DK.dtype.element_ty))
        desc_dv.store([pid_n * BLOCK_N, 0], dv.to(DV.dtype.element_ty))

    _kernels = (_fwd_kernel, _delta_kernel, _bwd_dq_kernel, _bwd_dkdv_kernel)

except Exception as exc:  # noqa: BLE001 - no triton means no kernel
    _import_error = exc


_allocator_device = None


def _ensure_allocator(device) -> None:
    """Give Triton somewhere to build the device-side tensor descriptors, unless the caller already did.

    ``tl.make_tensor_descriptor`` allocates its descriptor at launch time, so a real device buffer is needed; under a graph
    capture the graph's memory pool owns that buffer and replays it, ``debug/verify_attention_capture.py`` checks so.
    """
    if _triton is None:
        return
    global _allocator_device
    from triton.runtime import _allocation

    if not isinstance(_allocation._allocator.get(), _allocation.NullAllocator):
        return
    device = torch.device(device)
    _allocator_device = device
    _triton.set_allocator(lambda size, alignment, stream: torch.empty(size, dtype=torch.int8, device=_allocator_device))


_xmx_state = None


def _matrix_units_missing() -> bool:
    """True when Triton's compiler target has no subgroup matrix multiply, so ``tl.dot`` will be lowered to FMA.

    Only Triton's answer matters: torch's ``get_device_capability`` was seen to report the extension differently from one process to
    the next on this card, while Triton's target decides what the compiler emits (``docs/experiments.md`` 5c).  Cached because the
    driver fixes it on first use.
    """
    global _xmx_state
    if _xmx_state is None:
        try:
            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                _xmx_state = False  # not an Intel target; the question does not apply
            else:
                _xmx_state = not _triton.runtime.driver.active.get_current_target().arch.get(
                    "has_subgroup_matrix_multiply_accumulate", False)
        except Exception:  # noqa: BLE001 - under the interpreter, or before a driver exists, this says nothing
            _xmx_state = False
    return _xmx_state


def _warn_if_matrix_units_are_missing() -> None:
    """One warning when the compiler was told the card has no XMX.

    Triton asks an OpenCL platform for the device extensions when one exists.  A Mesa rusticl platform answers for the same card
    and does not list ``cl_intel_subgroup_matrix_multiply_accumulate``, so the target says "no matrix units" and ``tl.dot`` is
    lowered to FMA, which makes the kernel crawl (docs/experiments.md 5c).  The fix is the environment variable Triton's own knob
    reads; it has to be set before the first compile because the target is cached.
    """
    if _triton is None:
        return
    if _matrix_units_missing():
        log.warning("Triton's Intel backend compiled for a device without subgroup matrix multiply "
                    "(has_subgroup_matrix_multiply_accumulate=false), so tl.dot lowers to FMA and this attention will be slow. "
                    "A card with XMX can be misdetected because an OpenCL platform answered the extension query; set "
                    "TRITON_INTEL_DEVICE_EXTENSIONS='cl_intel_subgroup_matrix_multiply_accumulate "
                    "cl_intel_subgroup_2d_block_io cl_intel_bfloat16_conversions' before the first Triton compile. The last "
                    "extension is fastest but makes the rope's bit-exact self-test fall back (docs/experiments.md 5c).")


_warn_if_matrix_units_are_missing()


def enabled() -> bool:
    return _kernels is not None and os.environ.get("XPU_KERNELS_NO_ATTENTION", "") in ("", "0")


def config(head_dim: int, kernel: str = "forward") -> tuple[int, int, int, int]:
    """``(BLOCK_M, BLOCK_N, num_warps, num_stages)``: the environment's override, else the table for that kernel."""
    text = os.environ.get("XPU_KERNELS_ATTENTION_CONFIG", "")
    if text:
        parts = tuple(int(part) for part in text.split(","))
        if len(parts) != 4:
            raise ValueError("XPU_KERNELS_ATTENTION_CONFIG is BLOCK_M,BLOCK_N,num_warps,num_stages")
        return parts  # type: ignore[return-value]
    table = {"forward": DEFAULT_CONFIG, "dq": BWD_DQ_CONFIG, "dkdv": BWD_DKDV_CONFIG}[kernel]
    return table[head_dim]


def reference(q, k, v):
    """The eager attention the kernels replace (and the default reference of both gates)."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def _forward(q, k, v, scale):
    batch, heads, lq, d = q.shape
    lk = k.shape[2]
    block_m, block_n, warps, stages = config(d)
    _ensure_allocator(q.device)
    out = torch.empty_like(q)
    lse = torch.empty((batch, heads, lq), dtype=torch.float32, device=q.device)
    grid = (math.ceil(lq / block_m), batch * heads)
    _kernels[0][grid](q, k, v, out, lse, scale, lq, lk, D=d, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=warps, num_stages=stages)
    return out, lse


def _backward(q, k, v, out, lse, dout, scale):
    batch, heads, lq, d = q.shape
    lk = k.shape[2]
    bh = batch * heads
    _ensure_allocator(q.device)
    m, n, warps, stages = config(d, "dq")
    m1, n1, warps1, stages1 = config(d, "dkdv")
    delta = torch.empty_like(lse)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _kernels[1][(math.ceil(lq / m), bh)](out, dout, delta, lq, D=d, BLOCK_M=m, num_warps=4)
    _kernels[2][(math.ceil(lq / m), bh)](q, k, v, dout, lse, delta, dq, scale, lq, lk, D=d, BLOCK_M=m, BLOCK_N=n,
                                         num_warps=warps, num_stages=stages)
    _kernels[3][(math.ceil(lk / n1), bh)](q, k, v, dout, lse, delta, dk, dv, scale, lq, lk, D=d, BLOCK_M=m1, BLOCK_N=n1,
                                          num_warps=warps1, num_stages=stages1)
    return dq, dk, dv


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        _FORWARD.require()
        ctx.fused_backward = _BACKWARD.ok()
        ctx.scale = scale
        out, lse = _forward(q, k, v, scale)
        ctx.save_for_backward(q, k, v, out, lse)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        if ctx.fused_backward:
            dq, dk, dv = _backward(q, k, v, out, lse, dout.contiguous(), ctx.scale)
            return dq, dk, dv, None
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(True) for t in (q, k, v)]
            grads = torch.autograd.grad(reference(*leaves), leaves, dout)
        return (*grads, None)


def attention(q, k, v, scale: float | None = None):
    """``softmax(q k^T * scale) v`` for ``[batch, heads, length, head_dim]`` tensors; differentiable.

    ``scale`` defaults to ``1 / sqrt(head_dim)``, as ``scaled_dot_product_attention``.
    """
    q, k, v = (t if t.is_contiguous() else t.contiguous() for t in (q, k, v))
    if q.device.type not in ACCELERATORS:
        raise RuntimeError(f"the fused attention runs on {ACCELERATORS}, not {q.device.type}")
    return _Attention.apply(q, k, v, scale if scale is not None else 1.0 / math.sqrt(q.shape[-1]))


def usable(q, k=None, v=None) -> bool:
    """May the fused attention be used for these inputs?  (Checks the shapes of whatever is passed, then latches the gates.)

    Declines when the compiler was told the card has no matrix units: the kernels would still be right but ``tl.dot`` would
    lower to FMA, which loses to the math attention this exists to replace (``docs/experiments.md`` 5c).  A direct
    ``attention()`` call still runs; only the automatic choice refuses.
    """
    if not enabled() or q.dtype not in DTYPES or q.device.type not in ACCELERATORS or q.dim() != 4:
        return False
    if q.shape[-1] not in HEAD_DIMS:
        return False
    if _matrix_units_missing():
        return False
    for other in (k, v):
        if other is not None and (other.dim() != 4 or other.shape[-1] != q.shape[-1] or other.shape[:2] != q.shape[:2]
                                  or other.dtype != q.dtype or other.device != q.device):
            return False
    if k is not None and v is not None and k.shape != v.shape:
        return False
    _BACKWARD.ok()  # latch it here too: a first verification cannot happen inside a device-graph capture
    return _FORWARD.ok()


# SDXL: self-attention at 1024 tokens (10 heads), cross-attention to 77 text tokens (not a multiple of a block), 256 tokens (20 heads);
# 100 queries is not a multiple of a block either.
_SHAPES = (((1, 10, 1024, 64), (1, 10, 1024, 64)), ((1, 10, 1024, 64), (1, 10, 77, 64)),
           ((1, 20, 256, 64), (1, 20, 256, 64)), ((2, 4, 100, 64), (2, 4, 77, 64)))


def _inputs(qs, ks, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    return tuple(torch.randn(s, generator=gen, device=device).to(torch.bfloat16) for s in (qs, ks, ks))


def _probe_forward(ref):
    if _kernels is None:
        raise RuntimeError("the triton attention kernels are not available here")
    device = probe_device()
    pairs = {}
    for qs, ks in _SHAPES:
        q, k, v = _inputs(qs, ks, device, 0)
        pairs[f"forward {qs[2]}x{ks[2]}"] = (_forward(q, k, v, 1.0 / math.sqrt(qs[-1]))[0], ref(q, k, v))
    return pairs


def _probe_backward(ref):
    if _kernels is None:
        raise RuntimeError("the triton attention kernels are not available here")
    device = probe_device()
    pairs = {}
    for qs, ks in _SHAPES:
        q, k, v = _inputs(qs, ks, device, 1)
        scale = 1.0 / math.sqrt(qs[-1])
        gen = torch.Generator(device=device).manual_seed(2)
        dout = torch.randn(qs, generator=gen, device=device).to(torch.bfloat16)
        out, lse = _forward(q, k, v, scale)
        got = _backward(q, k, v, out, lse, dout, scale)
        # the truth is the float32 chain: a bf16 reference would carry its own rounding into the comparison
        with torch.enable_grad():
            leaves = [t.float().requires_grad_(True) for t in (q, k, v)]
            want = torch.autograd.grad(ref(*leaves), leaves, dout.float())
        for name, g, w in zip(("dq", "dk", "dv"), got, want):
            pairs[f"backward {name} {qs[2]}x{ks[2]}"] = (g, w)
    return pairs


#: Budgets on the relative L2 error.  A wrong formula or a missing mask is off by 1e-1 or more; bf16's own rounding of the probabilities
#: and gradients, against a float32 chain, is a few 1e-3.
REL_FORWARD = 2e-2
REL_BACKWARD = 5e-2

_FORWARD = Gate("attention", _probe_forward, default_reference=lambda: reference, env="XPU_KERNELS_NO_ATTENTION", rel=REL_FORWARD)
_BACKWARD = Gate("attention_backward", _probe_backward, default_reference=lambda: reference, env="XPU_KERNELS_NO_ATTENTION",
                 rel=REL_BACKWARD)


def set_reference(fn=None):
    """Set (or clear) the attention both gates compare against: ``fn(q, k, v) -> out``, differentiable."""
    _FORWARD.set_reference(fn)
    _BACKWARD.set_reference(fn)


def gates():
    return {"attention": _FORWARD, "attention_backward": _BACKWARD}
