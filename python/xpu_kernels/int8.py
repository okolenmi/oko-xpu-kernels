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
  kernels).  ``N`` must be a multiple of 16 for the dequant epilogue.
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

import ctypes
import glob
import os
import sys
from typing import Optional, Tuple

import torch

__all__ = [
    "available", "device_name", "load", "rotate", "quantize", "quantize_act",
    "prepare_weight", "dequant", "dequant_slice", "dequant_act_quantize",
    "linear", "linear_acc", "Int8Linear", "selftest",
]

_LIB = None
_TRIED = False
_PATH = None
_ERR = None

# oneAPI runtime libraries the .so links against; preloading them makes the
# library loadable when the process has not already pulled in the SYCL runtime.
_ONEAPI_RUNTIME = (
    "libsycl.so.8", "libsycl.so", "libOpenCL.so.1", "libze_loader.so.1",
    "libze_intel_gpu.so.1", "libimf.so", "libsvml.so", "libirng.so",
    "libintlc.so.5", "libstdc++.so.6",
)


def _candidate_paths():
    for var in ("XPU_KERNELS_LIB", "XPU_INT8_LIB"):   # the old name still works
        env = os.environ.get(var)
        if env:
            yield env
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in ("../../lib/libxpuint8.so", "../../build/libxpuint8.so",
                "../libxpuint8.so"):
        yield os.path.normpath(os.path.join(here, rel))
    yield os.path.join(sys.prefix, "lib", "libxpuint8.so")
    yield "/usr/local/lib/libxpuint8.so"


def _preload():
    """Best-effort RTLD_GLOBAL load of the oneAPI runtime, used only when the
    first ``CDLL`` of libxpuint8 fails (e.g. ``libsycl.so.8`` not on the loader
    path because torch was imported from a wheel with its own copies).  The
    directory list is explicit on purpose: a recursive glob of a oneAPI install
    walks hundreds of thousands of files and looks like a hang."""
    dirs = [os.path.join(sys.prefix, "lib")]
    root = os.environ.get("ONEAPI_ROOT")
    if root:
        dirs += [os.path.join(root, "lib"), os.path.join(root, "compiler", "lib")]
    dirs += sorted(glob.glob("/opt/intel/oneapi/compiler/*/lib"), reverse=True)[:2]
    home = os.path.expanduser("~/.local")
    if os.path.isdir(home):
        for base in sorted(glob.glob(os.path.join(home, "oneapi*")))[:4]:
            dirs += [os.path.join(base, "lib")]
            dirs += sorted(glob.glob(os.path.join(base, "opt", "intel", "oneapi",
                                                  "compiler", "*", "lib")), reverse=True)[:2]
    for d in dirs:
        for name in _ONEAPI_RUNTIME:
            p = os.path.join(d, name)
            if os.path.exists(p):
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def load():
    """Load and bind the library.  Returns the ctypes handle or None.

    Resolved from ``$XPU_KERNELS_LIB`` (or the legacy ``$XPU_INT8_LIB``), then
    ``lib/libxpuint8.so`` or ``build/libxpuint8.so`` next to this repo, then
    ``sys.prefix``.
    """
    global _LIB, _TRIED, _PATH, _ERR
    if _LIB is not None or _TRIED:
        return _LIB
    _TRIED = True
    preloaded = False
    for path in _candidate_paths():
        if not path or not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
        except OSError as exc:
            if preloaded:
                _ERR = f"{path}: {exc}"
                continue
            preloaded = True
            _preload()
            try:
                lib = ctypes.CDLL(path)
            except OSError as exc2:
                _ERR = f"{path}: {exc2}"
                continue
        try:
            lib.xpuint8_init.restype = ctypes.c_int
            lib.xpuint8_device_name.restype = ctypes.c_char_p
            lib.xpuint8_last_error.restype = ctypes.c_char_p
            lib.xpuint8_convrot_groupsize.restype = ctypes.c_int
            _bind(lib)
        except AttributeError as exc:
            _ERR = f"{path}: {exc}"
            continue
        if lib.xpuint8_init() != 0:
            _ERR = f"{path}: {lib.xpuint8_last_error()!r}"
            continue
        _LIB, _PATH = lib, path
        return lib
    return None


def _bind(lib):
    """Declare argtypes once; the stream variants are optional."""
    vp, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32
    lib.xpuint8_convrot_quantize.argtypes = [vp, vp, vp, i64, i64]
    lib.xpuint8_convrot_quantize.restype = ctypes.c_int
    lib.xpuint8_convrot_rotate_bf16.argtypes = [vp, vp, i64, i64]
    lib.xpuint8_convrot_rotate_bf16.restype = ctypes.c_int
    lib.xpuint8_dequant_epilogue.argtypes = [vp, vp, vp, vp, vp, i64, i64]
    lib.xpuint8_dequant_epilogue.restype = ctypes.c_int
    optional = {
        "xpuint8_convrot_quantize_stream": [vp] * 4 + [i64] * 2,
        "xpuint8_convrot_rotate_bf16_stream": [vp] * 3 + [i64] * 2,
        "xpuint8_convrot_quantize_act_stream": [vp] * 4 + [i64] * 2 + [i32],
        "xpuint8_dequant_epilogue_stream": [vp] * 6 + [i64] * 2,
        "xpuint8_dequant_epilogue_slice_stream": [vp] * 6 + [i64] * 3,
        "xpuint8_dequant_act_quantize_stream": [vp] * 6 + [i64] * 2 + [i32],
    }
    for name, args in optional.items():
        fn = getattr(lib, name, None)
        if fn is not None:
            fn.argtypes = args
            fn.restype = ctypes.c_int


def available() -> bool:
    """True when the library loaded and initialised on this machine."""
    return load() is not None


def device_name() -> Optional[str]:
    lib = load()
    if lib is None:
        return None
    name = lib.xpuint8_device_name()
    return name.decode() if name else None


def _lib():
    lib = load()
    if lib is None:
        raise RuntimeError(f"libxpuint8 unavailable: {_ERR or 'not found'}")
    return lib


def _err(lib, what):
    raise RuntimeError(f"{what}: {lib.xpuint8_last_error()!r}")


def _stream(device):
    return ctypes.c_void_p(torch.xpu.current_stream(device).sycl_queue)


def _fp32(t):
    return t.reshape(-1).to(dtype=torch.float32).contiguous()


def _flat2d(x):
    if x.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16, got {x.dtype}")
    if x.device.type != "xpu":
        raise TypeError(f"expected an XPU tensor, got {x.device}")
    x = x if x.is_contiguous() else x.contiguous()
    return x.reshape(-1, x.shape[-1]), x.shape


ACTIVATIONS = {"gelu_tanh": 1, "swiglu": 2, "situ": 3}


def rotate(x):
    """ConvRot rotation only: bf16 [.., K] in, bf16 [.., K] out, no quantization."""
    lib = _lib()
    x2, shape = _flat2d(x)
    out = torch.empty_like(x2)
    m, k = x2.shape
    rc = lib.xpuint8_convrot_rotate_bf16_stream(
        _stream(x.device), ctypes.c_void_p(x2.data_ptr()),
        ctypes.c_void_p(out.data_ptr()), m, k)
    if rc != 0:
        _err(lib, "rotate")
    return out.reshape(shape)


def quantize(x):
    """Rotate and quantize an activation: bf16 [.., K] -> (int8 [.., K], fp32 [..])."""
    lib = _lib()
    x2, shape = _flat2d(x)
    m, k = x2.shape
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    scale = torch.empty((m,), device=x.device, dtype=torch.float32)
    rc = lib.xpuint8_convrot_quantize_stream(
        _stream(x.device), ctypes.c_void_p(x2.data_ptr()),
        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(scale.data_ptr()), m, k)
    if rc != 0:
        _err(lib, "quantize")
    return q.reshape(*shape[:-1], k), scale.reshape(shape[:-1])


def quantize_act(x, activation: str):
    """Rotate+quantize with an activation fused into the load.

    ``gelu_tanh`` consumes [.., K], ``swiglu`` and ``situ`` consume [.., 2K]
    laid out as [hidden, gate]; the result is [.., K].  ``K`` is the *output*
    width in all cases -- the C API takes it that way, because for the pair
    activations it doubles as the stride between the two halves.
    """
    lib = _lib()
    fn = getattr(lib, "xpuint8_convrot_quantize_act_stream", None)
    if fn is None:
        raise RuntimeError("this build has no fused activation quantizer")
    x2, shape = _flat2d(x)
    m, k2 = x2.shape
    act = ACTIVATIONS[activation]
    k = k2 // 2 if act != 1 else k2
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    scale = torch.empty((m,), device=x.device, dtype=torch.float32)
    rc = fn(_stream(x.device), ctypes.c_void_p(x2.data_ptr()),
            ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(scale.data_ptr()),
            m, k, act)
    if rc != 0:
        _err(lib, "quantize_act")
    return q.reshape(*shape[:-1], k), scale.reshape(shape[:-1])


def prepare_weight(w, group: int = 256):
    """Prepare an INT8 weight for :func:`linear`.

    ``w`` is bf16 or fp32 [N, K] in the *unrotated* orientation, i.e. the
    matrix a plain ``x @ w.T`` would use.  Returns ``(int8 [N, K], fp32 [N])``:
    the row-rotated weights and the per-output-channel scale the dequant
    epilogue expects.  Round to nearest, symmetric, clipped to +-127.
    """
    if w.shape[1] % group:
        raise ValueError(f"K={w.shape[1]} is not a multiple of the group size {group}")
    wr = rotate(w.detach().to(device=w.device, dtype=torch.bfloat16).contiguous())
    amax = wr.abs().amax(dim=1).clamp_min(1e-30)
    scale = amax / 127.0
    q = torch.round(wr / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale.contiguous()


def dequant(acc, x_scale, w_scale, bias=None):
    """bf16 dequant of an int32 [M, N] accumulator with per-row and per-column scales."""
    lib = _lib()
    if acc.dim() != 2 or acc.dtype != torch.int32:
        raise TypeError("acc must be an int32 [M, N] tensor")
    m, n = acc.shape
    out = torch.empty((m, n), device=acc.device, dtype=torch.bfloat16)
    xs, ws = _fp32(x_scale), _fp32(w_scale)
    b = None if bias is None else _fp32(bias)
    rc = lib.xpuint8_dequant_epilogue_stream(
        _stream(acc.device), ctypes.c_void_p(acc.data_ptr()),
        ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(xs.data_ptr()),
        ctypes.c_void_p(ws.data_ptr()),
        ctypes.c_void_p(0 if b is None else b.data_ptr()), m, n)
    if rc != 0:
        _err(lib, "dequant")
    return out


def dequant_slice(acc, x_scale, w_scale, bias=None):
    """As :func:`dequant`, for a [M, N] column slice of a wider [M, S] accumulator.

    ``acc`` may be a strided view (``big[:, off:off+N]``); the kernel walks
    ``acc.stride(0)`` between rows, so the slice is read in place instead of
    copied.  Useful when one GEMM produces q|k|v and only one of them needs
    bf16.
    """
    lib = _lib()
    fn = getattr(lib, "xpuint8_dequant_epilogue_slice_stream", None)
    if fn is None:
        raise RuntimeError("this build has no row-strided dequant")
    if acc.dim() != 2 or acc.dtype != torch.int32:
        raise TypeError("acc must be an int32 [M, N] tensor")
    m, n = acc.shape
    out = torch.empty((m, n), device=acc.device, dtype=torch.bfloat16)
    xs, ws = _fp32(x_scale), _fp32(w_scale)
    b = None if bias is None else _fp32(bias)
    rc = fn(_stream(acc.device), ctypes.c_void_p(acc.data_ptr()),
            ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(xs.data_ptr()),
            ctypes.c_void_p(ws.data_ptr()),
            ctypes.c_void_p(0 if b is None else b.data_ptr()),
            m, n, acc.stride(0))
    if rc != 0:
        _err(lib, "dequant_slice")
    return out


def dequant_act_quantize(acc, x_scale, w_scale, activation="situ"):
    """Dequant, activate, rotate and quantize an up|gate accumulator in one pass.

    ``acc`` is int32 [M, 2K] holding [gate | up].  Returns ``(int8 [M, K],
    fp32 [M])`` -- the quantized activation for the *next* INT8 linear, with
    the bf16 intermediate never written.  This is what makes a
    ``down(situ(up(x)))`` block two kernels instead of eight.
    """
    lib = _lib()
    fn = getattr(lib, "xpuint8_dequant_act_quantize_stream", None)
    if fn is None:
        raise RuntimeError("this build has no fused dequant+activation+quantize")
    if acc.dim() != 2 or acc.dtype != torch.int32 or acc.shape[1] % 2:
        raise TypeError("acc must be an int32 [M, 2K] tensor")
    m, k2 = acc.shape
    k = k2 // 2
    q = torch.empty((m, k), device=acc.device, dtype=torch.int8)
    scale = torch.empty((m,), device=acc.device, dtype=torch.float32)
    xs, ws = _fp32(x_scale), _fp32(w_scale)
    rc = fn(_stream(acc.device), ctypes.c_void_p(acc.data_ptr()),
            ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(scale.data_ptr()),
            ctypes.c_void_p(xs.data_ptr()), ctypes.c_void_p(ws.data_ptr()),
            m, k, ACTIVATIONS[activation])
    if rc != 0:
        _err(lib, "dequant_act_quantize")
    return q, scale


def linear_acc(x, w8, w_scale, act: Optional[str] = None):
    """The GEMM without the output dequant: ``(int32 [M, N], fp32 [M])``.

    Use this when the consumer can walk the accumulator itself -- a fused
    dequant+activation, a rope/norm kernel, or a strided dequant of one slice
    of a fused projection.  ``act`` fuses an activation into the *input*
    quantizer (``gelu_tanh`` on [.., K], ``swiglu``/``situ`` on [.., 2K]).
    """
    if act is None:
        xq, xs = quantize(x)
    else:
        xq, xs = quantize_act(x, act)
    if w8.dtype != torch.int8 or w8.dim() != 2:
        raise TypeError("w8 must be an int8 [N, K] tensor")
    if xq.shape[-1] != w8.shape[1]:
        raise ValueError(f"K mismatch: activation {xq.shape[-1]} vs weight {w8.shape[1]}")
    acc = torch._int_mm(xq.reshape(-1, xq.shape[-1]), w8.t())
    return acc, xs.reshape(-1)


def linear(x, w8, w_scale, bias=None, act: Optional[str] = None):
    """INT8 ConvRot linear: bf16 [.., K] -> bf16 [.., N] (N = w8.shape[0])."""
    acc, xs = linear_acc(x, w8, w_scale, act)
    out = dequant(acc, xs, w_scale, bias)
    shape = (*x.shape[:-1], w8.shape[0])
    return out.reshape(shape)


class Int8Linear(torch.nn.Module):
    """Drop-in ``nn.Linear`` replacement for inference with prepared weights.

        mod = Int8Linear.from_linear(model.some_linear)
        model.some_linear = mod
    """

    def __init__(self, w8, w_scale, bias=None):
        super().__init__()
        self.register_buffer("w8", w8.detach())
        self.register_buffer("w_scale", w_scale.detach().to(torch.float32))
        self.bias = None if bias is None else torch.nn.Parameter(
            bias.detach().to(torch.float32), requires_grad=False)

    @classmethod
    def from_linear(cls, mod):
        q, scale = prepare_weight(mod.weight.detach())
        return cls(q, scale, getattr(mod, "bias", None))

    def forward(self, x):
        return linear(x, self.w8, self.w_scale, self.bias)

    def extra_repr(self):
        n, k = self.w8.shape
        return f"in_features={k}, out_features={n}, bias={self.bias is not None}"


def selftest(verbose: bool = True) -> bool:
    """Quick end-to-end check on this machine.  Returns True when it passes.

    Not a bit-exactness suite -- it checks that the pieces fit together (weight
    prep, quantization round trip, GEMM, dequant, the fused activation path)
    and that the INT8 path is close to bf16, as an INT8 quantization must be.
    The bit-exactness suite for the fused kernels lives in the project's
    ``debug/`` directory (``verify_situ.py``, ``verify_qkv_rope_fuse.py``).
    """
    if not available():
        if verbose:
            print(f"libxpuint8 unavailable: {_ERR}")
        return False
    dev = torch.device("xpu:0")
    torch.xpu.set_device(0)
    g = torch.Generator(device=dev).manual_seed(0)
    M, K, N = 512, 1024, 512
    x = (torch.randn(M, K, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    w = (torch.randn(N, K, generator=g, device=dev) * 0.05).to(torch.bfloat16)
    bias = torch.randn(N, generator=g, device=dev) * 0.1

    w8, ws = prepare_weight(w)
    y = linear(x, w8, ws, bias)
    ref = (x.float() @ w.to(torch.float32).t() + bias).to(torch.bfloat16)
    rel = ((y.float() - ref.float()).norm() / ref.float().norm()).item()

    # the accumulator path must agree with linear() exactly
    acc, xs = linear_acc(x, w8, ws)
    same = dequant(acc, xs, ws, bias).view(torch.int16).equal(y.view(torch.int16))

    # a column slice of a wider accumulator, read in place with its row stride
    acc_w, xs_w = linear_acc(x, torch.cat([w8, w8], 0), torch.cat([ws, ws]))
    half = acc_w[:, :N]
    sl = dequant_slice(half, xs_w, ws, bias)
    sl_ref = dequant(half.contiguous(), xs_w, ws, bias)
    slice_same = sl.view(torch.int16).equal(sl_ref.view(torch.int16))

    # the fused situ epilogue against the documented unfused equivalent:
    # dequant -> activation + rotate + quantize (that chain's numbers, not the
    # same bit pattern, because the fused one keeps the accumulator in fp32)
    kff = 1024
    wup = (torch.randn(2 * kff, K, generator=g, device=dev) * 0.05).to(torch.bfloat16)
    wup8, wup_s = prepare_weight(wup)
    xup = (torch.randn(M, K, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    acc2, xs2 = linear_acc(xup, wup8, wup_s)
    q2, s2 = dequant_act_quantize(acc2, xs2, wup_s)
    d = dequant(acc2, xs2, wup_s)
    q2_ref, s2_ref = quantize_act(d, "situ")
    code_diff = (q2.to(torch.int16) - q2_ref.to(torch.int16)).abs().max().item()
    scale_rel = ((s2 - s2_ref).abs() / s2_ref.clamp_min(1e-20)).max().item()
    ok = (same and slice_same and rel < 0.02 and q2.dtype == torch.int8
          and code_diff <= 1 and scale_rel < 0.05)
    if verbose:
        print(f"device            : {device_name()}")
        print(f"INT8 vs bf16      : relative error {rel:.4f} (expect < 0.02)")
        print(f"acc path == linear: {same}")
        print(f"strided slice     : {slice_same}")
        print(f"fused situ vs chain: max code diff {code_diff}, "
              f"max scale rel {scale_rel:.4f}")
        print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
