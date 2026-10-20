"""Loading and calling ``libxpuint8.so``: the ctypes surface, nothing else.

Kept apart from the ops so that a future C entry point can reuse the loader
without importing the whole API, and so the streaming/pointer plumbing has one
home.
"""

import ctypes
import glob
import os
import sys
from typing import Optional

import torch


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
    here = os.path.dirname(os.path.abspath(__file__))       # .../xpu_kernels/int8
    pkg = os.path.dirname(here)                             # .../xpu_kernels
    for base in (here, pkg):
        yield os.path.join(base, "libxpuint8.so")           # installed beside the code
    root = os.path.dirname(os.path.dirname(pkg))            # the repo, when run in place
    yield os.path.join(root, "build", "libxpuint8.so")
    yield os.path.join(root, "lib", "libxpuint8.so")
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
    if _ERR is None:
        tried = ", ".join(p for p in _candidate_paths() if p)
        _ERR = f"no libxpuint8.so found (tried: {tried}); run ./build.sh"
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
