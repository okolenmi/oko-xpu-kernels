"""Shared helpers for the debug/ instruments.

Every script here talks to the device directly, so they all share the same
ctypes bindings and the same min-of-N timing discipline (the B580's timings are
noisy enough that a single mean over 20 iterations is not trustworthy).
"""
import ctypes
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(REPO, "build", "libxpuint8.so")

if REPO not in sys.path:
    sys.path.insert(0, REPO)

_LIB = None


def load_lib(path=None):
    """Load libxpuint8.so with every entry point declared.

    NOTE: always use this rather than calling ctypes.CDLL yourself. ctypes
    defaults unannotated int arguments to a 32-bit C int, which silently
    truncates a 64-bit device pointer and makes the kernel write through
    garbage -- that presents as UR_RESULT_ERROR_DEVICE_LOST, not as an
    argument error.
    """
    global _LIB
    if _LIB is not None:
        return _LIB
    from benchmarks.bench_pipeline import load_lib as _load  # noqa
    return _load(path)


def bench(fn, n=20, warmup=5, reps=3):
    """Min-of-reps mean. Returns milliseconds."""
    import torch
    best = float("inf")
    for _ in range(reps):
        for _ in range(warmup):
            fn()
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.xpu.synchronize()
        best = min(best, (time.perf_counter() - t0) / n * 1000)
    return best


def device_ms(fn, n=20):
    """Mean *device* time per call from the profiler's kernel events: what a captured graph replays, host launch cost excluded.

    ``bench`` is wall time with the device synchronized at the end, which for kernels shorter than the host's launch cost
    (~50 us from Python, more for Triton) measures the launching, not the kernel.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.XPU]) as prof:
        for _ in range(n):
            fn()
        torch.xpu.synchronize()
    total = 0.0
    for event in prof.key_averages():
        t = getattr(event, "self_device_time_total", None)
        total += t if t is not None else getattr(event, "self_xpu_time_total", 0.0)
    return total / n / 1000.0


def native(name, extra_args=()):
    """Build (if needed) and run a probe from debug/native, return its stdout."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    build = os.path.join(here, "native", "build.sh")
    subprocess.run([build], check=True, capture_output=True)
    exe = os.path.join(here, "native", name)
    return subprocess.run([exe, *map(str, extra_args)],
                          capture_output=True, text=True).stdout


def require_lib():
    if not os.path.exists(LIB_PATH):
        raise SystemExit(f"{LIB_PATH} not found -- run ./build.sh first")
    return LIB_PATH