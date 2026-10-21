"""Is the INT8 forward capturable in a device graph?

The Triton kernels were verified capture-safe (`verify_graph_capture.py`), and
the INT8 *backward* is a plain matmul (verified in
`verify_capture_guards.py`).  The INT8 *forward* is the one that was never
checked, and it is the one a training step actually spends its time in: the
rotation+quantize and the dequant epilogue are SYCL kernels in
``libxpuint8.so``.

Two things have to be true, and only one of them is obvious:

* the C library must enqueue on **torch's** stream, or the work is not part of
  the captured graph at all.  The package does this: the ``*_stream`` entry
  points take the SYCL queue behind ``torch.xpu.current_stream()`` and enqueue
  without a host wait, so the work is ordered with torch's own ops.
* the capture must **replay**: the graph has to contain the kernels, not just
  the surrounding torch ops.  If the C work were enqueued on another queue, the
  capture would succeed and the replay would silently produce the *first*
  input's answer -- the worst kind of bug, because it is not an error, it is
  a stale gradient.

So this script checks the numbers, not the absence of an exception: capture
the INT8 forward, replay it on new data, and compare bit for bit with the
eager result on the same new data.

    PYTHONPATH=python LEVEL_ZERO_V1_SDK_PATH=~/.local/levelzero \\
        python debug/verify_int8_capture.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import xpu_kernels  # noqa: E402
from xpu_kernels import int8  # noqa: E402

DEV = "xpu"


def _weights(n=1024, k=3072, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = (torch.randn((n, k), generator=g, device=DEV) * 0.02).to(torch.bfloat16)
    w8, scale = xpu_kernels.prepare_weight(w)
    return w8, scale


def _check(name, got, want, failures):
    same = got.view(torch.int16).equal(want.view(torch.int16))
    print(f"  {name:34s} {'identical' if same else 'DIFFERS'}")
    if not same:
        failures.append(f"{name}: the replayed result differs from the eager one")
    return same


def main():
    torch.manual_seed(0)
    print(f"device: {torch.xpu.get_device_name(0)}")
    failures = []

    M, K, N = 512, 3072, 1024
    w8, w_scale = _weights(N, K)
    bias = (torch.randn(N, device=DEV) * 0.01).to(torch.bfloat16)

    x = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16)

    # -- the INT8 linear ----------------------------------------------------
    x_s = x.clone()
    g = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(g):
            y_c = int8.linear(x_s, w8, w_scale, bias)
    except Exception as exc:  # noqa: BLE001 - the failure mode is the finding
        print(f"\nCAPTURE FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
    print("capture: ok (INT8 linear: rotate+quantize, _int_mm, dequant epilogue)")

    x2 = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16)
    y2 = int8.linear(x2, w8, w_scale, bias)
    with torch.no_grad():
        x_s.copy_(x2)
    g.replay()
    torch.xpu.synchronize()
    _check("linear: replay vs eager", y_c, y2, failures)

    # -- the fused FFN ------------------------------------------------------
    up8, up_scale = _weights(2 * 4096, K, seed=1)
    down8, down_scale = _weights(K, 4096, seed=2)
    x_s2 = x.clone()
    g2 = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(g2):
            f_c = int8.ffn_situ(x_s2, up8, up_scale, down8, down_scale)
    except Exception as exc:  # noqa: BLE001
        print(f"\nCAPTURE FAILED (ffn_situ): {type(exc).__name__}: {exc}")
        raise SystemExit(1)
    print("capture: ok (ffn_situ: two GEMMs, one fused epilogue)")

    f2 = int8.ffn_situ(x2, up8, up_scale, down8, down_scale)
    with torch.no_grad():
        x_s2.copy_(x2)
    g2.replay()
    torch.xpu.synchronize()
    _check("ffn_situ: replay vs eager", f_c, f2, failures)

    # -- the same input must give the same answer (a stale graph would) -----
    with torch.no_grad():
        x_s.copy_(x)                      # back to the first input
    g.replay()
    torch.xpu.synchronize()
    y1 = int8.linear(x, w8, w_scale, bias)
    _check("linear: replay of the first input", y_c, y1, failures)

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        raise SystemExit(1)
    print("the INT8 forward captures, and the graph replays it: the work is "
          "really in the graph")


if __name__ == "__main__":
    main()
