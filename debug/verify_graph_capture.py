"""Is the fused rope forward+backward capturable in a device graph?

A training step that wants a captured graph needs three things from a kernel:
no host synchronisation, no allocation the allocator cannot serve from the
graph's pool, and no control flow that depends on a device value.  This script
checks the first two by actually capturing `rms_rope_split_half` and its
backward, replaying them on new data, and comparing the replayed gradient with
the eager one bit for bit -- a graph that changes the numbers is not a graph you
can ship.

Notes that matter when reading the result:

* The gate must be warm *before* capture: the first call probes the kernel
  against the reference, which synchronises and reads values back.  `status()`
  does the same for every kernel in one go.
* Two stages are checked: the rope forward+backward, and the adaLN one, whose
  backward allocates a partials buffer for its two-stage modulation reduction.
* Allocations made *inside* a capture are fine -- they are carved from the
  graph's private pool and replayed at the same addresses.  The requirement is
  that nothing inside the captured region waits on the device or moves data to
  the host, and replay must not need a *new* allocation size.
* Inputs are static buffers: data is copied in before each replay, which is how
  a trainer would use this.

    PYTHONPATH=python LEVEL_ZERO_V1_SDK_PATH=~/.local/levelzero \\
        python debug/verify_graph_capture.py
"""

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import xpu_kernels.triton as kt  # noqa: E402
from xpu_kernels.triton import adaln, reference, rope  # noqa: E402

B, S, H, D = 1, 3973, 24, 128
EPS = 1e-6


def timed(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    events = [(torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True))
              for _ in range(iters)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.xpu.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in events)


def main():
    dev = "xpu"
    # this script reuses the same leaf tensors every iteration, which makes
    # autograd's AccumulateGrad node outlive its stream; harmless here, and the
    # warning has nothing to do with the kernels
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    torch.manual_seed(0)
    print(f"device: {torch.xpu.get_device_name(0)}")

    # -- warm everything up: the gate probes read values back, and Triton
    #    compiles on first use.  Neither is legal inside a capture.
    for line in kt.status():
        if "backward" in line or line.startswith("rms_rope "):
            print("  " + line)
    assert rope._GATE_BWD.ok(), "backward gate is not active; nothing to capture"

    pe = torch.randn(B, S, 1, D // 2, 2, 2, device=dev).to(torch.bfloat16) * 0.1
    scale = torch.randn(H, D, device=dev).to(torch.bfloat16) * 0.05
    q = torch.randn(B, S, H, D, device=dev).to(torch.bfloat16)
    k = torch.randn(B, S, H, D, device=dev).to(torch.bfloat16)
    dyq = torch.randn(B, S, H, D, device=dev).to(torch.bfloat16)
    dyk = torch.randn(B, S, H, D, device=dev).to(torch.bfloat16)

    def eager(q_, k_, dyq_, dyk_):
        qr = q_.detach().clone().requires_grad_(True)
        kr = k_.detach().clone().requires_grad_(True)
        yq, yk = rope.rms_rope_split_half(qr, kr, pe, scale, scale, EPS)
        dq, dk = torch.autograd.grad((yq, yk), (qr, kr), (dyq_, dyk_))
        return yq, yk, dq, dk

    yq_ref, yk_ref, dq_ref, dk_ref = eager(q, k, dyq, dyk)
    print(f"\neager forward+backward: dx {tuple(dq_ref.shape)} {dq_ref.dtype}")

    # -- capture ------------------------------------------------------------
    q_s = q.clone().requires_grad_(True)
    k_s = k.clone().requires_grad_(True)
    dyq_s = dyq.clone()
    dyk_s = dyk.clone()

    g = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(g):
            yq_c, yk_c = rope.rms_rope_split_half(q_s, k_s, pe, scale, scale, EPS)
            dq_c, dk_c = torch.autograd.grad((yq_c, yk_c), (q_s, k_s), (dyq_s, dyk_s))
    except Exception as exc:  # noqa: BLE001 - the failure mode is the finding
        print(f"\nCAPTURE FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print("capture: ok (forward + backward for q and k in one graph)")

    # -- replay on new data -------------------------------------------------
    torch.manual_seed(1234)
    q2, k2 = torch.randn_like(q), torch.randn_like(k)
    dyq2, dyk2 = torch.randn_like(dyq), torch.randn_like(dyk)
    yq2, yk2, dq2, dk2 = eager(q2, k2, dyq2, dyk2)

    def load():
        # the static inputs are leaf tensors: copying into them is a write, not
        # a differentiable op, and autograd must not try to record it
        with torch.no_grad():
            q_s.copy_(q2)
            k_s.copy_(k2)
            dyq_s.copy_(dyq2)
            dyk_s.copy_(dyk2)

    load()
    g.replay()

    checks = {
        "q forward": yq_c.view(torch.int16).equal(yq2.view(torch.int16)),
        "k forward": yk_c.view(torch.int16).equal(yk2.view(torch.int16)),
        "q dq": dq_c.view(torch.int16).equal(dq2.view(torch.int16)),
        "k dk": dk_c.view(torch.int16).equal(dk2.view(torch.int16)),
    }
    print("replay:  " + ", ".join(f"{n} {'identical' if ok else 'DIFFERS'}"
                                  for n, ok in checks.items()))

    # -- does the graph actually save anything? -----------------------------
    def replay():
        load()
        g.replay()

    def eager_round():
        eager(q2, k2, dyq2, dyk2)

    t_replay = timed(replay)
    t_eager = timed(eager_round)
    print(f"\nreplay (4 copies + graph) {t_replay:6.3f} ms")
    print(f"eager  (autograd)         {t_eager:6.3f} ms")
    print("  the graph does not reduce device work -- there are four kernels either")
    print("  way. What it removes is host work, which is what a launch-bound step")
    print("  pays; that cannot be seen at this shape, where the GPU is the limit.")

    # -- host cost per step, at a shape where the GPU is not the limit -------
    # (a launch-bound trainer spends its step in the enqueue path, ~48 us per
    # launch on this machine; measure it where the kernels are small)
    Bs, Ss = 1, 256
    pe_s = pe[:, :Ss].contiguous()
    qs = torch.randn(Bs, Ss, H, D, device=dev).to(torch.bfloat16)
    ks = torch.randn(Bs, Ss, H, D, device=dev).to(torch.bfloat16)
    dys_q = torch.randn(Bs, Ss, H, D, device=dev).to(torch.bfloat16)
    dys_k = torch.randn(Bs, Ss, H, D, device=dev).to(torch.bfloat16)

    def host_us(fn, iters=200):
        fn()
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dt = time.perf_counter() - t0
        torch.xpu.synchronize()
        return dt / iters * 1e6

    qs_e = qs.clone().requires_grad_(True)
    ks_e = ks.clone().requires_grad_(True)

    def small_eager():
        yq, yk = rope.rms_rope_split_half(qs_e, ks_e, pe_s, scale, scale, EPS)
        torch.autograd.grad((yq, yk), (qs_e, ks_e), (dys_q, dys_k))

    gs = torch.xpu.XPUGraph()
    with torch.xpu.graph(gs):
        yq_s, yk_s = rope.rms_rope_split_half(qs_e, ks_e, pe_s, scale, scale, EPS)
        torch.autograd.grad((yq_s, yk_s), (qs_e, ks_e), (dys_q, dys_k))

    def small_replay():
        gs.replay()

    ds_gpu = timed(small_replay, iters=50)
    print(f"\nhost cost per step, S={Ss} (device time {ds_gpu:.3f} ms):")
    print(f"  eager forward+backward (autograd + 2 kernels): "
          f"{host_us(small_eager):7.1f} us")
    print(f"  graph replay                                 : "
          f"{host_us(small_replay):7.1f} us")
    print("  both are upper bounds on host cost when the device is the limit;")
    print("  the eager number is the one that scales with the number of kernels.")

    # -- adaLN: the case that allocates inside the capture ------------------
    # The rope backward writes into two tensors it sizes from the input; the
    # adaLN backward also allocates a *partials* buffer for the two-stage
    # modulation reduction (6 MB here).  That is the pattern the graph pool has
    # to absorb, so it is worth verifying separately rather than assuming the
    # rope result generalises.
    DA = 3072
    xa = torch.randn(B, S, DA, device=dev).to(torch.bfloat16)
    mod_s = torch.randn(1, 1, DA, device=dev) * 0.1
    mod_h = torch.randn(1, 1, DA, device=dev) * 0.1
    dya = torch.randn(B, S, DA, device=dev).to(torch.bfloat16)

    def eager_adaln(x_, s_, h_, dy_):
        xr = x_.detach().clone().requires_grad_(True)
        sr = s_.detach().clone().requires_grad_(True)
        hr = h_.detach().clone().requires_grad_(True)
        y = adaln.rms_adaln(xr, sr, hr, EPS)
        return (y,) + torch.autograd.grad(y, (xr, sr, hr), dy_)

    ya_ref, gx_ref, gs_ref, gh_ref = eager_adaln(xa, mod_s, mod_h, dya)

    xa_s = xa.clone().requires_grad_(True)
    s_s = mod_s.clone().requires_grad_(True)
    h_s = mod_h.clone().requires_grad_(True)
    dya_s = dya.clone()

    g2 = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(g2):
            ya_c = adaln.rms_adaln(xa_s, s_s, h_s, EPS)
            gx_c, gs_c, gh_c = torch.autograd.grad(ya_c, (xa_s, s_s, h_s), dya_s)
    except Exception as exc:  # noqa: BLE001 - the failure mode is the finding
        print(f"\nCAPTURE FAILED (adaLN): {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    torch.manual_seed(4321)
    xa2 = torch.randn_like(xa)
    s2 = torch.randn_like(mod_s) * 0.1
    h2 = torch.randn_like(mod_h) * 0.1
    dya2 = torch.randn_like(dya)
    ya2, gx2, gs2, gh2 = eager_adaln(xa2, s2, h2, dya2)

    with torch.no_grad():
        xa_s.copy_(xa2)
        s_s.copy_(s2)
        h_s.copy_(h2)
        dya_s.copy_(dya2)
    g2.replay()

    def same(a, b):
        return a.view(torch.int16).equal(b.view(torch.int16))

    checks.update({
        "adaln forward": same(ya_c, ya2),
        "adaln dx": same(gx_c, gx2),
        "adaln dscale": same(gs_c, gs2),
        "adaln dshift": same(gh_c, gh2),
    })
    print("\nadaLN (with the two-stage modulation reduction in the graph):")
    print("  capture: ok, replay " + ", ".join(
        f"{n.split()[1]} {'identical' if ok else 'DIFFERS'}"
        for n, ok in checks.items() if n.startswith("adaln")))

    # host cost, small shape
    Sa = 256
    xas = torch.randn(B, Sa, DA, device=dev).to(torch.bfloat16).requires_grad_(True)
    dys = torch.randn(B, Sa, DA, device=dev).to(torch.bfloat16)

    def small_eager_adaln():
        ref = adaln.rms_adaln(xas, s_s, h_s, EPS)
        torch.autograd.grad(ref, (xas, s_s, h_s), dys)

    ga = torch.xpu.XPUGraph()
    with torch.xpu.graph(ga):
        yas = adaln.rms_adaln(
            xas.detach().clone().requires_grad_(True), s_s, h_s, EPS)
        torch.autograd.grad(yas, (s_s, h_s), dys)

    print(f"host cost per step, S={Sa} (device time {timed(ga.replay, iters=50):.3f} ms):")
    print(f"  eager  (autograd + the torch chain): {host_us(small_eager_adaln):7.1f} us")
    print(f"  graph replay                       : {host_us(ga.replay):7.1f} us")

    ok = all(checks.values())
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
