"""Is piecewise capture viable on this stack, and how must the pools be held?

ComfyUI's shipped capture is whole-model-or-nothing: the flash attention cannot
be recorded (work-group scratch), so a captured forward runs the math backend --
slower device time and different numerics, and at device-bound shapes no entry
at all. The alternative is *piecewise* capture: capture the segments of the
forward between attention sites, and let the attention run eagerly (and fast,
flash) between them. Numeric fidelity is preserved; the launched-kernel count
still collapses to (segments + attention sites).

What this probe settles, on a toy that is deliberately launch-bound (more
kernels than work):

  1. does a graph -> eager-flash -> graph chain replay and match the eager
     flash forward (it should be exact: flash in both paths);
  2. can the chain's segments be captured into ONE shared pool, or does each
     graph need its own (the shared-pool capture failure seen in ComfyUI);
  3. the wall time of each design against the eager flash forward.

    LEVEL_ZERO_V1_SDK_PATH=/home/okolenmi/.local/levelzero \
        python debug/probe_piecewise_capture.py [rounds]
"""
import sys
import time

import torch

DEV = torch.device("xpu", 0)


def seg(dim=64, layers=10):
    """A segment: enough small kernels to be launch-bound."""
    lins = [torch.nn.Linear(dim, dim).to(DEV, torch.bfloat16) for _ in range(layers)]
    for l in lins:
        for p in l.parameters():
            p.requires_grad_(False)

    def run(x):
        for l in lins:
            x = torch.nn.functional.silu(l(x))
        return x.contiguous()

    return run


def sdpa_out(h):
    q, k, v = (h.reshape(1, 4, 32, 16) for _ in range(3))
    return torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(1, 32, 64)


def timed(fn, rounds):
    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    best = float("inf")
    for _ in range(rounds):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1000)
    return best


def diff(label, a, b):
    d = (a.float() - b.float()).abs().max().item()
    mag = b.float().abs().max().item()
    print(f"    {label:38s} max diff {d:.9f}  (magnitude {mag:.4f})")


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    torch.manual_seed(0)
    segA, segB = seg(), seg()
    x = torch.randn(1, 32, 64, device=DEV, dtype=torch.bfloat16)
    torch.xpu.synchronize()

    print(f"device: {torch.xpu.get_device_name(0)}   toy: segA(10x linear+silu) -> "
          f"flash SDPA -> segB(10x linear+silu), bf16, batch 1")

    # -- the eager flash reference -------------------------------------------
    def chain_eager(x):
        return segB(sdpa_out(segA(x)))

    out_eager = chain_eager(x)
    t_eager = timed(lambda: chain_eager(x), rounds)
    print(f"  eager (flash, all kernels launched):{t_eager:8.2f} ms")

    # -- 1+2: piecewise, each graph owning its pool ---------------------------
    s1 = x.clone()
    try:
        g1 = torch.xpu.XPUGraph()
        with torch.xpu.graph(g1):
            o1 = segA(s1)
        torch.xpu.synchronize()
    except RuntimeError as exc:
        print(f"  segment capture failed: {exc}")
        raise SystemExit

    b_in = torch.empty(1, 32, 64, device=DEV, dtype=torch.bfloat16)
    b_in.copy_(sdpa_out(o1))
    g2 = torch.xpu.XPUGraph()
    with torch.xpu.graph(g2):
        o2 = segB(b_in)
    torch.xpu.synchronize()

    def piecewise():
        s1.copy_(x)
        g1.replay()
        b_in.copy_(sdpa_out(o1))
        g2.replay()
        return o2

    out_pw = piecewise()
    t_pw = timed(piecewise, rounds)
    diff("piecewise (own pools) vs eager", out_pw, out_eager)
    print(f"  piecewise, own pools:               {t_pw:8.2f} ms   "
          f"({t_eager / t_pw:.2f}x against eager)")

    # -- 3: both segments captured into ONE shared pool ----------------------
    try:
        pool = torch.xpu.graph_pool_handle()
        s1s = x.clone()
        g1s = torch.xpu.XPUGraph()
        with torch.xpu.graph(g1s, pool=pool):
            o1s = segA(s1s)
        torch.xpu.synchronize()
        b2_in = torch.empty_like(b_in)
        b2_in.copy_(sdpa_out(o1s))
        g2s = torch.xpu.XPUGraph()
        with torch.xpu.graph(g2s, pool=pool):
            o2s = segB(b2_in)
        torch.xpu.synchronize()
        ok = True
    except RuntimeError as exc:
        print(f"  second shared-pool capture:         FAILED "
              f"({type(exc).__name__}: {str(exc).splitlines()[0][:70]})")
        ok = False
    if ok:
        def piecewise_shared():
            s1s.copy_(x)
            g1s.replay()
            b2_in.copy_(sdpa_out(o1s))
            g2s.replay()
            return o2s
        out_ps = piecewise_shared()
        t_ps = timed(piecewise_shared, rounds)
        diff("piecewise (shared pool) vs eager", out_ps, out_eager)
        print(f"  piecewise, one shared pool:         {t_ps:8.2f} ms   "
              f"({t_eager / t_ps:.2f}x against eager)")

        # 3b: capture a THIRD graph into the same pool *after* the others have
        # replayed (the trainer's live pattern: shape 2 arrives mid-run, after
        # shape 1 has replayed for many steps).
        try:
            s3 = x.clone()
            g3 = torch.xpu.XPUGraph()
            with torch.xpu.graph(g3, pool=pool):
                o1c = segA(s3)              # another pre-attention segment
            torch.xpu.synchronize()
            g3.replay()
            torch.xpu.synchronize()
            diff("3rd-graph-after-replays vs eager", o1c, segA(x))
            print("  3rd shared-pool capture (after replays of the earlier graphs): ok")
        except RuntimeError as exc:
            print(f"  3rd shared-pool capture (after replays): FAILED "
                  f"({type(exc).__name__}: {str(exc).splitlines()[0][:70]})")

    # -- 4: the shipped design (whole forward, math attention) ---------------
    from torch.nn.attention import SDPBackend, sdpa_kernel
    whole_in = torch.empty_like(x)
    whole_in.copy_(x)
    torch.xpu.synchronize()
    with sdpa_kernel(SDPBackend.MATH):
        chain_eager(whole_in)          # warm the math kernels before capture
        chain_eager(whole_in)
        gw = torch.xpu.XPUGraph()
        with torch.xpu.graph(gw):
            out_whole = chain_eager(whole_in)
    torch.xpu.synchronize()

    def whole():
        whole_in.copy_(x)
        gw.replay()
        return out_whole

    whole()
    t_whole = timed(whole, rounds)
    diff("whole math-graph vs eager flash", out_whole, out_eager)
    print(f"  whole graph, math attention:        {t_whole:8.2f} ms   "
          f"({t_eager / t_whole:.2f}x against eager)")


if __name__ == "__main__":
    main()
