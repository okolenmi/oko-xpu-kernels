"""ComfyUI's SDXL UNet: how much of the eager forward is the host?

The eager forward at 512 took 108 ms of wall but the captured replay only
33.6 ms -- so the eager path spends most of its time *issuing* kernels, not
executing them.  Measure:
  * the wall (sync) -- what the user waits for
  * the issue time (no sync) -- the host's cost to launch the kernels
  * the saturated device time (N iterations queued back to back) -- the device
    cost with the host running ahead
and the same for the math attention, plus a correctness check of the capture.
"""
import os, sys, time
sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import comfy.ops as ops
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

DEV = torch.device("xpu", 0)
DT = torch.bfloat16
RES = int(os.environ.get("RES", "512"))

CFG = {
    "use_checkpoint": False, "image_size": 32, "out_channels": 4,
    "use_spatial_transformer": True, "legacy": False, "num_classes": "sequential",
    "adm_in_channels": 2816, "in_channels": 4, "model_channels": 320,
    "num_res_blocks": [2, 2, 2], "transformer_depth": [0, 0, 2, 2, 10, 10],
    "channel_mult": [1, 2, 4], "transformer_depth_middle": 10,
    "use_linear_in_transformer": True, "context_dim": 2048, "num_head_channels": 64,
    "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
    "use_temporal_attention": False, "use_temporal_resblock": False,
}

torch.manual_seed(0)
unet = UNetModel(**CFG, device=DEV, operations=ops.pick_operations(DT, DT, load_device=DEV))
unet = unet.to(DT).eval()
# ComfyUI's UNetModel does not initialise its weights (the checkpoint does).
with torch.no_grad():
    for p in unet.parameters():
        if not p.any():      # only the zeroed ones
            p.normal_(0, 0.05)
for p in unet.parameters():
    p.requires_grad_(False)

ctx = torch.randn(1, 77, 2048, device=DEV, dtype=DT)
y = torch.randn(1, 2816, device=DEV, dtype=DT)
ts = torch.full((1,), 500.0, device=DEV, dtype=torch.float32)
x = torch.randn(1, 4, RES // 8, RES // 8, device=DEV, dtype=DT)


def wall(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    best = 1e9
    for _ in range(n):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1000)
    return best


def saturated(fn, n=10, warmup=2):
    """The device cost with the host running ahead: N iterations, no sync."""
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) * 1000 / n


def issue(fn, n=5, warmup=2):
    """The host's cost to issue the work (no sync in between)."""
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    best = 1e9
    for _ in range(n):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000)
        torch.xpu.synchronize()
    return best


print(f"=== {RES}x{RES}, ComfyUI's SDXL UNet, batch 1, bf16 ===")
with torch.no_grad():
    for name, backend in (("the default (flash)", None), ("MATH", SDPBackend.MATH)):
        ctxmgr = sdpa_kernel(backend) if backend is not None else torch.no_grad()
        with ctxmgr:
            w = wall(lambda: unet(x, ts, ctx, y))
            s = saturated(lambda: unet(x, ts, ctx, y))
            i = issue(lambda: unet(x, ts, ctx, y))
        print(f"  {name:22s}: the wall {w:7.1f} ms | the issue (host) {i:7.1f} ms"
              f" | the saturated device {s:7.1f} ms")

    # the capture, and whether it computes the same thing
    ref = unet(x, ts, ctx, y).clone()
    print(f"  the eager output: |max| {ref.abs().max().item():.4f}")
    g = torch.xpu.XPUGraph()
    torch.xpu.synchronize()
    try:
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.xpu.graph(g):
            out = unet(x, ts, ctx, y)
        torch.xpu.synchronize()
    except Exception as exc:
        print(f"  the capture: FAILED -- {type(exc).__name__}: {str(exc)[:90]}")
        sys.exit(0)
    with torch.no_grad():
        out.zero_()
        g.replay()
        torch.xpu.synchronize()
        d = (out.float() - ref.float()).abs()
        w_rep = wall(lambda: g.replay())
        s_rep = saturated(lambda: g.replay())
    print(f"  the capture (math): the replay the wall {w_rep:7.1f} ms | the saturated device {s_rep:7.1f} ms"
          f" | the output vs the eager: max {d.max().item():.6f} (|max| {out.abs().max().item():.4f})")
