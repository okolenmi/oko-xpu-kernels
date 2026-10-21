"""Independent check: how much device time is there inside the eager forward?

If the wall (109 ms) is much larger than the sum of the kernels' device time,
the eager path is launch/Python-bound and a graph (which replays the kernels
with no host work) would remove that gap.
"""
import os, sys, time
sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")
import torch
from torch.profiler import profile, ProfilerActivity
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
with torch.no_grad():
    for p in unet.parameters():
        if not p.any():
            p.normal_(0, 0.05)
for p in unet.parameters():
    p.requires_grad_(False)
ctx = torch.randn(1, 77, 2048, device=DEV, dtype=DT)
y = torch.randn(1, 2816, device=DEV, dtype=DT)
ts = torch.full((1,), 500.0, device=DEV, dtype=torch.float32)
x = torch.randn(1, 4, RES // 8, RES // 8, device=DEV, dtype=DT)

with torch.no_grad():
    for _ in range(3):
        unet(x, ts, ctx, y)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    unet(x, ts, ctx, y)
    torch.xpu.synchronize()
    wall = (time.perf_counter() - t0) * 1000

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        unet(x, ts, ctx, y)
        torch.xpu.synchronize()

events = prof.key_averages()
# only the device kernels: the aten:: rows are the same work, attributed twice
kernels = [e for e in events if e.device_time_total > 0 and e.count > 0
           and not e.key.startswith("aten::") and not e.key.startswith("cuda")]
n_kernels = sum(e.count for e in kernels)
dev_total = sum(e.device_time_total for e in kernels)
host_total = sum(e.self_cpu_time_total for e in events)
print(f"=== {RES}x{RES}: the eager forward's wall {wall:.1f} ms")
print(f"    the kernels: {n_kernels} launches, the device time summed {dev_total/1000:.1f} ms")
print(f"    the host (the CPU's own time in the profiler): {host_total/1000:.1f} ms")
print(f"    -> the gap the device spends idle: {wall - dev_total/1000:.1f} ms "
      f"({100*(wall-dev_total/1000)/wall:.0f} % of the wall)")
top = sorted(kernels, key=lambda e: -e.device_time_total)[:8]
print("    the top kernels by device time:")
for e in top:
    print(f"      {e.key[:58]:58s} {e.count:5d} x {e.device_time_total/1000:7.2f} ms")
