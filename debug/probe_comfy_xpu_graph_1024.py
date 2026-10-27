"""What the 1024x1024 step is made of, and what a capture can still take.

Times the real sampler at 1024x1024 (euler/normal, cfg 7, so the UNet runs a
batch-2 call), then profiles one step and prints the device-busy time, the
exposed host gap, and the kernels that hold the time.

    LEVEL_ZERO_V1_SDK_PATH=/home/okolenmi/.local/levelzero \
      /home/okolenmi/comfy/venv/bin/python debug/probe_comfy_xpu_graph_1024.py \
        --res 1024 --steps 4 --decode --table /tmp/p1024_kernels.txt

Graphs are off: this measures the eager path, which is what this shape runs.
The device-busy time is the sum of the kernel durations reported by
`torch.profiler` -- the profiler's own CPU overhead does not change a kernel's
duration -- so `step - busy` is the host issue that is *exposed*, i.e. the only
part a capture can remove.  `--decode` also times the VAE, which at 1024x1024
is seconds rather than milliseconds.
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")

parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=1024)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--profile-steps", type=int, default=1)
parser.add_argument("--ckpt", default="/home/okolenmi/comfy/ComfyUI/models/checkpoints/divA.safetensors")
parser.add_argument("--table", default="")
parser.add_argument("--decode", action="store_true", help="also time the VAE at this size")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")

import torch

import comfy.cli_args
import comfy.model_management as mm
import comfy.sample
import comfy.sd
import comfy.utils
import folder_paths

comfy.cli_args.args.xpu_graphs = False

torch.manual_seed(42)
model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
    args.ckpt, output_vae=True, output_clip=True,
    embedding_directory=folder_paths.get_folder_paths("embeddings"))[:4]

PROMPT = "a red fox in a snowy forest, photo"
cond = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
lat = args.res // 8
noise = torch.randn((1, 4, lat, lat), generator=torch.Generator("cpu").manual_seed(1234))
latent = torch.zeros_like(noise)


def sample(n):
    return comfy.sample.sample(model, noise, n, 7.0, "euler", "normal",
                               cond, cond, latent, denoise=1.0, seed=42,
                               disable_pbar=True)


def timed(n):
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    out = sample(n)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) * 1000, out


print(f"eager, {args.res}x{args.res}, cfg 7, euler/normal, batch 2")
ms, samples = timed(2)
print(f"  warm-up (loads the weights)   {ms:8.1f} ms  ({ms / 2:6.1f} ms/step)")

times = []
for i in range(3):
    ms, samples = timed(args.steps)
    times.append(ms / args.steps)
    print(f"  run {i + 1}                         {ms:8.1f} ms  "
          f"({ms / args.steps:6.1f} ms/step)")
times.sort()
best = times[0]
print(f"  the step: min {best:.1f} ms, median {times[1]:.1f} ms")

with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.XPU]) as prof:
    sample(args.profile_steps)
torch.xpu.synchronize()

events = prof.key_averages()
kernels = [e for e in events
           if not e.key.startswith("aten::") and e.self_device_time_total > 0]
busy = sum(e.self_device_time_total for e in kernels) / 1000 / args.profile_steps
print(f"  device busy (profile)         {busy:8.1f} ms/step  "
      f"({100 * busy / best:5.1f}% of the step)")
print(f"  host issue exposed            {best - busy:8.1f} ms/step")

print(f"\n  top kernels by device time ({args.profile_steps} step(s)):")
print(f"    {'kernel':54s} {'calls':>6s} {'ms/step':>9s} {'share':>6s}")
for e in sorted(kernels, key=lambda e: -e.self_device_time_total)[:16]:
    msp = e.self_device_time_total / 1000 / args.profile_steps
    print(f"    {e.key[:54]:54s} {e.count / args.profile_steps:6.0f} "
          f"{msp:9.2f} {100 * msp / busy:5.1f}%")

if args.table:
    with open(args.table, "w") as f:
        f.write(events.table(sort_by="self_device_time_total", row_limit=400))
    print(f"  the full table: {args.table}")

if args.decode:
    # The real node path: the UNet has to leave the card first, or this is the
    # untitled decode that OOMs at 1024x1024.  (Deleting the reference is what
    # frees the weights -- unloading through the patcher keeps them alive.)
    del model
    import gc
    gc.collect()
    mm.soft_empty_cache()
    vae_in = samples.to(vae.device)
    try:
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        vae.decode(vae_in)
        torch.xpu.synchronize()
        print(f"  VAE decode (first, warms)     {(time.perf_counter() - t0) * 1000:8.1f} ms")
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        vae.decode(vae_in)
        torch.xpu.synchronize()
        print(f"  VAE decode                    {(time.perf_counter() - t0) * 1000:8.1f} ms")
    except RuntimeError as exc:
        print(f"  VAE decode failed: {exc}")
