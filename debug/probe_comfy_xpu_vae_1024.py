"""The VAE decode as a generation actually runs it: after the denoiser is gone.

The VAE is post-processing.  By the time `VAEDecode` runs, the UNet should not
be on the card, and ComfyUI's own `VAE.decode` evicts it to RAM when it needs
the room (`load_models_gpu(..., memory_required=...)`).  This probe reproduces
that sequence: sample a latent with the real model, drop the probe's reference,
let model management evict the UNet, then decode and time it.

Two variants a user can pick, both inside the card:

  * fp16 (`--fp16-vae`), tiled (`VAEDecodeTiled`, tile_size 512),
  * fp32, tiled 512 (no dtype trade-off).

The untiled peak is linear in output pixels -- 0.0119 MB per pixel in fp16 --
so there is no leak to find (`--scaling` measures the curve):

    512x512   3 112 MB      768x768   6 884 MB (916 ms)
    1024x1024 12 332 MB (2 924 ms; above the card's 12.19 GB, so the driver
                        backs part of it from host RAM)

At 1024x1024 that is the whole card, and it is slower than tiling: tiled 512 is
1 078-1 165 ms in fp16 (619 MB peak) and 4 695-4 719 ms in fp32 (1 227 MB
peak).  The first decode of a session pays 4.7-6.0 s of kernel compilation on
top.  The script prints the fp16-vs-fp32 difference (max 0.0126, mean 0.00015
in [0,1] units) so the dtype choice is a measured one.

The allocator cache is returned between decodes (`soft_empty_cache`), which is
what makes a second decode possible at all.  `--untiled` additionally tries the
path that OOMs, and the process then hangs on teardown: do not run it casually.

    LEVEL_ZERO_V1_SDK_PATH=/home/okolenmi/.local/levelzero \
      /home/okolenmi/comfy/venv/bin/python debug/probe_comfy_xpu_vae_1024.py --res 1024
"""
import argparse
import gc
import sys
import time

sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")

parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=1024)
parser.add_argument("--ckpt", default="/home/okolenmi/comfy/ComfyUI/models/checkpoints/divA.safetensors")
parser.add_argument("--runs", type=int, default=2)
parser.add_argument("--untiled", action="store_true",
                    help="also try the fp16 untiled path (needs ~12 GB at 1024x1024)")
parser.add_argument("--scaling", action="store_true",
                    help="measure the untiled peak at 512/768/1024 on an empty card")
args = parser.parse_args()

import torch

import comfy.cli_args
import comfy.model_management as mm
import comfy.sample
import comfy.sd
import comfy.utils
import folder_paths

comfy.cli_args.args.xpu_graphs = False


def free_mb():
    return torch.xpu.mem_get_info()[0] / 2**20


torch.manual_seed(42)
model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
    args.ckpt, output_vae=True, output_clip=not args.scaling,
    output_model=not args.scaling,
    embedding_directory=folder_paths.get_folder_paths("embeddings"))[:4]

if args.scaling:
    # Untiled peak against output pixels: linear means no leak, and the slope
    # is what makes 1024x1024 the whole card.
    vae.vae_dtype = torch.float16
    vae.first_stage_model.to(dtype=torch.float16)
    for res in (512, 768, 1024):
        latent = torch.randn((1, 4, res // 8, res // 8), device="xpu", dtype=torch.float16)
        mm.soft_empty_cache()
        torch.xpu.reset_peak_memory_stats()
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        img = vae.decode(latent)
        torch.xpu.synchronize()
        peak = torch.xpu.max_memory_allocated() / 2**20
        print(f"  {res:5d}  {(time.perf_counter() - t0) * 1000:8.1f} ms   "
              f"peak {peak:7.0f} MB   {peak * 1e6 / (res * res):6.0f} MB/Mpx")
        del img, latent
    sys.exit(0)

cond = clip.encode_from_tokens_scheduled(clip.tokenize("a red fox in a snowy forest, photo"))
lat = args.res // 8
noise = torch.randn((1, 4, lat, lat), generator=torch.Generator("cpu").manual_seed(1234))
print(f"free before the step:          {free_mb():6.0f} MB")
samples = comfy.sample.sample(model, noise, 1, 7.0, "euler", "normal", cond, cond,
                              torch.zeros_like(noise), denoise=1.0, seed=1,
                              disable_pbar=True)
torch.xpu.synchronize()
print(f"free with the UNet on the card: {free_mb():6.0f} MB")

# The sequence a generation runs: the denoiser goes away before the VAE node.
del model, clip, cond, noise
gc.collect()
mm.unload_all_models()
mm.soft_empty_cache()
print(f"free after eviction to RAM:    {free_mb():6.0f} MB")


def decode_once(tile):
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    if tile:
        img = vae.decode_tiled(samples,
                               tile_x=tile // vae.downscale_ratio,
                               tile_y=tile // vae.downscale_ratio,
                               overlap=64 // vae.downscale_ratio)
    else:
        img = vae.decode(samples)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) * 1000, img


variants = [("fp16, tiled 512", torch.float16, 512),
            ("fp32, tiled 512", torch.float32, 512)]
if args.untiled:
    # The untiled path OOMs at 1024x1024 on this card even in fp16 with ~10 GB
    # free, and the process hangs on teardown after the driver error.  Kept
    # behind a flag so nobody runs it by accident.
    variants.append(("fp16, untiled", torch.float16, 0))

images = {}
for label, dtype, tile in variants:
    vae.vae_dtype = dtype
    vae.first_stage_model.to(dtype=dtype)
    gc.collect()
    mm.soft_empty_cache()
    print(f"\n{label}: free {free_mb():.0f} MB")
    kept = None
    for i in range(args.runs + 1):
        torch.xpu.reset_peak_memory_stats()
        ms, img = decode_once(tile)
        peak = torch.xpu.max_memory_allocated() / 2**20
        tag = "warm-up" if i == 0 else f"run {i}"
        print(f"  {tag:8s} {ms:8.1f} ms   peak alloc {peak:6.0f} MB   "
              f"free {free_mb():6.0f} MB")
        if i == 1:
            kept = img.float().cpu()
        del img
        gc.collect()
        mm.soft_empty_cache()
    if kept is not None:
        images[label] = kept

for a, b in (("fp16, tiled 512", "fp32, tiled 512"),):
    if a in images and b in images:
        d = (images[a] - images[b]).abs()
        print(f"\n{a} vs {b}: max |d| {d.max():.6f}, mean |d| {d.mean():.6f}, "
              f"p99.9 {d.flatten().kthvalue(int(d.numel() * 0.999)).values:.6f}")
