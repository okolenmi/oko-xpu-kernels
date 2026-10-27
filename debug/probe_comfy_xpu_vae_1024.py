"""The VAE decode as a generation actually runs it: after the denoiser is gone.

Run it under `torch.inference_mode()` or it lies.  The VAE's parameters require
grad, so a plain call builds an autograd graph that keeps every intermediate
alive: 11.5 KB per output pixel, 12 GB at 1024x1024 instead of 2.3 GB, four
times the time, and an OOM on a 12 GB card.  ComfyUI's `execution.py` runs the
whole graph inside `inference_mode`; this probe does the same.

The VAE is post-processing.  By the time `VAEDecode` runs the UNet should not be
on the card, and `VAE.decode` evicts it to RAM when it needs the room
(`load_models_gpu(..., memory_required=...)`).  This probe reproduces that
sequence -- sample a latent with the real model, drop the probe's reference, let
model management evict the UNet -- then times the decode variants a user can
pick, with the peak measured from the driver (torch's XPU peak counter does not
reset, so `max_memory_allocated` is useless here).

Warm 1024x1024 decode on the B580: fp16 untiled **281 ms / 2 316 MB**, fp32
untiled 1 318 ms / 5 448 MB, fp16 tiled 512 1 219 ms / 258 MB, fp32 tiled 512
4 970 ms / 884 MB.  The untiled path is the fast one; tiling only buys memory.
The fp16 image differs from fp32 by max 0.0126, mean 0.00015 in [0,1] units.

    LEVEL_ZERO_V1_SDK_PATH=/home/okolenmi/.local/levelzero \
      /home/okolenmi/comfy/venv/bin/python debug/probe_comfy_xpu_vae_1024.py --res 1024
    ... --scaling        # untiled peak against resolution, 512/768/1024
"""
import argparse
import gc
import sys
import threading
import time

sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")

parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=1024)
parser.add_argument("--ckpt", default="/home/okolenmi/comfy/ComfyUI/models/checkpoints/divA.safetensors")
parser.add_argument("--runs", type=int, default=2)
parser.add_argument("--scaling", action="store_true",
                    help="untiled peak against resolution, 512/768/1024")
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


def measure(fn):
    """Time one call and its peak from the driver, sampling free memory."""
    torch.xpu.synchronize()
    before = torch.xpu.mem_get_info()[0]
    low = [before]
    stop = [False]

    def monitor():
        while not stop[0]:
            low[0] = min(low[0], torch.xpu.mem_get_info()[0])
            time.sleep(0.002)

    thread = threading.Thread(target=monitor)
    thread.start()
    t0 = time.perf_counter()
    try:
        out = fn()
        torch.xpu.synchronize()
        err = None
    except Exception as e:  # noqa: BLE001 - a decode failure is a result here
        out = None
        err = f"{type(e).__name__}: {str(e)[:70]}"
    dt = (time.perf_counter() - t0) * 1000
    stop[0] = True
    thread.join()
    return dt, (before - low[0]) / 2**20, out, err


torch.manual_seed(42)
model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
    args.ckpt, output_vae=True, output_clip=not args.scaling,
    output_model=not args.scaling,
    embedding_directory=folder_paths.get_folder_paths("embeddings"))[:4]

if args.scaling:
    vae.vae_dtype = torch.float16
    vae.first_stage_model.to(dtype=torch.float16)
    for res in (512, 768, 1024):
        latent = torch.randn((1, 4, res // 8, res // 8), device="xpu", dtype=torch.float16)
        mm.soft_empty_cache()
        with torch.inference_mode():
            vae.decode(latent)  # compile
        mm.soft_empty_cache()

        def one():
            with torch.inference_mode():
                return vae.decode(latent)

        ms, peak, img, err = measure(one)
        print(f"  {res:5d}  {ms:8.1f} ms   peak {peak:7.0f} MB   "
              f"{peak * 1e6 / (res * res):6.0f} MB/Mpx   {err or 'ok'}")
        del img, latent
    sys.exit(0)

cond = clip.encode_from_tokens_scheduled(clip.tokenize("a red fox in a snowy forest, photo"))
lat = args.res // 8
noise = torch.randn((1, 4, lat, lat), generator=torch.Generator("cpu").manual_seed(1234))
with torch.inference_mode():
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


def decode(dtype, tile):
    def call():
        with torch.inference_mode():
            if tile:
                return vae.decode_tiled(samples, tile_x=tile // vae.downscale_ratio,
                                        tile_y=tile // vae.downscale_ratio,
                                        overlap=64 // vae.downscale_ratio)
            return vae.decode(samples)
    return call


variants = [("fp16, untiled", torch.float16, 0),
            ("fp32, untiled", torch.float32, 0),
            ("fp16, tiled 512", torch.float16, 512),
            ("fp32, tiled 512", torch.float32, 512)]

images = {}
for label, dtype, tile in variants:
    vae.vae_dtype = dtype
    vae.first_stage_model.to(dtype=dtype)
    gc.collect()
    mm.soft_empty_cache()
    print(f"\n{label}")
    for i in range(args.runs + 1):
        ms, peak, img, err = measure(decode(dtype, tile))
        tag = "warm-up" if i == 0 else f"run {i}"
        print(f"  {tag:8s} {ms:8.1f} ms   peak {peak:6.0f} MB   "
              f"free after {free_mb():6.0f} MB   {err or 'ok'}")
        if img is not None and label in ("fp16, untiled", "fp32, untiled") and i <= 1:
            images[label] = img.float().cpu()
        del img
        gc.collect()
        mm.soft_empty_cache()

if "fp16, untiled" in images and "fp32, untiled" in images:
    d = (images["fp16, untiled"] - images["fp32, untiled"]).abs()
    print(f"\nfp16 vs fp32: max |d| {d.max():.6f}, mean |d| {d.mean():.6f}, "
          f"p99.9 {d.flatten().kthvalue(int(d.numel() * 0.999)).values:.6f}")
