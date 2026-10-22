"""End to end: does --xpu-graphs speed up a real ComfyUI SDXL sampling run?

Loads a checkpoint the way ComfyUI does, samples with the real sampler, decodes
with the real VAE, and prints the time of the sampling loop, the image's mean
absolute difference against the same run without graphs, and the graph counters.

    LEVEL_ZERO_V1_SDK_PATH=/home/okolenmi/.local/levelzero \
      /home/okolenmi/comfy/venv/bin/python debug/probe_comfy_xpu_graph_e2e.py \
        --res 512 --steps 8 --graph 1

Run it twice (--graph 0 and --graph 1) and compare the two printed results.
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")

parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=512)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--graph", type=int, default=0)
parser.add_argument("--ckpt", default="/home/okolenmi/comfy/ComfyUI/models/checkpoints/divA.safetensors")
parser.add_argument("--out", default="")
parser.add_argument("--no-decode", action="store_true", help="skip the VAE, which needs ~3 GB at 1024")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")

import torch

import comfy.cli_args
import comfy.model_management as mm
import comfy.sample
import comfy.sd
import comfy.utils
import folder_paths

comfy.cli_args.args.xpu_graphs = bool(args.graph)

torch.manual_seed(42)
model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
    args.ckpt, output_vae=True, output_clip=True,
    embedding_directory=folder_paths.get_folder_paths("embeddings"))[:4]
print(f"loaded {os.path.basename(args.ckpt)}: "
      f"{comfy.utils.calculate_parameters(model.model.state_dict())} params, "
      f"dynamic={model.is_dynamic()}")

cond = clip.encode_from_tokens_scheduled(clip.tokenize("a red fox in a snowy forest, photo"))
lat = args.res // 8
noise = torch.randn((1, 4, lat, lat), generator=torch.Generator("cpu").manual_seed(1234))
latent = torch.zeros_like(noise)

samples = None
for label in ("warm-up (loads the weights)", "measured"):
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    samples = comfy.sample.sample(model, noise, args.steps, 7.0, "euler", "normal",
                                  cond, cond, latent, denoise=1.0, seed=42,
                                  disable_pbar=True)
    torch.xpu.synchronize()
    dt = time.perf_counter() - t0
    print(f"  {label:28s} {dt*1000:8.1f} ms  ({dt*1000/args.steps:6.1f} ms/step)")

if args.no_decode:
    print(f"  the samples: mean {samples.float().mean().item():.8f} "
          f"absmax {samples.float().abs().max().item():.4f} "
          f"md5 {__import__('hashlib').md5(samples.float().cpu().numpy().tobytes()).hexdigest()}")
    import comfy.xpu_graph as _xg
    print(f"  the graph counters: {_xg.stats()}")
    sys.exit(0)

mm.soft_empty_cache()
image = vae.decode(samples.to(vae.device))
print(f"  the image: {tuple(image.shape)} mean {image.float().mean().item():.8f} "
      f"absmax {image.float().abs().max().item():.4f}")

import comfy.xpu_graph as xg
print(f"  the graph counters: {xg.stats()}")

if args.out:
    import numpy
    from PIL import Image
    arr = (image[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(arr).save(args.out)
    import hashlib
    print(f"  saved {args.out} md5 {hashlib.md5(open(args.out,'rb').read()).hexdigest()}")
