"""How much of a 1024x1024 image is the VAE, and not the denoiser?

A fresh process that loads only the VAE (the UNet is never touched, so it never
reaches the card) and decodes a 1024x1024 latent through the same `VAE.decode`
the `VAEDecode` node calls.  The latent's content does not change the cost.

Measured on the B580: the default fp32 path sits *at* the memory ceiling.  The
first decode OOMs on sdpa, ComfyUI switches to sliced attention, and it lands in
5 954 ms; a second decode OOMs outright.  The fp16 (`--fp16-vae`) and tiled
paths are the ones a real 1024x1024 run should use, and are not measured here.

    LEVEL_ZERO_V1_SDK_PATH=/home/okolenmi/.local/levelzero \
      /home/okolenmi/comfy/venv/bin/python debug/probe_comfy_xpu_vae_1024.py --res 1024
"""
import argparse
import sys
import time

sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")

parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=1024)
parser.add_argument("--ckpt", default="/home/okolenmi/comfy/ComfyUI/models/checkpoints/divA.safetensors")
args = parser.parse_args()

import torch

import comfy.cli_args
import comfy.model_management as mm
import comfy.sd
import folder_paths

comfy.cli_args.args.xpu_graphs = False

torch.manual_seed(42)
_, _, vae, _ = comfy.sd.load_checkpoint_guess_config(
    args.ckpt, output_model=False, output_clip=False, output_vae=True,
    embedding_directory=folder_paths.get_folder_paths("embeddings"))

lat = torch.randn((1, 4, args.res // 8, args.res // 8),
                  device="xpu", dtype=torch.float16)


def timed():
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    image = vae.decode(lat)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) * 1000, image


ms, image = timed()
print(f"  warm-up (loads the VAE)       {ms:8.1f} ms")
runs = []
for i in range(2):
    ms, image = timed()
    runs.append(ms)
    print(f"  run {i + 1}                         {ms:8.1f} ms")
print(f"  VAE decode at {args.res}x{args.res}: min {min(runs):.1f} ms")
print(f"  the image: {tuple(image.shape)} mean {image.float().mean().item():.6f}")
print(f"  free after: {torch.xpu.mem_get_info()[0] / 2**20:.0f} MB")
