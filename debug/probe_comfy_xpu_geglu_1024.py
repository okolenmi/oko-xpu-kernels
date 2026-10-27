"""SDXL sampling at 1024x1024: eager GEGLU vs the fused xpu_kernels GEGLU.

The fused kernel replaces ``x = chunk(2); x * gelu(gate)`` with one pass over
the projection output.  It is a *different function in the last bit* (Triton's
erf is not torch's), so this measures both the time and the difference.

The two variants are interleaved in one process -- off, on, off, on -- because
inter-process step times on this card drift by ~10%, which is larger than the
effect being measured.

    LEVEL_ZERO_V1_SDK_PATH=/home/okolenmi/.local/levelzero \
      /home/okolenmi/comfy/venv/bin/python debug/probe_comfy_xpu_geglu_1024.py \
        --res 1024 --steps 4 --pairs 3
"""
import argparse
import logging
import sys
import time

sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")

parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=1024)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--pairs", type=int, default=3)
parser.add_argument("--ckpt", default="/home/okolenmi/comfy/ComfyUI/models/checkpoints/divA.safetensors")
args = parser.parse_args()

logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")

import torch

import comfy.cli_args
import comfy.model_management as mm
import comfy.sample
import comfy.sd
import comfy.utils
import folder_paths
from comfy.ldm.modules import attention as attn

comfy.cli_args.args.xpu_graphs = False

torch.manual_seed(42)
model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
    args.ckpt, output_vae=True, output_clip=True,
    embedding_directory=folder_paths.get_folder_paths("embeddings"))[:4]

cond = clip.encode_from_tokens_scheduled(clip.tokenize("a red fox in a snowy forest, photo"))
lat = args.res // 8
noise = torch.randn((1, 4, lat, lat), generator=torch.Generator("cpu").manual_seed(1234))
latent = torch.zeros_like(noise)


@torch.inference_mode()
def sample(n):
    return comfy.sample.sample(model, noise, n, 7.0, "euler", "normal",
                               cond, cond, latent, denoise=1.0, seed=42,
                               disable_pbar=True)


def timed(n):
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    out = sample(n)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) * 1000 / n, out


def run(fused, n):
    attn.FUSED_GEGLU = fused
    return timed(n)


print(f"\n{args.res}x{args.res} sampling, cfg 7, euler/normal, batch 2, "
      f"UNet dtype {model.model.get_dtype()}")
print(f"  fused GEGLU toggle: comfy.ldm.modules.attention.FUSED_GEGLU")

# warm up both variants (the first fused call also latches the kernel gate)
run(False, 2)
run(True, 2)
print(f"  gate: {attn._fused_geglu_module(torch.zeros((1, 2), dtype=torch.float16, device='xpu')) is not None}")

pairs = []
for i in range(args.pairs):
    off, out_off = run(False, args.steps)
    on, out_on = run(True, args.steps)
    pairs.append((off, on))
    print(f"  pair {i + 1}: eager {off:7.1f} ms/step   fused {on:7.1f} ms/step   "
          f"-> {off / on:5.3f}x  (saves {off - on:5.1f} ms)")

offs = sorted(p[0] for p in pairs)
ons = sorted(p[1] for p in pairs)
ratios = sorted(o / n for o, n in pairs)
print(f"\n  eager: min {offs[0]:.1f}  median {offs[len(offs) // 2]:.1f} ms/step")
print(f"  fused: min {ons[0]:.1f}  median {ons[len(ons) // 2]:.1f} ms/step")
print(f"  paired speedup: min {ratios[0]:.3f}x  median {ratios[len(ratios) // 2]:.3f}x  "
      f"max {ratios[-1]:.3f}x")

d = (out_on.float() - out_off.float()).abs()
print(f"  latent: max |diff| {d.max().item():.5f}  mean |diff| {d.mean().item():.6f}")
