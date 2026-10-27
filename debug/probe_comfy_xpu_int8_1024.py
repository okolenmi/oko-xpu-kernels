"""SDXL 1024x1024 sampling: bf16 vs the XPU ConvRot INT8 linear path.

The INT8 kernels in ``xpu_kernels.int8`` consume bf16 activations, so the
comparison is bf16-weights vs int8-weights: the fp16 default cannot enter this
path.  Sampling only -- no VAE, no capture, no decode.

    LEVEL_ZERO_V1_SDK_PATH=/home/okolenmi/.local/levelzero \
      /home/okolenmi/comfy/venv/bin/python debug/probe_comfy_xpu_int8_1024.py \
        --res 1024 --steps 4 --dtype bf16 --save /tmp/int8_bf16.pt
    ... --dtype bf16 --int8 --save /tmp/int8_int8.pt

Layers whose K is not a multiple of the ConvRot group size (256) cannot use
the kernels and stay bf16; the probe prints how many of each, and the reason.
"""
import argparse
import gc
import logging
import os
import sys
import time

sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")

parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=1024)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
parser.add_argument("--int8", action="store_true", help="quantize UNet linears to ConvRot INT8")
parser.add_argument("--patch", action="store_true",
                    help="route the plain Linear forward through the int8 kernels")
parser.add_argument("--only", default="all", choices=("all", "down", "up"),
                    help="which linears to quantize: all, the FFN down-projection, or the up-projection")
parser.add_argument("--group", type=int, default=256)
parser.add_argument("--save", default="")
parser.add_argument("--ckpt", default="/home/okolenmi/comfy/ComfyUI/models/checkpoints/divA.safetensors")
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
comfy.cli_args.args.bf16_unet = args.dtype == "bf16"
comfy.cli_args.args.fp16_unet = args.dtype == "fp16"

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


def free_mb():
    try:
        f, _ = torch.xpu.mem_get_info()
        return f / 2 ** 20
    except Exception:
        return float("nan")


def measure(tag):
    ms, out = timed(2)
    print(f"  {tag} warm-up                {ms:8.1f} ms  ({ms / 2:6.1f} ms/step)")
    times = []
    for i in range(args.runs):
        ms, out = timed(args.steps)
        times.append(ms / args.steps)
        print(f"  {tag} run {i + 1}                {ms:8.1f} ms  ({ms / args.steps:6.1f} ms/step)")
    times.sort()
    print(f"  {tag}: min {times[0]:.1f}  median {times[len(times) // 2]:.1f} ms/step"
          f"   driver free {free_mb():.0f} MB")
    return times[0], out


dm = model.model.diffusion_model
print(f"\n{args.res}x{args.res} sampling, cfg 7, euler/normal, batch 2, dtype {args.dtype}")
print(f"  UNet dtype {model.model.get_dtype()}   driver free {free_mb():.0f} MB")

base_best, base_out = measure("bf16" if args.dtype == "bf16" else "fp16")

if args.save:
    torch.save(base_out.detach().cpu(), args.save.replace(".pt", "_base.pt"))

if not args.int8:
    sys.exit(0)

# ---------------------------------------------------------------- quantize
from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout  # noqa: E402
import comfy.xpu_int8 as xpu_int8  # noqa: E402

print(f"\n  int8 library: {xpu_int8.library_path()}")
if xpu_int8.library() is None:
    raise SystemExit("libxpuint8.so not found -- cannot use the int8 path")

done = 0
skipped = {}
saved_mb = 0.0
for name, m in dm.named_modules():
    w = getattr(m, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 2:
        continue
    if isinstance(w, QuantizedTensor):
        continue
    if args.only == "down" and not name.endswith(".net.2"):
        continue
    if args.only == "up" and not name.endswith(".net.0.proj"):
        continue
    N, K = int(w.shape[0]), int(w.shape[1])
    if w.dtype != torch.bfloat16:
        skipped.setdefault(f"dtype {w.dtype}", 0)
        skipped[f"dtype {w.dtype}"] += 1
        continue
    if K % args.group:
        skipped.setdefault(f"K={K} not /{args.group}", 0)
        skipped[f"K={K} not /{args.group}"] += 1
        continue
    if N % 16:
        skipped.setdefault(f"N={N} not /16", 0)
        skipped[f"N={N} not /16"] += 1
        continue
    qdata, params = TensorWiseINT8Layout.quantize(
        w.detach().reshape(-1, K).contiguous(), is_weight=True, per_channel=True,
        convrot=True, convrot_groupsize=args.group)
    saved_mb += (w.nbytes - qdata.nbytes) / 2 ** 20
    m.weight = torch.nn.Parameter(
        QuantizedTensor(qdata, "TensorWiseINT8Layout", params), requires_grad=False)
    done += 1

gc.collect()
torch.xpu.empty_cache()
print(f"  quantized {done} linear weights, kept {sum(skipped.values())} bf16"
      f"   ({saved_mb:.0f} MB of fp32-free weight storage)")
for reason, n in sorted(skipped.items()):
    print(f"    {n:3d} x  {reason}")
print(f"  driver free after quantizing {free_mb():.0f} MB")

stats = {"fused": 0, "bf16": 0, "none": 0}

if args.patch:
    # SDXL reaches torch.nn.functional.linear through Linear.forward_comfy_cast_weights;
    # the int8 dispatch exists only in comfy.ops.linear_input_act, which newer models
    # call explicitly. This is the wiring the probe is measuring.
    import torch.nn.functional as F

    import comfy.ops as ops_mod

    def _int8_linear(x, weight, bias):
        if isinstance(weight, QuantizedTensor) and weight._layout_cls == "TensorWiseINT8Layout":
            r = comfy.xpu_int8.convrot_int8_linear(
                x, *TensorWiseINT8Layout.get_plain_tensors(weight), bias,
                convrot=getattr(weight._params, "convrot", False),
                convrot_groupsize=getattr(weight._params, "convrot_groupsize", 256))
            if r is not None:
                stats["fused"] += 1
                return r
            stats["none"] += 1
            weight = weight.dequantize()
        stats["bf16"] += 1
        return F.linear(x, weight, bias)

    def _forward(self, input):
        if isinstance(self.weight, QuantizedTensor) and self.weight._layout_cls == "TensorWiseINT8Layout":
            return _int8_linear(input, self.weight, self.bias)
        return orig_forward(self, input)

    orig_forward = ops_mod.disable_weight_init.Linear.forward
    ops_mod.disable_weight_init.Linear.forward = _forward
    print("  plain Linear forward routed through comfy.xpu_int8.convrot_int8_linear")

int8_best, int8_out = measure("int8")

if args.patch:
    print(f"  dispatched: {stats['fused']} fused, {stats['bf16']} bf16, "
          f"{stats['none']} int8 fallbacks (last 4 steps)")


if args.save:
    torch.save(int8_out.detach().cpu(), args.save)

d = (int8_out.float() - base_out.to(int8_out.dtype).float()).abs()
print(f"\n  int8 vs baseline latent: max |diff| {d.max().item():.5f}  "
      f"mean |diff| {d.mean().item():.6f}")
print(f"  step time: {base_best:.1f} -> {int8_best:.1f} ms  "
      f"({base_best / int8_best:.2f}x)")


# ------------------------------------------------- how many layers really fused
fused = {"ok": 0, "none": 0}
reasons = {}


class Counter:
    def __init__(self, fn):
        self.fn = fn
        self.n = 0

    def __call__(self, *a, **kw):
        self.n += 1
        r = self.fn(*a, **kw)
        if r is None:
            fused["none"] += 1
        else:
            fused["ok"] += 1
        return r


orig_linear = xpu_int8.convrot_int8_linear
orig_fallback = xpu_int8._fallback
xpu_int8.convrot_int8_linear = Counter(orig_linear)


def fallback(what, reason):
    key = f"{what}: {reason}"
    reasons[key] = reasons.get(key, 0) + 1
    return orig_fallback(what, reason)


xpu_int8._fallback = fallback
sample(1)
torch.xpu.synchronize()
xpu_int8.convrot_int8_linear = orig_linear
xpu_int8._fallback = orig_fallback
print(f"\n  one step issued {fused['ok'] + fused['none']} int8 linear calls: "
      f"{fused['ok']} fused, {fused['none']} fell back")
for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:6]:
    print(f"    {n:4d} x  {reason}")
