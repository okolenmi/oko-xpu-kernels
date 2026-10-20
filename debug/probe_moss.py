#!/usr/bin/env python
"""Run the MossImage workflow through the API and report timing.

Rebuilt from user/default/workflows/moss.json (UI format) into API format.

    python debug/probe_moss.py [steps] [seed] [--stats]

--stats asks the server to report the per-region INT8 counters
(XPU_INT8_TIMING=1 must be set on the server for the counters to appear).
"""
import asyncio
import json
import os
import random
import sys
import time

import aiohttp

BASE = "http://127.0.0.1:8188"

POS_DEFAULT = "amateur photo, shinomiya kaguya,"
NEG_DEFAULT = "beautiful, professional photo, mature, old, asian girl,"


def build_prompt(steps, seed=None):
    # MOSS_POS / MOSS_NEG override the two prompts.  Used by the CFG-batching
    # A/B: passing the same string twice makes both branches tokenize to the
    # same length, which is the only way to make ComfyUI batch the CFG pair.
    pos = os.environ.get("MOSS_POS", POS_DEFAULT)
    neg = os.environ.get("MOSS_NEG", NEG_DEFAULT)
    return {
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": pos, "clip": ["24", 0]}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": neg, "clip": ["24", 0]}},
        "6": {"class_type": "EmptyFlux2LatentImage",
              "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
        "16": {"class_type": "KSampler",
               "inputs": {"seed": seed if seed is not None else random.randint(1, 2**31 - 1),
                         "steps": steps,
                          "cfg": 3.5, "sampler_name": "euler",
                          "scheduler": "sgm_uniform", "denoise": 1.0,
                          "model": ["23", 0], "positive": ["4", 0],
                          "negative": ["5", 0], "latent_image": ["6", 0]}},
        "17": {"class_type": "VAEDecode",
               "inputs": {"samples": ["16", 0], "vae": ["22", 0]}},
        "18": {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "moss-image", "images": ["17", 0]}},
        "22": {"class_type": "VAELoader",
               "inputs": {"vae_name": "moss-image-sft-vae.safetensors"}},
        "23": {"class_type": "UNETLoader",
               "inputs": {"unet_name": "mossImage_sft_pruned_int8.safetensors",
                          "weight_dtype": "default"}},
        "24": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "moss-image-sft-qwen35-int8-convrot.safetensors",
                          "type": "moss_image", "device": "default"}},
    }


async def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    p = build_prompt(steps, seed)
    async with aiohttp.ClientSession() as s:
        t0 = time.monotonic()
        async with s.post(f"{BASE}/prompt", json={"prompt": p}) as r:
            resp = await r.json()
        if "error" in resp:
            print("ERROR:", json.dumps(resp["error"])[:600])
            return
        pid = resp["prompt_id"]
        while True:
            async with s.get(f"{BASE}/history/{pid}") as r:
                h = await r.json()
            if h:
                break
            await asyncio.sleep(0.2)
        dt = time.monotonic() - t0
        outs = h.get("outputs", {})
        for node, o in outs.items():
            for img in o.get("images", []):
                print(f"  wrote {img.get('filename')}")
        print(f"moss: wall={dt:.2f}s -> {dt / steps * 1000:.0f} ms/step "
              f"({steps} steps)")


asyncio.run(main())
