#!/usr/bin/env python
"""Count preview images actually delivered over the websocket, per sampler.

The fork's latent_preview.py decodes previews on a background thread
(XPUPreviewWorker) and rides them on progress-bar updates. A preview that is
never produced, or produced too late, never reaches the browser -- so the
symptom is "previews are skipped". This counts the preview frames that
actually arrive, and the progress messages, for each sampler.

Usage: sd_preview_count.py <sampler> [steps]
"""
import asyncio
import json
import os
import random
import struct
import sys
import time

import aiohttp

BASE = "http://127.0.0.1:8188"
WD = "/home/okolenmi/comfy/ComfyUI/user/default/workflows"


def build_prompt(sampler, steps, seed=None):
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "Dc1.safetensors"}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "a photograph of a cat sitting on a chair",
                         "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "blurry, low quality",
                         "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"seed": seed if seed is not None else random.randint(1, 2**31 - 1), "steps": steps, "cfg": 7.0,
                         "sampler_name": sampler, "scheduler": "normal",
                         "denoise": 1.0, "model": ["1", 0],
                         "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "sdprev", "images": ["6", 0]}},
    }


async def main():
    sampler = sys.argv[1] if len(sys.argv) > 1 else "euler"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    save_dir = sys.argv[3] if len(sys.argv) > 3 else None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    previews = 0
    progress = []
    done = False
    client_id = f"probe-{sampler}-{int(time.time())}"

    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(f"ws://127.0.0.1:8188/ws?clientId={client_id}") as ws:
            async def watch():
                nonlocal previews, done
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        event = struct.unpack(">I", msg.data[:4])[0]
                        if event == 1:
                            previews += 1
                            if save_dir:
                                with open(os.path.join(save_dir, f"p{previews:03d}.jpg"), "wb") as f:
                                    f.write(msg.data[8:])
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            m = json.loads(msg.data)
                        except Exception:
                            continue
                        if m.get("type") == "progress":
                            progress.append(m["data"].get("value"))
                        if m.get("type") == "execution_success":
                            done = True
                            return

            task = asyncio.create_task(watch())
            p = build_prompt(sampler, steps)
            async with s.post(f"{BASE}/prompt", json={"prompt": p, "client_id": client_id}) as r:
                resp = await r.json()
            if "error" in resp:
                print("ERROR:", json.dumps(resp["error"])[:500])
                return
            t0 = time.monotonic()
            while not done and time.monotonic() - t0 < 600:
                await asyncio.sleep(0.1)
            # give late previews a moment to arrive
            await asyncio.sleep(1.5)
            task.cancel()

    print(f"sampler={sampler:22s} steps={steps:3d}  previews={previews:3d}  "
          f"progress_updates={len(progress):3d}  progress_values={progress}")


asyncio.run(main())
