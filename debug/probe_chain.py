#!/usr/bin/env python
"""Chain several generations and watch for the OOM / hang.

The reported failure is a VRAM spike when a previous model is still being
lazily unloaded while the CLIP is fully resident and CLIP encoding starts. It
is slightly random and tends to appear after a few prompts in a row, so this
runs a chain and samples VRAM throughout.

    python debug/probe_chain.py [runs] [steps]

Monitors, from the client:
  * vram_free / torch_vram_free  (driver and torch-allocator level, via
    /system_stats) -- a spike shows as vram_free collapsing
  * the server's own log for unload/reload messages
  * wall time per run, so a hang is distinguishable from a slow run
"""
import asyncio
import json
import os
import sys
import time

import aiohttp

BASE = "http://127.0.0.1:8188"
PROMPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_moss.json")


async def vram(session):
    try:
        async with session.get(f"{BASE}/system_stats") as r:
            d = await r.json()
        dev = (d.get("devices") or [{}])[0]
        return (dev.get("vram_free", 0), dev.get("torch_vram_free", 0),
                dev.get("vram_total", 0))
    except Exception:
        return None


async def run_once(session, steps, sample_period=0.25):
    p = json.load(open(PROMPT))
    p["16"]["inputs"]["steps"] = steps
    p["16"]["inputs"]["seed"] = int(time.time() * 1000) % (2 ** 31)
    # A NEW prompt each time: identical text would be served from ComfyUI's
    # node cache and the CLIP would never actually encode, which is exactly
    # the condition the VRAM spike needs (CLIP resident + encoding running
    # while the previous model is still being unloaded).
    tag = f" {time.time_ns() % 100000}"
    p["4"]["inputs"]["text"] = p["4"]["inputs"]["text"] + tag
    p["5"]["inputs"]["text"] = p["5"]["inputs"]["text"] + tag

    state = {"min_vram_free": 1 << 62, "min_torch_free": 1 << 62,
             "samples": 0, "vram": [], "t0": time.monotonic()}

    async def watcher():
        while not state.get("done"):
            v = await vram(session)
            if v is not None:
                state["min_vram_free"] = min(state["min_vram_free"], v[0])
                state["min_torch_free"] = min(state["min_torch_free"], v[1])
                state["samples"] += 1
                state["vram"].append((round(time.monotonic() - state["t0"], 2),
                                      round(v[0] / 2**20), round(v[1] / 2**20)))
            await asyncio.sleep(sample_period)

    w = asyncio.create_task(watcher())
    try:
        async with session.post(f"{BASE}/prompt", json={"prompt": p}) as r:
            resp = await r.json()
        if "error" in resp:
            state["error"] = json.dumps(resp["error"])[:300]
            return state
        pid = resp["prompt_id"]
        while True:
            await asyncio.sleep(0.2)
            async with session.get(f"{BASE}/history/{pid}") as r:
                h = await r.json()
            if h:
                break
            if time.monotonic() - state["t0"] > 900:
                state["error"] = "TIMEOUT (hang?)"
                break
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"
    finally:
        state["done"] = True
        w.cancel()
    state["wall"] = time.monotonic() - state["t0"]
    return state


async def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    async with aiohttp.ClientSession() as s:
        for i in range(runs):
            try:
                st = await run_once(s, steps)
            except Exception as e:
                print(f"run {i + 1:2d}: SERVER GONE ({type(e).__name__}: {e})")
                break
            err = st.get("error")
            mark = "FAIL" if err else "ok  "
            end = st["vram"][-1] if st["vram"] else (0, 0, 0)
            print(f"run {i + 1:2d}: {mark} wall={st.get('wall', 0):6.1f}s  "
                  f"min vram_free={st['min_vram_free'] / 2**20:8.0f} MB  "
                  f"end vram_free={end[1]:8d} MB  "
                  f"min torch_free={st['min_torch_free'] / 2**20:8.0f} MB  "
                  f"end torch_free={end[2]:6d} MB"
                  + (f"  ERR={err}" if err else ""))
            if err and "TIMEOUT" in str(err):
                print("  last vram timeline (t, vram_free MB, torch_free MB):")
                for t, v, tv in st["vram"][-25:]:
                    print(f"    {t:7.2f} {v:9d} {tv:9d}")
                break
            if err:
                break


asyncio.run(main())
