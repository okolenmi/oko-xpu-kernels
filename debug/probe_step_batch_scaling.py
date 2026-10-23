"""Why does the bigger batch cost more than it should?

The trainer's step (bf16, the LoRA-shaped step, the checkpointing on) measured
at batch 1..6, with the batches *interleaved* in one process so the host's
clock drift is not a factor:

  * the wall per batch: where the host/device crossover actually is;
  * per-kernel device time from the XPU activity profile at batch 2 and batch 4,
    and the ratio: a kernel that simply does 2x more work is 2.00x; anything
    *above* 2x is the super-linear cost we are looking for;
  * the kernel **call count** per batch: gains that come not from slower
    kernels but from more kernels (the checkpointing's per-block decisions,
    chunked ops, fill kernels);
  * `--no-checkpoint`: the same comparison without the recompute, to split the
    checkpointing's share out.

    PYTHONPATH=python LEVEL_ZERO_V1_SDK_PATH=/home/okolenmi/.local/levelzero \
        python debug/probe_step_batch_scaling.py [--no-checkpoint]

No graph capture here: the question is the eager path's device time, and a
failed profiling step loses the device, so the profiles run alone, one per
batch.
"""
import os
import sys
import time

TRAINER = os.environ.get("TRAINER_TREE", "/home/okolenmi/Desktop/B580-diffusion-training")
sys.path.insert(0, TRAINER)

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from next.engine.runtime.checkpointing import ActivationCheckpointer  # noqa: E402
from next.families.sdxl.denoiser import build_unet  # noqa: E402

DEV = torch.device("xpu", 0)


def build(checkpoint):
    torch.manual_seed(0)
    model = build_unet()
    model.to_empty(device=DEV)
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.normal_(0, 0.05)
            else:
                p.zero_()
    model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    if checkpoint:
        # the trainer's own checkpointing: the blocks' activations are
        # recomputed in backward instead of being held
        model.set_checkpointer(ActivationCheckpointer(preserve_rng=False), 1.0)
    return model


def inputs(batch, lat):
    h, w = lat if isinstance(lat, tuple) else (lat, lat)
    g = torch.Generator(device=DEV).manual_seed(batch)
    x = torch.randn(batch, 4, h, w, generator=g, device=DEV, dtype=torch.bfloat16,
                    requires_grad=True)
    ctx = torch.randn(batch, 77, 2048, generator=g, device=DEV, dtype=torch.bfloat16)
    y = torch.randn(batch, 2816, generator=g, device=DEV, dtype=torch.bfloat16)
    ts = torch.full((batch,), 500.0, device=DEV, dtype=torch.float32)
    return x, ts, ctx, y


def timed_step(model, x, ts, ctx, y):
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    model(x, ts, ctx, y).float().pow(2).mean().backward()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) * 1000


def wall_per_batch(model, h=64, w=64):
    """Interleaved: for each batch, the best of R rounds, the order mirrored so
    the drift cancels."""
    walls = {}
    for b in (1, 2, 3, 4, 6):
        x, ts, ctx, y = inputs(b, (h, w))
        t = timed_step(model, x, ts, ctx, y)          # the warmup (and the compile)
        best = t
        for _ in range(3):
            best = min(best, timed_step(model, x, ts, ctx, y))
        for _ in range(3):                            # the mirrored order
            best = min(best, timed_step(model, x, ts, ctx, y))
        walls[b] = best
        print(f"  batch {b}: {best:8.1f} ms   ({best/b:6.1f} ms/sample)")
        del x, ts, ctx, y
    return walls


def profile_batch(model, batch, lat=(64, 64)):
    x, ts, ctx, y = inputs(batch, lat)
    model(x, ts, ctx, y).float().pow(2).mean().backward()
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.XPU]) as prof:
        model(x, ts, ctx, y).float().pow(2).mean().backward()
        torch.xpu.synchronize()
    rows = {}
    for e in prof.key_averages():
        key = e.key
        if key.startswith("aten::"):
            continue          # the aten rows double-count their kernels
        d = float(getattr(e, "device_time_total", 0.0))
        if d <= 0:
            continue
        name = " ".join(key.split())[:64]
        short = name.split("(")[0][:44]
        if short in rows:
            rows[short][0] += d
            rows[short][1] += e.count
        else:
            rows[short] = [d, e.count]
    del x, ts, ctx, y
    torch.xpu.empty_cache()
    return rows


def main():
    checkpoint = "--no-checkpoint" not in sys.argv
    lat = (64, 64)
    for arg in sys.argv[1:]:
        if arg.startswith("--lat="):
            lat = tuple(int(v) for v in arg.split("=", 1)[1].split(","))  # H,W
    lat_profile = lat if len(lat) == 2 else (lat[0], lat[0])
    model = build(checkpoint)
    torch.manual_seed(0)
    print(f"device: {torch.xpu.get_device_name(0)}   "
          f"checkpointing: {checkpoint}   latents {lat_profile[0]}x{lat_profile[1]}")
    print("=== the wall, the batches interleaved in one process ===")
    walls = wall_per_batch(model, *lat_profile)

    print("=== the per-kernel device time, batch 2 against batch 4 ===")
    p2 = profile_batch(model, 2, lat_profile)
    p4 = profile_batch(model, 4, lat_profile)
    both = sorted(set(p2) & set(p4), key=lambda k: -(p4[k][0] / max(p2[k][0], 1e-9)))
    print(f"{'kernel':46s} {'4xBD ms':>9} {'ratio':>7} {'calls 2':>8} {'calls 4':>8}")
    t2 = sum(p2[k][0] for k in p2)
    t4 = sum(p4[k][0] for k in p4)
    print(f"  the device sum: {t2:7.1f} -> {t4:7.1f} ms  ({t4/t2:.2f}x for a "
          f"2x batch)")
    for k in both[:18]:
        v2, c2 = p2[k]
        v4, c4 = p4[k]
        r = v4 / max(v2, 1e-9)
        print(f"{k:46s} {v4:9.1f} {r:6.2f}x  {c2:8d} {c4:8d}")

    print("=== the host's share: the walls against the device sums ===")
    for b in (2, 4):
        p = {2: p2, 4: p4}[b]
        t = {2: t2, 4: t4}[b]
        wall = walls[b]
        print(f"  batch {b}: the wall {wall:7.1f} ms, the device {t:7.1f} ms "
              f"-> the host or idle {(wall - t):7.1f} ms "
              f"({100*(wall-t)/wall:4.1f} % of the step)")


if __name__ == "__main__":
    main()
