"""The SDXL step, in the training project's own UNet: where the time goes, and
what this library's INT8 path does to it.

The training project's current implementation lives in `next/` (its own SDXL
UNet, its own GEGLU: `x, gate = proj(x).chunk(2, -1); x * F.gelu(gate)`).  This
script borrows that model -- read-only, no change to its tree -- and measures:

  * the phase split: the forward, the loss, the backward;
  * the host cost, the call count and the per-kernel device time (torch's
    profiler with the XPU activity -- it works; see the note below);
  * the eager step against the captured step (the trainer's own capture
    pattern: forward, loss and backward inside one device graph);
  * the same with the library's INT8 path:
      A  the trainer's UNet, bf16                       (the reference)
      B  every eligible linear replaced by the library's Int8Linear
      C  B, and the GEGLU's FFN replaced by the library's *fused* FFN
         (`int8.ffn(..., act="geglu_erf")`)

The base is frozen and the input requires grad, i.e. the LoRA-shaped step:
the backward computes the activation gradient and no weight gradients.

Memory: the trainer's own checkpointing is on (`ActivationCheckpointer`, its
default), because without it the SDXL step's activations do not leave room
for a graph capture on a 12 GB card -- the capture's pool is the step's
activations again.  The up-projection's bias is zeroed first, so that arm C
(whose fused FFN takes no up-bias) computes the same function as A and B.

    PYTHONPATH=python python debug/probe_sdxl_trainer_step.py [batch] [resolution]
        [--no-checkpoint]

The results are in `docs/sdxl.md` (this library) and in the training
project's `docs/sdxl-int8-measurements.md`.
"""
import os
import sys
import time

TRAINER = os.environ.get("TRAINER_TREE", "/home/okolenmi/Desktop/B580-diffusion-training")
sys.path.insert(0, TRAINER)

import torch  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import xpu_kernels  # noqa: E402
from xpu_kernels import int8  # noqa: E402

from next.engine.runtime.checkpointing import ActivationCheckpointer  # noqa: E402
from next.engine.runtime.profiler import BlockProfiler  # noqa: E402
from next.families.general.ldm_unet.attention import FeedForward, GEGLU  # noqa: E402
from next.families.sdxl.denoiser import build_unet  # noqa: E402

DEV = torch.device("xpu", 0)


def eligible(mod):
    """The library's own rule: the contraction a multiple of 256, the output a
    multiple of 16."""
    w = mod.weight
    return w.shape[1] % 256 == 0 and w.shape[0] % 16 == 0


class FusedFFN(torch.nn.Module):
    """The trainer's GEGLU + the FFN's second projection, as one fused FFN.

    The library's `ffn` takes the up-projection's weight as [2*ffn, K] and the
    down-projection's as [K, ffn] -- the same layout nn.Linear has -- so this
    is a drop-in for the pair.  The up-projection's bias must be zero (the
    fused kernel adds no bias before the activation); the script zeroes it.
    """

    def __init__(self, geglu, down):
        super().__init__()
        self.up8, self.up_scale = int8.prepare_weight_fused(geglu.proj.weight.detach())
        self.down8, self.down_scale = int8.prepare_weight_fused(down.weight.detach())
        self.down_bias = None if down.bias is None else down.bias.detach().clone()
        self.out = down.out_features

    def forward(self, x):
        y = int8.ffn(x, self.up8, self.up_scale, self.down8, self.down_scale,
                     self.down_bias, act="geglu_erf")
        return y.reshape(*x.shape[:-1], self.out)


def swap_linears(model):
    n = 0
    for mod in list(model.modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear) and eligible(child):
                setattr(mod, child_name, int8.Int8Linear.from_linear(child, backward="int8"))
                n += 1
    return n


def swap_geglu(model):
    n = 0
    for mod in list(model.modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, FeedForward):
                seq = [c for c in child.net if not isinstance(c, torch.nn.Dropout)]
                if len(seq) == 2 and isinstance(seq[0], GEGLU) and isinstance(seq[1], torch.nn.Linear):
                    setattr(mod, child_name, FusedFFN(seq[0], seq[1]))
                    n += 1
    return n


def build(seed=0, checkpoint=True):
    torch.manual_seed(seed)
    model = build_unet()
    model.to_empty(device=DEV)
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.normal_(0, 0.05)
            else:
                p.zero_()
        for mod in model.modules():
            if isinstance(mod, GEGLU) and mod.proj.bias is not None:
                mod.proj.bias.zero_()
    model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    if checkpoint:
        # the trainer's own checkpointing: the blocks' activations are
        # recomputed in backward instead of being held
        model.set_checkpointer(ActivationCheckpointer(preserve_rng=False), 1.0)
    return model


def block_profile(model, step):
    """The trainer's own block profiler: with the checkpointing on, the
    recompute in backward is the block's forward, timed for free."""
    checkpointer = ActivationCheckpointer(preserve_rng=False)
    model.set_checkpointer(checkpointer, 1.0)
    prof = BlockProfiler(checkpointer, model.checkpoint_blocks(),
                         lambda: torch.xpu.memory_allocated(), torch.xpu.synchronize)
    step()   # the warmup: the first block pays the kernel compilation
    prof.reset()
    with prof:
        step()
    costs = prof.costs()
    return sorted(((c.recompute_ms, c.activation_mb, name)
                 for name, c in costs.items()), reverse=True)


def timeit(fn, n=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    best = float("inf")
    for _ in range(n):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1000)
    return best


def host_profile(model, x, ctx, y, ts, device_table=False):
    """The host cost, the call count and (optionally) the device-time table
    of one step.

    The XPU activity of the profiler works on this card: it registers PTI
    callbacks with the Level Zero loader, prints a registration warning, and
    still produces a complete device-time table.  What is *not* survivable is
    an out-of-memory during the profiled step: the device is lost
    (UR_RESULT_ERROR_DEVICE_LOST), the profiler's cleanup then fails with
    PTI_ERROR_NOT_IMPLEMENTED, and that exception escapes a destructor -- so
    the process dies instead of raising.  The profiler is the victim, not the
    cause: keep the step inside the memory budget.
    """
    acts = [ProfilerActivity.CPU, ProfilerActivity.XPU] if device_table else [ProfilerActivity.CPU]
    with profile(activities=acts, acc_events=True) as prof:
        model(x, ts, ctx, y).float().pow(2).mean().backward()
        torch.xpu.synchronize()
    table = prof.key_averages()
    calls = sum(e.count for e in table if e.key.startswith("aten::"))
    host_us = sum(e.self_cpu_time_total for e in table)
    if device_table:
        dev = sorted((e for e in table if e.self_device_time_total > 0),
                     key=lambda e: -e.self_device_time_total)
        total = sum(e.self_device_time_total for e in dev)
        print(f"    the device-time table: {total / 1e6:.3f} s over "
              f"{sum(e.count for e in dev)} kernel calls")
        for e in dev[:12]:
            print(f"      {e.self_device_time_total / 1e3:8.2f} ms  x{e.count:<6d} "
                  f"{e.key[:64]}")
    return calls, host_us / 1000.0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    checkpoint = "--no-checkpoint" not in sys.argv
    batch = int(args[0]) if args else 1
    res = int(args[1]) if len(args) > 1 else 512
    lat = res // 8
    print(f"device: {xpu_kernels.device_name()}   batch {batch}, {res}x{res} "
          f"(latents {lat}x{lat})   checkpointing: {checkpoint}")
    print("the base is frozen and the input requires grad: the LoRA-shaped step")

    x = torch.randn(batch, 4, lat, lat, device=DEV, dtype=torch.bfloat16,
                    requires_grad=True)
    ctx = torch.randn(batch, 77, 2048, device=DEV, dtype=torch.bfloat16)
    y = torch.randn(batch, 2816, device=DEV, dtype=torch.bfloat16)
    ts = torch.full((batch,), 500.0, device=DEV, dtype=torch.float32)

    rows = {}
    for arm in ("A: bf16", "B: INT8 linears", "C: INT8 + the fused GEGLU"):
        model = build(checkpoint=checkpoint)
        nlin = ngeglu = 0
        if "B" in arm or "C" in arm:
            nlin = swap_linears(model)
        if "C" in arm:
            ngeglu = swap_geglu(model)
        model.eval()
        torch.xpu.synchronize()
        torch.xpu.reset_peak_memory_stats()
        weights = torch.xpu.memory_allocated()

        def step():
            model(x, ts, ctx, y).float().pow(2).mean().backward()

        step()   # the warmup: the first backward pays the kernel compilation
        torch.xpu.synchronize()
        torch.xpu.reset_peak_memory_stats()

        # the phases, one at a time; nothing is kept alive between them
        t_fwd = timeit(lambda: model(x, ts, ctx, y))
        out = model(x, ts, ctx, y)
        loss = out.float().pow(2).mean()
        t_loss = timeit(lambda: out.float().pow(2).mean(), n=3)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        torch.xpu.synchronize()
        t_bwd = (time.perf_counter() - t0) * 1000
        del out, loss
        torch.xpu.empty_cache()

        t_eager = timeit(step)
        peak = torch.xpu.max_memory_allocated()

        # the captured step (the trainer's pattern: forward, loss, backward)
        t_cap = float("nan")
        try:
            xs = x.detach().clone().requires_grad_(True)
            graph = torch.xpu.XPUGraph()
            torch.xpu.synchronize()
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION), torch.xpu.graph(graph):
                l = model(xs, ts, ctx, y).float().pow(2).mean()
                l.backward()
            torch.xpu.synchronize()
            t_cap = timeit(lambda: graph.replay(), n=10, warmup=3)
            del graph, l, xs
        except Exception as exc:  # noqa: BLE001
            print(f"  {arm}: the capture failed: {type(exc).__name__}: {exc}")
        torch.xpu.empty_cache()

        # the two profilers last: they allocate, and the capture's pool needed
        # the room
        calls, host = host_profile(model, x, ctx, y, ts, device_table=arm.startswith("A"))
        torch.xpu.empty_cache()
        blocks = block_profile(model, step)
        total = sum(b[0] for b in blocks)
        print(f"  {arm}: the checkpointed blocks by forward time (the trainer's "
              f"profiler): {len(blocks)} blocks, {total:.1f} ms in total")
        for ms, mb, name in blocks[:5]:
            print(f"      {ms:7.2f} ms  {mb / 2**20:7.1f} MB  {name}")
        torch.xpu.empty_cache()
        del model
        torch.xpu.empty_cache()
        rows[arm] = (t_fwd, t_loss, t_bwd, t_eager, t_cap, peak, weights, calls, host)

    print()
    print(f"{'arm':26s} {'forward':>9} {'loss':>7} {'backward':>9} {'step':>8} "
          f"{'captured':>9} {'host':>8} {'calls':>7} {'weights':>8} {'peak':>8}")
    for arm, (f, l, b, e, c, p, w, n, h) in rows.items():
        print(f"{arm:26s} {f:8.1f}m {l:6.1f}m {b:8.1f}m {e:7.1f}m "
              f"{c:8.1f}m {h:7.1f}m {n:7d} {w / 2**30:7.2f}G {p / 2**30:7.2f}G")
    a = rows["A: bf16"]
    print()
    for arm, r in rows.items():
        print(f"{arm:26s} eager {r[3] / a[3]:5.2f}x   captured "
              f"{r[4] / a[4]:5.2f}x   host {r[8] / a[8]:5.2f}x   "
              f"the weights {r[6] / a[6]:5.2f}x   the peak {r[5] / a[5]:5.2f}x")


if __name__ == "__main__":
    main()
