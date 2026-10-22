"""Does comfy.xpu_graph capture ComfyUI's SDXL UNet, and is the replay correct?

Runs the diffusion model's forward through the new hook three ways -- the eager
path, the capture, and a replay on *new* input data -- and compares all three.
Then changes a weight under the graph (a LoRA patch does that) and checks that
the result is still the eager one.

    LEVEL_ZERO_V1_SDK_PATH=/home/okolenmi/.local/levelzero \
        /home/okolenmi/comfy/venv/bin/python debug/probe_comfy_xpu_graph.py
"""
import logging
import sys

sys.path.insert(0, "/home/okolenmi/comfy/ComfyUI")

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")

import torch

import comfy.cli_args
import comfy.ops as ops
import comfy.xpu_graph as xg
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

DEV = torch.device("xpu", 0)
DT = torch.bfloat16

CFG = {
    "use_checkpoint": False, "image_size": 32, "out_channels": 4,
    "use_spatial_transformer": True, "legacy": False, "num_classes": "sequential",
    "adm_in_channels": 2816, "in_channels": 4, "model_channels": 320,
    "num_res_blocks": [2, 2, 2], "transformer_depth": [0, 0, 2, 2, 10, 10],
    "channel_mult": [1, 2, 4], "transformer_depth_middle": 10,
    "use_linear_in_transformer": True, "context_dim": 2048, "num_head_channels": 64,
    "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
    "use_temporal_attention": False, "use_temporal_resblock": False,
}


class Wrapper:
    """The bits of comfy.model_base.BaseModel the hook reads."""

    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model
        self.current_patcher = None


def compare(what, got, want):
    a, b = got.float(), want.float()
    diff = (a - b).abs().max().item()
    mag = b.abs().max().item()
    rel = diff / mag if mag else 0.0
    ok = diff <= 1e-3 + 0.05 * mag
    print(f"  {what:38s} max |diff| {diff:.6f} ({100*rel:5.2f} % of {mag:.4f})  "
          f"{'OK' if ok else 'MISMATCH'}")
    return ok


def differs(what, got, want):
    """The opposite check: a replay must not hand back the previous step's tensor."""
    diff = (got.float() - want.float()).abs().max().item()
    ok = diff > 1e-2
    print(f"  {what:38s} max |diff| {diff:.6f}  {'OK' if ok else 'STALE'}")
    return ok


def main():
    comfy.cli_args.args.xpu_graphs = True
    torch.manual_seed(0)
    unet = UNetModel(**CFG, device=DEV, operations=ops.pick_operations(DT, DT, load_device=DEV))
    unet = unet.to(DT).eval()
    with torch.no_grad():          # ComfyUI's UNetModel expects the checkpoint to init it
        for p in unet.parameters():
            if not p.any():
                p.normal_(0, 0.05)
    for p in unet.parameters():
        p.requires_grad_(False)
    model = Wrapper(unet)

    def tensors(seed, res=64):
        g = torch.Generator(device=DEV).manual_seed(seed)
        return dict(
            x=torch.randn(1, 4, res, res, device=DEV, dtype=DT, generator=g),
            t=torch.full((1,), 500.0, device=DEV, dtype=torch.float32),
            ctx=torch.randn(1, 77, 2048, device=DEV, dtype=DT, generator=g),
            y=torch.randn(1, 2816, device=DEV, dtype=DT, generator=g),
        )

    def call(batch, routed):
        opts = {"original_shape": list(batch["x"].shape)}
        if routed:
            return xg.diffusion_forward(model, batch["x"], batch["t"], batch["ctx"],
                                        None, opts, {"y": batch["y"]})
        return unet(batch["x"], batch["t"], context=batch["ctx"], y=batch["y"],
                    transformer_options=opts)

    first = tensors(1)
    second = tensors(2)

    print("=== the first call (this captures) ===")
    with torch.no_grad():
        want = call(first, routed=False)
        got = call(first, routed=True)
    good = compare("the capture vs the eager", got, want)
    print(f"  the counters: {xg.stats()}")

    print("=== the second call, on new input data (this replays) ===")
    with torch.no_grad():
        want2 = call(second, routed=False)
        got2 = call(second, routed=True)
    good &= compare("the replay vs the eager", got2, want2)
    # and the replay must not be returning the previous step's result
    good &= differs("the replay vs the *first* result (stale?)", got2, want)
    print(f"  the counters: {xg.stats()}")

    print("=== a weight changes under the graph (a LoRA patch) ===")
    with torch.no_grad():
        unet.input_blocks[0][0].weight.add_(0.01)
        want3 = call(second, routed=False)
        got3 = call(second, routed=True)
    good &= compare("the re-capture vs the eager", got3, want3)

    print("=== the same, a second time (the key gets disabled, not thrashing) ===")
    with torch.no_grad():
        unet.input_blocks[0][0].weight.add_(0.01)
        want4 = call(second, routed=False)
        got4 = call(second, routed=True)
    good &= compare("the eager fallback vs the eager", got4, want4)
    print(f"  the counters: {xg.stats()}")
    if xg._CACHE is not None:
        for key, why in xg._CACHE.disabled.items():
            print(f"  disabled {key[1]}: {why}")

    print("=== the timing: eager against the replay ===")
    with torch.no_grad():
        eager_ms = xg._time(lambda: call(first, routed=False), 5)
        xg.clear()
        print(f"  the eager path      : {eager_ms:7.1f} ms")
        call(first, routed=True)      # re-capture
        replay_ms = xg._time(lambda: call(first, routed=True), 5)
        print(f"  the hook (replaying): {replay_ms:7.1f} ms  ({eager_ms/replay_ms:.2f}x)")

    print("=== the hook's cost with the flag off ===")
    comfy.cli_args.args.xpu_graphs = False
    with torch.no_grad():
        direct_ms = xg._time(lambda: call(first, routed=False), 5)
        hooked_ms = xg._time(lambda: call(first, routed=True), 5)
    print(f"  the direct call      : {direct_ms:7.1f} ms")
    print(f"  through the hook     : {hooked_ms:7.1f} ms  "
          f"({100*(hooked_ms-direct_ms)/direct_ms:+.2f} %)")
    comfy.cli_args.args.xpu_graphs = True

    print("PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
