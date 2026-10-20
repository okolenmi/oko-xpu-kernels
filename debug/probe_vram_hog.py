"""Hold VRAM on the GPU, like the ComfyUI browser tab does.

The user reports the ComfyUI tab keeps ~1 GB of VRAM resident on the shared
Arc even after the server is closed. That is what pushes the peak over the
edge, so the reproducer needs the same pressure.

    python debug/probe_vram_hog.py [MB] [seconds]

Allocates and touches the memory (so it is really resident, not just
reserved) and holds it.
"""
import sys
import time

import torch

mb = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 3600

n = mb * 1024 * 1024 // 2
print(f"allocating {mb} MB on xpu:0 ...", flush=True)
t = torch.empty(n, dtype=torch.bfloat16, device="xpu:0")
# touch every page so the driver actually backs it
for i in range(0, n, 1 << 20):
    t[i] = 1.0
torch.xpu.synchronize()
print(f"held: allocated={torch.xpu.memory_allocated() / 2**20:.0f} MB "
      f"reserved={torch.xpu.memory_reserved() / 2**20:.0f} MB", flush=True)

end = time.time() + seconds
while time.time() < end:
    time.sleep(5)
print("released", flush=True)
