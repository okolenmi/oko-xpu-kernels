#!/usr/bin/env python
"""Compare two PNGs: max/mean abs difference and PSNR (8-bit RGB)."""
import sys

import numpy as np
from PIL import Image

a = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.int16)
b = np.asarray(Image.open(sys.argv[2]).convert("RGB"), dtype=np.int16)
if a.shape != b.shape:
    print(f"shape differs: {a.shape} vs {b.shape}")
    sys.exit(1)
d = np.abs(a - b)
mse = float((d.astype(np.float64) ** 2).mean())
psnr = float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)
print(f"size={a.shape} max={int(d.max())} mean={d.mean():.4f} "
      f"p99={np.percentile(d, 99):.1f} changed={100.0 * (d > 0).mean():.3f}% "
      f"psnr={psnr:.2f} dB")
