"""
Why the hand-written INT8 GEMM cannot beat torch._int_mm: a shape sweep
showing it is L2-bandwidth bound (~4 TB/s flat) rather than DPAS-issue bound.

    ./native/build.sh && python3 probe_gemm_shapes.py
"""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import native

GV3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native", "bin", "gemm_v3")
shapes = [(4096,3072,3072),(4096,256,3072),(4096,16,3072),(4096,3072,512),
          (512,3072,3072),(8192,3072,3072),(4096,3072,12288)]
print(f"{'M x N x K':<20} {'ms':>9} {'TOPS':>8} {'B+A GB':>9} {'TB/s':>7}  note")
for (M,N,K) in shapes:
    r = subprocess.run([GV3, str(M), str(N), str(K), "8"], capture_output=True, text=True).stdout.strip().splitlines()[-1].split()
    ms, tops = float(r[0]), float(r[2])
    rows = 8
    b = (M//rows)*N*K      # B re-read once per row-tile
    a = M*K*(N//16)       # A re-read once per column-tile
    tot = (b+a)/1e9
    note = ""
    if N <= 256: note = "B fits in L2"
    if K <= 512:  note = "K tiny: A/B tiles hot in L1"
    print(f"{M}x{N}x{K:<12} {ms:9.4f} {tops:8.2f} {tot:9.2f} {tot/(ms/1e3)/1e3:7.2f}  {note}")