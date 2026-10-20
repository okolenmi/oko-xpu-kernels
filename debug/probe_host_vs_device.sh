#!/usr/bin/env bash
# Is a step host-bound, CPU-bound or device-bound?  One server, four phases.
#
#   debug/probe_host_vs_device.sh [steps]
#
# What it records, all from the same warm session and the same prompt:
#   * per-forward host enqueue time and process CPU time (COMFY_HOST_TRACE in
#     comfy/model_base.py), no synchronisation added;
#   * the wall and CPU time spent inside every torch.xpu.synchronize()
#     (XPU_TRACE_SYNC=1) -- a spinning wait shows cpu ~= wall, a blocking one
#     shows cpu ~= 0;
#   * the server's CPU cores, sampled from /proc/<pid>/stat;
#   * end-to-end wall per phase.
#
# Phases: short (5) to get the fixed per-run cost, idle (N), three busy CPU
# loops on the other cores, and a light GPU matmul loop.  If CPU load moves the
# wall it is CPU-bound; if only GPU load moves it, it is device-bound and the
# 100% core in `top` is a spinning wait.
set -u
cd /home/okolenmi/comfy/xpu-kernels
PY=/home/okolenmi/comfy/venv/bin/python
OUT=/tmp/opencode/hostdev
STEPS=${1:-12}
rm -rf "$OUT"; mkdir -p "$OUT"

stop() { pkill -f "ComfyUI/[m]ain.py" 2>/dev/null; sleep 6; }
wait_ready() {
    for _ in $(seq 1 150); do
        curl -s --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && return 0
        sleep 2
    done
    return 1
}
mark() { date +%s.%N >"$OUT/mark_$1"; }
phase() {   # name steps seed
    local name=$1 steps=$2 seed=$3
    mark "${name}_start"
    "$PY" -u debug/probe_moss.py "$steps" "$seed" 2>&1 |
        sed "s/^/$name /" | tee -a "$OUT/walls.txt"
    mark "${name}_end"
}

stop
rm -f /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png
COMFY_HOST_TRACE="$OUT/trace.csv" XPU_TRACE_SYNC=1 \
    bash debug/server.sh --deterministic >/dev/null 2>&1 &
wait_ready || { echo "server failed"; exit 1; }
PID=$(pgrep -f "ComfyUI/[m]ain.py" | head -1)
echo "server pid $PID"

"$PY" - "$PID" >"$OUT/cpu.csv" 2>/dev/null <<'EOF' &
import sys, time
pid = int(sys.argv[1])
while True:
    try:
        with open(f"/proc/{pid}/stat") as f:
            p = f.read().split()
    except OSError:
        break
    print(f"{time.time():.3f} {int(p[13]) + int(p[14])}", flush=True)
    time.sleep(0.5)
EOF
SAMPLER=$!

phase warm 3 9001           # model load + prompt cache, not reported
sleep 1
phase short 5 9002          # fixed-cost calibration
sleep 1
phase idle "$STEPS" 9003

HOGS=""
for _ in 1 2 3; do ( while :; do :; done ) & HOGS="$HOGS $!"; done
sleep 2
phase cpuload "$STEPS" 9004
kill $HOGS 2>/dev/null; wait $HOGS 2>/dev/null
sleep 2

"$PY" - >/dev/null 2>&1 <<'EOF' &
import torch
a = torch.randn(1024, 1024, device="xpu", dtype=torch.bfloat16)
while True:
    b = a @ a
    torch.xpu.synchronize()
EOF
GPUHOG=$!
sleep 3
phase gpuload "$STEPS" 9005
kill $GPUHOG 2>/dev/null

kill $SAMPLER 2>/dev/null
stop

"$PY" - "$OUT" "$STEPS" <<'EOF'
import re
import statistics as st
import sys
from pathlib import Path

out = Path(sys.argv[1])
steps = int(sys.argv[2])
walls = {}
for line in (out / "walls.txt").read_text().splitlines():
    m = re.match(r"(\w+)\s+moss: wall=([\d.]+)s", line)
    if m:
        walls[m.group(1)] = float(m.group(2))

trace = [(float(a), float(b), float(c)) for _, a, b, c in
         (l.split(",") for l in (out / "trace.csv").read_text().splitlines()[1:] if l.strip())]
cpu = [(float(a), int(b)) for a, b in
       (l.split() for l in (out / "cpu.csv").read_text().splitlines() if l.strip())]
mark = {p.stem[5:]: float(p.read_text())
        for p in out.glob("mark_*")}

def phase_stats(name, n):
    t0, t1 = mark[f"{name}_start"] * 1e3, mark[f"{name}_end"] * 1e3
    fwd = [r for r in trace if t0 <= r[0] <= t1]
    fwd = fwd[2:]                       # first two forwards re-encode the prompt
    sub = st.median([r[1] for r in fwd]) if fwd else float("nan")
    c = [v for t, v in cpu if t0 / 1e3 <= t <= t1 / 1e3]
    cores = (max(c) - min(c)) / 100.0 / (t1 / 1e3 - t0 / 1e3) if len(c) > 2 else float("nan")
    wall = walls.get(name, float("nan"))
    return wall, (wall * 1000 - (walls["short"] * 1000 - 5 * 0)) / n, sub, cores, len(fwd)

slope = (walls["idle"] - walls["short"]) / (steps - 5) * 1000
fixed = walls["short"] * 1000 - 5 * slope
print(f"\nfixed per run {fixed:.0f} ms, marginal {slope:.0f} ms/step "
      f"(from short={walls['short']:.2f}s, idle={walls['idle']:.2f}s)\n")
print(f"{'phase':<9} {'wall':>7} {'step*':>7} {'vs idle':>8} {'submit/fwd':>11} "
      f"{'server cores':>13} {'forwards':>8}")
base = (walls["idle"] * 1000 - fixed) / steps
for name in ("idle", "cpuload", "gpuload"):
    if name not in walls:
        continue
    wall = walls[name]
    step = (wall * 1000 - fixed) / steps
    t0, t1 = mark[f"{name}_start"] * 1e3, mark[f"{name}_end"] * 1e3
    fwd = [r for r in trace if t0 <= r[0] <= t1][2:]
    sub = st.median([r[1] for r in fwd]) if fwd else float("nan")
    c = [v for t, v in cpu if t0 / 1e3 <= t <= t1 / 1e3]
    cores = (max(c) - min(c)) / 100.0 / (t1 / 1e3 - t0 / 1e3) if len(c) > 2 else float("nan")
    print(f"{name:<9} {wall:>6.2f}s {step:>6.0f}ms {100 * (step / base - 1):>+7.1f}% "
          f"{sub:>10.1f}ms {cores:>12.2f} {len(fwd):>8}")
print("\n* step = (wall - fixed) / steps; with the sampler running ahead the "
      "per-step cost is device time, not the host enqueue time.")
EOF

echo
echo "=== synchronize() waits (server log) ==="
grep "\[sync\]" debug/server.log | tail -8
echo
echo "=== sum of sync wall / cpu for the last prompt ==="
grep "\[sync\]" debug/server.log | tail -12 |
    sed 's/.*wall=\([0-9.]*\)ms cpu=\([0-9.]*\)ms.*/\1 \2/' |
    awk '{w+=$1; c+=$2; n++} END {if (n) printf "%d waits, wall %.0f ms, cpu %.0f ms -> %s\n", n, w, c, (c > 0.7*w ? "SPINS" : "blocks")}'
