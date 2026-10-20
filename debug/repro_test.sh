#!/usr/bin/env bash
# How reproducible is a 3-step render, really?
#
# Starts a fresh server for every run (same library, same seed, same prompt,
# --deterministic) and records the PNG hash. Two hashes with a stable
# split means the model is not bit-reproducible across processes, which makes
# a PNG-md5 gate useless for A/Bs and needs to be chased; one hash means the
# earlier flip was a one-off (a cache or VRAM state change).
#
#   debug/repro_test.sh [runs]
set -u
cd /home/okolenmi/comfy/xpu-kernels
PY=/home/okolenmi/comfy/venv/bin/python
OUT=/tmp/opencode/repro
mkdir -p "$OUT"
: > "$OUT/hashes.txt"
RUNS=${1:-6}

stop() { pkill -f "ComfyUI/[m]ain.py" 2>/dev/null; sleep 6; }
wait_ready() {
    for _ in $(seq 1 150); do
        curl -s --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && return 0
        sleep 2
    done
    return 1
}

for i in $(seq 1 "$RUNS"); do
    stop
    bash debug/server.sh --deterministic >/dev/null 2>&1 &
    wait_ready || { echo "server failed (run $i)"; exit 1; }
    rm -f /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png
    local_start=$(date +%s)
    timeout 900 "$PY" -u debug/probe_moss.py 3 1234 >/dev/null 2>&1
    img=$(ls -t /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png 2>/dev/null | head -1)
    md5=$(md5sum "$img" | cut -d' ' -f1)
    cp "$img" "$OUT/run${i}_$md5.png"
    # which INT8 kernels ran how often: a changed count means a changed path
    calls=$(grep -c "xpu_kernels" debug/server.log 2>/dev/null)
    echo "run$i  md5=$md5  wall=$(( $(date +%s) - local_start ))s  $calls int8 log lines" | tee -a "$OUT/hashes.txt"
done
stop
echo; sort "$OUT/hashes.txt" | awk '{print $2}' | sort | uniq -c
