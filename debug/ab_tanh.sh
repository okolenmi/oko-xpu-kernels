#!/usr/bin/env bash
# Paired, interleaved A/B of the situ tanh: sycl::tanh (shipped) vs the
# expm1 form (comfy/lib/xpu_int8/src.cpp:tanh_fast).
#
#   debug/ab_tanh.sh [rounds]
#
# The two arms are two prebuilt libxpuint8.so files, installed before the
# server starts (the library is loaded once per process):
#   ref  /tmp/opencode/tanh_ab/libxpuint8_tanh_ref.so   sycl::tanh
#   new  /tmp/opencode/tanh_ab/libxpuint8_expm1.so      expm1
#
# Per arm and round: a 3-step render (fixed seed, PNG md5 must match) with
# COMFY_PROFILE_FWD set, so one forward is profiled and the fused kernel's
# device time can be read straight out of the table. Device time, not wall
# clock: the effect is ~3% of a step, below this box's per-run wall noise.
set -u
cd /home/okolenmi/comfy/xpu-kernels
PY=/home/okolenmi/comfy/venv/bin/python
OUT=/tmp/opencode/tanh_ab
INSTALLED=/home/okolenmi/comfy/ComfyUI/comfy/lib/libxpuint8.so
ROUNDS=${1:-2}
SEED_IMG=1234
mkdir -p "$OUT"
: > "$OUT/summary.txt"

stop() { pkill -f "ComfyUI/[m]ain.py" 2>/dev/null; sleep 6; }
wait_ready() {
    for _ in $(seq 1 150); do
        curl -s --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && return 0
        sleep 2
    done
    return 1
}
vram_counters() {
    cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | tr '\n' ' '
}

run_arm() {
    local label="$1" lib="$2" round="$3"
    stop
    cp "$lib" "$INSTALLED"
    echo "--- arm=$label round=$round lib=$(basename "$lib") ---"
    COMFY_PROFILE_FWD="$OUT/prof_${label}_r${round}.txt" \
        bash debug/server.sh --deterministic >"$OUT/srv_${label}_r${round}.log" 2>&1 &
    wait_ready || { echo "server failed for $label"; return 1; }

    ( while true; do
        echo "$(date +%s) cpu/mem:$(ps -o %cpu=,rss= -C python 2>/dev/null | awk '{c+=$1; r+=$2} END{printf "%.0f%% %.0fMB", c, r/1024}') sys:$(free -m | awk '/Mem:/{print $3"MB used"}') vram:$(vram_counters)"
        sleep 3
      done ) >>"$OUT/res_${label}_r${round}.log" 2>&1 &
    local sampler=$!

    rm -f /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png
    timeout 900 "$PY" -u debug/probe_moss.py 3 "$SEED_IMG" >/dev/null 2>&1
    local img md5
    img=$(ls -t /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png 2>/dev/null | head -1)
    cp "$img" "$OUT/${label}_r${round}_3step.png"
    md5=$(md5sum "$OUT/${label}_r${round}_3step.png" | cut -d' ' -f1)
    kill $sampler 2>/dev/null
    stop

    # device time of the fused kernel, from the single profiled forward
    local row
    row=$(awk '/dequant_situ|dequant_act_quantize/ {
                   for (i = 1; i <= NF; i++) if ($i == "count" || $(i) ~ /^[0-9]+$/) {}
                 } /dequant/ { print; exit }' "$OUT/prof_${label}_r${round}.txt" 2>/dev/null)
    echo "$label r$round  md5=$md5  $row" | tee -a "$OUT/summary.txt"
}

for round in $(seq 1 "$ROUNDS"); do
    run_arm ref /tmp/opencode/tanh_ab/libxpuint8_tanh_ref.so "$round"
    run_arm new /tmp/opencode/tanh_ab/libxpuint8_expm1.so "$round"
done

# leave the candidate installed; the caller decides whether to keep it
cp /tmp/opencode/tanh_ab/libxpuint8_expm1.so "$INSTALLED"

echo
echo "=== summary ==="
cat "$OUT/summary.txt"
echo
echo "PNG hashes must all be equal (bit-exact):"
md5sum "$OUT"/*_3step.png
