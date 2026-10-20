#!/usr/bin/env bash
# Paired, interleaved A/B for the single-pass rms_rope kernel (comfy/xpu_rope.py).
#
#   debug/ab_rope.sh [rounds]
#
# Arm "rows"  : the new single-pass Triton kernel (default, gate on).
# Arm "base"  : MOSS_NO_ROPE_ROWS=1 -> comfy-kitchen's shipped kernel.
#
# Per arm and round: a 3-step render (fixed seed) whose PNG must hash the same
# in both arms, then a short and a long render whose wall clocks give the
# marginal ms/step by slope -- (wall_long - wall_short) / (long - short) -- the
# only comparison that is free of the fixed ~1 s/run cost.
#
# A resource sampler runs alongside every arm (process CPU/RSS, system memory,
# and the Xe VRAM counter if the driver exposes one) so both arms are shown to
# run under the same conditions.
set -u
cd /home/okolenmi/comfy/xpu-kernels
PY=/home/okolenmi/comfy/venv/bin/python
OUT=/tmp/opencode/rope_ab
SHORT=${SHORT:-5}
LONG=${LONG:-25}
ROUNDS=${1:-2}
SEED_IMG=1234
SEED_TIME=777
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
    # i915/xe expose per-tile VRAM counters; print whatever exists
    cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | tr '\n' ' '
}

wall_of() {  # parse "wall=12.34s" out of the probe output
    sed -n 's/.*wall=\([0-9.]*\)s.*/\1/p' | tail -1
}

run_arm() {
    local label="$1" skipenv="$2" round="$3"
    stop
    echo "--- arm=$label round=$round env='${skipenv:-<none>}' ---"
    env $skipenv bash debug/server.sh --deterministic >"$OUT/srv_${label}_r${round}.log" 2>&1 &
    wait_ready || { echo "server failed for $label"; return 1; }

    ( while true; do
        echo "$(date +%s) cpu/mem:$(ps -o %cpu=,rss= -C python 2>/dev/null | awk '{c+=$1; r+=$2} END{printf "%.0f%% %.0fMB", c, r/1024}') sys:$(free -m | awk '/Mem:/{print $3"MB used"}') vram:$(vram_counters)"
        sleep 3
      done ) >>"$OUT/res_${label}_r${round}.log" 2>&1 &
    local sampler=$!

    rm -f /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png
    timeout 900 "$PY" -u debug/probe_moss.py 3 "$SEED_IMG" >/dev/null 2>&1
    local img
    img=$(ls -t /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png 2>/dev/null | head -1)
    cp "$img" "$OUT/${label}_r${round}_3step.png"
    local md5
    md5=$(md5sum "$OUT/${label}_r${round}_3step.png" | cut -d' ' -f1)

    local ws wl
    ws=$(timeout 900 "$PY" -u debug/probe_moss.py "$SHORT" "$SEED_TIME" 2>/dev/null | wall_of)
    wl=$(timeout 900 "$PY" -u debug/probe_moss.py "$LONG" "$SEED_TIME" 2>/dev/null | wall_of)
    kill $sampler 2>/dev/null
    stop

    local slope
    slope=$(awk -v a="$wl" -v b="$ws" -v l="$LONG" -v s="$SHORT" 'BEGIN{printf "%.1f", (a-b)/(l-s)*1000}')
    echo "$label r$round  md5=$md5  short(${SHORT})=${ws}s  long(${LONG})=${wl}s  marginal=${slope} ms/step" \
        | tee -a "$OUT/summary.txt"
}

for round in $(seq 1 "$ROUNDS"); do
    run_arm rows "" "$round"
    run_arm base "MOSS_NO_ROPE_ROWS=1" "$round"
done

echo
echo "=== summary ==="
cat "$OUT/summary.txt"
echo
echo "image hashes must all be equal (bit-exact):"
for f in "$OUT"/*_3step.png; do md5sum "$f"; done
