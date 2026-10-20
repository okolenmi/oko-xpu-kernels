#!/usr/bin/env bash
# Step-level paired A/B for the situ tanh fix: shipped sycl::tanh vs expm1.
#
#   debug/ab_tanh_slope.sh [rounds]
#
# Arms are two prebuilt libxpuint8.so files (the library is loaded once per
# process, so the server is restarted between arms):
#   ref  /tmp/opencode/tanh_ab/libxpuint8_tanh_ref.so
#   new  /tmp/opencode/tanh_ab/libxpuint8_expm1.so
#
# Per arm and round: a 3-step render (fixed seed) whose PNG hash must be equal
# in both arms, then a short and a long render of the SAME prompt/seed, whose
# wall clocks give the marginal ms/step by slope -- (long - short)/(long-short)
# -- which is free of the fixed ~1 s/run cost. Interleaved, so a drift in
# machine state hits both arms.
#
# Expected effect is -25 to -39 ms/step (~3%): the arms must be interleaved and
# repeated, a single pair is inside this box's noise.
set -u
cd /home/okolenmi/comfy/xpu-kernels
PY=/home/okolenmi/comfy/venv/bin/python
OUT=/tmp/opencode/tanh_slope
SHORT=${SHORT:-5}
LONG=${LONG:-25}
ROUNDS=${1:-2}
SEED_IMG=1234
INSTALLED=/home/okolenmi/comfy/ComfyUI/comfy/lib/libxpuint8.so
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
vram_counters() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | tr '\n' ' '; }
wall_of() { sed -n 's/.*wall=\([0-9.]*\)s.*/\1/p' | tail -1; }

run_arm() {
    local label="$1" lib="$2" round="$3"
    stop
    cp "$lib" "$INSTALLED"
    echo "--- arm=$label round=$round ---"
    bash debug/server.sh --deterministic >/dev/null 2>&1 &
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

    # Warm the prompt cache with a *different* seed: the text encode is cached
    # per prompt text, but ComfyUI also caches whole node results per
    # (steps, seed), so a warm-up that matches a timed run makes that run
    # return in 0.2 s and the "slope" becomes a single wall time. Every timed
    # run below therefore uses a seed nothing else has used.
    timeout 900 "$PY" -u debug/probe_moss.py 3 9001 >/dev/null 2>&1
    local ws wl
    ws=$(timeout 900 "$PY" -u debug/probe_moss.py "$SHORT" 9002 2>/dev/null | wall_of)
    wl=$(timeout 900 "$PY" -u debug/probe_moss.py "$LONG" 9003 2>/dev/null | wall_of)
    kill $sampler 2>/dev/null
    stop

    local slope
    slope=$(awk -v a="$wl" -v b="$ws" -v l="$LONG" -v s="$SHORT" 'BEGIN{printf "%.1f", (a-b)/(l-s)*1000}')
    if awk -v b="$ws" 'BEGIN{exit !(b < 2)}'; then
        echo "WARNING: short run was ${ws}s -- cached, the slope is invalid" | tee -a "$OUT/summary.txt"
    fi
    echo "$label r$round  md5=$md5  short(${SHORT})=${ws}s  long(${LONG})=${wl}s  slope=${slope} ms/step" \
        | tee -a "$OUT/summary.txt"
}

for round in $(seq 1 "$ROUNDS"); do
    run_arm ref /tmp/opencode/tanh_ab/libxpuint8_tanh_ref.so "$round"
    run_arm new /tmp/opencode/tanh_ab/libxpuint8_expm1.so "$round"
done

# leave the candidate installed
cp /tmp/opencode/tanh_ab/libxpuint8_expm1.so "$INSTALLED"

echo
echo "=== summary ==="
cat "$OUT/summary.txt"
echo
echo "PNG hashes must all be equal:"
md5sum "$OUT"/*_3step.png
echo
echo "per-round slope delta (new - ref):"
awk '/^(ref|new) r[0-9]/ {
        lab=$1; rnd=$2; split($0, f, " "); v=0
        for (i in f) if (f[i] ~ /^slope=/) { sub("slope=", "", f[i]); v=f[i]+0 }
        if (lab=="ref") ref[rnd]=v; else printf "  %s: %+.1f ms/step\n", rnd, v-ref[rnd]
     }' "$OUT/summary.txt"
