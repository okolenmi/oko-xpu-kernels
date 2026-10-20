#!/usr/bin/env bash
# Is the PNG difference caused by COMFY_PROFILE_FWD, or by the tanh change?
# Runs the same 3-step render (seed 1234) three ways, fresh server each time:
#   ref      shipped sycl::tanh library, no profiler
#   new      expm1 library, no profiler
#   new+prof expm1 library, COMFY_PROFILE_FWD set (what ab_tanh.sh does)
set -u
cd /home/okolenmi/comfy/xpu-kernels
PY=/home/okolenmi/comfy/venv/bin/python
OUT=/tmp/opencode/tanh_ab/md5check
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

run_case() {
    local label="$1" lib="$2" prof="$3"
    stop
    cp "$lib" "$INSTALLED"
    if [[ -n "$prof" ]]; then
        COMFY_PROFILE_FWD="$OUT/prof_${label}.txt" \
            bash debug/server.sh --deterministic >"$OUT/srv_${label}.log" 2>&1 &
    else
        bash debug/server.sh --deterministic >"$OUT/srv_${label}.log" 2>&1 &
    fi
    wait_ready || { echo "server failed for $label"; return 1; }
    rm -f /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png
    timeout 900 "$PY" -u debug/probe_moss.py 3 1234 >/dev/null 2>&1
    local img md5
    img=$(ls -t /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png 2>/dev/null | head -1)
    cp "$img" "$OUT/${label}_3step.png"
    md5=$(md5sum "$OUT/${label}_3step.png" | cut -d' ' -f1)
    stop
    echo "$label  md5=$md5" | tee -a "$OUT/summary.txt"
}

REF=/tmp/opencode/tanh_ab/libxpuint8_tanh_ref.so
NEW=/tmp/opencode/tanh_ab/libxpuint8_expm1.so
run_case ref      "$REF" ""
run_case new      "$NEW" ""
run_case new_prof "$NEW" "yes"
run_case ref_prof "$REF" "yes"

echo; cat "$OUT/summary.txt"
