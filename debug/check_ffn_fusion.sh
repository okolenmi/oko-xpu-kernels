#!/usr/bin/env bash
# End-to-end check of the dequant+situ+quantize fusion:
#   - fused three-step image must be pixel-identical to the eager one
#   - profile one warm forward to see the up-projection dequant disappear
set -u
cd /home/okolenmi/comfy/xpu-kernels
PY=/home/okolenmi/comfy/venv/bin/python
OUT=/tmp/opencode/ffn_fusion
mkdir -p "$OUT"

stop() { pkill -f "ComfyUI/[m]ain.py" 2>/dev/null; sleep 6; }
wait_ready() {
    for _ in $(seq 1 120); do
        curl -s --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && return 0
        sleep 2
    done
    return 1
}

run_mode() {
    local label="$1" steps="$2" skipenv="$3" prof="$4"
    stop
    echo "=== $label ==="
    env $skipenv XPU_INT8_TIMING=1 ${prof:+COMFY_PROFILE_FWD=/tmp/opencode/prof_${label}.txt} \
        ${prof:+COMFY_PROFILE_FWD_SKIP=1} \
        bash debug/server.sh --deterministic >/tmp/opencode/srv_ffnf_${label}.log 2>&1 &
    wait_ready || { echo "server failed"; return 1; }
    rm -f /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png
    timeout 900 "$PY" -u debug/probe_moss.py "$steps" 1234 2>&1 | tail -1
    local img
    img=$(ls -t /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png 2>/dev/null | head -1)
    cp "$img" "$OUT/${label}.png" && echo "saved $label"
    stop
}

run_mode fused_prof 3 "" 1
run_mode eager_prof 3 "MOSS_NO_SITU_FUSE=1" 1
echo "ffn fusion check done"
