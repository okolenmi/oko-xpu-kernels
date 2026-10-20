#!/usr/bin/env bash
# Short-horizon check: does the fused path agree with eager before chaos
# amplifies rounding?  Also a same-mode repeat to prove --deterministic holds.
set -u
cd /home/okolenmi/comfy/xpu-kernels
PY=/home/okolenmi/comfy/venv/bin/python
OUT=/tmp/opencode/situ_short
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
    local label="$1" steps="$2" skipenv="$3"
    stop
    echo "=== $label ($steps steps) ==="
    env $skipenv bash debug/server.sh --deterministic >/tmp/opencode/srv_short_${label}.log 2>&1 &
    wait_ready || { echo "server failed"; return 1; }
    rm -f /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png
    timeout 600 "$PY" -u debug/probe_moss.py "$steps" 1234 2>&1 | tail -1
    local img
    img=$(ls -t /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png 2>/dev/null | head -1)
    cp "$img" "$OUT/${label}.png" && echo "saved $label"
    stop
}

run_mode fused_1 1 ""
run_mode eager_1 1 "MOSS_NO_SITU_FUSE=1"
run_mode eager_1_rep 1 "MOSS_NO_SITU_FUSE=1"
run_mode fused_1_rep 1 ""
echo "short runs done"
