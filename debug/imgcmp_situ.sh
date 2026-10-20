#!/usr/bin/env bash
# Same seed, deterministic, fused vs eager: quantify the situ rounding effect.
set -u
cd /home/okolenmi/comfy/xpu-kernels
PY=/home/okolenmi/comfy/venv/bin/python
OUT=/tmp/opencode/situ_imgcmp
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
    local label="$1" skipenv="$2"
    stop
    echo "=== $label ==="
    env $skipenv bash debug/server.sh --deterministic >/tmp/opencode/srv_img_${label}.log 2>&1 &
    wait_ready || { echo "server failed"; return 1; }
    rm -f /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png
    timeout 900 "$PY" -u debug/probe_moss.py 20 1234 2>&1 | tail -2
    local img
    img=$(ls -t /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png 2>/dev/null | head -1)
    cp "$img" "$OUT/${label}.png" && echo "saved $label -> $OUT/${label}.png"
    stop
}

run_mode fused ""
run_mode eager "MOSS_NO_SITU_FUSE=1"
echo "image comparison runs done"
