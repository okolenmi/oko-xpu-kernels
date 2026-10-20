#!/usr/bin/env bash
# Paired PNG gate for the qkv dequant -> rms_rope fusion, same .so, same seeds.
set -u
cd /home/okolenmi/comfy/xpu-kernels
PY=/home/okolenmi/comfy/venv/bin/python
stop() { pkill -f "ComfyUI/[m]ain.py" 2>/dev/null; sleep 6; }
ready() { for _ in $(seq 1 150); do curl -s --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }
run() {  # $1 = label, $2 = MOSS_NO_QKV_ROPE
    stop
    rm -f /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png
    MOSS_NO_QKV_ROPE=$2 bash debug/server.sh --deterministic >/dev/null 2>&1 &
    ready || { echo "$1: server failed"; return 1; }
    local t0=$(date +%s)
    timeout 900 "$PY" -u debug/probe_moss.py ${STEPS:-3} $3 >/dev/null 2>&1
    local img; img=$(ls -t /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png 2>/dev/null | head -1)
    local used; used=$(grep -c "qkv dequant fused" debug/server.log)
    echo "$1  md5=$(md5sum "$img" | cut -d' ' -f1)  wall=$(( $(date +%s) - t0 ))s  fused_log=$used"
}
run "fused  s=1234" 0 1234
run "ref    s=1234" 1 1234
run "fused  s=2345" 0 2345
run "ref    s=2345" 1 2345
stop
