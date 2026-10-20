#!/usr/bin/env bash
# Profile one warm model forward in fused and eager situ modes.
# The activation chain should disappear from the device table in fused mode.
set -u
cd /home/okolenmi/comfy/xpu-kernels
PY=/home/okolenmi/comfy/venv/bin/python

stop() {
    pkill -f "ComfyUI/[m]ain.py" 2>/dev/null
    sleep 6
}

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
    env $skipenv XPU_INT8_TIMING=1 \
        COMFY_PROFILE_FWD="/tmp/opencode/prof_${label}.txt" \
        COMFY_PROFILE_FWD_SKIP=1 \
        bash debug/server.sh >/tmp/opencode/srv_prof_${label}.log 2>&1 &
    wait_ready || { echo "server failed"; return 1; }
    timeout 600 "$PY" -u debug/probe_moss.py 4 1234 2>&1 | tail -2
    stop
}

run_mode fused ""
run_mode eager "MOSS_NO_SITU_FUSE=1"
echo "profiles done"
