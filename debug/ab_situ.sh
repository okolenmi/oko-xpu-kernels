#!/usr/bin/env bash
# Interleaved A/B of the fused FFN activation (situ inside the INT8 quantizer).
#
# Restarts the server per round because the switch is read at import time, and
# alternates fused/unfused so a drift in desktop GPU load cannot masquerade as
# an effect.  Prints two 20-step wall clocks per round.
set -u
cd /home/okolenmi/comfy/xpu-kernels

wait_ready() {
    for _ in $(seq 1 120); do
        curl -s --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && return 0
        sleep 2
    done
    echo "server did not come up" >&2
    return 1
}

stop_server() {
    pkill -f "ComfyUI/main.py" 2>/dev/null
    sleep 6
}

run_round() {
    local label="$1"; shift
    stop_server
    echo "--- $label ---"
    env "$@" XPU_INT8_TIMING=1 bash debug/server.sh >"/tmp/opencode/srv_${label}.log" 2>&1 &
    wait_ready || return 1
    for _ in 1 2; do
        timeout 900 /home/okolenmi/comfy/venv/bin/python -u debug/probe_moss.py 20 $((RANDOM + 1)) 2>&1 | tail -1
    done
    stop_server
}

for round in 1 2; do
    run_round "fused_r${round}"
    run_round "eager_r${round}" MOSS_NO_SITU_FUSE=1
done
echo "done"
