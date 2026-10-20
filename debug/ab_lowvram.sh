#!/usr/bin/env bash
# Interleaved paired A/B: the debug server runs with --lowvram.  Does the INT8
# model fit resident, and is it faster without the offload machinery?
#
#   debug/ab_lowvram.sh
set -u
cd /home/okolenmi/comfy/xpu-kernels
PY=/home/okolenmi/comfy/venv/bin/python
ROOT=/home/okolenmi/comfy
OUT=/tmp/opencode/ab_lowvram
mkdir -p "$OUT"

stop() { pkill -f "ComfyUI/[m]ain.py" 2>/dev/null; sleep 6; }
wait_ready() {
    for _ in $(seq 1 120); do
        curl -s --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && return 0
        sleep 2
    done
    return 1
}

base_args=(--enable-manager --enable-triton-backend --gpu-text-encoder
           --supports-int8-compute --keep-int8-native --deterministic)

start_server() {   # $1 = "lowvram" | "resident"
    stop
    local extra=()
    [[ "$1" == "lowvram" ]] && extra=(--lowvram)
    (
        cd "$ROOT"
        export LEVEL_ZERO_V1_SDK_PATH=/home/okolenmi/.local/levelzero
        source venv/bin/activate
        exec python ComfyUI/main.py "${base_args[@]}" "${extra[@]}"
    ) >/tmp/opencode/srv_lv_$1.log 2>&1 &
}

run_mode() {
    local label="$1" cfg="$2"
    echo "--- $label ($cfg) ---"
    start_server "$cfg" || { echo "start failed"; return 1; }
    wait_ready || { echo "server did not come up"; return 1; }
    # first prompt pays the model load / Triton JIT; the second is the warm one
    timeout 900 "$PY" -u debug/probe_moss.py 3 1111 2>&1 | tail -1
    timeout 900 "$PY" -u debug/probe_moss.py 20 4242 2>&1 | tail -1
    grep -aoE "OOM|out of memory|CUDA error|device lost" /tmp/opencode/srv_lv_$cfg.log | tail -2
    stop
}

for round in 1 2; do
    run_mode "lowvram_r${round}"  "lowvram"
    run_mode "resident_r${round}" "resident"
done
echo "lowvram A/B done"
