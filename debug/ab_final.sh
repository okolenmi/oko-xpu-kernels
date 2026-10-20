#!/usr/bin/env bash
# Final A/B: eager (pre-situ) vs fully fused (situ + dequant fused), both under
# --deterministic, interleaved.  Also checks that turning off
# fill_uninitialized_memory leaves the image bit-identical and reproducible:
# the 3-step image of every round must equal the one produced before that change.
set -u
cd /home/okolenmi/comfy/xpu-kernels
PY=/home/okolenmi/comfy/venv/bin/python
OUT=/tmp/opencode/final_ab
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
    local label="$1" skipenv="$2" prof="$3"
    stop
    echo "--- $label ---"
    env $skipenv XPU_INT8_TIMING=1 ${prof:+COMFY_PROFILE_FWD=/tmp/opencode/prof_fillfix.txt} \
        ${prof:+COMFY_PROFILE_FWD_SKIP=1} \
        bash debug/server.sh --deterministic >/tmp/opencode/srv_final_${label}.log 2>&1 &
    wait_ready || { echo "server failed"; return 1; }
    rm -f /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png
    timeout 900 "$PY" -u debug/probe_moss.py 3 1234 2>&1 | tail -1
    local img
    img=$(ls -t /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png 2>/dev/null | head -1)
    cp "$img" "$OUT/${label}_3step.png"
    for _ in 1 2; do
        timeout 900 "$PY" -u debug/probe_moss.py 20 777 2>&1 | tail -1
    done
    stop
}

for round in 1 2; do
    run_mode "fused_r${round}" "" "$([[ $round == 1 ]] && echo 1)"
    run_mode "eager_r${round}" "MOSS_NO_SITU_FUSE=1" ""
done
echo "final A/B done"
