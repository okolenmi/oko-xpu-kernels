#!/usr/bin/env bash
# End-to-end check of the qkv dequant -> rms_rope fusion by marginal step time.
#
# (wall(50 steps) - wall(10 steps)) / 40 per arm, two interleaved rounds, fresh
# server per render and a distinct seed each time (the node cache is keyed by
# prompt+seed, and a cached run returns in 0.2 s).
#
# ONLY RUN THIS WITH THE DESKTOP IDLE.  Any GPU activity (scrolling, video,
# window animation, even a compositor repaint) costs up to +20% per step and
# attacks one arm at random: a run in interactive use gave reference rounds of
# 1128 and 1221 ms/step while the fused arm sat at 1149/1107 -- the noise was
# several times the ~27 ms effect.  The device profile (COMFY_PROFILE_FWD) is
# unaffected and is the metric to use while the machine is in use.
#
#   debug/ab_qkv_slope.sh          # ~8 renders, ~10 min
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/home/okolenmi/comfy/venv/bin/python
stop() { pkill -f "ComfyUI/[m]ain.py" 2>/dev/null; sleep 5; }
wait_ready() { for _ in $(seq 1 150); do curl -s --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }
one() {  # gate steps seed -> wall seconds
    local gate="$1" steps="$2" seed="$3"
    stop
    MOSS_NO_QKV_ROPE=$gate bash debug/server.sh --deterministic >/dev/null 2>&1 &
    wait_ready || { echo "0"; return 1; }
    timeout 1200 "$PY" -u debug/probe_moss.py "$steps" "$seed" 2>&1 | sed -n 's/.*wall=\([0-9.]*\)s.*/\1/p'
}
seed=${SEED0:-4000}
for r in ${ROUNDS:-1 2}; do
    for arm in fused ref; do
        gate=$([ "$arm" = fused ] && echo 0 || echo 1)
        seed=$((seed+1)); w10=$(one "$gate" 10 "$seed")
        seed=$((seed+1)); w50=$(one "$gate" 50 "$seed")
        "$PY" -c "print(f'{\"$arm\":<5} r$r  10s={$w10:6.2f}  50s={$w50:6.2f}  slope={(($w50)-($w10))/40*1000:7.1f} ms/step')"
    done
done
stop
