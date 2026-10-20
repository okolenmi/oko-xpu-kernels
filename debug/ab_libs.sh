#!/usr/bin/env bash
# Interleaved paired A/B of the old and the new libxpuint8.so.
#
# The noise floor on this box is +/-5-10%, so the two libraries have to be
# measured in the same session, interleaved, with the server restarted
# between them (the library is loaded at server start).
#
#   bash debug/ab_libs.sh [rounds] [runs_per_round]
set -u

ROUNDS="${1:-2}"
RUNS="${2:-2}"
ROOT=/home/okolenmi/comfy
SIDE=$ROOT/xpu-kernels
# Pristine copies.  Never copy onto the installed path from the installed
# path: the A/B replaces the installed .so, so the "new" source has to live
# outside it or the second round silently measures the old library twice.
NEW=/tmp/opencode/libxpuint8_new.so
OLD=/tmp/opencode/libxpuint8_old.so
INSTALLED=$ROOT/ComfyUI/comfy/lib/libxpuint8.so
PY=$ROOT/venv/bin/python

cd "$SIDE"

start_server() {
    pkill -f "ComfyUI/main.py" >/dev/null 2>&1
# leave the new library installed
cp "$NEW" "$INSTALLED"
    sleep 4
    XPU_INT8_TIMING=1 bash "$SIDE/debug/server.sh" >/dev/null 2>&1 &
    for _ in $(seq 1 90); do
        curl -s --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && return 0
        sleep 2
    done
    echo "server did not come up" >&2
    return 1
}

run_one() {
    # $1 = label; prints "label seconds"
    local t0 t1
    t0=$(date +%s.%N)
    timeout 900 "$PY" -u "$SIDE/debug/probe_moss.py" 50 $((RANDOM + 1)) >/dev/null 2>&1
    t1=$(date +%s.%N)
    awk -v a="$t0" -v b="$t1" -v l="$1" 'BEGIN{printf "%s %.2f\n", l, b-a}'
}

for r in $(seq 1 "$ROUNDS"); do
    for which in new old; do
        case "$which" in
            new) src=$NEW ;;
            old) src=$OLD ;;
        esac
        cp "$src" "$INSTALLED"
        start_server || exit 1
        # first prompt is cold (model load); discard it
        timeout 900 "$PY" -u "$SIDE/debug/probe_moss.py" 50 $((RANDOM + 1)) >/dev/null 2>&1
        for i in $(seq 1 "$RUNS"); do
            run_one "round$r-$which"
        done
    done
done

pkill -f "ComfyUI/main.py" >/dev/null 2>&1
