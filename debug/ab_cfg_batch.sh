#!/usr/bin/env bash
# Interleaved paired A/B: CFG pair as two forwards vs one batched forward.
#
#   debug/ab_cfg_batch.sh [short_steps] [long_steps] [rounds]
#
# ComfyUI batches the two CFG branches into a single forward (batch=2) iff
# can_concat_cond() accepts the pair, and for MossImage that needs both
# c_crossattn to carry the same token count.  The two arms therefore use two
# different negative prompts tokenizing to 22 and 21 tokens against the same
# 21-token positive prompt; both arms do the same token-forwards
# (3973+3974 vs 2x3973), so the difference is purely the batching decision.
#
# probe_moss.py reports wall/steps, which is dominated by the per-run fixed
# cost (CLIP encode, VAE decode, graph setup) when steps are few.  This script
# instead reports the sampling-only slope (wall_long - wall_short)/dsteps,
# which cancels that fixed cost, and warms the prompt cache before each timed
# pair: a run whose prompt changed since the previous run pays a multi-second
# text-encoder reload (see docs/xpu_int8_perf_findings.md), which would land in
# whichever of the two runs comes first and corrupt the slope.
#
# Measured result: two-forward 1117 ms/step vs batched 1107 ms/step (-0.9%).
set -u
SHORT="${1:-5}"
LONG="${2:-25}"
ROUNDS="${3:-2}"
WARM="${4:-3}"
PY=/home/okolenmi/comfy/venv/bin/python
cd "$(dirname "${BASH_SOURCE[0]}")"

POS="amateur photo, shinomiya kaguya,"                              # 21 tokens
NEG_TWO="beautiful, professional photo, mature, old, asian girl,"   # 22 -> no batch
NEG_BAT="beautiful, professional photo, mature, old, girl,"          # 21 -> batch

wall() {  # $1=negative prompt  $2=steps
    MOSS_POS="$POS" MOSS_NEG="$1" "$PY" probe_moss.py "$2" 2>&1 \
        | sed -n 's/.*wall=\([0-9.]*\)s.*/\1/p'
}

slope() {  # $1=negative prompt -> ms/step of the sampling loop alone
    local w_short w_long
    # Warm the prompt cache first: a run whose prompt differs from the previous
    # run pays a multi-second CLIP reload, which would land in whichever of the
    # two timings runs first and corrupt the slope.
    wall "$1" "$WARM" >/dev/null
    w_short=$(wall "$1" "$SHORT")
    w_long=$(wall "$1" "$LONG")
    printf '  [%s steps=%ss %s steps=%ss]\n' "$SHORT" "$w_short" "$LONG" "$w_long" >&2
    awk -v s="$w_short" -v l="$w_long" -v n="$((LONG - SHORT))" \
        'BEGIN { printf "%.0f", (l - s) / n * 1000 }'
}

sum_two=0
sum_bat=0
for r in $(seq 1 "$ROUNDS"); do
    a=$(slope "$NEG_TWO")
    b=$(slope "$NEG_BAT")
    sum_two=$((sum_two + a))
    sum_bat=$((sum_bat + b))
    printf 'round %d: two-forward=%4s ms/step   batched=%4s ms/step\n' "$r" "$a" "$b"
done

avg_two=$((sum_two / ROUNDS))
avg_bat=$((sum_bat / ROUNDS))
awk -v a="$avg_two" -v b="$avg_bat" 'BEGIN {
    printf "mean: two-forward=%d ms/step  batched=%d ms/step  delta=%+.1f%%\n",
           a, b, (b - a) / a * 100
}'
