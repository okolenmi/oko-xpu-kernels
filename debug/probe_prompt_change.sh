#!/usr/bin/env bash
# Is the per-prompt (changed text) cost paid on every prompt, or once?
#
#   debug/probe_prompt_change.sh [steps]
#
# The text encoder only runs when the prompt differs from the previous run:
# identical text is served from ComfyUI's node cache, so the encoder is never
# invoked and its weights are never pulled back into VRAM.  A workflow that
# changes the prompt on every render therefore pays this cost on every render,
# which is what the run-i/r1..r2 sequence below distinguishes from a one-time
# start-up cost.
set -u
STEPS="${1:-2}"
PY=/home/okolenmi/comfy/venv/bin/python
cd "$(dirname "${BASH_SOURCE[0]}")"

POS="amateur photo, shinomiya kaguya,"
NEG=(
    "beautiful, professional photo, mature, old, asian girl,"   # 22 tokens
    "beautiful, professional photo, mature, old, girl,"         # 21 tokens
    "beautiful, professional photo, mature, old lady,"          # 21 tokens
)

wall() {  # $1=negative prompt
    MOSS_POS="$POS" MOSS_NEG="$1" "$PY" probe_moss.py "$STEPS" 2>&1 \
        | sed -n 's/.*wall=\([0-9.]*\)s.*/\1/p'
}

# Throwaway: the first run after a server start pays the model load.
wall "${NEG[0]}" > /dev/null

for r in 1 2; do
    for i in "${!NEG[@]}"; do
        printf 'prompt %s run %s: %ss\n' "$i" "$r" "$(wall "${NEG[$i]}")"
    done
done
