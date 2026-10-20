#!/usr/bin/env bash
# Measure wall time against step count, to separate the per-run fixed cost
# (model (re)load, CLIP encode, VAE decode) from the marginal cost of one
# sampling step.
#
#   debug/probe_scaling.sh 2 4 8 16 32
#
# probe_moss.py reports wall/steps, which is only meaningful once the fixed
# cost is amortised -- every A/B must therefore use the same step count in
# both arms, and short runs exaggerate the fixed part.
set -u
PY=/home/okolenmi/comfy/venv/bin/python
cd "$(dirname "${BASH_SOURCE[0]}")"

for s in "$@"; do
    w=$("$PY" probe_moss.py "$s" 2>&1 | sed -n 's/.*wall=\([0-9.]*\)s.*/\1/p')
    awk -v s="$s" -v w="$w" \
        'BEGIN { printf "%3s steps  wall=%8.2fs  wall/step=%6.0f ms\n", s, w, w / s * 1000 }'
done
