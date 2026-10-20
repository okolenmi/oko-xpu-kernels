#!/usr/bin/env bash
# Start the ComfyUI server used by the debug/ probes.
#
#   debug/server.sh              # normal flags
#   debug/server.sh --previews   # also enable latent2rgb previews
#
# Writes the pid to debug/server.pid and the log to debug/server.log.
set -u

ROOT=/home/okolenmi/comfy
PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LEVEL_ZERO_V1_SDK_PATH=/home/okolenmi/.local/levelzero
export XPU_INT8_TIMING="${XPU_INT8_TIMING:-0}"

ARGS=(--enable-manager --lowvram --enable-triton-backend
      --gpu-text-encoder --supports-int8-compute --keep-int8-native)
CPU_TEXT_ENCODER=0
for a in "$@"; do
    [[ "$a" == "--previews" ]] && ARGS+=(--preview-method latent2rgb)
    [[ "$a" == "--force-non-blocking" ]] && ARGS+=(--force-non-blocking)
    [[ "$a" == "--deterministic" ]] && ARGS+=(--deterministic)
    [[ "$a" == "--no-triton" ]] && ARGS+=(--disable-triton-backend)
    # --gpu-text-encoder makes the text encoder evict the unet from VRAM, so a
    # render whose prompt changed pays a unet reload; --cpu-text-encoder drops
    # it (see docs/xpu_int8_perf_findings.md).
    [[ "$a" == "--cpu-text-encoder" ]] && CPU_TEXT_ENCODER=1
done
if [[ "$CPU_TEXT_ENCODER" == "1" ]]; then
    KEPT=()
    for a in "${ARGS[@]}"; do
        [[ "$a" == "--gpu-text-encoder" ]] || KEPT+=("$a")
    done
    ARGS=("${KEPT[@]}")
fi

cd "$ROOT"
source venv/bin/activate
exec python ComfyUI/main.py "${ARGS[@]}" > "$PROBE_DIR/server.log" 2>&1
