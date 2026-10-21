#!/usr/bin/env bash
# Check the *published* wheel, not the tree.
#
# A release is only proven when the artifact a consumer downloads has been
# run.  This script takes the wheel from PyPI, points ComfyUI's loader at the
# `.so` inside it (XPU_KERNELS_LIB is the loader's first choice), and re-runs
# the PNG gate: the same seed, the same three steps, the recorded md5.
#
# The Python side of the published wheel is the tagged tree, so the binary is
# what needs checking -- it was compiled on a different image (ubuntu-22.04)
# than the local build, and a different compiler build can move a last bit.
#
#   bash debug/verify_published_wheel.sh [version]
#
# It restarts the ComfyUI server twice (once with the published .so, once
# without it, to leave the machine as it found it).
set -uo pipefail

ROOT=/home/okolenmi/comfy/xpu-kernels
VERSION="${1:-0.3.0}"
EXPECTED=a2aacfda45409493a490b0402b61f850
VENV=/home/okolenmi/comfy/venv/bin/python
WORK=$(mktemp -d /tmp/opencode/pubwheel.XXXXXX)

cleanup() {
    # always leave a server running, the way we found it
    pkill -f "ComfyUI/[m]ain.py" 2>/dev/null
    sleep 3
    (cd "$ROOT" && setsid nohup bash debug/server.sh --deterministic >/dev/null 2>&1 &)
}
trap cleanup EXIT

echo "== downloading xpu-kernels==$VERSION from PyPI"
pip download "xpu-kernels==$VERSION" --no-deps --only-binary :all: -d "$WORK" -q || exit 1
ls -la "$WORK"

echo
echo "== unpacking the wheel"
unzip -o -q "$WORK"/*.whl -d "$WORK/x" || exit 1
SO="$WORK/x/xpu_kernels/int8/libxpuint8.so"
test -f "$SO" || { echo "the wheel carries no .so"; exit 1; }
md5sum "$SO"
readelf -d "$SO" | grep NEEDED | sed 's/.*\[/  needs /;s/\]//'

echo
echo "== the wheel imports and its gates verify"
(cd "$WORK" && PYTHONPATH="$WORK/x" "$VENV" - <<'PY'
import xpu_kernels
print("version:", xpu_kernels.__version__)
import xpu_kernels.triton as kt
for line in kt.status():
    print("  ", line)
PY
) || exit 1

echo
echo "== the PNG gate, with the published .so loaded"
pkill -f "ComfyUI/[m]ain.py" 2>/dev/null
sleep 3
(cd "$ROOT" && XPU_KERNELS_LIB="$SO" setsid nohup bash debug/server.sh --deterministic >/dev/null 2>&1 &)
for i in $(seq 1 40); do
    curl -s -m 3 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && break
    sleep 3
done
pid=$(pgrep -f "ComfyUI/[m]ain.py" | head -1)
echo "server pid $pid, loaded:"
grep -o "/[^ ]*libxpuint8.so" "/proc/$pid/maps" | sort -u

"$VENV" -u debug/probe_moss.py 3 1234 | tail -2
newest=$(ls -t /home/okolenmi/comfy/ComfyUI/output/moss-image_*.png | head -1)
echo "$newest"
md5sum "$newest"
got=$(md5sum "$newest" | cut -d' ' -f1)
if [ "$got" = "$EXPECTED" ]; then
    echo
    echo "PASS: the published wheel renders the recorded image"
else
    echo
    echo "FAIL: the published wheel renders $got, expected $EXPECTED"
    exit 1
fi
