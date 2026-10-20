#!/usr/bin/env bash
# Build libxpuint8.so for Intel Arc (Xe2) with DPC++ (icpx).
#
#   ./build.sh                                  # -> build/libxpuint8.so
#   SYCL_ROOT=/opt/intel/oneapi/compiler/2025.3 ./build.sh
#
# The compiler version is pinned on purpose. The library links against the SYCL
# runtime the *embedding process* uses -- for torch XPU wheels that is
# libsycl.so.8 (oneAPI 2025.x). A 2026.x icpx produces a .so that links
# libsycl.so.9 and fails to load with an undefined
# submit_kernel_direct_with_event_impl: the 2025 runtime exports no
# submit_kernel* symbol at all, so it cannot be shimmed with -L or an rpath.
# Override the check with XPUINT8_SKIP_VERSION_CHECK=1 at your own risk.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$ROOT/build}"
WANT_MAJOR="${WANT_MAJOR:-2025}"

# Resolution order: an explicit SYCL_ROOT, then the usual install layouts
# preferring one whose version matches the pin, then whatever icpx is on PATH
# (the version check below will reject it if it does not match).
matches() { [[ -x "$1" ]] && "$1" --version 2>/dev/null | grep -q "$WANT_MAJOR\."; }

if [[ -z "${ICPX:-}" ]]; then
    if [[ -n "${SYCL_ROOT:-}" ]]; then
        ICPX="$SYCL_ROOT/bin/icpx"
    else
        ICPX=""
        for cand in /opt/intel/oneapi/compiler/*/bin/icpx \
                    "$HOME"/.local/oneapi*/opt/intel/oneapi/compiler/*/bin/icpx \
                    "$HOME"/.local/oneapi-*/bin/icpx \
                    "$HOME"/.local/oneapi-*/*/bin/icpx; do
            if matches "$cand"; then ICPX="$cand"; break; fi
        done
        if [[ -z "$ICPX" ]] && command -v icpx >/dev/null 2>&1; then
            ICPX="$(command -v icpx)"
        fi
    fi
fi

if [[ ! -x "${ICPX:-/nonexistent}" ]]; then
    echo "error: no icpx at ${ICPX:-<unset>}" >&2
    echo "set SYCL_ROOT to a oneAPI compiler install ($WANT_MAJOR.x for this torch)" >&2
    exit 1
fi

if [[ "${XPUINT8_SKIP_VERSION_CHECK:-0}" != "1" ]]; then
    "$ICPX" --version | grep -q "$WANT_MAJOR\." || {
        echo "error: $ICPX is not $WANT_MAJOR.x; the .so would link a SYCL runtime" >&2
        echo "       the loading process does not have (see the note above)" >&2
        "$ICPX" --version | head -2 >&2
        exit 1
    }
fi

# icpx needs its own runtime on the library path to compile; the embedding
# process supplies libsycl at run time.
SYSLIB="$(cd "$(dirname "$ICPX")/.." && pwd)/lib"
export LD_LIBRARY_PATH="$SYSLIB:$SYSLIB/x64:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT"
echo "building $OUT/libxpuint8.so with $ICPX"
"$ICPX" -O3 -fPIC -shared -fsycl -std=c++17 \
    -I"$ROOT/include" \
    "$ROOT/src/xpuint8.cpp" \
    -o "$OUT/libxpuint8.so" \
    ${XPUINT8_EXTRA_FLAGS:-}

echo "done -> $OUT/libxpuint8.so"
echo "check it: PYTHONPATH=$ROOT/python python -m xpu_kernels"
