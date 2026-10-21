#!/usr/bin/env bash
# Build distributables.
#
#   ./wheel.sh            # wheel with libxpuint8.so inside -> dist/
#   ./wheel.sh --sdist    # source only, no .so, no toolchain needed -> dist/
#
# The wheel is machine-specific in exactly one way: its libxpuint8.so links
# against the SYCL runtime it was built with (libsycl.so.8, oneAPI 2025.x -- see
# build.sh).  It therefore has to be built where that toolchain is, and installed
# into a process using the same runtime.  That is why this builds the .so first
# rather than shipping a prebuilt one, and why a published release is tagged
# with its toolchain (v0.2.0-oneapi2025.3) instead of a "latest" that silently
# stops working when oneAPI moves.  The *version* stays plain: PyPI must not
# accept local version identifiers such as 0.2.0+oneapi2025.3 (PEP 440).
#
# The .so is staged inside the package because that is where the loader looks
# (python/xpu_kernels/int8/libxpuint8.so, see int8/_lib.py).  It is removed again
# on exit: a staged copy shadows build/libxpuint8.so for every later in-tree run
# and would silently pin a stale kernel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGED="$ROOT/python/xpu_kernels/int8/libxpuint8.so"

cd "$ROOT"

# torch is not needed to build the archive, but setuptools>=64 is, and pip
# resolves the backend in-process with --no-build-isolation so this works on a
# machine with no network.  `python -m build` is not used because this repo has
# a build/ directory, which shadows the module when the cwd is the repo root.
if [[ -z "${PYTHON:-}" ]]; then
    for cand in python3 python; do
        if command -v "$cand" >/dev/null 2>&1 \
           && "$cand" -c 'import setuptools, sys; sys.exit(setuptools.__version__ < "64")' 2>/dev/null; then
            PYTHON="$cand"; break
        fi
    done
fi
if [[ -z "${PYTHON:-}" ]]; then
    echo "error: no python with setuptools>=64 (set PYTHON=/path/to/python)" >&2
    exit 1
fi

if [[ "${1:-}" == "--sdist" ]]; then
    # the same cwd caveat as below applies to this probe
    if (cd /tmp && "$PYTHON" -c 'import build' 2>/dev/null); then
        # from a neutral cwd: this repo's own build/ directory would shadow the
        # module, and `python -m` puts the cwd first on sys.path
        (cd /tmp && "$PYTHON" -m build --sdist --outdir "$ROOT/dist" "$ROOT")
    else
        echo "error: an sdist needs the 'build' module: $PYTHON -m pip install build" >&2
        echo "       (the wheel does not: ./wheel.sh)" >&2
        exit 1
    fi
    echo
    echo "sdist built without libxpuint8.so: the consumer runs ./build.sh or"
    echo "sets XPU_KERNELS_LIB.  int8/_lib.py has the search order."
    exit 0
fi

if [[ -e "$STAGED" ]]; then
    echo "error: $STAGED already exists -- remove it first" >&2
    echo "       (a stale staged .so shadows build/libxpuint8.so)" >&2
    exit 1
fi

"$ROOT/build.sh"

cleanup() { rm -f "$STAGED"; }
trap cleanup EXIT

cp "$ROOT/build/libxpuint8.so" "$STAGED"
"$PYTHON" -m pip wheel . --no-deps --no-build-isolation -w "$ROOT/dist"

echo
echo "built:"
ls -1 "$ROOT/dist"/*.whl | tail -1
echo
echo "it carries the libxpuint8.so that build.sh just produced; after"
echo "installing, check it with:"
echo "  python -c 'import xpu_kernels; print(xpu_kernels.available())'"
