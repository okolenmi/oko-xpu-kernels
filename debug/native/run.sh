#!/usr/bin/env bash
# Run a built native probe with the oneAPI runtime this machine keeps under ~/.local (no root, no system-wide install):
#   ./debug/native/run.sh probe_dequant_weight 1280 5120
# The SYCL runtime loads its Level Zero adapter with dlopen, and that adapter's own dependencies (umf, tbb) are only found
# through LD_LIBRARY_PATH, which an rpath in the probe does not cover: without them the probe reports "No device of requested type".
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
name="$1"; shift

paths=()
for root in "$HOME/.local/oneapi/opt/intel/oneapi" "$HOME/.local/oneapi-2025.3/2025.3"; do
    [[ -d "$root" ]] || continue
    for dir in "$root"/compiler/*/lib "$root"/umf/*/lib "$root"/tbb/*/lib/intel64/gcc4.8 "$root"/lib "$root"/lib/x64; do
        [[ -d "$dir" ]] && paths+=("$dir")
    done
    break  # one toolchain: mixing the 2026.0 and 2025.3 runtimes is how "libsycl.so.9" and ".so.8" end up in one process
done
IFS=: LD_LIBRARY_PATH="${paths[*]}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$HERE/bin/$name" "$@"
