#!/usr/bin/env bash
# Build the native probes in debug/native and debug/native/rejected.
#
# These are standalone SYCL programs: each one measures or falsifies a specific
# hypothesis and prints its result. They are the evidence behind the tables in
# docs/performance.md, so they are kept in the repo rather than in /tmp.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICPX="${ICPX:-icpx}"
OUT="${OUT:-$HERE/bin}"

# DPC++ needs its own runtime on the library path (no root, no system install).
if [[ -d /home/okolenmi/.local/oneapi/opt/intel/oneapi/compiler/2026.0/lib ]]; then
    export LD_LIBRARY_PATH="/home/okolenmi/.local/oneapi/opt/intel/oneapi/compiler/2026.0/lib:/home/okolenmi/.local/oneapi/opt/intel/oneapi/umf/1.1/lib:/home/okolenmi/.local/oneapi/opt/intel/oneapi/tbb/2023.0/lib/intel64/gcc4.8:${LD_LIBRARY_PATH:-}"
fi

# icpx is not on PATH here: look where the oneAPI trees of this machine live. 2026.0 first, because the runtime path above is
# the 2026.0 one and a compiler must run against its own runtime.
if ! command -v "$ICPX" >/dev/null 2>&1; then
    for candidate in "$HOME/.local/oneapi/opt/intel/oneapi/compiler/2026.0/bin/icpx" "$HOME/.local/oneapi-2025.3/2025.3/bin/icpx"; do
        if [[ -x "$candidate" ]]; then ICPX="$candidate"; break; fi
    done
fi
command -v "$ICPX" >/dev/null 2>&1 || { echo "icpx not found: set ICPX=/path/to/icpx" >&2; exit 1; }

build_one() {
    local src="$1" name="$2"
    "$ICPX" -O3 -fsycl -std=c++17 "$src" -o "$OUT/$name"
    echo "  built $OUT/$name"
}

mkdir -p "$OUT"

# `./build.sh probe_dequant_weight [...]` builds only the named probes
if [[ $# -gt 0 ]]; then
    for name in "$@"; do build_one "$HERE/$name.cpp" "$name"; done
    exit 0
fi


echo "building native probes -> $OUT"
build_one "$HERE/probe_bandwidth.cpp"           probe_bandwidth
build_one "$HERE/probe_rotate_ablation.cpp"     probe_rotate_ablation
build_one "$HERE/probe_epilogue.cpp"            probe_epilogue
build_one "$HERE/probe_ffn_situ.cpp"            probe_ffn_situ
build_one "$HERE/probe_tanh_exact.cpp"          probe_tanh_exact
build_one "$HERE/probe_gelu_exact.cpp"          probe_gelu_exact
build_one "$HERE/probe_lane_utilisation.cpp"    probe_lane_utilisation
build_one "$HERE/probe_bf16_rotate.cpp"         probe_bf16_rotate
build_one "$HERE/probe_load_store_width.cpp"    probe_load_store_width
build_one "$HERE/probe_cross_lane.cpp"         probe_cross_lane
build_one "$HERE/probe_dequant_weight.cpp"      probe_dequant_weight
# Candidate for an upstream report: sycl::reduce_over_group() kills the host
# process on this runtime. Kept buildable so it can be re-tested after a bump.
build_one "$HERE/reproduce_reduce_over_group.cpp" reproduce_reduce_over_group

# GEMM probes: correctness-verified against a CPU reference, used to show that
# the hand-written GEMM is L2-bandwidth bound rather than DPAS bound.
if [[ -f "$HERE/../../kernels/int8_gemm_v3.cpp" ]]; then
    build_one "$HERE/../../kernels/int8_gemm_v3.cpp" gemm_v3
    build_one "$HERE/../../kernels/int8_gemm_v4.cpp" gemm_v4
fi

# oneDNN probe: needs the oneDNN 3.9 (2025.3) runtime from Intel's apt repo.
# It is not part of the library -- it is the probe that closed the "fuse the
# dequant epilogue into the GEMM" lever -- so it is built only if the
# dependency has been fetched.
DNNL_ROOT="${DNNL_ROOT:-$HOME/.local/oneapi-dnnl-2025.3}"
# libdnnl.so is built against libsycl.so.8, so it only links with the 2025.3
# compiler. With 2026.0 the link fails on ~20 undefined sycl::_V1 symbols --
# which used to abort this script (set -e) before the rejected probes were built.
DNNL_ICPX="${DNNL_ICPX:-$HOME/.local/oneapi-2025.3/2025.3/bin/icpx}"
if [[ -d "$DNNL_ROOT/include" && -d "$DNNL_ROOT/lib" ]]; then
    if [[ ! -x "$DNNL_ICPX" ]]; then
        echo "  skipping probe_dnnl_fused: libdnnl needs the 2025.3 compiler" \
             "(set DNNL_ICPX; not found at $DNNL_ICPX)"
        DNNL_ICPX=""
    fi
fi
if [[ -n "${DNNL_ICPX:-}" && -d "$DNNL_ROOT/include" && -d "$DNNL_ROOT/lib" ]]; then
    "$DNNL_ICPX" -O3 -fsycl -std=c++17 "$HERE/probe_dnnl_fused.cpp" -o "$OUT/probe_dnnl_fused" \
        -I"$DNNL_ROOT/include" -L"$DNNL_ROOT/lib" -ldnnl -Wl,-rpath,"$DNNL_ROOT/lib"
    echo "  built $OUT/probe_dnnl_fused"
elif [[ -z "${DNNL_ICPX:-}" ]]; then
    :  # already explained above
else
    echo "  skipping probe_dnnl_fused (no oneDNN at $DNNL_ROOT; see docs)"
fi

# Negative-result experiments, kept because their failure modes are the reason
# several plausible optimisations are not in the library.
REJ="$HERE/rejected"
for f in "$REJ"/*.cpp; do
    [[ -e "$f" ]] || continue
    build_one "$f" "rej_$(basename "$f" .cpp)"
done

echo "done."