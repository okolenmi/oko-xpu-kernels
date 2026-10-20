#!/usr/bin/env bash
# Fetch the oneDNN 3.9 (2025.3) runtime from Intel's apt repo, without root.
#
# The build has to be 2025.3: it links the same libsycl.so.8 the venv's
# torch uses, so a sycl::queue can be handed over directly.  The CachyOS
# `extra/onednn` package is built against oneAPI 2026 and would bring its own
# libsycl.so.9 -- a different runtime instance, which breaks the hand-off.
#
# There is no dpkg on this box, but a .deb is an ar archive.
#
#   bash debug/native/fetch_dnnl.sh
set -euo pipefail
DEST="${1:-$HOME/.local/oneapi-dnnl-2025.3}"
BASE=https://apt.repos.intel.com/oneapi/pool/main
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for f in intel-oneapi-runtime-dnnl-2025.3.0-409_amd64.deb \
         intel-oneapi-dnnl-devel-2025.3-2025.3.0-409_amd64.deb; do
    echo "fetching $f"
    curl -sL "$BASE/$f" -o "$TMP/$f"
    (cd "$TMP" && bsdtar -xf "$f" && tar -xf data.tar.xz)
done

mkdir -p "$DEST/lib"
cp -a "$TMP/opt/intel/oneapi/dnnl/2025.3/include" "$DEST/"
cp -a "$TMP/opt/intel/oneapi/redist/lib/libdnnl.so"* "$DEST/lib/"
echo "oneDNN installed in $DEST"
ldd "$DEST/lib/libdnnl.so.3" | grep sycl
