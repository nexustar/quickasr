#!/bin/bash
# Fetch a pinned ggml release (checksum-verified) and build it for `make gpu`:
# static with Metal on macOS; shared, with Vulkan and CPU as loadable backend
# modules, elsewhere. GGML_SRC=dir builds a local checkout instead. The Vulkan
# build needs glslc and the Vulkan headers and loader, found through VULKAN_SDK
# or the system (scripts/build-vulkan-tools.sh builds the first two).
#
# Usage: scripts/build-ggml.sh [outdir]    (default .ggml)
set -euo pipefail

GGML_VERSION=0.27.0
GGML_SHA256=a4bec5021627a204984f8abd34781eb3bf5dee586b6337246c952d2103d7ef45

out=${1:-.ggml}
src=${GGML_SRC:-$out/src}
build=$out/build

if [ -z "${GGML_SRC:-}" ] && [ "$(cat "$src/.version" 2>/dev/null)" != "$GGML_VERSION" ]; then
    rm -rf "$src" "$build"
    mkdir -p "$out"
    tarball="$out/ggml-$GGML_VERSION.tar.gz"
    curl -fsSL -o "$tarball" "https://github.com/ggml-org/ggml/archive/refs/tags/v$GGML_VERSION.tar.gz"
    if command -v sha256sum >/dev/null; then sum=$(sha256sum "$tarball"); else sum=$(shasum -a 256 "$tarball"); fi
    if [ "${sum%% *}" != "$GGML_SHA256" ]; then
        echo "ggml $GGML_VERSION: checksum mismatch" >&2
        exit 1
    fi
    # .pi/ holds only a symlink (agent notes), which MSYS2 tar cannot create
    tar -xzf "$tarball" -C "$out" --exclude="ggml-$GGML_VERSION/.pi"
    rm "$tarball"
    mv "$out/ggml-$GGML_VERSION" "$src"
    echo "$GGML_VERSION" > "$src/.version"
fi

# No OpenMP: libgomp would be one more runtime dependency to ship
opts=(-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF
      -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF)
case "$(uname -s)" in
    Darwin)
        opts+=(-DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON) ;;
    *)
        # Unversioned sonames: shipped as plain files next to the binary
        opts+=(-DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_VULKAN=ON
               -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON) ;;
esac
gen=()
if [ -n "${MSYSTEM:-}" ]; then
    # MinGW runtime (libstdc++, libgcc, winpthread) linked into the DLLs, but
    # not exported: ggml-base re-exported _Unwind_Resume, which then clashed
    # with the copy linked into ggml-cpu and ggml-vulkan
    gen=(-G "MSYS Makefiles")
    rt="-static -Wl,--exclude-libs,libgcc.a:libgcc_eh.a:libstdc++.a:libwinpthread.a"
    opts+=("-DCMAKE_SHARED_LINKER_FLAGS=$rt" "-DCMAKE_MODULE_LINKER_FLAGS=$rt")
fi

cmake -S "$src" -B "$build" ${gen[@]+"${gen[@]}"} "${opts[@]}" >/dev/null
cmake --build "$build" -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" >/dev/null
echo "ggml $GGML_VERSION built in $build"
