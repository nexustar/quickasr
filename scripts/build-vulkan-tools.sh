#!/bin/bash
# Build glslc (shaderc) and install Vulkan-Headers and SPIRV-Headers for
# ggml's Vulkan backend where the system lacks them (manylinux_2_28). Sources
# are pinned and checksum-verified; the shaderc dependencies are the revisions
# in its DEPS.
# Point `make gpu` at the result with VULKAN_SDK=<outdir>; the Vulkan loader
# to link against still comes from the system (vulkan-loader-devel).
#
# Usage: scripts/build-vulkan-tools.sh [outdir]    (default .vulkan)
set -euo pipefail

SHADERC_VERSION=2026.4
VULKAN_HEADERS_VERSION=1.4.363.0
# name|url|sha256|directory under shaderc/ (empty: not part of shaderc)
SOURCES=(
    "shaderc|https://github.com/google/shaderc/archive/refs/tags/v$SHADERC_VERSION.tar.gz|f06ce5bcca94e5df7f34e115743597d0ad2e13c5fe9213c67dc8c76031241947|"
    "glslang|https://github.com/KhronosGroup/glslang/archive/e1b562a8bed273a02f30b59b66a5d499793cede5.tar.gz|907174a24713c6202c146f164bf81783f1fbc79c8cb821a30f18f159eb980312|third_party/glslang"
    "spirv-headers|https://github.com/KhronosGroup/SPIRV-Headers/archive/04fd3caa1e8267e4d95c806cad901181728e1006.tar.gz|392f4801409aad9c4f1b77745f179952fd6264e4c8bd0fc1bc45dfa6807bf6d0|third_party/spirv-headers"
    "spirv-tools|https://github.com/KhronosGroup/SPIRV-Tools/archive/ef96ed763b43b59b33b31b362f09a02b729fa1c9.tar.gz|82c62146083fd558735a3171cf97cfc47903ca7d368482e87f94bd44883c0f00|third_party/spirv-tools"
    "vulkan-headers|https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-$VULKAN_HEADERS_VERSION.tar.gz|4a078be12bef21cfebc09d878b77a63cff9d68f899254a0b00d0e37ef73e7f7e|"
)
stamp="shaderc-$SHADERC_VERSION vulkan-headers-$VULKAN_HEADERS_VERSION"

out=${1:-.vulkan}
if [ "$(cat "$out/.version" 2>/dev/null)" = "$stamp" ]; then exit 0; fi

rm -rf "$out"
work="$out/work"
mkdir -p "$work" "$out/bin"
for entry in "${SOURCES[@]}"; do
    IFS='|' read -r name url sha sub <<< "$entry"
    curl -fsSL -o "$work/$name.tar.gz" "$url"
    if command -v sha256sum >/dev/null; then sum=$(sha256sum "$work/$name.tar.gz"); else sum=$(shasum -a 256 "$work/$name.tar.gz"); fi
    if [ "${sum%% *}" != "$sha" ]; then
        echo "$name: checksum mismatch" >&2
        exit 1
    fi
    mkdir "$work/$name"
    tar -xzf "$work/$name.tar.gz" -C "$work/$name" --strip-components=1
    rm "$work/$name.tar.gz"
    if [ -n "$sub" ]; then mv "$work/$name" "$work/shaderc/$sub"; fi
done

cmake -S "$work/shaderc" -B "$work/build" -DCMAKE_BUILD_TYPE=Release \
    -DSHADERC_SKIP_TESTS=ON -DSHADERC_SKIP_EXAMPLES=ON -DSHADERC_SKIP_COPYRIGHT_CHECK=ON \
    -DSPIRV_SKIP_TESTS=ON -DSPIRV_SKIP_EXECUTABLES=ON \
    -DENABLE_GLSLANG_BINARIES=OFF -DGLSLANG_TESTS=OFF >/dev/null
cmake --build "$work/build" --target glslc_exe -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" >/dev/null
cp "$work/build/glslc/glslc" "$out/bin/glslc"
cp -R "$work/vulkan-headers/include" "$out/include"
# ggml-vulkan also finds SPIRV-Headers as a CMake package
cmake -S "$work/shaderc/third_party/spirv-headers" -B "$work/spirv-headers" \
    -DCMAKE_INSTALL_PREFIX="$out" -DSPIRV_HEADERS_ENABLE_TESTS=OFF >/dev/null
cmake --install "$work/spirv-headers" >/dev/null
rm -rf "$work"
echo "$stamp" > "$out/.version"
echo "glslc $SHADERC_VERSION and Vulkan headers $VULKAN_HEADERS_VERSION in $out"
