#!/bin/bash
# Build the Linux release binary for the host arch (x86_64: x86-64-v3 with
# HASWELL kernels, aarch64: NEOVERSEN1 kernels, ARMv8.2+) with GPU support
# through ggml's Vulkan backend. OpenBLAS is linked statically; libggml and
# the Vulkan and CPU backend modules go next to the binary. Run in
# manylinux_2_28 for glibc >= 2.28, with the Vulkan loader for linking:
#
#   podman run --rm -v "$PWD":/src -w /src quay.io/pypa/manylinux_2_28_x86_64 \
#       sh -c 'dnf install -y vulkan-loader-devel && scripts/build-linux.sh'

set -euo pipefail

OPENBLAS_VERSION=0.3.34
OPENBLAS_SHA256=cd7e129868320cc2d033afa920e31202dfe0b8066a5b66661900ccc0f197dfed

case "$(uname -m)" in
    x86_64)
        key=linux-x64
        arch_flags=-march=x86-64-v3
        openblas_target=(TARGET=HASWELL DYNAMIC_ARCH=0)
        ;;
    aarch64)
        key=linux-arm64
        arch_flags=-march=armv8-a
        openblas_target=(TARGET=NEOVERSEN1 DYNAMIC_ARCH=0)
        ;;
    *)
        echo "unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

OPENBLAS_PREFIX=${OPENBLAS_PREFIX:-$PWD/.openblas/$OPENBLAS_VERSION-$key}
JOBS=$(nproc)

if [ ! -f "$OPENBLAS_PREFIX/lib/libopenblas.a" ]; then
    work=$(mktemp -d)
    curl -fsSL -o "$work/openblas.tar.gz" \
        "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$OPENBLAS_VERSION/OpenBLAS-$OPENBLAS_VERSION.tar.gz"
    echo "$OPENBLAS_SHA256  $work/openblas.tar.gz" | sha256sum -c -
    tar -xzf "$work/openblas.tar.gz" -C "$work"

    # NUM_THREADS caps runtime threads; it defaults to the build machine's cores
    opts=("${openblas_target[@]}" USE_OPENMP=0 USE_THREAD=1 NUM_THREADS=256
          NO_LAPACK=1 NO_LAPACKE=1 NO_FORTRAN=1 NO_SHARED=1)
    make -C "$work/OpenBLAS-$OPENBLAS_VERSION" -j"$JOBS" "${opts[@]}" libs >/dev/null
    make -C "$work/OpenBLAS-$OPENBLAS_VERSION" "${opts[@]}" PREFIX="$OPENBLAS_PREFIX" install >/dev/null
    cp "$work/OpenBLAS-$OPENBLAS_VERSION/LICENSE" "$OPENBLAS_PREFIX/LICENSE"
    rm -rf "$work"
fi

scripts/build-vulkan-tools.sh .vulkan
export VULKAN_SDK=$PWD/.vulkan

make gpu ARCH_FLAGS="$arch_flags" \
    OPENBLAS_CFLAGS="-I$OPENBLAS_PREFIX/include" \
    OPENBLAS_LIBS="$OPENBLAS_PREFIX/lib/libopenblas.a"
strip qwen_asr libggml*.so

dir="npm/platforms/$key"
cp qwen_asr libggml*.so "$dir/"
cp LICENSE "$dir/LICENSE"
cp "$OPENBLAS_PREFIX/LICENSE" "$dir/LICENSE-OpenBLAS"
cp .ggml/src/LICENSE "$dir/LICENSE-ggml"
cp LICENSE npm/LICENSE

echo "glibc requirement: $(objdump -T qwen_asr libggml*.so | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)"
for f in qwen_asr libggml*.so; do echo "$f needs: $(objdump -p "$f" | awk '/NEEDED/{print $2}' | tr '\n' ' ')"; done
ls -l "$dir"
