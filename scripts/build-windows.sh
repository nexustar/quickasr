#!/bin/bash
# Build the Windows release binary (x86-64-v3 with HASWELL kernels). Run in
# an MSYS2 UCRT64 shell with mingw-w64-ucrt-x86_64-gcc, make, curl and perl.
# OpenBLAS and the MinGW runtime are linked statically.

set -euo pipefail

OPENBLAS_VERSION=0.3.34
OPENBLAS_SHA256=cd7e129868320cc2d033afa920e31202dfe0b8066a5b66661900ccc0f197dfed

case "$(uname -m)" in
    x86_64)
        key=win32-x64
        arch_flags=-march=x86-64-v3
        openblas_target=(TARGET=HASWELL DYNAMIC_ARCH=0)
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

make blas ARCH_FLAGS="$arch_flags" \
    OPENBLAS_CFLAGS="-I$OPENBLAS_PREFIX/include" \
    OPENBLAS_LIBS="$OPENBLAS_PREFIX/lib/libopenblas.a -static"
strip qwen_asr.exe

mkdir -p "npm/prebuilds/$key"
cp qwen_asr.exe "npm/prebuilds/$key/qwen_asr.exe"
cp "$OPENBLAS_PREFIX/LICENSE" "npm/prebuilds/$key/LICENSE-OpenBLAS"
cp LICENSE npm/LICENSE

echo "DLL imports: $(objdump -p qwen_asr.exe | awk '/DLL Name/{print $3}' | tr '\n' ' ')"
ls -l "npm/prebuilds/$key/qwen_asr.exe"
