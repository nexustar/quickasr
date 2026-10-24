#!/bin/bash
# Build the macOS release binary for the host arch (arm64: M1+, with GPU support
# through ggml's Metal backend; x86_64: x86-64-v3). Accelerate's new LAPACK
# interface needs macOS 13.3.

set -euo pipefail

case "$(uname -m)" in
    arm64)
        key=darwin-arm64
        arch_flags=-mcpu=apple-m1
        target=gpu
        ;;
    x86_64)
        key=darwin-x64
        arch_flags=-march=x86-64-v3
        target=blas
        ;;
    *)
        echo "unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

export MACOSX_DEPLOYMENT_TARGET=13.3

make "$target" ARCH_FLAGS="$arch_flags"
strip qwen_asr
# strip breaks the ad-hoc signature arm64 requires
codesign --force --sign - qwen_asr

dir="npm/platforms/$key"
cp qwen_asr "$dir/qwen_asr"
cp LICENSE npm/LICENSE
cp LICENSE "$dir/LICENSE"
if [ "$target" = gpu ]; then cp .ggml/src/LICENSE "$dir/LICENSE-ggml"; fi

echo "minimum macOS: $(otool -l qwen_asr | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}')"
ls -l "$dir"
