#!/bin/bash
# Build the macOS release binary for the host arch (arm64: M1+, x86_64:
# x86-64-v3). Accelerate's new LAPACK interface needs macOS 13.3.

set -euo pipefail

case "$(uname -m)" in
    arm64)
        key=darwin-arm64
        arch_flags=-mcpu=apple-m1
        ;;
    x86_64)
        key=darwin-x64
        arch_flags=-march=x86-64-v3
        ;;
    *)
        echo "unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

export MACOSX_DEPLOYMENT_TARGET=13.3

make blas ARCH_FLAGS="$arch_flags"
strip qwen_asr
# strip breaks the ad-hoc signature arm64 requires
codesign --force --sign - qwen_asr

mkdir -p "npm/prebuilds/$key"
cp qwen_asr "npm/prebuilds/$key/qwen_asr"
cp LICENSE npm/LICENSE

echo "minimum macOS: $(otool -l qwen_asr | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}')"
ls -l "npm/prebuilds/$key/qwen_asr"
