# qwen-asr

Prebuilt CLI for [qwen-asr](https://github.com/nexustar/qwen-asr), a pure C
inference engine for Qwen3-ASR (0.6B / 1.7B) speech-to-text.

## Install

```bash
npm install -g qwen-asr
```

Supported platforms:

- Linux x86-64 with AVX2/FMA (x86-64-v3, Haswell or newer), glibc >= 2.28.
- Linux arm64 with ARMv8.2 (Cortex-A55/A76 or newer, e.g. Raspberry Pi 5,
  Neoverse servers, Apple Silicon under Asahi Linux), glibc >= 2.28.
  Raspberry Pi 3/4 are not supported.
- On Linux, OpenBLAS is statically linked; the only other runtime dependency
  is libstdc++ (for the bundled ggml libraries). `--device gpu` runs on the
  GPU through Vulkan when the system has a Vulkan driver (`libvulkan.so.1`), and
  on the CPU otherwise.
- macOS 13.3+ on Apple Silicon or Intel (every Intel Mac that runs macOS 13
  has the AVX2/FMA the binary needs), using the system Accelerate framework.
  On Apple Silicon, `--device gpu` runs on the GPU (Metal); the first run
  compiles the GPU shaders, which takes 10-20 s.
- Windows x86-64 with AVX2/FMA (x86-64-v3, Haswell or newer). OpenBLAS and
  the MinGW runtime are statically linked; no DLLs to install.
  `--device gpu` runs on the GPU through Vulkan (included with current GPU
  drivers).
- With several GPUs, `qwen-asr --list-devices` lists them and
  `--device <name>` (e.g. `Vulkan1`) picks one instead of the default.

The binary for each platform is in its own package (`qwen-asr-darwin-arm64`,
`qwen-asr-linux-x64`, ...), an optional dependency of `qwen-asr`; npm
installs only the one for the machine.

## Usage

Model weights are not included. Download them from Hugging Face
(`Qwen/Qwen3-ASR-0.6B` or `Qwen/Qwen3-ASR-1.7B`), e.g. with the repository's
`download_model.sh`, then:

```bash
qwen-asr -d qwen3-asr-0.6b -i audio.wav
cat audio.wav | qwen-asr -d qwen3-asr-0.6b --stdin
qwen-asr -h
```

See the [repository README](https://github.com/nexustar/qwen-asr#readme) for
all options (segmented mode, streaming, language forcing, ...).

## From Node

Programs that run the binary themselves (rather than through the `qwen-asr`
launcher) can get its path:

```js
const { binaryPath } = require('qwen-asr');

const bin = binaryPath();  // e.g. .../node_modules/qwen-asr-darwin-arm64/qwen_asr
```

`binaryPath()` picks the arm64 binary for x64 Node under Rosetta, and throws
an Error with `code` `QWEN_ASR_UNSUPPORTED_PLATFORM` (no binary for this
OS / CPU architecture) or `QWEN_ASR_UNSUPPORTED_CPU` (CPU below the build
baseline) when none can run here, for example when optional dependencies
were not installed (`--omit=optional`). Add the package to
`optionalDependencies` so installs on other platforms still succeed.

## License

MIT. The Linux and Windows binaries statically link OpenBLAS (BSD-3-Clause,
see `LICENSE-OpenBLAS` in their packages); the Apple Silicon binary statically
links ggml (MIT, see `LICENSE-ggml`).
