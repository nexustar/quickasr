# Building, Releasing, Testing

## Building

```bash
make blas       # BLAS acceleration (Accelerate on macOS, OpenBLAS on Linux and Windows)
make gpu        # blas plus GPU support (Metal on macOS, Vulkan elsewhere)
make test       # Run regression checks (requires built binary + model files)
make test-stream-cache  # Check stream cache on/off equivalence
make clean      # Clean build artifacts
```

For Linux, install OpenBLAS first:
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora
sudo dnf install openblas-devel
```

For Windows, build in an [MSYS2](https://www.msys2.org/) UCRT64 shell
(MSVC is not supported):
```bash
pacman -S make mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-openblas
make blas       # produces quickasr.exe
```

`make gpu` needs CMake. It runs `scripts/build-ggml.sh`, which downloads a pinned ggml
release (checksum-verified) into `.ggml/` and builds it; `GGML_SRC=/path/to/ggml` builds a
local checkout instead. On macOS ggml is linked statically and its Metal shaders are
embedded as source. On Linux and Windows it also needs `glslc` and the Vulkan and
SPIR-V headers and loader: the Vulkan SDK (found through `VULKAN_SDK`), distribution
packages (e.g. `glslc libvulkan-dev spirv-headers` on Debian/Ubuntu;
`mingw-w64-ucrt-x86_64-{cmake,vulkan-devel,shaderc,spirv-headers}` in MSYS2), or
`scripts/build-vulkan-tools.sh`, which builds `glslc` and installs the headers into
`.vulkan/` (then `VULKAN_SDK=$PWD/.vulkan make gpu`; the loader still comes from the
system). libggml and the Vulkan and CPU backend modules are copied next to the binary and
loaded at run time, so a machine without a Vulkan driver still runs on the CPU.

## Prebuilt binary (npm)

Linux x86-64 with AVX2/FMA (Haswell or newer) or Linux arm64 with ARMv8.2
(Cortex-A55/A76 or newer, Asahi Linux), both with glibc >= 2.28, macOS
13.3+ (Apple Silicon or Intel), or Windows x86-64 with AVX2/FMA:
```bash
npm install -g quickasr
quickasr -d qwen3-asr-0.6b -i samples/jfk.wav
```

Each platform's binary is in its own package (`@quickasr/darwin-arm64`,
`@quickasr/linux-x64`, ...), an optional dependency of `quickasr`, so npm
downloads only the one for the machine. All but the Intel Mac binary are
`make gpu` builds and support `--device gpu` (Metal on Apple Silicon, Vulkan on
Linux and Windows).

Release binaries are built by `scripts/build-linux.sh` (inside
`manylinux_2_28`; x86-64-v3 or ARMv8.2, OpenBLAS linked statically, `make gpu`
with Vulkan) and
`scripts/build-darwin.sh` (Apple M1 with `make gpu`, or x86-64-v3; Accelerate) and
`scripts/build-windows.sh` (MSYS2 UCRT64; x86-64-v3, OpenBLAS and MinGW
runtime linked statically, `make gpu` with Vulkan). The Makefile
accepts the same overrides for custom builds:
```bash
make blas ARCH_FLAGS=-march=x86-64-v3 OPENBLAS_CFLAGS=-I/opt/openblas/include \
    OPENBLAS_LIBS=/opt/openblas/lib/libopenblas.a
```

Pushing a `v*` tag runs `.github/workflows/release.yml`, which builds all
platforms into `npm/platforms/<platform>/`, runs a transcription smoke test on
each, and publishes the platform packages and then `quickasr` to npm.


## Regression Tests

The repository includes `asr_regression.py` (repo root), a stdlib-only regression harness.
It scans `samples/**/*.wav` recursively:
- quality regression runs on WAV files that already have a sibling `.txt` reference
- focused checks (segmented conditioning, streaming, stream-cache equivalence) use fixed targets

Generate references (using the larger model and full-context decode):

```bash
./asr_regression.py --generate-missing \
    --binary ./quickasr --model-dir qwen3-asr-1.7b
```

Run regression checks:

```bash
./asr_regression.py \
    --binary ./quickasr --model-dir qwen3-asr-1.7b
```

Or run the default regression profile via make:

```bash
make test
```

Streaming cache equivalence regression (cache on vs off):

```bash
./asr_regression.py --stream-cache-check-only \
    --binary ./quickasr --stream-cache-model-dir qwen3-asr-0.6b
```

Or via make:

```bash
make test-stream-cache
```

Output format:
- Each sample starts with a progress line: `START i/N`.
- Live model text is shown while that sample is transcribed.
- The sample closes with `DONE: OK i/N` (only `OK` is green) or `DONE: FAIL i/N` (status in red).

Example:

```text
[START 1/22] jfk.wav ...
And so, my fellow Americans, ask not what your country can do for you...
[DONE: OK 1/22] jfk.wav | exact 0/108 (0.000) | norm 0/104 (0.000) | 2.6s
```

Per sample, the tool reports two distances:
- `exact`: character-level Levenshtein distance on raw text.
- `norm`: character-level Levenshtein distance after normalization
  (punctuation -> spaces, lowercase, whitespace collapsed).


## C API

The library exposes a simple callback-based API:

**Offline transcription:**

```c
#include "qwen_asr.h"

qwen_ctx_t *ctx = qwen_load("qwen3-asr-0.6b");

/* Optional: set a callback to receive tokens as they are decoded */
qwen_set_token_callback(ctx, my_token_handler, userdata);

/* Optional: force language or set system prompt */
qwen_set_force_language(ctx, "Italian");
qwen_set_prompt(ctx, "Preserve spelling: PostgreSQL, Redis");

/* Transcribe — returns malloc'd string */
char *text = qwen_transcribe(ctx, "audio.wav");
printf("%s\n", text);
free(text);

/* Or from raw samples */
char *text2 = qwen_transcribe_audio(ctx, samples, n_samples);

qwen_free(ctx);
```

**Streaming transcription:**

```c
/* Load audio first */
float *samples = qwen_load_wav("long_audio.wav", &n_samples);

/* Stream-transcribe with prefix rollback */
qwen_set_token_callback(ctx, my_token_handler, userdata);
char *text = qwen_transcribe_stream(ctx, samples, n_samples);
free(text);
free(samples);
```

Tokens are emitted via the callback as they become "fixed" (past the rollback window). The returned string contains the full concatenated text.
