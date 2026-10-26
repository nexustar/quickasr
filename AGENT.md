This file is the practical guide for agents working on this repository.
It is intentionally implementation-oriented: what to change, where, how to test,
and which behaviors are considered contractually stable.

## Project Scope

quickasr: low-latency (especially streaming latency) speech-to-text runtime
for consumer hardware, forked from antirez/qwen-asr. It runs the Qwen3-ASR
models:
- `Qwen3-ASR-0.6B`
- `Qwen3-ASR-1.7B`

Primary target is CPU inference (BLAS + architecture-specific SIMD paths);
`make gpu` adds an optional ggml GPU path (`--device gpu`).

## Source Of Truth

When docs and code disagree, trust these files first:
- CLI behavior and options: `main.c`
- Public API and runtime state: `qwen_asr.h`
- Offline + segmented + streaming orchestration: `qwen_asr.c`
- Encoder math + load path: `qwen_asr_encoder.c`
- Decoder math + KV cache path: `qwen_asr_decoder.c`
- Kernel dispatch and hot loops: `qwen_asr_kernels*.c`, `qwen_asr_kernels_impl.h`
- Test harness: `asr_regression.py`
- Build targets: `Makefile`

Architecture/background references:
- `MODEL.md`
- `MODEL_CARD_OFFICIAL.md`

## Supported Runtime Modes

- Offline full-context (default): `-S 0`
- Offline segmented: `-S <secs>`
- Streaming: `--stream`
- Input from file: `-i file.wav`
- Input from stdin: `--stdin` (WAV or raw s16le 16k mono)
- Device: `--device cpu|gpu|<name>` (default `cpu`, this program's CPU path, never ggml's
  CPU backend); `gpu`/`<name>` need a `make gpu` build (encoder and decoder through ggml;
  same stream logic). `--list-devices` lists them. `gpu` with no GPU falls back to the CPU
  path; an unknown name fails `qwen_load`

## User-Facing Behavior Contract (Do Not Break)

- `--silent` must still print transcription to stdout.
- `--silent` suppresses status/debug noise (stderr), not the text output.
- Without `--debug`, stderr should be concise:
  - model loading info
  - final inference summary lines
- `--debug` enables verbose internal diagnostics.
- `--language` is the only language forcing flag (no `--force-language`).
- `--past-text` accepted values are exactly `yes|no|auto`.
- `--past-text auto` means:
  - `yes` for `--stream`
  - `no` for non-stream modes

## Model + Inference Facts

- Model variant is auto-detected from weights (0.6B vs 1.7B).
- Encoder uses per-chunk Conv2D + windowed attention.
- Decoder uses causal Qwen3 with KV cache and prefill reuse.
- Encoder weights are loaded as f32 (converted at load where needed).
- Decoder large weights are bf16 mmapped and consumed via bf16 kernels.

## Important Defaults

From `qwen_load()` and CLI:
- Segment mode default: `-S 0` (full-audio decode)
- Segment cut search window: `-W 3.0`
- Stream chunk: `2.0s`
- Stream rollback: `5` tokens
- Stream unfixed chunks: `2`
- Stream max new tokens/chunk: `32`
- Encoder infer attention window: `8s` (`--enc-window-sec` in `[1,8]`)

## Repository Map

- `main.c`
  - CLI parsing, defaults, reporting, callback wiring
- `qwen_asr.c`
  - high-level transcription flows
  - segmented logic + optional past-text cleanup path
  - streaming chunk loop, encoder-window cache, rollback commit logic
- `qwen_asr_encoder.c`
  - audio tower load + forward
- `qwen_asr_decoder.c`
  - decoder load + prefill + token step + KV cache
- `qwen_asr_audio.c`
  - WAV/stdin decoding, resampling, mel prep helpers
- `qwen_asr_tokenizer.c`
  - tokenizer encode/decode
- `qwen_asr_safetensors.c`
  - safetensors loading and mmap
- `qwen_asr_kernels.c`
  - common math, threading, BLAS paths
- `qwen_asr_kernels_generic.c`
  - generic hot kernels
- `qwen_asr_kernels_neon.c`
  - ARM NEON hot kernels
- `qwen_asr_kernels_avx.c`
  - x86 AVX hot kernels
- `qwen_asr_kernels_impl.h`
  - architecture dispatch macros
- `qwen_asr_ggml.c`
  - GPU path: encoder and decoder as ggml graphs, device KV cache (only with `USE_GGML`)
- `scripts/build-ggml.sh`
  - fetches the pinned ggml release (sha256) into `.ggml/` and builds it for `make gpu`
- `scripts/build-vulkan-tools.sh`
  - builds pinned glslc + Vulkan/SPIR-V headers into `.vulkan/` (Linux release builds; `VULKAN_SDK=.vulkan`)
- `asr_regression.py`
  - quality + focused regression checks
- `bench/`
  - real-time streaming latency benchmark (public AISHELL / LibriSpeech test sets, downloaded)
- `download_model.sh`
  - interactive small/large model downloader

## Build + Run

Build:
```bash
make blas
```

GPU build (fetches ggml; needs CMake, and off macOS glslc + Vulkan/SPIR-V headers + loader):
```bash
make gpu
```

Smoke run:
```bash
./quickasr -d qwen3-asr-0.6b -i samples/jfk.wav
```

Stdin path:
```bash
cat samples/jfk.wav | ./quickasr -d qwen3-asr-0.6b --stdin
```

## Regression Workflow

Primary suite:
```bash
make test
# equivalent to:
./asr_regression.py --binary ./quickasr --model-dir qwen3-asr-1.7b
```

Focused checks:
```bash
./asr_regression.py --segment-check-only --binary ./quickasr --model-dir qwen3-asr-1.7b
./asr_regression.py --stream-check-only --binary ./quickasr --model-dir qwen3-asr-1.7b
./asr_regression.py --stream-cache-check-only --binary ./quickasr --stream-cache-model-dir qwen3-asr-0.6b
```

Notes:
- Quality regression only runs on WAVs that already have sibling `.txt` refs.
- `make test` includes stream-cache equivalence check by default.
- This means both model dirs are typically required:
  - main model (`--model-dir`, default `qwen3-asr-1.7b`)
  - stream-cache model (`--stream-cache-model-dir`, default `qwen3-asr-0.6b`)

Reference management:
```bash
./asr_regression.py --generate-missing --binary ./quickasr --model-dir qwen3-asr-1.7b
./asr_regression.py --refresh-refs --binary ./quickasr --model-dir qwen3-asr-1.7b
```

## Streaming Implementation Notes

Current streaming behavior in `qwen_transcribe_stream()`:
- Chunk-by-chunk audio growth (default 2s)
- Encoder cache for completed local-attention windows
- Re-encode only current partial tail window
- Decoder prefill reuse by longest unchanged embedding prefix
- Rolled-back tokens from the previous step checked in one decoder pass (draft reuse)
- Prefix rollback policy for token stability
- Text is committed, and fed back as prefix, only once two consecutive steps decoded it the same
- Monotonic commit frontier (no retracting already-emitted text)

Debug/env switch:
- `QWEN_STREAM_NO_ENC_CACHE=1` disables encoder window cache (debug/regression only)
- `QWEN_GPU_ENC=0` with a GPU `--device` keeps the encoder on the CPU (debug/A-B only)
- `QWEN_STREAM_NO_AGREE=1` commits without waiting for two steps to agree (debug/A-B only)
- `QWEN_STREAM_NO_DRAFT=1` disables reusing the previous step's tokens as a one-pass draft (debug/regression only; output is identical either way)

Important caveat:
- In streaming mode, if no token callback is installed (for example CLI `--silent`),
  the code uses direct final refinement instead of interactive chunk emission.
  This path is not representative of interactive stream throughput.

## Segmented Mode Notes

When `-S > 0`:
- split points are chosen near low-energy regions inside `-W`
- default emission is token-by-token ASAP

When `--past-text yes` in segmented mode:
- boundary cleanup/post-processing path is enabled
- output is buffered per segment before emission
- collapse guardrails can retry segments unconditioned and disable conditioning after repeated collapses

## Performance Reporting Contract

Final stderr summary line format is:
```text
Inference: <ms> ms, <tokens> text tokens (<tok/s> tok/s, encoding: <ms>ms, decoding: <ms>ms)
Audio: <audio_s> s processed in <infer_s> s (<x>x realtime)
```

`encoding` = mel + encoder time
`decoding` = decoder prefill + autoregressive decode

## Kernel/Optimization Rules

- Architecture dispatch is centralized in `qwen_asr_kernels_impl.h`.
- Keep generic/NEON/AVX variants functionally equivalent.
- If you optimize one path, verify no regression on others.
- Favor meaningful speedups; avoid complexity for tiny wins.

## Change Checklist For Agents

Before editing:
1. Identify behavioral contract impacted (CLI, output, speed, quality, memory).
2. Read corresponding source-of-truth file(s).

After editing:
1. Build: `make blas`
2. Run focused sanity command(s) for changed area.
3. Run regression:
   - at minimum relevant focused checks
   - ideally full `make test` for non-trivial changes
4. Update `README.md` and `docs/` if CLI/runtime behavior changed.
5. Keep `AGENT.md` aligned if workflow/test defaults changed.

## Local-Only Artifacts (Do Not Depend On In Commits)

Common local directories/files are intentionally ignored:
- `qwen3-asr-0.6b/`, `qwen3-asr-1.7b/`
- `Qwen3-ASR/`
- `samples/extra/`
- `TODO.md`
- virtualenv folders

Do not make code rely on these being present unless guarded by checks.
