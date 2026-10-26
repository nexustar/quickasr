# Usage

Offline transcription, segmented mode, GPU, language and prompt options, stdin input.
Streaming has its own page: [streaming.md](streaming.md).

## Normal Mode (Default)

```bash
./quickasr -d qwen3-asr-0.6b -i recording.wav
```

This is the default mode, and defaults to `-S 0` (full-audio offline decode).
The model sees the entire recording in one shot, which is usually best for short/medium files.
For long files, memory/time grow with sequence length, so segmented mode (`-S 20` or similar) is often preferable.

Tokens stream to stdout as they are generated. By default, timing info is printed to stderr (`Inference: ...` and `Audio: ... (Xx realtime)`). Use `--silent` or `--debug` to control verbosity:

```bash
./quickasr -d qwen3-asr-0.6b -i audio.wav --silent    # no stderr output
./quickasr -d qwen3-asr-0.6b -i audio.wav --debug      # per-layer/per-chunk details
```

Token emission behavior depends on mode:
- With `-S 0`, text is emitted token-by-token as soon as each decode step produces it.
- With segmented mode (`-S > 0`), default behavior is still token-by-token ASAP.
- With segmented mode plus `--past-text yes`, boundary cleanup is enabled automatically and output is emitted once per segment after post-processing.

For very long files, decoder cost still grows with sequence history. Use `--stream` when you need incremental output while audio is arriving.

## Which Mode To Use (By File Length)

- **Up to ~60s**: use `-S 0` (which is the default) for best quality if speed is acceptable.
- **Large prerecorded files**: use segmented offline mode, e.g. `-S 20` (or `-S 30`, or even more).
- **Long live/continuous audio or low-latency UI needs**: use `--stream`.
- **Batch/offline file transcription**: prefer `-S 20`/`-S 30`; it is usually much faster than interactive `--stream`.
- **If segmented output drops/warps around boundaries**: try a different segment size and keep default `--past-text auto`.
- **If you want stronger continuity across segments/chunks**: try `--past-text yes` (can help continuity, can also cause drift on some files).

Large-file tradeoff summary:
- `-S 20`: offline segmented decode, usually best throughput on long files, stable memory, and token-by-token output.
- `-S 20 --past-text yes`: buffered per-segment output with boundary cleanup and continuity bias.
- `--stream`: incremental output while audio arrives, lower interaction latency, but usually higher total compute for full prerecorded files.

## Segment Splitting (`-S`)

```bash
./quickasr -d qwen3-asr-0.6b -i long_recording.wav -S 20
```

Splits audio into segments of ~N seconds, finding segment-cutting silence boundaries within a search window (`-W`, default 3 seconds). Segments are transcribed and concatenated.

Default segmented behavior (`-S > 0`) emits tokens ASAP, like full offline mode.

When `--past-text yes` is used, segmented mode switches to buffered per-segment emission and enables boundary post-processing:
- Split points are chosen near low-energy (silence-like) regions within the `-W` window to avoid cutting in the middle of words.
- If past-text conditioning causes a segment collapse (too short for its duration) or large duplicate span, that segment is retried without conditioning.
- If collapses keep happening, past-text conditioning is disabled for the remainder of the run.
- Boundary whitespace and spacing are normalized when segments are appended.

By default (`--past-text auto`), segmented mode does **not** use past-text conditioning. This is usually more stable on long files.
If you want extra continuity bias across boundaries, enable conditioning explicitly:

```bash
./quickasr -d qwen3-asr-0.6b -i lecture.wav -S 20 --past-text yes
```

If repeated conditioned segment collapses are detected, conditioning is disabled automatically for the rest of the run (fail-open behavior).

`--past-text` also applies to `--stream`, where `auto` means `yes`.

## Silence Skipping (`--skip-silence`)

```bash
./quickasr -d qwen3-asr-0.6b -i recording.wav --skip-silence
```

When enabled, long silent spans are removed before transcription (short pauses are kept). This reduces compute on recordings with long dead-air sections.

Tradeoffs:
- Useful for podcasts, meetings, and captured audio with long pauses.
- Can slightly alter timing-sensitive boundary behavior and punctuation.
- Disabled by default to preserve baseline behavior.

## GPU (`--device`)

```bash
make gpu
./quickasr -d qwen3-asr-1.7b --stdin --stream --device gpu
```

With a `make gpu` build, `--device gpu` runs the encoder and decoder on the GPU through
ggml; the streaming logic is the same as on the CPU. On an Apple M1 Pro (14-core GPU) it
is 1.6-3.3x faster than the CPU path (see [performance.md](performance.md)), and a
real-time stream leaves the CPU almost idle (5-8% of one core).
The device holds an f16 copy of the weights, so memory use is higher (about 1.4 GB for
0.6B, 1.2-2.9 GB for 1.7B), and loading 1.7B takes about 4 s longer.

`--device` takes `cpu` (the default), `gpu`, or a GPU name from `--list-devices`:
```text
$ quickasr --list-devices
cpu: CPU (6 threads) [default]
MTL0: Apple M1 Pro (gpu, 10922 MiB free of 10922 MiB) [gpu]
```

For programs, `--list-devices json` prints the same as one JSON array; `default` marks
what runs without `--device`, `gpu_default` what `--device gpu` picks:
```text
$ quickasr --list-devices json
[{"name":"cpu","description":"CPU","type":"cpu","threads":6,"default":true,"gpu_default":false},{"name":"MTL0","description":"Apple M1 Pro","type":"gpu","free_mib":10922,"total_mib":10922,"default":false,"gpu_default":true}]
```

Notes:
- `gpu` is the first discrete GPU, else the first integrated one (marked `[gpu]` in the
  list); with several, name one (e.g. `--device Vulkan1`). With no GPU, or no Vulkan
  driver on Linux and Windows, `--device gpu` prints a notice and runs on the CPU; a
  name that matches no GPU is an error.
- The first run compiles the GPU shaders, which takes 10-20 s on macOS; the system caches
  the result, so later runs start in about a second.
- `cpu` is this program's own CPU path, not ggml's CPU backend, which is 1.5-2.5x slower
  here (M1 Pro, same output).
- Without a `make gpu` build, `--device gpu` prints a notice and runs on the CPU.
- Vulkan (Linux, Windows) has not been benchmarked yet.

## Language (`--language`)

```bash
./quickasr -d qwen3-asr-0.6b -i audio.wav --language Italian
```

Forces the model to transcribe (or translate) into the specified language by adding language tokens into the decoder prompt. If omitted, language is usually auto-detected from the audio. The 0.6b and 1.7b models can behave differently, with the smaller model being more likely to translate when forced into a different language than the source audio.

Example of this behavior:

```
$ ./quickasr -d qwen3-asr-0.6b -i samples/jfk.wav --language Italian --silent
E così, miei amici americani, chiedete non ciò che il vostro paese può
fare per voi, chiedete ciò che voi possiate fare per il vostro paese.
```

Supported languages: Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian.

## System Prompt (`--prompt`)

```bash
./quickasr -d qwen3-asr-0.6b -i audio.wav --prompt "Preserve spelling: PostgreSQL, Redis, CUDA"
```

Injects a system prompt into the model's chat template. This slightly biases the model without changing the fundamental transcription behavior. Useful for:

- **Preserving technical terms**: `--prompt "Preserve spelling: PostgreSQL, CUDA, FFmpeg"`
- **Domain context**: `--prompt "This is a medical consultation about cardiology."`
- **Style hints**: `--prompt "Use formal punctuation and capitalization."`

The prompt is encoded once and prepended to every segment/chunk. Its effect is subtle — it nudges the model's token probabilities rather than forcing specific output.

## Reading Audio from Stdin

The **`--stdin` flag** reads audio from standard input. The format is auto-detected: if the data starts with a RIFF header it is parsed as WAV, otherwise it is treated as **raw signed 16-bit little-endian, 16 kHz, mono** (`s16le`).

```bash
# Transcribe an MP3 file
ffmpeg -i podcast.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
    ./quickasr -d qwen3-asr-0.6b --stdin

# Pipe a WAV directly
cat recording.wav | ./quickasr -d qwen3-asr-0.6b --stdin

# Live transcription of a web radio stream
curl -sL http://stream.live.vc.bbcmedia.co.uk/bbc_world_service | \
    ffmpeg -i pipe:0 -ar 16000 -ac 1 -f s16le pipe:1 2>/dev/null | \
    ./quickasr -d qwen3-asr-0.6b --stdin --stream --monitor

# Same flow, but keep WAV framing on stdin
curl -sL http://stream.live.vc.bbcmedia.co.uk/bbc_world_service | \
    ffmpeg -i pipe:0 -ar 16000 -ac 1 -f wav pipe:1 2>/dev/null | \
    ./quickasr -d qwen3-asr-0.6b --stdin --stream --monitor
```

To convert files to WAV format, just use ffmpeg:

    ffmpeg -i input.ogg output.wav

Example WAV files are under `samples/`: `jfk.wav`, `test_speech.wav` and clips from *Night of the Living Dead* (1968, public domain) in `samples/night_of_the_living_dead_1968/`.
