# quickasr

quickasr is a speech-to-text runtime for consumer hardware (laptops, desktops, small ARM boards), focused on latency, streaming latency above all: how soon text appears after it is spoken. It currently runs the [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) models (0.6B and 1.7B). The core is plain C with no dependencies beyond the C standard library and a BLAS implementation (Accelerate on macOS, OpenBLAS on Linux and Windows); `make gpu` adds an optional GPU path (Metal, Vulkan) through ggml.

quickasr started as a fork of [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

## How Fast Is It?

Times faster than real time on an **Apple M1 Pro** (16 GB) for
`samples/night_of_the_living_dead_1968/119s_theres_supposed_to_be_another_broadcast.wav`
(119 s, in the repository), excluding model loading, and the delay before live speech
appears on screen:

| | 0.6B CPU | 0.6B GPU | 1.7B CPU | 1.7B GPU | 1.7B GPU, 1 s chunks |
|---|---|---|---|---|---|
| Offline, whole file (`-S 0`) | 13.1x | 23.7x | 7.0x | 11.3x | — |
| Offline, 30 s segments (`-S 30`) | 14.7x | 24.8x | 7.1x | 11.8x | — |
| Streaming (`--stream`) | 4.3x | 10.1x | 1.9x | 4.9x | 3.2x |
| Live streaming latency, p50 (Chinese / English) | 1.33 / 1.54 s | 1.08 / 1.27 s | 1.90 / 2.14 s | 1.26 / 1.45 s | 0.65 / 0.85 s |

Streaming re-decodes recent audio every chunk (2 s by default, `--stream-chunk-sec`) to
keep the text stable, so it costs more than offline decoding; anything above 1x keeps up
with live audio, which both models do on the CPU alone. Smaller chunks show text sooner
for more compute. The latency row is audio fed in real time (180 s each of AISHELL-1 and
LibriSpeech test speech), from the end of a word to its first appearance. All numbers,
how they were measured and memory use are in [docs/performance.md](docs/performance.md).

## Install

Prebuilt binaries via npm (Linux x86-64/arm64, macOS 13.3+, Windows x86-64):
```bash
npm install -g quickasr
```

Or build from source:
```bash
make blas       # CPU (Accelerate on macOS; on Linux first: apt install libopenblas-dev)
make gpu        # CPU plus --device gpu (Metal on macOS, Vulkan elsewhere; needs CMake)
```

Then download a model (interactive: small = 0.6B, large = 1.7B):
```bash
./download_model.sh
```

The 1.7B model is more accurate and about twice as slow as 0.6B.

## Usage

```bash
# Transcribe a file (text streams to stdout as it is decoded)
./quickasr -d qwen3-asr-0.6b -i audio.wav

# Any format through ffmpeg
ffmpeg -i audio.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
    ./quickasr -d qwen3-asr-0.6b --stdin

# Live microphone (macOS): streaming mode, on the GPU
ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -f s16le - 2>/dev/null | \
    ./quickasr -d qwen3-asr-1.7b --stdin --stream --device gpu

# Long recordings: segmented offline mode (much faster than --stream for files)
./quickasr -d qwen3-asr-0.6b -i lecture.wav -S 30
```

For applications, `--stream-json` prints one object per line: `commit` is final text to
append, `tail` is a provisional guess to show and replace:
```
{"commit":"And so, my fellow","tail":" Americans, ask not."}
{"commit":" Americans, ask not what your","tail":" country can do."}
```

Common options:

| Option | |
|---|---|
| `--stream` | Streaming mode: 2 s chunks (`--stream-chunk-sec`), text as soon as it is stable |
| `--stream-json` | JSON lines with `commit` / `tail` (with `--stream`) |
| `--device cpu\|gpu\|<name>` | Where to run; `--list-devices` lists the GPUs |
| `-S <secs>` | Segmented offline mode for long files |
| `--language <name>` | Force the language (otherwise auto-detected), e.g. `Chinese`, `English` |
| `--prompt <text>` | System prompt to bias spelling, e.g. `"Preserve spelling: PostgreSQL"` |
| `--skip-silence` | Drop long silent spans |
| `--silent` / `--debug` | Only the text on stdout / verbose diagnostics on stderr |

`./quickasr -h` lists every option.

## Documentation

- [docs/streaming.md](docs/streaming.md): how streaming works, defaults, `--stream-json` and its event log, picking `--stream-chunk-sec`, `--monitor`
- [docs/usage.md](docs/usage.md): offline and segmented modes, `--past-text`, silence skipping, GPU devices, language, prompt, stdin
- [docs/performance.md](docs/performance.md): all benchmarks and memory use
- [docs/development.md](docs/development.md): building (Linux, Windows, Vulkan), npm packages and releases, regression tests, C API
- [MODEL.md](MODEL.md): model architecture and weight format

## License

MIT
