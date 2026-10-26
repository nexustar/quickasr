# Streaming (`--stream`)

```bash
./quickasr -d qwen3-asr-0.6b -i long_recording.wav --stream
```

Streaming mode processes audio in **2-second chunks** with rollback text conditioning:

1. Audio arrives chunk by chunk.
2. Encoder uses local windows (`--enc-window-sec`, default `8s`); completed windows are cached, only the current partial tail window is re-encoded.
3. Decoder prompt includes previous output minus a rollback suffix (5 tokens by default) to stabilize chunk boundaries.
4. Per chunk decode is bounded by `--stream-max-new-tokens` (default `32`). A step that covers more than one chunk of audio (after skipped silence, or decoding kept audio again after a reset) may write up to 10 tokens per second of that audio instead.
5. Only stable text is emitted: text is committed, and fed back to the decoder as prefix, once two consecutive steps decoded it the same. A step that has heard only the first word of a sentence can guess the rest; the next step, with more audio, overwrites the guess instead of building on it. The final chunk flushes remaining text.
6. With live stdin, a pause also flushes: when no audio arrives for `--stream-idle-flush-ms` (default `800`), or with `--skip-silence` once the audio has stayed near its noise floor for a second after speech (checked as audio arrives, not only at chunk boundaries), the buffered audio is decoded and the held-back text is emitted. The stream stays open and decoding continues from there when audio resumes. Without this a source that stops sending during silence (a voice gate, a paused capture) leaves the end of each utterance unprinted until the speaker resumes.
7. With live stdin, after a longer pause (`--stream-pause-reset-ms`, default `3000`; an input stall, or with `--skip-silence` audio near its noise floor) the next speech is decoded without the earlier text as context, so what was said before the break stops shaping what follows (for example Japanese text turning a switch to English into comma-separated single words). Without `--language` the language is detected again, so a speaker who switches language after a pause is followed. Speech already printed is not transcribed again.

This keeps encoder recomputation under control. For long streams, a **sliding window** automatically bounds both encoder and decoder context so that memory and compute stay flat indefinitely:

- **Encoder window eviction**: only the most recent 4 encoder attention windows (~32 s of audio context) are kept; older windows are freed.
- **Decoder prefix capping**: only the most recent ~150 tokens of previously decoded text are fed as decoder prefix context. Internal token history is still tracked for streaming commit/dedup decisions, while only the capped tail is embedded into the decoder input.

These limits activate automatically when the stream is long enough to exceed them. For short files or live sessions under ~40 s, they have no effect.

`--stream --silent` has a special non-interactive behavior for file input: it skips chunk-by-chunk streaming and runs one direct final refinement pass. (For live stdin streaming, chunked mode is still used.)

`--stream-json` replaces the plain token stream with one JSON object per line:

```
{"commit":"And so, my fellow","tail":" Americans, ask not."}
{"commit":" Americans, ask not what your","tail":" country can do."}
```

`commit` is text that will not change; append it. `tail` is the model's current guess at what follows, including the rollback tokens that plain mode holds back and everything decoded during the first `unfixed_chunks`; it replaces the previous tail and can be rewritten or withdrawn. Show it for low latency, but split sentences on `commit` only: the tail routinely ends in a period that the next chunk takes back.

With `--stream-json` (unless `--silent`) the stream also logs to stderr, one line per event, stamped with the position in the input: pause, stall and periodic resets, loop recoveries with their cause and the looping text, a loop broken with the runner-up token, the detected language when it changes, and, live, input waiting to be decoded when decoding falls behind:

```
[stream 4.0s] language Japanese
[stream 32.0s] loop broken with the runner-up token
[stream 68.0s] recovery: loop while decoding, "嬉しい"
[stream 96.0s] pause reset
[stream 128.0s] 12.0s of input not decoded yet
```

Default stream settings:
- `chunk_size`: 2s (`--stream-chunk-sec`, range `1..8`)
- `encoder_window`: 8s (`--enc-window-sec`, range `1..8`)
- `rollback`: 5 tokens
- `unfixed_chunks`: 2
- `max_new_tokens`: 32 (`--stream-max-new-tokens`)
- `past_text`: `auto` by default (effectively `yes` for `--stream`, `no` otherwise)
- `idle_flush`: 800 ms (`--stream-idle-flush-ms`, `0` disables; live stdin only)
- `pause_reset`: 3000 ms (`--stream-pause-reset-ms`, `0` disables; live stdin only)

Streaming tuning:

```bash
# default streaming
./quickasr -d qwen3-asr-0.6b -i audio.wav --stream

# larger chunk for slower hardware (see notes below)
./quickasr -d qwen3-asr-1.7b -i audio.wav --stream --stream-chunk-sec 4

# narrower encoder window (less compute per chunk; may reduce quality)
./quickasr -d qwen3-asr-0.6b -i audio.wav --stream --enc-window-sec 4

# allow more text generation per chunk
./quickasr -d qwen3-asr-0.6b -i audio.wav --stream --stream-max-new-tokens 64
```

Picking `--stream-chunk-sec`:

Each chunk's audio has to arrive before it is decoded, so a smaller chunk shows text
sooner and costs more compute (more decoder passes per second of audio). Fed in real
time on an Apple M1 Pro, the share of the audio duration spent computing, and the median
delay before a word is shown (180 s of Chinese speech):

| Model, device | `--stream-chunk-sec 2` (default) | `--stream-chunk-sec 1` |
|---|---|---|
| 0.6B, CPU | 28% busy, shown after 1.4 s | 48% busy, shown after 0.7 s |
| 0.6B, GPU | 14% busy, shown after 1.1 s | 22% busy, shown after 0.5 s |
| 1.7B, CPU | 60% busy, shown after 2.0 s | 96% busy, shown after 1.5 s; falls behind on English |
| 1.7B, GPU | 25% busy, shown after 1.3 s | 38% busy, shown after 0.6 s |

Use 1 s when there is headroom (the GPU, or 0.6B on a fast CPU); raise it on slower
hardware once the busy share approaches 100% and text starts to lag behind the audio.
Prefer raising `--stream-chunk-sec` over lowering `--enc-window-sec`: the chunk size is
a scheduling knob with no quality cost, while a smaller encoder window also shrinks the
model's attention context.

## Monitor Mode (`--monitor`)

```bash
./quickasr -d qwen3-asr-0.6b -i audio.wav --stream --monitor
```

Shows inline Unicode symbols on stderr alongside the transcription, useful for diagnosing streaming pipeline behavior in real time:

| Symbol | Meaning |
|--------|---------|
| `▶` | Encoder chunk processed |
| `·` | Decoder prefill completed |
| `▪` | Decode step (normal speed) |
| `▸` | Decode step (slow, >30 ms/token) |
| `⟳` | Encoder window evicted (sliding window) |

Example output (stderr + stdout interleaved):
```
▶·▪▶·▪▶·▪And so, my fellow Americans,▶·▪ ask not what your country...
```

Monitor output goes to stderr and does not affect the transcription text on stdout. It can be combined with `--debug` for full diagnostics, or used alone for a lightweight visual heartbeat.
