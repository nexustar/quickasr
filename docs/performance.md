# Performance and Memory

Measured on an **Apple M1 Pro** (6 performance + 2 efficiency cores, 14-core GPU, 16 GB)
with a `make gpu` build and the default threads (the 6 performance cores). Times are the
`Inference` line of the summary (model loading excluded), median of 3 runs. The offline
and whole-file streaming tables use the clips in `samples/`.

## Real-time Streaming Latency

The main use case: audio fed to `--stdin --stream --stream-json` at 1x, 180 s each of
Chinese (AISHELL-1 test) and English (LibriSpeech test-clean) speech, built and scored
with the scripts in [bench/](../bench/README.md) (the audio is downloaded, not in the
repository). Latency is the time from the end of a
word in the audio to its appearance: *shown* counts the provisional `tail`, *committed*
only final text. Seconds, p50 / p90:

| Model, device, chunk | zh shown | zh committed | en shown | en committed | busy (zh / en) | CER zh / WER en |
|---|---|---|---|---|---|---|
| 0.6B, CPU, 2 s | 1.35 / 2.24 | 3.99 / 5.42 | 1.58 / 2.57 | 3.69 / 5.14 | 28% / 32% | 2.7% / 2.5% |
| 0.6B, GPU, 2 s | 1.11 / 1.94 | 3.74 / 5.03 | 1.30 / 2.27 | 3.37 / 4.75 | 14% / 17% | 2.5% / 2.5% |
| 1.7B, CPU, 2 s | 1.96 / 2.93 | 4.72 / 6.02 | 2.22 / 3.27 | 4.34 / 5.74 | 60% / 66% | 1.1% / 2.5% |
| 1.7B, GPU, 2 s | 1.29 / 2.17 | 4.01 / 5.78 | 1.46 / 2.48 | 3.54 / 5.04 | 25% / 29% | 1.1% / 1.8% |
| 0.6B, CPU, 1 s | 0.74 / 1.37 | 3.12 / 4.27 | 1.00 / 1.73 | 2.39 / 3.29 | 48% / 53% | 2.5% / 1.8% |
| 0.6B, GPU, 1 s | 0.52 / 1.12 | 2.89 / 4.07 | 0.73 / 1.45 | 2.11 / 3.06 | 22% / 26% | 2.5% / 2.5% |
| 1.7B, GPU, 1 s | 0.62 / 1.21 | 3.18 / 4.30 | 0.83 / 1.57 | 2.27 / 3.14 | 38% / 44% | 0.9% / 1.8% |
| 1.7B, CPU, 1 s | 1.52 / 2.74 | 4.27 / 5.85 | 4.87 / 9.60 | 6.55 / 11.37 | 96% / 104% | 0.9% / 1.8% |

*busy* is compute time as a share of the audio duration. Everything keeps up except 1.7B
on the CPU with 1 s chunks, which runs at the limit and on English falls behind, so its
latency grows over the stream.
Committed text trails the shown text: the last 5 tokens (rollback) stay open, text is
committed only once two steps decoded it the same, and the first chunks are not
committed at all. A pause commits everything at once. Word end times come from segment timestamps interpolated per character, so the
latencies are approximate.

## Offline Mode (Full + Segmented)

| Setup | Audio | 0.6B CPU | 0.6B GPU | 1.7B CPU | 1.7B GPU |
|-------|-------|----------|----------|----------|----------|
| `samples/jfk.wav -S 0` | 11.0 s | 0.89 s (12.4x) | 0.51 s (21.6x) | 1.91 s (5.8x) | 1.21 s (9.1x) |
| `45s_dont_be_afraid_of_me.wav -S 30 -W 3` | 45.0 s | 2.09 s (21.5x) | 1.12 s (40.2x) | 4.16 s (10.8x) | 2.54 s (17.7x) |
| `89s_ill_come_back_down_as_soon_as.wav -S 30 -W 3` | 88.9 s | 8.88 s (10.0x) | 5.43 s (16.4x) | 18.81 s (4.7x) | 11.62 s (7.7x) |

(x = times faster than real time.)

## Streaming Mode, Whole File (interactive `--stream`, 45 s clip)

| Setup | 0.6B | 1.7B |
|-------|------|------|
| CPU, encoder cache on (default) | 7.37 s (6.1x) | 13.22 s (3.4x) |
| CPU, encoder cache off (`QWEN_STREAM_NO_ENC_CACHE=1`, single run) | 14.46 s (3.1x) | 24.50 s (1.8x) |
| GPU | 2.90 s (15.5x) | 4.06 s (11.1x) |

## Long-file Example (`119s_theres_supposed_to_be_another_broadcast.wav`, 119 s)

| Mode | 0.6B CPU | 0.6B GPU | 1.7B CPU | 1.7B GPU |
|------|----------|----------|----------|----------|
| `--stream` | 27.6 s (4.3x) | 11.8 s (10.1x) | 61.6 s (1.9x) | 24.2 s (4.9x) |
| `-S 0` | 9.1 s (13.1x) | 5.0 s (23.7x) | 17.1 s (7.0x) | 10.6 s (11.3x) |
| `-S 30` | 8.1 s (14.7x) | 4.8 s (24.8x) | 16.7 s (7.1x) | 10.1 s (11.8x) |

For prerecorded files, segmented offline mode is several times faster than `--stream`.

## Memory Requirements

Memory usage has two parts:
- Static model footprint (allocated once at load time).
- Runtime footprint (depends on input length and decoding mode).

### Static Footprint (Model Load)

These numbers come from the current implementation and model files:
- Safetensors are memory-mapped.
- Encoder BF16 weights are converted to F32 and kept in heap memory.
- Decoder builds a fused gate/up matrix copy for faster decode.

| Component | 0.6B | 1.7B |
|-----------|------|------|
| safetensors mmap files | 1.747 GiB | 4.376 GiB |
| encoder copied F32 weights | 0.694 GiB | 1.183 GiB |
| decoder extra heap (fused + norms) | 0.328 GiB | 1.313 GiB |
| static total (theoretical) | 2.770 GiB | 6.871 GiB |

### Runtime Scaling (Why `-S 0` Grows)

For one segment, dominant runtime allocations scale with sequence length:

- `mel_frames ~= floor(audio_seconds * 100)`
- `enc_tokens ~= 13 * floor(mel_frames / 100) + ceil((mel_frames % 100) / 8)`
- `total_seq = enc_tokens + 15` (plus prompt/language/past-text tokens if used)
- `prefill_len = total_seq - 1`
- `pref_cap = next_pow2(prefill_len)`
- `kv_max = prefill_len + 1024`

Main growing buffers:
- KV cache: `2 * 28 * kv_max * 1024 * 4` bytes
- Prefill buffers:
  - 0.6B: `77,824 * pref_cap` bytes
  - 1.7B: `131,072 * pref_cap` bytes

Implications:
- `-S 0` (full-audio decode) lets `total_seq` grow with audio duration, so peak memory increases with file length.
- `-S 20` (or any segmented mode) bounds per-segment `total_seq`, so memory stays nearly flat as file length increases.
- Enabling `--past-text yes` adds previous text tokens to each segment/chunk prompt and can increase memory again.

### Measured Peak RSS (`--silent`)

Measured on the Apple M1 Pro above. With `--device gpu` the f16 weights on the device
count too (macOS reports unified GPU memory in the process RSS).

| Audio length | 0.6B `-S 0` | 0.6B `-S 20` | 0.6B GPU `-S 0` | 1.7B `-S 0` | 1.7B `-S 20` | 1.7B GPU `-S 0` |
|--------------|------------:|-------------:|----------------:|------------:|-------------:|----------------:|
| 10.000s | 2.669 GiB | 2.674 GiB | 4.076 GiB | 6.572 GiB | 6.570 GiB | 8.558 GiB |
| 45.000s | 2.848 GiB | 2.763 GiB | 4.099 GiB | 6.753 GiB | 6.664 GiB | 9.601 GiB |
| 88.890s | 3.147 GiB | 2.783 GiB | 4.106 GiB | 7.099 GiB | 6.691 GiB | 9.516 GiB |
| 119.262s | 3.271 GiB | 2.777 GiB | 4.142 GiB | 7.263 GiB | 6.678 GiB | 8.500 GiB |

In practice:
- For long files, segmented mode is safer for both speed and memory.
- Default is `-S 0`, so for large files explicitly pick segmented mode (`-S 20` or `-S 30`).
- Use `-S 0` mainly for short files where full-context quality is worth the extra memory/time.
