# Streaming latency benchmark

Measures how long after a word is spoken it appears, with audio fed to
`quickasr --stdin --stream --stream-json` in real time. These scripts produced the
latency table in [docs/performance.md](../docs/performance.md). The test audio is not in
the repository; it comes from two public test sets on Hugging Face:

| Stream | Source | License |
|---|---|---|
| Chinese | [AudioLLMs/aishell_1_zh_test](https://huggingface.co/datasets/AudioLLMs/aishell_1_zh_test), `data/test-00000-of-00003.parquet` (AISHELL-1 test) | Apache 2.0 |
| English | [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr), `all/test.clean/0000.parquet` (LibriSpeech test-clean) | CC BY 4.0 |

`make_stream.py` joins the utterances of a file, in order, into one long stream with
0.3-1.5 s of silence between them (fixed seed), and keeps each utterance's start and end
time. The latency of a character is the time it first appears minus its time in the
audio, interpolated linearly inside its utterance, so it is approximate.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pyarrow soundfile rapidfuzz cn2an zhconv whisper-normalizer huggingface_hub

.venv/bin/hf download AudioLLMs/aishell_1_zh_test data/test-00000-of-00003.parquet \
    --repo-type dataset --local-dir data/aishell
.venv/bin/hf download openslr/librispeech_asr all/test.clean/0000.parquet \
    --repo-type dataset --local-dir data/librispeech

mkdir -p streams
.venv/bin/python make_stream.py zh data/aishell/data/test-00000-of-00003.parquet streams/zh__aishell00
.venv/bin/python make_stream.py en data/librispeech/all/test.clean/0000.parquet streams/en__libri00
```

Stream names must start with the language (`zh__`, `en__`); `score.py` picks the metric
from it.

## Run and score

```bash
mkdir -p out
for s in zh__aishell00 en__libri00; do
  .venv/bin/python run_stream.py streams/$s.raw out/$s.0.6b-cpu --realtime --limit-sec 180 -- \
      ../quickasr -d ../qwen3-asr-0.6b --stdin --stream --stream-json
done
.venv/bin/python score.py streams out 0.6b-cpu
```

The output stem is `out/<stream>.<tag>`; pass the same tags to `score.py`. Any quickasr
option can follow `--` (`--device gpu`, `--stream-chunk-sec 1`, ...). Each run takes the
length of the audio, here 180 s.

`score.py` prints, per stream and tag:

- `err%`: CER (Chinese) or WER (English) of the final text, with substitutions, deletions
  and insertions
- `busy`: compute time (the `Inference` line) as a share of the audio duration; it must
  stay below 100% to keep up
- `cpu/aud`: CPU seconds per second of audio, `rssMB`: peak memory
- `commit p50/p90`: seconds until a character is committed (final)
- `shown p50/p90`: seconds until it is first shown, committed or in the provisional tail
