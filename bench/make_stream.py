#!/usr/bin/env python3
"""Concatenate test-set utterances into one long 16 kHz s16le stream.

usage: make_stream.py {zh|en} PARQUET OUT_STEM [--sec 600] [--skip N]
                      [--gap MIN MAX] [--noise-dbfs DB]

PARQUET is a Hugging Face parquet shard with an audio column (FLAC/WAV bytes,
16 kHz) and a transcript column. Utterances are taken in file order, separated
by 0.3-1.5 s of silence (fixed seed), until --sec seconds; --skip N starts
at the N-th utterance, for a second stream from the same file. --gap sets
the silence range in seconds; --noise-dbfs adds white noise at that RMS level
to the whole stream (room noise instead of digital silence). Writes OUT_STEM.raw
and OUT_STEM.json: {"ref": text, "segments": [{i, start, end, text}]}.
"""
import argparse, io, json, random, sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SR = 16000


def utterances(path, lang):
    t = pq.read_table(path)
    cols = t.column_names
    acol = next(c for c in cols if "audio" in c.lower() or "context" in c.lower())
    tcol = next(c for c in cols if c.lower() in ("answer", "text", "sentence", "transcript"))
    for row in t.to_pylist():
        pcm, sr = sf.read(io.BytesIO(row[acol]["bytes"]), dtype="int16")
        assert sr == SR, sr
        text = row[tcol]
        yield pcm, text.lower() if lang == "en" else text.replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang", choices=["zh", "en"])
    ap.add_argument("parquet")
    ap.add_argument("out")
    ap.add_argument("--sec", type=float, default=600)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--gap", type=float, nargs=2, default=[0.3, 1.5])
    ap.add_argument("--noise-dbfs", type=float)
    a = ap.parse_args()
    rng = random.Random(1234)
    pieces, segs, t = [], [], 0.0
    for k, (pcm, text) in enumerate(utterances(a.parquet, a.lang)):
        if k < a.skip:
            continue
        if t + len(pcm) / SR > a.sec:
            break
        gap = np.zeros(int(rng.uniform(*a.gap) * SR), dtype="<i2")
        if pieces:
            pieces.append(gap)
            t += len(gap) / SR
        segs.append({"i": len(segs), "start": round(t, 3), "end": round(t + len(pcm) / SR, 3), "text": text})
        pieces.append(pcm.astype("<i2"))
        t += len(pcm) / SR
    pcm = np.concatenate(pieces)
    if a.noise_dbfs is not None:
        noise = np.random.default_rng(1234).normal(0, 32768 * 10 ** (a.noise_dbfs / 20), len(pcm))
        pcm = np.clip(pcm + noise, -32768, 32767).astype("<i2")
    pcm.tofile(a.out + ".raw")
    sep = " " if a.lang == "en" else ""
    json.dump({"ref": sep.join(s["text"] for s in segs), "segments": segs},
              open(a.out + ".json", "w"), ensure_ascii=False, indent=0)
    print(f"{a.out}: {len(segs)} utterances, {t:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
