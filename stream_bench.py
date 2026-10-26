#!/usr/bin/env python3
"""
Streaming benchmark for quickasr --stdin --stream.

Measures what the streaming layer loses or delays relative to the same
model's offline transcript, so model recognition errors cancel out:

  cover   fraction of the offline transcript that also appears in the
          streaming output (normalized characters, order-preserving match)
  extra   fraction of the streaming output that matches nothing in the
          offline transcript (duplicates, hallucinated text)
  gap     longest stretch of the offline transcript missing from the
          streaming output, converted to seconds of audio
  first   seconds until the first text arrives            (--pace only;
          with --stream-json a provisional tail counts)
  stuck   characters that only arrive after stdin closes  (--pace only)
  tail    seconds from end of audio to the last text      (--pace only)

Usage examples:
  # Long-run losses, fed as fast as the binary reads
  ./stream_bench.py -d qwen3-asr-0.6b long/*.mp3

  # Latency and end-of-utterance behaviour: real-time feed, then 6 s with
  # stdin open and no audio, the way a live client pauses
  ./stream_bench.py -d qwen3-asr-0.6b --pace --hold 6 clips/*.wav

Arguments after a literal -- replace the default streaming flags
(--stream-max-new-tokens 32 --skip-silence --past-text yes):
  ./stream_bench.py -d qwen3-asr-0.6b --pace clips/*.wav -- --stream-json --past-text yes

Audio in any format ffmpeg reads is converted to 16 kHz mono s16le; files
ending in .raw are assumed to be in that format already. No audio ships with
the repository.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import wave
from pathlib import Path
from typing import List, Optional, Tuple

SAMPLE_RATE = 16000
BYTES_PER_SEC = SAMPLE_RATE * 2
FEED_BYTES = BYTES_PER_SEC * 40 // 1000  # 40 ms, a typical capture frame


def normalize_text(text: str) -> str:
    """Keep letters and digits only, lowercased: punctuation and spacing
    differ between streaming and offline decoding and are not losses."""
    return "".join(c.lower() for c in text if unicodedata.category(c)[0] in "LN")


def load_pcm(path: Path) -> bytes:
    if path.suffix == ".raw":
        return path.read_bytes()
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode(errors='replace')[:200]}")
    return proc.stdout


def offline_reference(binary: str, model_dir: str, path: Path, pcm: bytes, ref_dir: Path) -> str:
    key = hashlib.sha1(pcm).hexdigest()[:12]
    ref_path = ref_dir / f"{path.stem}.{Path(model_dir).name}.{key}.txt"
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        with wave.open(tmp_name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
        proc = subprocess.run([binary, "-d", model_dir, "-i", tmp_name, "--silent"],
                              capture_output=True)
    finally:
        os.unlink(tmp_name)
    if proc.returncode != 0:
        raise RuntimeError(f"offline transcription failed on {path}")
    text = proc.stdout.decode("utf-8", errors="replace").strip()
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(text, encoding="utf-8")
    return text


def run_stream(cmd: List[str], pcm: bytes, pace: bool, hold: float
               ) -> Tuple[List[Tuple[float, str]], float, float]:
    """Feed pcm to cmd's stdin. Returns (timestamped output pieces,
    time audio ended, time stdin closed), all relative to the first byte fed."""
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=0)
    assert proc.stdin and proc.stdout
    events: List[Tuple[float, bytes]] = []
    t0 = time.monotonic()

    def reader() -> None:
        fd = proc.stdout.fileno()
        while True:
            buf = os.read(fd, 4096)
            if not buf:
                return
            events.append((time.monotonic() - t0, buf))

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        if pace:
            for i in range(0, len(pcm), FEED_BYTES):
                proc.stdin.write(pcm[i:i + FEED_BYTES])
                delay = t0 + (i + FEED_BYTES) / BYTES_PER_SEC - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        else:
            proc.stdin.write(pcm)
        t_audio_end = time.monotonic() - t0
        if hold > 0:
            time.sleep(hold)
        t_close = time.monotonic() - t0
        proc.stdin.close()
    except BrokenPipeError:
        t_audio_end = t_close = time.monotonic() - t0
    th.join()
    proc.wait()
    # Decode per event; a multi-byte character split across reads is rare
    # enough at these sizes that replacement is acceptable for timing.
    return [(t, b.decode("utf-8", errors="replace")) for t, b in events], t_audio_end, t_close


def parse_json_updates(events: List[Tuple[float, str]]
                       ) -> Tuple[List[Tuple[float, str]], Optional[float]]:
    """Turn --stream-json lines into (time, committed text) pieces plus the
    time anything, committed or provisional, was first reported."""
    commits: List[Tuple[float, str]] = []
    first_visible: Optional[float] = None
    pending = ""
    for t, chunk in events:
        pending += chunk
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            if not line.strip():
                continue
            try:
                upd = json.loads(line)
            except ValueError:
                continue
            commit, tail = upd.get("commit", ""), upd.get("tail", "")
            if first_visible is None and (commit.strip() or tail.strip()):
                first_visible = t
            if commit:
                commits.append((t, commit))
    return commits, first_visible


def compare(ref: str, out: str, dur: float) -> Tuple[float, float, float]:
    a, b = normalize_text(ref), normalize_text(out)
    if not a:
        return (1.0, 0.0 if not b else 1.0, 0.0)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    blocks = [m for m in sm.get_matching_blocks() if m.size]
    matched = sum(m.size for m in blocks)
    gap = 0
    prev_end = 0
    for m in blocks:
        gap = max(gap, m.a - prev_end)
        prev_end = m.a + m.size
    gap = max(gap, len(a) - prev_end)
    extra = (len(b) - matched) / len(b) if b else 0.0
    return matched / len(a), extra, gap / len(a) * dur


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+", type=Path, help="audio files")
    ap.add_argument("-d", "--model-dir", required=True)
    ap.add_argument("--binary", default="./quickasr")
    ap.add_argument("--pace", action="store_true", help="feed in real time (needed for latency metrics)")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="seconds to keep stdin open with no audio after the file ends")
    ap.add_argument("--ref-dir", type=Path, default=Path(".stream_bench_refs"),
                    help="cache directory for offline reference transcripts")
    ap.add_argument("--jobs", type=int, default=1,
                    help="files to run concurrently (keep at 1 with --pace: contention skews timing)")
    ap.add_argument("--json", type=Path, help="write per-file results here")
    ap.add_argument("--label", default="", help="free-form label stored in the JSON output")
    # Everything after a literal "--" goes to the streaming run. Split it off
    # by hand: argparse would hand those words to the `audio` list.
    argv = sys.argv[1:]
    stream_args: List[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, stream_args = argv[:cut], argv[cut + 1:]
    args = ap.parse_args(argv)
    args.stream_args = stream_args
    return args


def main() -> int:
    args = parse_args()
    extra_args = list(args.stream_args)
    if not extra_args:
        extra_args = ["--stream-max-new-tokens", "32", "--skip-silence", "--past-text", "yes"]
    cmd = [args.binary, "-d", args.model_dir, "--stdin", "--stream"] + extra_args
    json_updates = "--stream-json" in extra_args

    results = []
    hdr = f"{'file':38s} {'dur':>6s} {'cover':>6s} {'extra':>6s} {'gap':>6s}"
    if args.pace:
        hdr += f" {'first':>6s} {'stuck':>6s} {'tail':>6s}"
    print(hdr)
    def run_file(path: Path):
        pcm = load_pcm(path)
        ref = offline_reference(args.binary, args.model_dir, path, pcm, args.ref_dir)
        return (len(pcm) / BYTES_PER_SEC, ref) + run_stream(cmd, pcm, args.pace, args.hold)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs))
    futures = [pool.submit(run_file, path) for path in args.audio]
    for path, fut in zip(args.audio, futures):
        try:
            dur, ref, events, t_audio_end, t_close = fut.result()
        except RuntimeError as e:
            print(f"{path.name[:38]:38s} ERROR {e}")
            continue
        first_visible: Optional[float] = None
        if json_updates:
            events, first_visible = parse_json_updates(events)
        out = "".join(s for _, s in events)
        cover, extra, gap = compare(ref, out, dur)
        row = {"file": path.name, "dur": round(dur, 1), "cover": round(cover, 4),
               "extra": round(extra, 4), "gap_sec": round(gap, 1),
               "ref_chars": len(normalize_text(ref)), "out_chars": len(normalize_text(out))}
        line = f"{path.name[:38]:38s} {dur:6.1f} {cover:6.1%} {extra:6.1%} {gap:5.1f}s"
        if args.pace:
            first: Optional[float] = first_visible if json_updates else next(
                (t for t, s in events if s.strip()), None)
            before = [(t, s) for t, s in events if t < t_close]
            stuck = len(normalize_text("".join(s for t, s in events if t >= t_close)))
            last_before = max((t for t, s in before if s.strip()), default=None)
            tail = (last_before - t_audio_end) if (last_before is not None and stuck == 0) else None
            row.update({"first_sec": None if first is None else round(first, 2),
                        "stuck_chars": stuck,
                        "tail_sec": None if tail is None else round(tail, 2)})
            line += (f" {'-' if first is None else format(first, '5.1f') + 's':>6s}"
                     f" {stuck:6d}"
                     f" {'-' if tail is None else format(tail, '5.1f') + 's':>6s}")
        print(line, flush=True)
        results.append(row)

    if results:
        total_ref = sum(r["ref_chars"] for r in results)
        w_cover = sum(r["cover"] * r["ref_chars"] for r in results) / total_ref if total_ref else 0.0
        summary = (f"{len(results)} files, cover {w_cover:.1%} (weighted), "
                   f"worst gap {max(r['gap_sec'] for r in results):.1f}s")
        if args.pace:
            firsts = sorted(r["first_sec"] for r in results if r.get("first_sec") is not None)
            if firsts:
                summary += f", median first {firsts[len(firsts) // 2]:.1f}s"
            summary += f", files with stuck text {sum(1 for r in results if r.get('stuck_chars'))}"
        print(summary)
    if args.json:
        args.json.write_text(json.dumps({"label": args.label, "cmd": cmd, "pace": args.pace,
                                         "hold": args.hold, "results": results},
                                        ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
