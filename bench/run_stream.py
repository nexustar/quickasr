#!/usr/bin/env python3
"""Feed a 16 kHz s16le stream to quickasr --stream-json over stdin and log what it shows.

usage: run_stream.py RAW OUT_STEM [--realtime] [--limit-sec S] -- QUICKASR_CMD...

QUICKASR_CMD must include --stdin --stream --stream-json. The stream clock starts
once the model has loaded. Writes OUT_STEM.txt (final text), OUT_STEM.stderr and
OUT_STEM.events.json: {"audio_sec", "wall_sec", "cpu_sec", "maxrss_mb", "realtime",
"events": [[t, committed, shown], ...]}, t in seconds since the first audio byte.
"""
import argparse, json, os, subprocess, sys, threading, time

SR = 16000
BLOCK = SR * 2 // 10  # 100 ms of s16le


def main():
    k = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--realtime", action="store_true", help="feed at 1x (needed for latency)")
    ap.add_argument("--limit-sec", type=float, default=0)
    a = ap.parse_args(sys.argv[1:k])
    cmd = sys.argv[k + 1:]

    data = open(a.raw, "rb").read()
    if a.limit_sec:
        data = data[: int(a.limit_sec * SR) * 2]
    audio_sec = len(data) / 2 / SR

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open(a.out + ".stderr", "w"))
    t0 = [None]

    def feeder():
        while "Model loaded." not in open(a.out + ".stderr").read():
            time.sleep(0.02)
        t0[0] = time.monotonic()
        for off in range(0, len(data), BLOCK):
            if a.realtime:
                d = t0[0] + off / 2 / SR - time.monotonic()
                if d > 0:
                    time.sleep(d)
            p.stdin.write(data[off:off + BLOCK])
            p.stdin.flush()
        p.stdin.close()

    threading.Thread(target=feeder, daemon=True).start()

    committed, tail, events = "", "", []
    for line in p.stdout:
        now = time.monotonic()
        try:
            o = json.loads(line.decode("utf-8", "replace"))
        except ValueError:
            continue
        if t0[0] is None:
            continue
        committed += o.get("commit", "")
        tail = o.get("tail", "")
        events.append([round(now - t0[0], 3), committed, committed + tail])
    _, status, ru = os.wait4(p.pid, 0)
    wall = time.monotonic() - t0[0]
    open(a.out + ".txt", "w").write(committed + tail + "\n")
    json.dump({"audio_sec": audio_sec, "wall_sec": round(wall, 2),
               "cpu_sec": round(ru.ru_utime + ru.ru_stime, 2),
               "maxrss_mb": round(ru.ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10), 1),
               "realtime": a.realtime, "exit": status, "events": events},
              open(a.out + ".events.json", "w"), ensure_ascii=False)
    print(f"{a.out}: audio {audio_sec:.0f}s wall {wall:.1f}s cpu {ru.ru_utime + ru.ru_stime:.1f}s "
          f"exit {status}", file=sys.stderr)


if __name__ == "__main__":
    main()
