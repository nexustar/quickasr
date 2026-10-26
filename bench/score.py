#!/usr/bin/env python3
"""Score streaming runs against reference streams.

usage: score.py STREAM_DIR OUT_DIR TAG [TAG ...]
  STREAM_DIR/LANG__NAME.json   from make_stream.py (LANG is zh or en)
  OUT_DIR/LANG__NAME.TAG.*     from run_stream.py with OUT_STEM OUT_DIR/LANG__NAME.TAG

Accuracy: CER for zh (NFKC, letters+digits only, t2s, digits->hanzi), WER for en
(whisper EnglishTextNormalizer). busy: the run's Inference time / audio duration.
Latency (realtime runs only): ref chars get an audio time by linear interpolation
inside their segment; matched hyp chars get the wall time when they were first
committed and first displayed (display = committed + provisional tail).
"""
import json, os, re, statistics, sys, unicodedata
from rapidfuzz.distance import Levenshtein


def simple(t):
    t = unicodedata.normalize("NFKC", t).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in "LN")


def norm_zh(t):
    import cn2an, zhconv
    t = zhconv.convert(unicodedata.normalize("NFKC", t), "zh-cn")
    try:
        t = cn2an.transform(t, "an2cn")
    except Exception:
        pass
    t = re.sub(r"\d", lambda m: "零一二三四五六七八九"[int(m.group())], t)
    return simple(t)


_en = None


def norm_en_words(t):
    global _en
    if _en is None:
        from whisper_normalizer.english import EnglishTextNormalizer
        _en = EnglishTextNormalizer()
    return _en(t).split()


def accuracy(lang, ref, hyp):
    if lang == "en":
        r, h = norm_en_words(ref), norm_en_words(hyp)
    elif lang == "zh":
        r, h = norm_zh(ref), norm_zh(hyp)
    else:
        r, h = simple(ref), simple(hyp)
    ops = Levenshtein.editops(r, h)
    s = sum(o.tag == "replace" for o in ops)
    d = sum(o.tag == "delete" for o in ops)
    i = sum(o.tag == "insert" for o in ops)
    return len(r), s, d, i


def latency(segs, ev):
    """Return (commit_lat list, display_lat list) for matched chars."""
    rt, rc = [], []
    for s in segs:
        t = simple(s["text"])
        for k, c in enumerate(t):
            rc.append(c)
            rt.append(s["start"] + (s["end"] - s["start"]) * (k + 1) / len(t))
    events = ev["events"]
    if not events:
        return [], []
    final = simple(events[-1][2])
    ct, dt = [None] * len(final), [None] * len(final)
    for t, com, disp in events:
        for arr, txt in ((ct, simple(com)), (dt, simple(disp))):
            for k in range(len(txt)):
                if k < len(arr) and arr[k] is None:
                    arr[k] = t
    end = events[-1][0]
    ct = [end if x is None else x for x in ct]
    dt = [end if x is None else x for x in dt]
    cl, dl = [], []
    for blk in Levenshtein.opcodes("".join(rc), final):
        if blk.tag != "equal":
            continue
        for a, b in zip(range(blk.src_start, blk.src_end), range(blk.dest_start, blk.dest_end)):
            cl.append(ct[b] - rt[a]); dl.append(min(dt[b], ct[b]) - rt[a])
    return cl, dl


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def main():
    sdir, odir, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
    names = sorted(f[:-5] for f in os.listdir(sdir) if f.endswith(".json"))
    print(f"{'stream':<16}{'tag':<14}{'err%':>7}{'sub':>6}{'del':>6}{'ins':>6}"
          f"{'busy':>7}{'cpu/aud':>8}{'rssMB':>7}{'commit p50/p90':>16}{'shown p50/p90':>15}")
    tot = {}
    for name in names:
        lang = name.split("__")[0]
        ref = json.load(open(f"{sdir}/{name}.json"))
        for tag in tags:
            hp = f"{odir}/{name}.{tag}.txt"
            if not os.path.exists(hp):
                continue
            ev = json.load(open(f"{odir}/{name}.{tag}.events.json"))
            segs = ref["segments"]
            ref_text = ref["ref"]
            if ev["audio_sec"] < segs[-1]["end"] - 1:      # --limit-sec run
                segs = [s for s in segs if s["end"] <= ev["audio_sec"]]
                ref_text = (" " if lang == "en" else "").join(s["text"] for s in segs)
            n, s, d, i = accuracy(lang, ref_text, open(hp).read())
            t = tot.setdefault((lang, tag), [0, 0, 0, 0])
            for k, v in enumerate((n, s, d, i)):
                t[k] += v
            m = re.search(r"Inference: (\d+) ms", open(f"{odir}/{name}.{tag}.stderr").read())
            busy = f"{int(m.group(1)) / 10 / ev['audio_sec']:6.0f}%" if m else f"{'?':>7}"
            lat = ""
            if ev["realtime"]:
                cl, dl = latency(segs, ev)
                lat = f"{pct(cl, .5):7.2f}/{pct(cl, .9):<6.2f}  {pct(dl, .5):6.2f}/{pct(dl, .9):<6.2f}"
            print(f"{name:<16}{tag:<14}{100 * (s + d + i) / n:7.2f}{s:6d}{d:6d}{i:6d}"
                  f"{busy}{ev['cpu_sec'] / ev['audio_sec']:8.2f}{ev['maxrss_mb']:7.0f}  {lat}")
    print()
    for (lang, tag), (n, s, d, i) in sorted(tot.items()):
        print(f"TOTAL {lang} {'WER' if lang == 'en' else 'CER'} {tag:<14}{100 * (s + d + i) / n:6.2f}%  "
              f"(sub {s} del {d} ins {i} / {n})")


if __name__ == "__main__":
    main()
