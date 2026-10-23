"""vlab find: search narration transcripts for a section and (optionally) export it for Yap.

    vlab find "openclaw setup walkthrough"                 # ranked candidate sections, last 30 days
    vlab find "openclaw config" --since 2026-10-20 --json
    vlab find "openclaw setup walkthrough" --export 1      # write a Yap-ready folder for candidate #1

Read-only on vlab.db (opened with mode=ro; never through Store, which writes on open).

Matching (per transcript row): keyword overlap weighted by rarity, with light normalisation ("open-claw",
"open claw" and "OpenClaw" all match "openclaw"; set up/setting up/setup; walk through/walkthrough), fuzzy
token match (difflib) for Whisper misspellings, and, when the cached local MiniLM ONNX model is present, a
semantic score (all-MiniLM-L6-v2 via onnxruntime, offline). --semantic off disables it.

Section boundaries:
  start  the nearest spoken start cue ("I'm going to walk through...", "let me show...", "I've got to...",
         "ok, now...") at or before the first matching row, within --lookback seconds and with no pause longer
         than --pause between; otherwise the first matching row.
  end    the first of: a pause longer than --pause, an end cue ("that's it", "moving on", "ok now ..."), a new
         start cue that does not match the query, more than --topic-gap seconds without a matching row, or
         --max-len seconds.

Export (--export N): <out>/screen.mp4 (screenrec, trimmed, H.264), audio.m4a (retained narration placed on the
wall clock, silence where nothing was kept), clip.mp4 (both muxed: import this into Yap), frames/ (10 s
screenshots), transcript.srt + transcript.json (clip-relative, with word timings), manifest.json and
yap-steps.md (asset.import + transcript.import). Everything stays on this Mac (default ~/Datasets/vision-lab-clips).
"""
import argparse
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
STOP = set("a an the and or of to in on for with my i i'm im me we our is it this that be am are was were "
           "do doing how what where when which find section part bit where about some".split())
SYN = {"setup": ["setup", "set up", "setting up", "configure", "configuring", "configuration", "config", "install",
                 "installing"],
       "config": ["config", "configuration", "configure", "settings", "setup"],
       "walkthrough": ["walkthrough", "walk through", "walking through", "walk you through", "show", "demo", "tour"],
       "update": ["update", "updating", "change", "changing", "modify", "tweak"],
       "fix": ["fix", "fixing", "debug", "debugging", "repair"]}
START_CUES = re.compile(
    r"\b(i'?m going to|i am going to|i'?m gonna|gonna|let me|let'?s|i'?ve got to|i have to|i need to|i want to|"
    r"next,? i'?m|now i'?m|okay,? so|ok,? so|ok,? now|okay,? now|alright,? so|all right,? so|"
    r"so now|first,? i|we'?re going to)\b", re.I)
END_CUES = re.compile(r"\b(that'?s it|that'?s all|that'?s done|and done|we'?re done|i'?m done|moving on|"
                      r"that'?s the (setup|walkthrough|demo)|wrap(ping)? up)\b", re.I)


# ---------------------------------------------------------------- text
def norm(s):
    s = (s or "").lower().replace("’", "'")
    s = re.sub(r"[-_/.]+", " ", s)
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def compact(s):
    return norm(s).replace(" ", "").replace("'", "")


def stem(t):
    for suf in ("ations", "ation", "ings", "ing", "ed", "es", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def query_terms(q):
    """[(term, [variants])]: stopwords dropped; 'set up'/'walk through' style pairs folded into one term."""
    # "open-claw" / "open_claw" in a query is one name: fold to "openclaw" (text matching still finds "Open Claw")
    q = re.sub(r"\b([A-Za-z0-9]+)(?:[-_]([A-Za-z0-9]+))+\b",
               lambda m: m.group(0) if m.group(0).lower().replace("-", " ").replace("_", " ") in
               {v for vs in SYN.values() for v in vs} else re.sub(r"[-_]", "", m.group(0)), q)
    n = norm(q)
    for k, vs in SYN.items():
        for v in vs:
            if " " in v:
                n = re.sub(rf"\b{v}\b", k, n)
    terms = []
    for t in n.split():
        if t in STOP or len(t) < 2:
            continue
        key = next((k for k, vs in SYN.items() if t == k or t in vs), None)
        terms.append((t, sorted(set([t] + (SYN[key] if key else [])))))
    return terms


def _fuzzy_hit(variant, toks, comp):
    v = variant.replace(" ", "")
    if len(v) >= 5 and v in comp:
        return 1.0                                   # "open claw" / "open-claw" -> "openclaw"
    best = 0.0
    sv = stem(v)
    for t in toks:
        if t == variant or stem(t) == sv:
            return 1.0
        if len(v) >= 5 and len(t) >= 4 and abs(len(t) - len(v)) <= 3:
            best = max(best, SequenceMatcher(None, v, t).ratio())
    # neighbouring token pairs, for Whisper splitting a word ("open clock" vs "openclaw")
    for a, b in zip(toks, toks[1:]):
        if len(v) >= 6:
            best = max(best, SequenceMatcher(None, v, a + b).ratio())
    return 1.0 if best >= 0.86 else (0.6 if best >= 0.78 else 0.0)


def keyword_score(terms, text, weights):
    if not terms:
        return 0.0, []
    n = norm(text)
    toks, comp = n.split(), compact(text)
    phrase_n = " " + n + " "
    got, total, hit = 0.0, 0.0, []
    for t, variants in terms:
        w = weights.get(t, 1.0)
        total += w
        s = 0.0
        for v in variants:
            if " " in v and f" {v} " in phrase_n:
                s = 1.0
            else:
                s = max(s, _fuzzy_hit(v, toks, comp))
            if s == 1.0:
                break
        if s:
            hit.append(t)
        got += w * s
    return (got / total if total else 0.0), hit


def term_weights(terms, texts):
    """Rarer query terms count more (log-idf over the searched rows)."""
    N = max(len(texts), 1)
    out = {}
    comps = [(norm(x).split(), compact(x)) for x in texts]
    for t, variants in terms:
        df = sum(1 for toks, comp in comps if any(_fuzzy_hit(v, toks, comp) for v in variants if " " not in v))
        out[t] = 1.0 + math.log((N + 1) / (df + 1))
    return out


# ---------------------------------------------------------------- semantic (optional, local only)
class Embedder:
    MODEL = "models--sentence-transformers--all-MiniLM-L6-v2"

    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub" / self.MODEL / "snapshots"
        snap = next(p for p in sorted(hub.iterdir()) if (p / "onnx" / "model.onnx").exists())
        self.tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
        self.tok.enable_truncation(256)
        self.tok.enable_padding()
        self.sess = ort.InferenceSession(str(snap / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def __call__(self, texts, batch=64):
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[i:i + batch])
            ids = np.array([e.ids for e in enc], np.int64)
            mask = np.array([e.attention_mask for e in enc], np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.vstack(out) if out else np.zeros((0, 384), np.float32)


def get_embedder(mode="auto"):
    if mode == "off":
        return None
    try:
        return Embedder()
    except Exception as e:
        if mode == "on":
            raise
        return None


# ---------------------------------------------------------------- data (read-only)
def connect_ro(path):
    db = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def default_db():
    return os.environ.get("VLAB_DB") or str(ROOT / "data" / "vlab.db")


def _has(db, table):
    return bool(db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone())


def load_rows(db, t0, t1):
    if not _has(db, "voice_transcripts"):
        return []
    cols = {r[1] for r in db.execute("PRAGMA table_info(voice_transcripts)")}
    extra = ", ".join(c for c in ("audio_segment_id", "audio_offset_s", "words") if c in cols)
    q = (f"SELECT id, ts_start, ts_end, text, avg_logprob, no_speech_prob{', ' + extra if extra else ''} "
         "FROM voice_transcripts WHERE ts_start >= ? AND ts_start < ? AND TRIM(COALESCE(text,'')) <> '' ORDER BY ts_start")
    try:
        from .voicefilter import is_junk_transcript
    except Exception:
        is_junk_transcript = lambda t, r: False
    return [dict(r) for r in db.execute(q, (t0, t1)) if not is_junk_transcript(r["text"], dict(r))]


# ---------------------------------------------------------------- sections
def score_rows(query, rows, embed=None):
    terms = query_terms(query)
    texts = [r["text"] for r in rows]
    w = term_weights(terms, texts)
    sem = None
    if embed is not None and rows:
        ctx = [" ".join(texts[max(0, i - 1): i + 2]) for i in range(len(texts))]   # a row plus its neighbours
        e = embed([query] + ctx)
        sem = (e[1:] @ e[0]).astype(float)
    out = []
    for i, r in enumerate(rows):
        kw, hit = keyword_score(terms, r["text"], w)
        s = kw if sem is None else 0.65 * kw + 0.35 * max(0.0, float(sem[i]))
        out.append({"kw": round(kw, 3), "sem": None if sem is None else round(float(sem[i]), 3),
                    "score": round(s, 3), "terms": hit})
    return out


def is_hit(sc, min_kw=0.34, min_sem=0.5):
    return sc["kw"] >= min_kw or (sc["sem"] is not None and sc["sem"] >= min_sem)


def find_sections(query, rows, embed=None, pause_s=120.0, lookback_s=600.0, topic_gap_s=300.0, max_len_s=2700.0,
                  min_kw=0.34, min_sem=0.5, limit=5):
    """rows sorted by ts_start. Returns candidate sections, best first."""
    if not rows:
        return []
    scores = score_rows(query, rows, embed)
    hits = [i for i, s in enumerate(scores) if is_hit(s, min_kw, min_sem)]
    sections, used = [], set()
    for i in sorted(hits, key=lambda k: -scores[k]["score"]):
        if i in used:
            continue
        # start: nearest start cue at/before i, same run (no long pause), within lookback
        st = i
        j = i
        while j >= 0 and rows[i]["ts_start"] - rows[j]["ts_start"] <= min(lookback_s, max_len_s):
            if START_CUES.search(rows[j]["text"]) and (is_hit(scores[j], min_kw * 0.5, min_sem * 0.85) or j == i
                                                        or j >= i - 3):
                st = j
                break
            if j > 0 and rows[j]["ts_start"] - rows[j - 1]["ts_end"] > pause_s:
                break
            j -= 1
        # end
        en, last_hit_end = i, rows[i]["ts_end"]
        natural = True                                             # ended by a pause, an end cue or the data
        k = i + 1
        while k < len(rows):
            r, prev = rows[k], rows[k - 1]
            if r["ts_start"] - prev["ts_end"] > pause_s:
                break
            if r["ts_end"] - rows[st]["ts_start"] > max_len_s:
                natural = False; break
            h = is_hit(scores[k], min_kw, min_sem)
            if not h and START_CUES.search(r["text"]) and r["ts_start"] - last_hit_end > 30:
                break                                              # a new, unrelated "ok, now I'm going to..."
            if not h and r["ts_start"] - last_hit_end > topic_gap_s:
                natural = False; break
            en = k
            if h:
                last_hit_end = r["ts_end"]
            if END_CUES.search(r["text"]):
                break
            k += 1
        # drifted off topic (no natural boundary): trim trailing non-matching rows past the last hit + a short tail
        while not natural and en > i and not is_hit(scores[en], min_kw, min_sem) and rows[en]["ts_start"] - last_hit_end > 60 \
                and not END_CUES.search(rows[en]["text"]):
            en -= 1
        span = range(st, en + 1)
        used.update(span)
        sec_hits = [x for x in span if is_hit(scores[x], min_kw, min_sem)]
        cue = START_CUES.search(rows[st]["text"]) is not None
        best = max(scores[x]["score"] for x in sec_hits)
        conf = best + 0.05 * min(len(sec_hits), 6) + (0.15 if cue else 0.0)
        sections.append({"start": rows[st]["ts_start"], "end": rows[en]["ts_end"], "score": round(conf, 3),
                         "start_row": st, "end_row": en, "start_cue": cue, "hits": sec_hits,
                         "rows": list(span)})
    # merge overlaps
    sections.sort(key=lambda s: s["start"])
    merged = []
    for s in sections:
        if merged and s["start"] <= merged[-1]["end"] + 5 and max(s["end"], merged[-1]["end"]) - merged[-1]["start"] <= max_len_s:
            m = merged[-1]
            m["end"] = max(m["end"], s["end"]); m["score"] = max(m["score"], s["score"])
            m["hits"] = sorted(set(m["hits"]) | set(s["hits"])); m["rows"] = sorted(set(m["rows"]) | set(s["rows"]))
            m["start_cue"] = m["start_cue"] or s["start_cue"]; m["end_row"] = max(m["end_row"], s["end_row"])
        else:
            merged.append(dict(s))
    merged.sort(key=lambda s: -s["score"])
    for s in merged:
        s["scores"] = {x: scores[x] for x in s["hits"]}
    return merged[:limit]


# ---------------------------------------------------------------- screen + audio linkage (time joins)
def screens_dir():
    return Path(os.environ.get("VLAB_SCREENS_DIR") or ROOT / "data" / "screens")


def screenrec_dir():
    v = os.environ.get("VLAB_SCREENREC_DIR")
    if v:
        return Path(v)
    try:
        cfg = json.loads(Path(os.environ.get("PLANLOOP_CONFIG") or ROOT / "data" / "planloop.json").read_text())
        if cfg.get("screenrec_dir"):
            return Path(cfg["screenrec_dir"])
    except (OSError, ValueError):
        pass
    return ROOT / "data" / "screenrec"


def frames_near(ts, within=30.0, root=None):
    """Nearest 10 s planner screenshot (data/screens/YYYY-MM-DD/HHMMSS.jpg) to `ts`, or None."""
    root = Path(root or screens_dir())
    best = None
    for day in {datetime.fromtimestamp(ts - within).date(), datetime.fromtimestamp(ts + within).date()}:
        d = root / day.strftime("%Y-%m-%d")
        if not d.is_dir():
            continue
        for f in d.glob("*.jpg"):
            if not re.fullmatch(r"\d{6}", f.stem):
                continue
            t = datetime.combine(day, datetime.strptime(f.stem, "%H%M%S").time()).timestamp()
            if abs(t - ts) <= within and (best is None or abs(t - ts) < best[0]):
                best = (abs(t - ts), f)
    return str(best[1]) if best else None


def frames_between(t0, t1, every_s=10.0, root=None):
    root = Path(root or screens_dir())
    out = []
    day = datetime.fromtimestamp(t0).date()
    while datetime.combine(day, datetime.min.time()).timestamp() < t1:
        d = root / day.strftime("%Y-%m-%d")
        if d.is_dir():
            for f in sorted(d.glob("*.jpg")):
                if re.fullmatch(r"\d{6}", f.stem):
                    t = datetime.combine(day, datetime.strptime(f.stem, "%H%M%S").time()).timestamp()
                    if t0 <= t < t1:
                        out.append((t, f))
        day += timedelta(days=1)
    keep, last = [], -1e18
    for t, f in out:
        if t - last >= every_s - 0.5:
            keep.append((t, f)); last = t
    return keep


def screen_chunks(t0, t1, root=None):
    idx = Path(root or screenrec_dir()) / "index.db"
    if not idx.exists():
        return []
    db = connect_ro(idx)
    try:
        return [dict(r) for r in db.execute("SELECT rec_id, path, start_ts, end_ts FROM chunks WHERE end_ts > ? AND "
                                            "start_ts < ? ORDER BY start_ts", (t0, t1))]
    finally:
        db.close()


def screen_at(ts, root=None):
    """{path, offset_s} of the screen recording covering `ts`, or None."""
    c = screen_chunks(ts, ts + 0.001, root)
    return {"path": c[0]["path"], "offset_s": round(ts - c[0]["start_ts"], 2)} if c else None


def audio_segments(db, t0, t1):
    if not _has(db, "audio_segments"):
        return []
    return [dict(r) for r in db.execute("SELECT * FROM audio_segments WHERE deleted_at IS NULL AND ts_end > ? AND "
                                        "ts_start < ? ORDER BY ts_start", (t0, t1))]


# ---------------------------------------------------------------- presentation
def tz_label(ts):
    return datetime.fromtimestamp(ts).astimezone().strftime("%Z")


def fmt(ts):
    d = datetime.fromtimestamp(ts)
    return d.strftime("%a %b %-d, %Y %-I:%M:%S %p ") + tz_label(ts)


def describe(sec, rows, db=None, quotes=3):
    t0, t1 = sec["start"], sec["end"]
    qs = sorted(sec["hits"], key=lambda x: -sec["scores"][x]["score"])[:quotes]
    cue = rows[sec["start_row"]]["text"].strip() if sec["start_cue"] else None
    segs = audio_segments(db, t0, t1) if db is not None else []
    covered = sum(min(s["ts_end"], t1) - max(s["ts_start"], t0) for s in segs)
    return {"start": t0, "end": t1, "start_local": fmt(t0), "end_local": fmt(t1),
            "duration_s": round(t1 - t0, 1), "score": sec["score"],
            "start_cue": cue[:200] if cue else None,
            "quotes": [{"ts": rows[x]["ts_start"], "local": fmt(rows[x]["ts_start"]), "text": rows[x]["text"].strip()[:200],
                        "match": sec["scores"][x]} for x in sorted(qs, key=lambda x: rows[x]["ts_start"])],
            "frames": {"before_start": frames_near(t0 - 15), "start": frames_near(t0), "end": frames_near(t1),
                       "after_end": frames_near(t1 + 15)},
            "screen": {"start": screen_at(t0), "end": screen_at(t1)},
            "audio": {"segments": len(segs), "coverage": round(covered / max(t1 - t0, 1e-9), 2)}}


def print_human(query, found):
    if not found:
        print(f"No sections found for {query!r}."); return
    for n, s in enumerate(found, 1):
        print(f"#{n}  {s['start_local']}  ->  {datetime.fromtimestamp(s['end']).strftime('%-I:%M:%S %p')} "
              f"{tz_label(s['end'])}   ({s['duration_s'] / 60:.1f} min, score {s['score']})")
        if s["start_cue"]:
            print(f"    starts: \"{s['start_cue']}\"")
        for q in s["quotes"]:
            print(f"    {datetime.fromtimestamp(q['ts']).strftime('%-I:%M:%S %p')}  \"{q['text']}\"")
        for k, v in s["frames"].items():
            if v:
                print(f"    frame {k:<12} {v}")
        sc = s["screen"]["start"]
        if sc:
            print(f"    screen video  {sc['path']} @ {sc['offset_s']} s")
        print(f"    audio kept    {s['audio']['segments']} segment(s), {s['audio']['coverage'] * 100:.0f}% of the range")
        print()


# ---------------------------------------------------------------- export
def _ff():
    from .audiostore import ffmpeg_bin
    return ffmpeg_bin()


def build_audio(segs, t0, t1, dest):
    """Place each retained chunk at its wall-clock start (silence elsewhere) -> AAC m4a. Returns seconds of audio."""
    from . import audiostore
    n = int(round((t1 - t0) * SR))
    buf = np.zeros(n, np.float32)
    placed = 0.0
    for s in segs:
        p = Path(s["path"])
        if not p.exists():
            continue
        out = subprocess.run([_ff(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(p), "-ac", "1",
                              "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, timeout=120)
        a = np.frombuffer(out.stdout, np.float32)
        off = int(round((s["ts_start"] - t0) * SR))
        lo, hi = max(off, 0), min(off + len(a), n)
        if hi > lo:
            buf[lo:hi] = a[lo - off: hi - off]
            placed += (hi - lo) / SR
    audiostore.encode(buf, dest)
    return placed


def _fps(path):
    try:
        out = subprocess.run([str(Path(_ff()).with_name("ffprobe")) if Path(_ff()).with_name("ffprobe").exists() else
                              "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=30).stdout.strip()
        n, d = out.split("/")
        f = float(n) / float(d)
        return round(f, 3) if 0 < f <= 120 else 30
    except Exception:
        return 30


def build_screen(chunks, t0, t1, dest, tmpdir):
    """screenrec chunks -> one trimmed H.264 file (VideoToolbox) on the wall clock: a gap between chunks holds the last
    frame, a late start is black. Gaps are reported in the manifest."""
    if not chunks:
        return None, []
    lst = Path(tmpdir) / "concat.txt"
    # each entry lasts until the next chunk starts on the wall clock, so a gap holds the last frame instead of
    # pulling later footage earlier (keeps the screen in sync with the audio and transcript)
    lines = []
    for a, b in zip(chunks, chunks[1:] + [None]):
        lines.append(f"file '{a['path']}'\n" + (f"duration {b['start_ts'] - a['start_ts']:.3f}\n" if b else ""))
    lst.write_text("".join(lines))
    gaps = [{"after": a["end_ts"], "gap_s": round(b["start_ts"] - a["end_ts"], 2)}
            for a, b in zip(chunks, chunks[1:]) if b["start_ts"] - a["end_ts"] > 1.0]
    ss = t0 - chunks[0]["start_ts"]
    lead = max(0.0, -ss)                                  # recording started after t0: pad the front
    fps = _fps(chunks[0]["path"])
    vf = f"fps={fps}" + (f",tpad=start_duration={lead:.3f}:start_mode=add:color=black" if lead > 0 else "")
    cmd = [_ff(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
           "-ss", f"{max(0.0, ss):.3f}", "-an", "-vf", vf, "-t", f"{t1 - t0:.3f}", "-c:v", "h264_videotoolbox",
           "-b:v", "6M", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)]
    p = subprocess.run(cmd, capture_output=True, timeout=1800)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-400:])
    return str(dest), gaps


def srt_time(s):
    s = max(0.0, s); h, r = divmod(int(s * 1000), 3600000); m, r = divmod(r, 60000); sec, ms = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def export(sec, rows, db, out, pad_s=5.0, query=""):
    t0, t1 = sec["start"] - pad_s, sec["end"] + pad_s
    out = Path(out); (out / "frames").mkdir(parents=True, exist_ok=True)
    tmp = out / ".tmp"; tmp.mkdir(exist_ok=True)
    man = {"query": query, "start": t0, "end": t1, "start_local": fmt(t0), "end_local": fmt(t1),
           "pad_s": pad_s, "created": time.time(), "files": {}}
    segs = audio_segments(db, t0, t1)
    if segs:
        man["audio_seconds"] = round(build_audio(segs, t0, t1, out / "audio.m4a"), 1)
        man["files"]["audio"] = "audio.m4a"
    chunks = screen_chunks(t0, t1)
    try:
        v, gaps = build_screen(chunks, t0, t1, out / "screen.mp4", tmp)
        if v:
            man["files"]["screen"] = "screen.mp4"; man["screen_gaps"] = gaps
    except Exception as e:
        man["screen_error"] = str(e)[:300]
    if "audio" in man["files"] and "screen" in man["files"]:
        p = subprocess.run([_ff(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(out / "screen.mp4"),
                            "-i", str(out / "audio.m4a"), "-map", "0:v", "-map", "1:a", "-c", "copy", "-movflags",
                            "+faststart", str(out / "clip.mp4")], capture_output=True, timeout=600)
        if p.returncode == 0:
            man["files"]["clip"] = "clip.mp4"
    frames = frames_between(t0, t1)
    for t, f in frames:
        shutil.copy2(f, out / "frames" / f"{int(round(t - t0)):05d}s-{datetime.fromtimestamp(t).strftime('%H%M%S')}.jpg")
    man["files"]["frames"] = f"frames/ ({len(frames)})"
    span = [rows[i] for i in range(len(rows)) if rows[i]["ts_end"] > t0 and rows[i]["ts_start"] < t1]
    with open(out / "transcript.srt", "w") as f:
        for n, r in enumerate(span, 1):
            f.write(f"{n}\n{srt_time(r['ts_start'] - t0)} --> {srt_time(r['ts_end'] - t0)}\n{r['text'].strip()}\n\n")
    tj = []
    for r in span:
        words = json.loads(r["words"]) if r.get("words") else []
        tj.append({"start": round(r["ts_start"] - t0, 3), "end": round(r["ts_end"] - t0, 3), "text": r["text"].strip(),
                   "words": [{"start": round(w[0] - t0, 3), "end": round(w[1] - t0, 3), "word": w[2]} for w in words]})
    (out / "transcript.json").write_text(json.dumps(tj, indent=1))
    man["files"]["transcript"] = ["transcript.srt", "transcript.json"]
    man["note"] = ("Audio is placed on the wall clock per kept chunk; the mic delivers ~89% of nominal samples"
                   ", so speech inside a chunk can lead the screen by up to ~2 s.")
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    clip = out / ("clip.mp4" if "clip" in man["files"] else "screen.mp4")
    (out / "yap-steps.md").write_text(
        "# Yap import (local)\n\n"
        f"echo '{{\"requestId\":\"import-{out.name}\",\"path\":\"{clip}\"}}' | ~/.local/bin/yap asset.import --params - --wait\n\n"
        "Then attach the transcript to the clip's audio stream (assetId/streamId from the import receipt):\n\n"
        f"echo '{{\"requestId\":\"tx-{out.name}\",\"assetId\":\"<assetId>\",\"streamId\":\"<audio streamId>\","
        f"\"path\":\"{out / 'transcript.srt'}\",\"format\":\"srt\"}}' | ~/.local/bin/yap transcript.import --params - --wait\n")
    shutil.rmtree(tmp, ignore_errors=True)
    return man


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "section"


# ---------------------------------------------------------------- CLI
def _ts(s, end=False):
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        d = datetime.fromisoformat(s)
        if end and len(s) <= 10:
            d += timedelta(days=1)
        return d.timestamp()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vlab find", description=__doc__.splitlines()[0])
    ap.add_argument("query")
    ap.add_argument("--days", type=float, default=30.0, help="search the last N days (default 30)")
    ap.add_argument("--since"); ap.add_argument("--until")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--semantic", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--pause", type=float, default=120.0, help="silence (s) that ends a section")
    ap.add_argument("--lookback", type=float, default=600.0)
    ap.add_argument("--topic-gap", type=float, default=300.0)
    ap.add_argument("--max-len", type=float, default=2700.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--export", type=int, metavar="N", help="export candidate #N as a Yap-ready folder")
    ap.add_argument("--out", help="export folder (default ~/Datasets/vision-lab-clips/<start>-<query>)")
    ap.add_argument("--pad", type=float, default=5.0)
    ap.add_argument("--db", default=None)
    a = ap.parse_args(argv)
    now = time.time()
    t1 = _ts(a.until, end=True) or now + 1
    t0 = _ts(a.since) or (t1 - a.days * 86400)
    db = connect_ro(a.db or default_db())
    rows = load_rows(db, t0, t1)
    embed = get_embedder(a.semantic)
    secs = find_sections(a.query, rows, embed, a.pause, a.lookback, a.topic_gap, a.max_len, limit=a.limit)
    found = [describe(s, rows, db) for s in secs]
    res = {"query": a.query, "searched": {"from": fmt(t0), "to": fmt(min(t1, now)), "rows": len(rows),
                                          "semantic": embed is not None}, "sections": found}
    if a.export:
        if not 1 <= a.export <= len(secs):
            print(f"--export {a.export}: only {len(secs)} candidate(s)", file=sys.stderr); return 2
        s = secs[a.export - 1]
        out = Path(a.out or Path.home() / "Datasets" / "vision-lab-clips" /
                   f"{datetime.fromtimestamp(s['start']).strftime('%Y%m%d-%H%M')}-{slug(a.query)}").expanduser()
        res["export"] = {"dir": str(out), **export(s, rows, db, out, a.pad, a.query)}
    if a.json:
        print(json.dumps(res, indent=1, default=str))
    else:
        print(f"Searched {res['searched']['rows']} transcript rows, {res['searched']['from']} to {res['searched']['to']}"
              f"{' (keyword + fuzzy + local semantic)' if embed else ' (keyword + fuzzy)'}\n")
        print_human(a.query, found)
        if a.export:
            print(f"Exported to {res['export']['dir']}: {', '.join(str(v) for v in res['export']['files'].values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
