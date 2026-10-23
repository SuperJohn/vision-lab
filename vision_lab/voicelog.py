"""Voice log: every captured voice item as text, grouped by local day (read-only).

GET /api/voice/log?days=7&before=<epoch s>&replies=1&limit=2000

Pagination is by day: a page holds the `days` most recent local calendar days that have entries
and start before `before` (default: now). `next_before` is the start of the oldest day returned;
pass it as `before` to get the next page, or it is null when nothing older exists.
Days are newest first, and entries inside a day are newest first too.

Sources are pluggable (SOURCES). Each one is fn(store, t0, t1) -> [entry] where an entry is
    {"id": "<source>:<row id>", "ts": float, "ts_end": float|None, "kind": "speech"|"reply"|"sound"|"transcript",
     "source": str, "text": str, "label": str|None, "confidence": float|None, "interrupted": bool}
Built in: `utterances` (voice chat turns: user = speech, assistant = reply) and `transcripts`
(recording-mode mic audio transcribed by mlx-whisper; the audio is kept, audiostore.py). Junk transcript rows
(Whisper noise/hallucinations, vision_lab/voicefilter.py) are hidden here but never deleted.
Room sounds (AST classifier, audio `events`) are NOT shown: the log is only what was said.
Old sound rows stay in the DB untouched; `sounds=1` still returns them for debugging, off by default.

Nothing here writes to the DB, starts capture, or touches the mic; Privacy mode only stops new capture,
so the log shows exactly what was already stored."""
import json
import time
from datetime import datetime, timedelta

from fastapi import APIRouter, Query

from .voicefilter import is_junk_transcript

router = APIRouter()
STORE = {"store": None}
CAPTURE = {"state": None}   # set by the server: () -> {"privacy": bool, "mic_enabled": bool}
MAX_DAYS, MAX_LIMIT = 60, 5000


def _utterances(store, t0, t1):
    rows = store._query(
        "SELECT id, turn_id, role, text, ts_start, ts_end, interrupted, error FROM utterances "
        "WHERE ts_start >= ? AND ts_start < ? AND text IS NOT NULL AND TRIM(text) <> ''", (t0, t1))
    return [{"id": f"utterance:{r['id']}", "ts": r["ts_start"], "ts_end": r["ts_end"],
             "kind": "speech" if r["role"] == "user" else "reply", "source": "voice chat",
             "text": r["text"].strip(), "label": None, "confidence": None,
             "interrupted": bool(r["interrupted"]), "turn_id": r["turn_id"]} for r in rows]


def _sounds(store, t0, t1):
    rows = store._query(
        "SELECT id, ts, cls, text, data FROM events WHERE modality = 'audio' AND ts >= ? AND ts < ?", (t0, t1))
    out = []
    for r in rows:
        try:
            d = json.loads(r["data"]) if r["data"] else {}
        except ValueError:
            d = {}
        out.append({"id": f"event:{r['id']}", "ts": r["ts"], "ts_end": None, "kind": "sound", "source": "room sound",
                    "text": d.get("audioset_label") or r["text"], "label": r["cls"] or d.get("label"),
                    "confidence": d.get("confidence"), "interrupted": False})
    return out


def _has_transcripts(store):
    return bool(store._query("SELECT 1 FROM sqlite_master WHERE type='table' AND name='voice_transcripts'"))


def _transcripts(store, t0, t1):
    """Recording-mode voice: Whisper text from captured mic audio (audio retained)."""
    if not _has_transcripts(store):
        return []
    rows = store._query("SELECT id, ts_start, ts_end, text, avg_logprob, no_speech_prob, source FROM voice_transcripts "
                        "WHERE ts_start >= ? AND ts_start < ? AND TRIM(COALESCE(text,'')) <> ''", (t0, t1))
    rows = [r for r in rows if not is_junk_transcript(r["text"], r)]   # hide old junk; rows stay in the DB
    return [{"id": f"transcript:{r['id']}", "ts": r["ts_start"], "ts_end": r["ts_end"], "kind": "transcript",
             "source": r["source"] or "recording mode", "text": r["text"].strip(), "label": None,
             "confidence": None, "interrupted": False} for r in rows]


SOURCES = {"utterances": _utterances, "transcripts": _transcripts}
OPTIONAL = {"sounds": _sounds}          # opt-in only (sounds=1); never in the default output
CAPTURED = {"speech", "sound", "transcript"}     # what a day's count means (replies are not captured voice)


def _day_start(ts):
    d = datetime.fromtimestamp(ts)
    return datetime(d.year, d.month, d.day).timestamp()


def _day_bounds(ts):
    start = datetime.fromtimestamp(_day_start(ts))
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def _latest_before(store, before, sources):
    """Newest entry ts strictly before `before` across sources (cheap MAX queries)."""
    qs = {"utterances": "SELECT MAX(ts_start) FROM utterances WHERE ts_start < ? AND text IS NOT NULL AND TRIM(text) <> ''",
          "sounds": "SELECT MAX(ts) FROM events WHERE modality = 'audio' AND ts < ?",
          "transcripts": "SELECT MAX(ts_start) FROM voice_transcripts WHERE ts_start < ?"}
    best = None
    for name in sources:
        if name == "transcripts" and not _has_transcripts(store):
            continue
        if name in qs:
            v = list(store._query(qs[name], (before,))[0].values())[0]
        else:   # plug-in source without a fast path: scan back up to a year
            got = _src(name)(store, before - 366 * 86400, before)
            v = max((e["ts"] for e in got), default=None)
        if v is not None and (best is None or v > best):
            best = v
    return best


def _src(name):
    return SOURCES.get(name) or OPTIONAL[name]   # looked up live, so plug-in sources work


def build_log(store, days=7, before=None, replies=True, sounds=False, limit=2000):
    days = max(1, min(int(days), MAX_DAYS))
    limit = max(1, min(int(limit), MAX_LIMIT))
    before = float(before) if before else time.time() + 1
    names = list(SOURCES) + (["sounds"] if sounds else [])
    out_days, cursor, total = [], before, 0
    while len(out_days) < days and total < limit:
        last = _latest_before(store, cursor, names)
        if last is None:
            cursor = None
            break
        d0, d1 = _day_bounds(last)
        hi = min(d1, cursor)
        entries = [e for n in names for e in _src(n)(store, d0, hi)]
        if not replies:
            entries = [e for e in entries if e["kind"] != "reply"]
        entries.sort(key=lambda e: (e["ts"], e["id"]), reverse=True)
        entries = entries[: limit - total]
        cursor = d0
        if not entries:
            continue
        cap = [e for e in entries if e["kind"] in CAPTURED]
        span = cap or entries
        dt = datetime.fromtimestamp(d0)
        out_days.append({"date": dt.strftime("%Y-%m-%d"), "label": dt.strftime("%A, %B %-d, %Y"),
                         "day_start": d0, "count": len(cap), "replies": len(entries) - len(cap),
                         "first_ts": min(e["ts"] for e in span), "last_ts": max(e["ts"] for e in span),
                         "today": d0 == _day_start(time.time()), "entries": entries})
        total += len(entries)
    more = cursor is not None and _latest_before(store, cursor, names) is not None
    cap = CAPTURE["state"]() if CAPTURE["state"] else {}
    return {"days": out_days, "next_before": cursor if more else None, "order": "newest first",
            "sources": names, "capture": {"recording_mode": False, **cap}}


@router.get("/api/voice/log")
def voice_log(days: int = Query(7, ge=1, le=MAX_DAYS), before: float | None = None, replies: bool = True,
              sounds: bool = False, limit: int = Query(2000, ge=1, le=MAX_LIMIT)):
    return build_log(STORE["store"], days, before, replies, sounds, limit)
