"""Timeline "Captures" box: when Vision Lab was actually capturing video, audio and the screen (read-only).

GET /api/timeline/captures?start=<epoch s>&end=<epoch s>   (default: today, local time)

Rows, each a list of [start, end] bars in epoch seconds, clipped to the window:
  video   Camera capture sessions. Recording mode logs real video_started / video_stopped events
          (vision_lab/recording.py); those spans are used as-is (a start with no stop yet runs to now).
          Before recording mode there was no start/stop log, so older sessions are derived from
          frame activity (keyframes plus scene / frame_probe / track events): points closer than
          VIDEO_GAP_S join into one bar. Privacy mode (capture_paused..capture_resumed) and camera pauses
          (camera_paused..camera_resumed) then cut the bars, so a privacy gap always shows as a gap.
  audio   Only moments where speech was actually captured and transcribed, never whole sessions:
          recording-mode transcript segments (voice_transcripts) and the user's voice-chat utterances
          (ts_start..ts_end, or AUDIO_MIN_S wide when there is no end). Room-sound events (AST classifier)
          are not used. Pluggable via AUDIO_SOURCES.
  screen  The screenrec segment index (chunks table; recordings without chunks but with an end time are
          used as a fallback). Segments closer than SCREEN_GAP_S merge into one session bar.
Never writes: vlab.db through the Store, the screenrec index opened with mode=ro."""
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

from .voicefilter import is_junk_transcript

router = APIRouter()
STORE = {"store": None}
VIDEO_GAP_S, SCREEN_GAP_S, AUDIO_MIN_S, SOUND_CLIP_S = 180.0, 15.0, 2.0, 2.0
ROOT = Path(__file__).resolve().parent.parent


def merge(bars, gap=0.0):
    """Sort and merge [a, b] bars whose gap is <= `gap` seconds."""
    out = []
    for a, b in sorted([float(a), float(max(a, b))] for a, b in bars):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def subtract(bars, cuts):
    """Remove every cut interval from the bars (both sorted/merged or not)."""
    cuts = merge(cuts)
    out = []
    for a, b in merge(bars):
        pieces = [[a, b]]
        for c, d in cuts:
            nxt = []
            for x, y in pieces:
                if d <= x or c >= y:
                    nxt.append([x, y]); continue
                if c > x: nxt.append([x, c])
                if d < y: nxt.append([d, y])
            pieces = nxt
        out += pieces
    return [p for p in out if p[1] > p[0]]


def clip(bars, t0, t1):
    return [[max(a, t0), min(b, t1)] for a, b in bars if b > t0 and a < t1 and min(b, t1) > max(a, t0)] \
        + [[a, a] for a, b in bars if a == b and t0 <= a < t1]


def paused_intervals(store, t0, t1, now=None):
    """[a, b] spans where capture was paused (privacy) or the camera was paused, overlapping [t0, t1]."""
    now = now or time.time()
    out = []
    for on, off in (("capture_paused", "capture_resumed"), ("camera_paused", "camera_resumed")):
        rows = store._query("SELECT ts, kind FROM events WHERE kind IN (?, ?) AND ts < ? ORDER BY ts", (on, off, t1))
        start = None
        for r in rows:
            if r["kind"] == on and start is None:
                start = r["ts"]
            elif r["kind"] == off and start is not None:
                if r["ts"] > t0:
                    out.append([start, r["ts"]])
                start = None
        if start is not None:            # still paused (or paused past the window)
            out.append([start, max(t1, now) if t1 > now else t1])
    return merge(out)


def recorded_video(store, t0, t1, now=None):
    """[start, stop] spans from recording mode's video_started / video_stopped events."""
    now = now or time.time()
    rows = store._query("SELECT ts, kind FROM events WHERE kind IN ('video_started', 'video_stopped') AND ts < ? "
                        "ORDER BY ts, id", (t1,))
    out, start = [], None
    for r in rows:
        if r["kind"] == "video_started":
            start = r["ts"] if start is None else start
        elif start is not None:
            if r["ts"] > t0:
                out.append([start, r["ts"]])
            start = None
    if start is not None:
        out.append([start, max(start, min(now, t1))])
    return out


def video_bars(store, t0, t1, gap=VIDEO_GAP_S):
    pts = [r["ts"] for r in store._query("SELECT ts FROM keyframes WHERE ts >= ? AND ts < ?", (t0 - gap, t1 + gap))]
    pts += [r["ts"] for r in store._query(
        "SELECT ts FROM events WHERE kind IN ('scene', 'frame_probe', 'track') AND ts >= ? AND ts < ?", (t0 - gap, t1 + gap))]
    bars = [b for b in merge([[p, p] for p in pts], gap) if b[1] > b[0]] + recorded_video(store, t0, t1)
    return clip(subtract(merge(bars), paused_intervals(store, t0, t1)), t0, t1)


def _user_utterances(store, t0, t1):
    rows = store._query("SELECT ts_start, ts_end FROM utterances WHERE role = 'user' AND text IS NOT NULL "
                        "AND TRIM(text) <> '' AND ts_start < ? AND COALESCE(ts_end, ts_start) >= ?", (t1, t0))
    return [[r["ts_start"], r["ts_end"] if r["ts_end"] and r["ts_end"] > r["ts_start"] else r["ts_start"] + AUDIO_MIN_S]
            for r in rows]


def _sound_events(store, t0, t1):
    rows = store._query("SELECT ts FROM events WHERE modality = 'audio' AND ts >= ? AND ts < ?", (t0, t1 + SOUND_CLIP_S))
    return [[r["ts"] - SOUND_CLIP_S, r["ts"]] for r in rows]


def _transcripts(store, t0, t1):
    """Recording-mode speech (voice_transcripts, written by voicetx after the audio is transcribed and deleted)."""
    if not store._query("SELECT 1 FROM sqlite_master WHERE type='table' AND name='voice_transcripts'"):
        return []
    rows = store._query("SELECT ts_start, ts_end, text, avg_logprob, no_speech_prob FROM voice_transcripts "
                        "WHERE ts_start < ? AND ts_end >= ?", (t1, t0))
    return [[r["ts_start"], max(r["ts_end"], r["ts_start"] + AUDIO_MIN_S)] for r in rows
            if not is_junk_transcript(r["text"], r)]          # Whisper noise is not speech (voicefilter)


AUDIO_SOURCES = {"utterances": _user_utterances, "transcripts": _transcripts}   # no room-sound events


def audio_bars(store, t0, t1):
    return clip(merge([b for f in AUDIO_SOURCES.values() for b in f(store, t0, t1)]), t0, t1)


def screenrec_index():
    if os.environ.get("VLAB_SCREENREC_INDEX"):
        return Path(os.environ["VLAB_SCREENREC_INDEX"])
    d = os.environ.get("VLAB_SCREENREC_DIR")
    if not d:
        try:
            cfg = json.loads((Path(os.environ.get("PLANLOOP_CONFIG") or ROOT / "data" / "planloop.json")).read_text())
            d = cfg.get("screenrec_dir")
        except (OSError, ValueError):
            d = None
    return Path(d or ROOT / "data" / "screenrec") / "index.db"


def screen_bars(t0, t1, index=None, gap=SCREEN_GAP_S):
    p = Path(index or screenrec_index())
    if not p.exists():
        return []
    try:
        db = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        try:
            bars = db.execute("SELECT start_ts, end_ts FROM chunks WHERE end_ts > ? AND start_ts < ?", (t0, t1)).fetchall()
            bars += db.execute("SELECT r.start_ts, r.end_ts FROM recordings r WHERE r.end_ts IS NOT NULL AND r.end_ts > ? "
                               "AND r.start_ts < ? AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.rec_id = r.rec_id)",
                               (t0, t1)).fetchall()
        finally:
            db.close()
    except sqlite3.Error:
        return []
    return clip(merge([list(b) for b in bars], gap), t0, t1)


def captures(store, start=None, end=None, index=None):
    if start is None:
        d = datetime.now()
        start = datetime(d.year, d.month, d.day).timestamp()
    end = end if end is not None and end > start else start + 86400
    rows = [("video", "Video", video_bars(store, start, end)),
            ("audio", "Audio", audio_bars(store, start, end)),
            ("screen", "Screen capture", screen_bars(start, end, index))]
    return {"start": start, "end": end, "privacy": clip(paused_intervals(store, start, end), start, end),
            "rows": [{"key": k, "label": l, "bars": b, "seconds": round(sum(y - x for x, y in b))} for k, l, b in rows]}


@router.get("/api/timeline/captures")
def timeline_captures(start: float | None = None, end: float | None = None):
    return captures(STORE["store"], start, end)
