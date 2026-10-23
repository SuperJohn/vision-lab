"""Retained mic audio: recording-mode chunks are kept as compressed audio, linked in vlab.db.

Keep the raw narration audio so demo sections can be found from what was said and cut in Yap.
This replaces the 10/20 rule "mic audio is transcribed, then deleted" (retention.py). The old behaviour is
still one setting away (VLAB_AUDIO_RETAIN=0).

Settings (environment; run.sh / the app pass them through to the server):
  VLAB_AUDIO_RETAIN     default on. "0", "off", "no", "false" = old behaviour (delete after transcription).
  VLAB_AUDIO_DIR        default ~/Datasets/vision-lab-audio. Files go to <dir>/YYYY/MM/DD/voice-<start ms>.m4a.
                        Local disk, outside the repo and outside iCloud (Desktop/Documents sync is not used).
  VLAB_AUDIO_BITRATE    default 32k. AAC-LC, mono, 16 kHz: ~14.4 MB per recorded hour, <=0.35 GB per 24 h day.
  VLAB_AUDIO_KEEP_DAYS  default keep forever. A positive number deletes archived files older than that many days
                        (same rules as VLAB_VIDEO_KEEP_DAYS; unset/0/none/forever/keep = keep everything).

DB (vlab.db, additive only; back up the live DB before the first run):
  audio_segments(id, ts_start, ts_end, duration_s, path UNIQUE, codec, bitrate, sample_rate, bytes, source,
                 created_at, deleted_at)
      ts_start = wall clock when the chunk's first audio arrived; ts_end = wall clock when it closed (falls back
      to ts_start + duration_s). duration_s = audio actually in the file. The two can differ: the mic delivers
      ~89% of nominal samples (drift), so a 15 s wall chunk holds ~13.3 s of audio.
  voice_transcripts += audio_segment_id, audio_offset_s (seconds into that file), words (JSON
      [[ts_start, ts_end, word, probability], ...] in epoch seconds).
  The screen timeline (screenrec index.db chunks, data/screens/YYYY-MM-DD/HHMMSS.jpg, batch_annotations) uses
  the same epoch clock, so linking is a time join (vision_lab/find.py: screen_at / frames_near).
"""
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

FFMPEG = os.environ.get("VLAB_FFMPEG", "/opt/homebrew/bin/ffmpeg")
SR = 16000
OFF = ("0", "off", "no", "false")
FOREVER = ("", "0", "none", "forever", "keep")
AUDIO_DDL = """CREATE TABLE IF NOT EXISTS audio_segments (
  id INTEGER PRIMARY KEY, ts_start REAL NOT NULL, ts_end REAL NOT NULL, duration_s REAL, path TEXT UNIQUE,
  codec TEXT, bitrate INTEGER, sample_rate INTEGER, bytes INTEGER, source TEXT, created_at REAL, deleted_at REAL);
CREATE INDEX IF NOT EXISTS audio_segments_ts ON audio_segments(ts_start, ts_end);"""
TX_COLUMNS = {"audio_segment_id": "INTEGER", "audio_offset_s": "REAL", "words": "TEXT"}
MIGRATION = "hou189_audio_retain_v1"


def ffmpeg_bin():
    return FFMPEG if Path(FFMPEG).exists() else "ffmpeg"


def retain_enabled():
    return (os.environ.get("VLAB_AUDIO_RETAIN") or "1").strip().lower() not in OFF


def audio_dir():
    return Path(os.environ.get("VLAB_AUDIO_DIR") or Path.home() / "Datasets" / "vision-lab-audio").expanduser()


def bitrate():
    return (os.environ.get("VLAB_AUDIO_BITRATE") or "32k").strip()


def bitrate_bps(b=None):
    b = (b or bitrate()).lower()
    return int(float(b[:-1]) * 1000) if b.endswith("k") else int(float(b))


def keep_days():
    """None = keep forever (default). A positive number only when VLAB_AUDIO_KEEP_DAYS says so."""
    v = (os.environ.get("VLAB_AUDIO_KEEP_DAYS") or "").strip().lower()
    if v in FOREVER:
        return None
    try:
        d = float(v)
    except ValueError:
        return None                      # unparseable: fail safe = keep
    return d if d > 0 else None


def gb_per_day(hours=24.0, b=None):
    return round(bitrate_bps(b) / 8 * 3600 * hours / 1e9, 3)


def settings():
    return {"retain": retain_enabled(), "dir": str(audio_dir()), "bitrate": bitrate(),
            "keep_days": keep_days() or "forever", "gb_per_24h": gb_per_day(), "gb_per_8h": gb_per_day(8)}


# ---------------------------------------------------------------- schema
def _columns(db, table):
    return {r[1] for r in db.execute(f"PRAGMA table_info({table})")}


def migrate(db):
    """Idempotent, additive: new table + index, three nullable columns on voice_transcripts. Returns what changed.
    Takes a raw sqlite3 connection (tests run it on a COPY of vlab.db)."""
    changed = []
    have = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "audio_segments" not in have:
        changed.append("create audio_segments")
    db.executescript(AUDIO_DDL)
    if "voice_transcripts" in have:
        cols = _columns(db, "voice_transcripts")
        for c, t in TX_COLUMNS.items():
            if c not in cols:
                db.execute(f"ALTER TABLE voice_transcripts ADD COLUMN {c} {t}")
                changed.append(f"voice_transcripts.{c}")
        db.execute("CREATE INDEX IF NOT EXISTS voice_transcripts_audio ON voice_transcripts(audio_segment_id)")
    db.commit()
    return changed


def ensure_schema(store):
    with store.lock:
        migrate(store.db)


# ---------------------------------------------------------------- files
def path_for(start_ts, root=None):
    d = datetime.fromtimestamp(start_ts)
    return Path(root or audio_dir()) / d.strftime("%Y/%m/%d") / f"voice-{int(round(start_ts * 1000))}.m4a"


def encode(audio, dest, b=None):
    """float32 16 kHz mono (in memory) -> AAC m4a at `dest` (atomic: .tmp then rename). Returns bytes written."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "f32le", "-ar", str(SR),
           "-ac", "1", "-i", "-", "-c:a", "aac", "-b:a", b or bitrate(), "-movflags", "+faststart", "-f", "mp4", str(tmp)]
    p = subprocess.run(cmd, input=audio.astype("float32").tobytes(), capture_output=True, timeout=120)
    if p.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(p.stderr.decode()[-300:] or "ffmpeg encode failed")
    os.replace(tmp, dest)
    return dest.stat().st_size


def archive(store, audio, start_ts, end_ts=None, source="recording mode", root=None):
    """Keep one chunk: encode, then insert its audio_segments row. Returns the row id (existing id on replay)."""
    ensure_schema(store)
    dur = len(audio) / SR
    end_ts = end_ts if end_ts and end_ts > start_ts else start_ts + dur
    dest = path_for(start_ts, root)
    row = store._query("SELECT id FROM audio_segments WHERE path=?", (str(dest),))
    if row and dest.exists():
        return row[0]["id"]
    n = encode(audio, dest)
    store._exec("INSERT OR REPLACE INTO audio_segments (ts_start, ts_end, duration_s, path, codec, bitrate, sample_rate, "
                "bytes, source, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (start_ts, end_ts, round(dur, 4), str(dest), "aac", bitrate_bps(), SR, n, source, time.time()))
    return store._query("SELECT id FROM audio_segments WHERE path=?", (str(dest),))[0]["id"]


def segments(db_query, t0, t1):
    """Retained (not deleted) segments overlapping [t0, t1), oldest first. db_query = store._query."""
    return db_query("SELECT * FROM audio_segments WHERE deleted_at IS NULL AND ts_end > ? AND ts_start < ? "
                    "ORDER BY ts_start", (t0, t1))


def prune(store, now=None, days=None, dry_run=False):
    """Delete archived audio older than VLAB_AUDIO_KEEP_DAYS (default: keep forever = never deletes).
    Rows stay (deleted_at set) so transcripts keep their timing; only files under the audio dir are touched."""
    days = keep_days() if days is None else days
    if not days or days <= 0:
        return []
    ensure_schema(store)
    cutoff = (now or time.time()) - days * 86400
    root = audio_dir().resolve()
    rows = store._query("SELECT id, path FROM audio_segments WHERE deleted_at IS NULL AND ts_end < ?", (cutoff,))
    gone = []
    for r in rows:
        p = Path(r["path"])
        try:
            p.resolve().relative_to(root)
        except ValueError:
            continue                     # never delete outside the audio dir
        if not dry_run:
            p.unlink(missing_ok=True)
            store._exec("UPDATE audio_segments SET deleted_at=? WHERE id=?", (time.time(), r["id"]))
        gone.append(p)
    return gone


def words_json(words):
    return json.dumps([[round(w[0], 3), round(w[1], 3), w[2], None if w[3] is None else round(w[3], 3)] for w in words])
