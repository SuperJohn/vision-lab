"""SQLite storage for events, presence spans, keyframes, zones and rules."""
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KEYFRAME_DIR = DATA_DIR / "keyframes"
# Recordings go to ~/Downloads unless VLAB_RECORDINGS points elsewhere.
RECORDING_DIR = Path(os.environ.get("VLAB_RECORDINGS", Path.home() / "Downloads")).expanduser()

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY, ts REAL, kind TEXT, track_id INTEGER, cls TEXT,
  text TEXT, color TEXT, keyframe_id INTEGER, vlm_text TEXT, rule_id TEXT);
CREATE TABLE IF NOT EXISTS presence (
  id INTEGER PRIMARY KEY, track_id INTEGER, cls TEXT, start REAL, end REAL);
CREATE TABLE IF NOT EXISTS keyframes (
  id INTEGER PRIMARY KEY, ts REAL, path TEXT, caption TEXT, embedding BLOB);
CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS presence_start ON presence(start);
CREATE TABLE IF NOT EXISTS utterances (
  id INTEGER PRIMARY KEY, turn_id TEXT, role TEXT, text TEXT, ts_start REAL, ts_end REAL,
  voice TEXT, model TEXT, latencies TEXT, interrupted INTEGER DEFAULT 0, error TEXT,
  keyframe_id INTEGER, snapshot TEXT, written_at REAL, source_key TEXT UNIQUE);
CREATE INDEX IF NOT EXISTS utterances_ts ON utterances(ts_start);
"""

# Columns added after the first release. ALTER ... ADD COLUMN keeps every existing row;
# old rows read the DEFAULT, so all pre-existing events count as vision.
MIGRATIONS = [
    ("events", "modality", "TEXT DEFAULT 'vision'"),
    ("events", "written_at", "REAL"),
    ("events", "data", "TEXT"),
    ("events", "source_key", "TEXT"),
]


class Store:
    def __init__(self, path=None):
        path = path or os.environ.get("VLAB_DB") or DATA_DIR / "vlab.db"
        DATA_DIR.mkdir(exist_ok=True)
        KEYFRAME_DIR.mkdir(exist_ok=True)
        RECORDING_DIR.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)
            for table, col, decl in MIGRATIONS:
                have = {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
                if col not in have:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
            self.db.execute("CREATE INDEX IF NOT EXISTS events_modality_ts ON events(modality, ts)")
            self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS events_source_key ON events(source_key)")
            # Close spans left open by a previous run.
            self.db.execute("UPDATE presence SET end = start WHERE end IS NULL")
            self.db.commit()

    def _exec(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur.lastrowid

    def _query(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    # events
    def add_event(self, kind, text, color="#A3AFBD", track_id=None, cls=None,
                  keyframe_id=None, rule_id=None, ts=None, modality="vision", data=None, source_key=None):
        return self._exec(
            "INSERT OR IGNORE INTO events (ts, kind, track_id, cls, text, color, keyframe_id, rule_id, "
            "modality, written_at, data, source_key) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts or time.time(), kind, track_id, cls, text, color, keyframe_id, rule_id, modality,
             time.time(), json.dumps(data) if data is not None else None, source_key))

    # ---------- multimodal (step 2) ----------
    def add_utterance(self, turn_id, role, text, ts_start, ts_end, voice=None, model=None, latencies=None,
                      interrupted=False, error=None, keyframe_id=None, snapshot=None, source_key=None):
        return self._exec(
            "INSERT OR IGNORE INTO utterances (turn_id, role, text, ts_start, ts_end, voice, model, latencies, "
            "interrupted, error, keyframe_id, snapshot, written_at, source_key) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (turn_id, role, text, ts_start, ts_end, voice, model, json.dumps(latencies or {}), int(bool(interrupted)),
             error, keyframe_id, json.dumps(snapshot) if snapshot is not None else None, time.time(), source_key))

    def nearest_keyframe(self, ts, window=600):
        rows = self._query("SELECT id, ts, caption FROM keyframes WHERE ts BETWEEN ? AND ? "
                           "ORDER BY ABS(ts - ?) LIMIT 1", (ts - window, ts + window, ts))
        return rows[0] if rows else None

    def merged(self, since=0, until=None, modalities=None, q=None, limit=500):
        """One time-ordered stream across modalities: events (vision/audio/voice/system),
        utterances (voice) and keyframe captions (vision)."""
        until = until or time.time() + 1
        like = f"%{q}%" if q else None
        sql = [
            "SELECT 'event' AS source, id, ts, NULL AS ts_end, COALESCE(modality,'vision') AS modality, kind, "
            "cls AS label, text, vlm_text AS detail, keyframe_id, data, written_at FROM events "
            "WHERE ts BETWEEN :s AND :u" + (" AND (text LIKE :q OR vlm_text LIKE :q)" if q else ""),
            "SELECT 'utterance', id, ts_start, ts_end, 'voice', 'utterance', role, text, latencies, keyframe_id, "
            "snapshot, written_at FROM utterances WHERE ts_start BETWEEN :s AND :u" + (" AND text LIKE :q" if q else ""),
            "SELECT 'keyframe', id, ts, NULL, 'vision', 'caption', NULL, caption, NULL, id, NULL, NULL FROM keyframes "
            "WHERE caption IS NOT NULL AND ts BETWEEN :s AND :u" + (" AND caption LIKE :q" if q else ""),
        ]
        args = {"s": since, "u": until, "q": like, "n": limit}
        where = ""
        if modalities:
            args.update({f"m{i}": m for i, m in enumerate(modalities)})
            where = "WHERE modality IN (" + ",".join(f":m{i}" for i in range(len(modalities))) + ")"
        rows = self._query(f"SELECT * FROM ({' UNION ALL '.join(sql)}) {where} ORDER BY ts DESC LIMIT :n", args)
        for r in rows:
            for k in ("data", "detail"):
                if isinstance(r.get(k), str) and r[k][:1] in "{[":
                    try:
                        r[k] = json.loads(r[k])
                    except ValueError:
                        pass
        return rows

    def set_event_vlm(self, event_id, text):
        self._exec("UPDATE events SET vlm_text = ? WHERE id = ?", (text, event_id))

    def events(self, since=0, limit=200):
        return self._query(
            "SELECT * FROM events WHERE ts >= ? ORDER BY ts DESC LIMIT ?", (since, limit))

    # presence spans for the timeline
    def open_span(self, track_id, cls, start):
        return self._exec("INSERT INTO presence (track_id, cls, start) VALUES (?,?,?)",
                          (track_id, cls, start))

    def close_span(self, span_id, end):
        self._exec("UPDATE presence SET end = ? WHERE id = ?", (end, span_id))

    def spans(self, since):
        return self._query(
            "SELECT track_id, cls, start, COALESCE(end, ?) AS end FROM presence "
            "WHERE COALESCE(end, ?) >= ? ORDER BY start", (time.time(), time.time(), since))

    # keyframes
    def add_keyframe(self, ts, path, caption=None):
        return self._exec("INSERT INTO keyframes (ts, path, caption) VALUES (?,?,?)",
                          (ts, str(path), caption))

    def set_keyframe_caption(self, kid, caption):
        self._exec("UPDATE keyframes SET caption = ? WHERE id = ?", (caption, kid))

    def set_keyframe_embedding(self, kid, emb):
        self._exec("UPDATE keyframes SET embedding = ? WHERE id = ?",
                   (np.asarray(emb, dtype=np.float32).tobytes(), kid))

    def keyframe(self, kid):
        rows = self._query("SELECT id, ts, path, caption FROM keyframes WHERE id = ?", (kid,))
        return rows[0] if rows else None

    def keyframes_missing_embedding(self, limit=16):
        return self._query(
            "SELECT id, path FROM keyframes WHERE embedding IS NULL ORDER BY id DESC LIMIT ?",
            (limit,))

    def keyframe_embeddings(self, since):
        with self.lock:
            rows = self.db.execute(
                "SELECT id, ts, caption, embedding FROM keyframes "
                "WHERE ts >= ? AND embedding IS NOT NULL", (since,)).fetchall()
        return [(r["id"], r["ts"], r["caption"], np.frombuffer(r["embedding"], dtype=np.float32))
                for r in rows]

    def keyframes_text_search(self, q, since, limit=12):
        return self._query(
            "SELECT id, ts, caption FROM keyframes WHERE ts >= ? AND caption LIKE ? "
            "ORDER BY ts DESC LIMIT ?", (since, f"%{q}%", limit))

    def count_keyframes(self, since):
        return self._query("SELECT COUNT(*) AS n FROM keyframes WHERE ts >= ?", (since,))[0]["n"]

    # config (zones, rules, settings) as JSON blobs
    def get_config(self, key, default):
        rows = self._query("SELECT value FROM config WHERE key = ?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def set_config(self, key, value):
        self._exec("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                   (key, json.dumps(value)))
