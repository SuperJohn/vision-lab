"""Build tests/fixtures/vlab_fixture.db: the real vlab.db schema filled with SYNTHETIC rows only (HOU-84).

No real home data: every caption, path, label and embedding below is invented and seeded, so the
file is reproducible.  It also plants known data-quality cases for vision_lab.dq (HOU-85):
  * couch detections on screen captures (impossible: a couch can't be on a desktop screenshot)
  * couch boxes that sit on top of a bed box (bed/couch confusion), and low-confidence couches
  * duplicate person boxes in one frame (IoU > 0.7) and overlapping duplicate person tracks
    whose keyframe embeddings are near-identical

    .venv/bin/python tests/fixtures/make_fixture.py [out.db]
"""
import json, random, sqlite3, struct, sys
from pathlib import Path

OUT = Path(__file__).with_name("vlab_fixture.db")
T0 = 1_790_000_000.0          # fixed synthetic epoch (2026-09-21), not a real capture time
EMB_DIM = 768                 # SigLIP base: 768 float32 = 3072 bytes, same as production

# Exact DDL of data/vlab.db as of HOU-84 (store.py + planloop.ensure_schema + batch.py migrations).
SCHEMA = """
CREATE TABLE events (
  id INTEGER PRIMARY KEY, ts REAL, kind TEXT, track_id INTEGER, cls TEXT,
  text TEXT, color TEXT, keyframe_id INTEGER, vlm_text TEXT, rule_id TEXT, modality TEXT DEFAULT 'vision', written_at REAL, data TEXT, source_key TEXT);
CREATE TABLE presence (
  id INTEGER PRIMARY KEY, track_id INTEGER, cls TEXT, start REAL, end REAL);
CREATE TABLE keyframes (
  id INTEGER PRIMARY KEY, ts REAL, path TEXT, caption TEXT, embedding BLOB);
CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX events_ts ON events(ts);
CREATE INDEX presence_start ON presence(start);
CREATE INDEX events_modality_ts ON events(modality, ts);
CREATE UNIQUE INDEX events_source_key ON events(source_key);
CREATE TABLE utterances (
  id INTEGER PRIMARY KEY, turn_id TEXT, role TEXT, text TEXT, ts_start REAL, ts_end REAL,
  voice TEXT, model TEXT, latencies TEXT, interrupted INTEGER DEFAULT 0, error TEXT,
  keyframe_id INTEGER, snapshot TEXT, written_at REAL, source_key TEXT UNIQUE);
CREATE INDEX utterances_ts ON utterances(ts_start);
CREATE TABLE plan_blocks (
  id TEXT PRIMARY KEY, day TEXT, task TEXT, task_id TEXT, task_rank INTEGER, linear_issue TEXT, kind TEXT,
  planned_start REAL, planned_end REAL, start_local TEXT, end_local TEXT, budget_min INTEGER,
  zone TEXT DEFAULT 'desk', status TEXT DEFAULT 'planned', source TEXT, tags TEXT, updated_at REAL);
CREATE INDEX plan_blocks_day ON plan_blocks(day, planned_start);
CREATE TABLE plan_changes (
  id INTEGER PRIMARY KEY, ts REAL, block_id TEXT, change TEXT, before TEXT, after TEXT, source TEXT);
CREATE INDEX plan_changes_block ON plan_changes(block_id, ts);
CREATE TABLE batch_annotations (
  id INTEGER PRIMARY KEY, source_type TEXT, source_path TEXT, keyframe_id INTEGER, capture_ts REAL,
  processed_ts REAL, created_at REAL, model TEXT, caption TEXT, label TEXT, detections TEXT,
  gemma_ms INTEGER, detect_ms INTEGER, status TEXT, error TEXT, sensitive INTEGER NOT NULL DEFAULT 0, exportable INTEGER NOT NULL DEFAULT 1, UNIQUE(source_type, source_path));
CREATE INDEX batch_ann_capture ON batch_annotations(source_type, capture_ts);
"""


def emb(rng, base=None, noise=0.002):
    v = [rng.gauss(0, 1) for _ in range(EMB_DIM)] if base is None else [b + rng.gauss(0, noise) for b in base]
    n = sum(x * x for x in v) ** 0.5
    v = [x / n for x in v]
    return v, struct.pack(f"<{EMB_DIM}f", *v)


def det(cls, conf, box):
    return {"cls": cls, "conf": conf, "box": [round(x, 3) for x in box]}


def build(path=OUT):
    path = Path(path)
    if path.exists():
        path.unlink()
    rng = random.Random(84)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.execute("INSERT INTO config VALUES ('settings', ?)", (json.dumps({"capture_paused": False, "camera": 0}),))
    db.execute("INSERT INTO config VALUES ('mic_enabled', 'true')")

    # keyframes: 40 synthetic, embeddings seeded; kf 31/32 are a near-duplicate pair (same person, two IDs)
    vecs = {}
    for k in range(1, 41):
        base = vecs.get(31) if k == 32 else None
        v, blob = emb(rng, base)
        vecs[k] = v
        db.execute("INSERT INTO keyframes VALUES (?,?,?,?,?)",
                   (k, T0 + k * 60, f"data/keyframes/fixture_{k:03d}.jpg", f"Synthetic scene {k}: a test person at a desk.", blob))

    # presence + track events. Tracks 1..20 normal; tracks 101/102 = duplicate person IDs (overlap, near-same start)
    tid, eid = 0, 0
    classes = ["person", "chair", "couch", "bed", "dog", "cup"]
    for i in range(1, 21):
        cls = classes[i % len(classes)]
        start = T0 + i * 90
        db.execute("INSERT INTO presence (track_id, cls, start, end) VALUES (?,?,?,?)", (i, cls, start, start + 30 + i))
        db.execute("INSERT INTO events (ts, kind, track_id, cls, text, color, keyframe_id, modality, written_at) VALUES (?,?,?,?,?,?,?,?,?)",
                   (start, "track", i, cls, f"{cls.title()} #{i} came into view", "#A3AFBD", i, "vision", start + 0.1))
    for t, kf, st in ((101, 31, T0 + 1900.0), (102, 32, T0 + 1900.8)):
        db.execute("INSERT INTO presence (track_id, cls, start, end) VALUES (?,?,?,?)", (t, "person", st, st + 120))
        db.execute("INSERT INTO events (ts, kind, track_id, cls, text, color, keyframe_id, modality, written_at) VALUES (?,?,?,?,?,?,?,?,?)",
                   (st, "track", t, "person", f"Person #{t} came into view", "#A3AFBD", kf, "vision", st + 0.1))
    # a second, legitimately separate person overlapping in time but with an unrelated embedding
    db.execute("INSERT INTO presence (track_id, cls, start, end) VALUES (?,?,?,?)", (103, "person", T0 + 1901.5, T0 + 1960))
    db.execute("INSERT INTO events (ts, kind, track_id, cls, text, color, keyframe_id, modality, written_at) VALUES (?,?,?,?,?,?,?,?,?)",
               (T0 + 1901.5, "track", 103, "person", "Person #103 came into view", "#A3AFBD", 33, "vision", T0 + 1901.6))

    # other modalities
    for j, (mod, kind, data) in enumerate([
        ("desktop", "desktop_sample", {"app": "Editor", "title": "fixture.py", "idle_s": 1.0, "error": None}),
        ("vision", "frame_probe", {"ok": True, "w": 1280, "h": 720, "boxes": [det("person", 0.9, [0.1, 0.1, 0.5, 0.9])]}),
        ("activity", "activity_slice", {"guess": "at_desk_work", "evidence": {"desktop_samples": 15}}),
        ("audio", "sound", {"kind": "audio_event", "label": "knock", "confidence": 0.6}),
        ("voice", "interrupt", {"kind": "voice_interrupt", "reason": "barge-in"}),
        ("vision", "capture_paused", None), ("vision", "capture_resumed", None),
    ]):
        ts = T0 + 3000 + j * 10
        db.execute("INSERT INTO events (ts, kind, text, modality, written_at, data, source_key) VALUES (?,?,?,?,?,?,?)",
                   (ts, kind, f"synthetic {kind}", mod, ts + 0.2, json.dumps(data) if data else None, f"fixture:{kind}:{j}"))
    db.execute("INSERT INTO events (ts, kind, text, modality, rule_id, keyframe_id, vlm_text) VALUES (?,?,?,?,?,?,?)",
               (T0 + 3200, "rule", "Rule fired: test rule", "vision", "r1", 5, "A synthetic description."))

    # utterances: 3 user/assistant turns, each linked to a keyframe (production: 100% linked)
    for n in range(3):
        ts = T0 + 4000 + n * 30
        for role, txt in (("user", f"Synthetic question {n}?"), ("assistant", f"Synthetic answer {n}.")):
            db.execute("INSERT INTO utterances (turn_id, role, text, ts_start, ts_end, voice, model, latencies, keyframe_id, written_at, source_key) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (f"turn-{n}", role, txt, ts, ts + 2, "test-voice", "test-model", json.dumps({"first_token_ms": 300}),
                        10 + n, ts + 2.5, f"fixture:utt:{n}:{role}"))
            ts += 3

    # batch annotations (fast/slow loop output): keyframes + screens with planted DQ cases
    rows = []
    for k in range(1, 31):                        # 30 keyframe annotations
        d = [det("person", 0.9, [0.1, 0.2, 0.4, 0.95])]
        if k <= 6:                                # 6 couches; 3 of them sit on a bed box, 2 are low confidence
            d.append(det("couch", 0.82 if k <= 4 else 0.31, [0.5, 0.55, 0.95, 1.0]))
            if k <= 3:
                d.append(det("bed", 0.7, [0.48, 0.5, 0.97, 1.0]))
        if k in (7, 8):                           # duplicate person boxes in one frame (NMS miss)
            d.append(det("person", 0.6, [0.11, 0.21, 0.41, 0.94]))
        if k == 9:                                # two real people side by side: low IoU, NOT a dup
            d.append(det("person", 0.88, [0.55, 0.2, 0.85, 0.95]))
        rows.append(("keyframe", f"data/keyframes/fixture_{k:03d}.jpg", k, T0 + k * 60, d))
    for s in range(1, 11):                        # 10 screen captures; 2 with an impossible couch
        d = [det("tv", 0.5, [0.0, 0.0, 1.0, 1.0])] if s % 2 else []
        if s in (3, 7):
            d.append(det("couch", 0.45, [0.2, 0.6, 0.6, 0.9]))
        rows.append(("screen", f"data/screens/2026-09-21/fixture_{s:03d}.jpg", None, T0 + 5000 + s * 10, d))
    for st, p, kf, ts, d in rows:
        db.execute("INSERT INTO batch_annotations (source_type, source_path, keyframe_id, capture_ts, processed_ts, created_at, model, caption, label, detections, gemma_ms, detect_ms, status, sensitive, exportable) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (st, p, kf, ts, ts + 5, ts + 5, "fixture-model", "Synthetic caption.", "test", json.dumps(d), 900, 12, "ok",
                    1 if st == "screen" else 0, 0 if st == "screen" else 1))

    db.execute("INSERT INTO plan_blocks (id, day, task, kind, planned_start, planned_end, start_local, end_local, budget_min, status, source, updated_at) "
               "VALUES ('blk1','2026-09-21','Synthetic task','work',?,?,'09:00','10:00',60,'planned','fixture',?)", (T0, T0 + 3600, T0))
    db.execute("INSERT INTO plan_changes (ts, block_id, change, before, after, source) VALUES (?,?,?,?,?,?)",
               (T0 + 10, "blk1", "status", "planned", "done", "fixture"))
    db.commit()
    db.execute("VACUUM")
    db.close()
    return path


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else OUT))
