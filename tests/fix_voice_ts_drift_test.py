"""tools/fix_voice_ts_drift.py on synthetic data: a drifting recorder (mic at ~80%) with known true times.

Temp DBs only. PYTHONPATH=. .venv/bin/python tests/fix_voice_ts_drift_test.py
"""
import random, sqlite3, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import fix_voice_ts_drift as fx

SCHEMA = """CREATE TABLE voice_transcripts (id INTEGER PRIMARY KEY, ts_start REAL, ts_end REAL, text TEXT, model TEXT,
  avg_logprob REAL, no_speech_prob REAL, source TEXT, source_key TEXT UNIQUE, written_at REAL);
CREATE TABLE events (id INTEGER PRIMARY KEY, ts REAL, kind TEXT, text TEXT);"""
L = 1.2


def build(sessions, post_fix_at=None, seed=7):
    """sessions: [(wall_start, wall_duration_s, rate)] -> (db path, truth {id: (start, end)}, t_fix)."""
    rnd = random.Random(seed)
    path = Path(tempfile.mkdtemp(prefix="fixts-")) / "vlab.db"
    db = sqlite3.connect(path); db.executescript(SCHEMA)
    truth = {}

    def add(ts_s, ts_e, w, key, true):
        cur = db.execute("INSERT INTO voice_transcripts (ts_start, ts_end, text, source, source_key, written_at) "
                         "VALUES (?,?,?,?,?,?)", (ts_s, ts_e, "x", "recording mode", key, w))
        truth[cur.lastrowid] = true

    for s0, dur, rate in sessions:
        db.execute("INSERT INTO events (ts, kind) VALUES (?, 'mic_started')", (s0,))
        wall, samp = s0, s0                            # true wall start / drifted name of the current chunk
        while wall < s0 + dur:
            r = rate * (1 + 0.02 * rnd.uniform(-1, 1))
            speech = rnd.random() < 0.75
            segs, off = [], rnd.uniform(0, 1)
            talk = rnd.choice([6.0, 9.0, 12.0, 16.0, 25.0]) if speech else 0
            while speech and off < min(talk, 19.5):
                seg = min(rnd.uniform(1, 5), 19.9 - off)
                segs.append((off, off + seg)); off += seg + rnd.uniform(0, 0.8)
            # recorder rule: close at the first quiet block once >= 10 s, never past 20 s
            d = min(20.0, max(10.0, round(segs[-1][1] + 0.1, 1))) if segs else 10.0
            wall_end = wall + d / r
            w = wall_end + L + rnd.uniform(0, 0.15)
            for k, (a, b) in enumerate(segs):
                b = min(b, d)
                add(samp + a, samp + b, w, f"file:voice-{int(samp * 1000)}.pcm:{k}", (wall + a / r, wall + b / r))
            wall, samp = wall_end, samp + d
        db.execute("INSERT INTO events (ts, kind) VALUES (?, 'mic_stopped')", (wall,))
    t_fix = None
    if post_fix_at:                                   # fixed recorder: wall-clock names, rate 0.9, L latency
        t_fix = post_fix_at
        db.execute("INSERT INTO events (ts, kind) VALUES (?, 'mic_started')", (t_fix,))
        wall = t_fix
        for i in range(40):
            d_wall = 10 / 0.9
            add(wall + 0.5, wall + 3, wall + d_wall + L + rnd.uniform(0, 0.1), f"file:voice-{int(wall * 1000)}.pcm:0",
                None)
            wall += d_wall
    db.commit(); db.close()
    return path, truth, t_fix


def run(path, t_fix, apply=True, latency=None):
    args = ["--db", str(path), "--t-fix", str(t_fix)] + (["--apply"] if apply else []) + \
           (["--latency", str(latency)] if latency is not None else [])
    return fx.main(args)


def test_recovers_true_times_two_sessions():
    T = 1_792_520_000.0
    path, truth, t_fix = build([(T, 1800, 0.89), (T + 3600, 6 * 3600, 0.80)], post_fix_at=T + 8 * 3600)
    rep = run(path, t_fix)
    assert abs(rep["latency_s"] - (L + 0.05)) < 0.2, rep["latency_s"]
    db = sqlite3.connect(path)
    errs = []
    for i, s, e, tag in db.execute("SELECT id, ts_start, ts_end, ts_fix FROM voice_transcripts"):
        if truth[i] is None:
            assert tag is None, "post-fix rows must stay untouched"
            continue
        assert tag == "rewrite-v1"
        errs.append(abs(s - truth[i][0]))
    errs.sort()
    assert errs[len(errs) // 2] < 0.3 and errs[int(len(errs) * 0.99)] < 1.0 and errs[-1] < 3.0, (errs[len(errs) // 2], errs[-1])
    c = rep["checks"]
    assert c["ts_start_lt_written_at"] and c["within_mic_started_and_t_fix"], c
    assert rep["total_rows_before_after"][0] == rep["total_rows_before_after"][1]
    # originals kept: undo restores them exactly
    orig = dict(db.execute("SELECT id, ts_start_orig FROM voice_transcripts WHERE ts_fix='rewrite-v1'"))
    db.execute("UPDATE voice_transcripts SET ts_start = ts_start_orig, ts_end = ts_end_orig, ts_fix = NULL "
               "WHERE ts_fix = 'rewrite-v1'"); db.commit()
    now = dict(db.execute("SELECT id, ts_start FROM voice_transcripts WHERE id IN (%s)" % ",".join(map(str, orig))))
    assert now == orig


def test_dry_run_writes_nothing_and_rerun_is_noop():
    T = 1_792_520_000.0
    path, truth, t_fix = build([(T, 3600, 0.8)], post_fix_at=T + 7200)
    before = sqlite3.connect(path).execute("SELECT group_concat(ts_start) FROM voice_transcripts").fetchone()
    rep = run(path, t_fix, apply=False)
    assert rep["updates"] > 100
    db = sqlite3.connect(path)
    assert "ts_fix" not in {r[1] for r in db.execute("PRAGMA table_info(voice_transcripts)")}
    assert db.execute("SELECT group_concat(ts_start) FROM voice_transcripts").fetchone() == before
    db.close()
    run(path, t_fix)
    again = run(path, t_fix)
    assert again["rows_in_scope"] == 0 and again["updates"] == 0


def test_plan_handles_gaps_and_last_chunk():
    B = 1_792_520_000.0
    rows = [dict(id=1, ts_start=B + 0.5, ts_end=B + 3.0, written_at=B + 13.7, key=f"file:voice-{int(B * 1000)}.pcm:0"),
            dict(id=2, ts_start=B + 60.2, ts_end=B + 65.0, written_at=B + 100.0, key=f"file:voice-{int((B + 60) * 1000)}.pcm:0")]
    up, chunks = fx.plan(rows, [B - 1], latency=1.2)
    c0 = next(c for c in chunks if c["n"] == B)
    assert c0["d"] == 10.0                             # gap 60 s > 20 s and speech ended at 3 s -> 10 s chunk
    c1 = next(c for c in chunks if c["n"] == B + 60)
    assert abs(c1["d"] - 5.1) < 1e-6                   # last chunk of a session: length from its speech
    assert all(s <= e for _, s, e in up)


if __name__ == "__main__":
    import io, contextlib, time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                fn()
            print(f"PASS {name} ({time.time() - t:.1f}s)", flush=True)
    print("fix_voice_ts_drift: OK")
