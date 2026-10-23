"""recording-mode mic audio is kept (AAC m4a), linked in vlab.db, prunable by setting; mute still works.

Synthetic audio only (numpy tones, ffmpeg lavfi), temp dirs and temp DBs. Never opens the real mic and never
touches the live app, vlab.db or the real audio archive (VLAB_AUDIO_DIR is forced to a temp dir).

    PYTHONPATH=. .venv/bin/python tests/audio_retain.py
"""
import json, os, sqlite3, subprocess, sys, tempfile, time, types
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ARCHIVE = Path(tempfile.mkdtemp(prefix="vlab-audio-test-"))
os.environ["VLAB_AUDIO_DIR"] = str(ARCHIVE)          # never the real archive
for k in ("VLAB_AUDIO_RETAIN", "VLAB_AUDIO_KEEP_DAYS", "VLAB_AUDIO_BITRATE"):
    os.environ.pop(k, None)
from vision_lab import audiostore, recording, retention, voicetx
from vision_lab.store import Store

SR = 16000
T0 = datetime(2026, 10, 1, 10, 0, 0).timestamp()
TONE = ["-re", "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=16000"]


def store():
    return Store(str(Path(tempfile.mkdtemp(prefix="vlab-db-")) / "vlab.db"))


def tone(seconds, freq=300.0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def fake_stt(audio):
    n = len(audio) / SR
    return [{"start": 0.5, "end": 2.0, "text": " Synthetic first sentence.", "avg_logprob": -0.2, "no_speech_prob": 0.01,
             "words": [{"word": " Synthetic", "start": 0.5, "end": 1.0, "probability": 0.9},
                       {"word": " first", "start": 1.0, "end": 1.4, "probability": 0.8},
                       {"word": " sentence.", "start": 1.4, "end": 2.0, "probability": 0.95}]},
            {"start": 3.0, "end": min(4.5, n), "text": " Synthetic second.", "avg_logprob": -0.3, "no_speech_prob": 0.02}]


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,channels,sample_rate:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True)
    return json.loads(out.stdout)


def pcm_file(d, start, seconds, wall_s=None):
    end = start + (wall_s or seconds)
    f = Path(d) / f"voice-{int(start * 1000)}-{int(end * 1000)}.pcm"
    f.write_bytes(tone(seconds).tobytes())
    return f


def m4as():
    return sorted(ARCHIVE.rglob("*.m4a"))


def clear_archive():
    for f in ARCHIVE.rglob("*"):
        if f.is_file():
            f.unlink()


# ------------------------------------------------------------------ retention + linkage
def test_default_keeps_audio_and_links_rows():
    clear_archive()
    st, d = store(), Path(tempfile.mkdtemp())
    f = pcm_file(d, T0, 6.0, wall_s=6.75)                 # mic delivered ~89%: 6 s of audio over 6.75 s of wall time
    r = voicetx.transcribe_file(st, f, stt=fake_stt)
    assert r["transcripts"] == 2 and r.get("audio_segment_id"), r
    assert not f.exists(), "the temp chunk is still removed"
    kept = audiostore.path_for(T0)
    assert kept.exists() and m4as() == [kept], m4as()
    assert kept.relative_to(ARCHIVE).parts[:3] == ("2026", "10", "01"), kept
    p = probe(kept)
    s = p["streams"][0]
    assert s["codec_name"] == "aac" and s["channels"] == 1 and s["sample_rate"] == "16000", p
    assert abs(float(p["format"]["duration"]) - 6.0) < 0.15, p
    seg = st._query("SELECT * FROM audio_segments")
    assert len(seg) == 1, seg
    seg = seg[0]
    assert seg["ts_start"] == T0 and abs(seg["ts_end"] - (T0 + 6.75)) < 1e-3 and abs(seg["duration_s"] - 6.0) < 1e-3, seg
    assert seg["codec"] == "aac" and seg["bitrate"] == 32000 and seg["bytes"] == kept.stat().st_size and seg["path"] == str(kept)
    rows = st._query("SELECT ts_start, ts_end, audio_segment_id, audio_offset_s, words FROM voice_transcripts ORDER BY ts_start")
    assert [x["audio_segment_id"] for x in rows] == [seg["id"]] * 2
    assert [x["audio_offset_s"] for x in rows] == [0.5, 3.0]
    assert rows[0]["ts_start"] == T0 + 0.5
    w = json.loads(rows[0]["words"])
    assert [x[2] for x in w] == ["Synthetic", "first", "sentence."] and w[0][0] == T0 + 0.5 and w[2][1] == T0 + 2.0, w
    assert rows[1]["words"] is None
    # a segment row is a time join away from any transcript row (and from the screen timeline)
    j = st._query("SELECT t.ts_start - a.ts_start AS off, t.audio_offset_s FROM voice_transcripts t "
                  "JOIN audio_segments a ON a.id = t.audio_segment_id")
    assert all(abs(x["off"] - x["audio_offset_s"]) < 1e-6 for x in j), j


def test_retain_off_is_the_old_rule():
    clear_archive()
    os.environ["VLAB_AUDIO_RETAIN"] = "0"
    try:
        st, d = store(), Path(tempfile.mkdtemp())
        f = pcm_file(d, T0 + 100, 5.0)
        r = voicetx.transcribe_file(st, f, stt=fake_stt)
        assert r["transcripts"] == 2 and "audio_segment_id" not in r and not f.exists() and not m4as()
        assert st._query("SELECT COUNT(*) n FROM audio_segments")[0]["n"] == 0
        assert retention.policy()["voice_audio"] == "delete after transcription"
    finally:
        os.environ.pop("VLAB_AUDIO_RETAIN")
    assert retention.policy()["voice_audio"] == "forever"


def test_whisper_failure_still_keeps_audio():
    clear_archive()
    st, d = store(), Path(tempfile.mkdtemp())
    f = pcm_file(d, T0 + 200, 4.0)

    def boom(audio):
        raise RuntimeError("model unavailable")
    r = voicetx.transcribe_file(st, f, stt=boom)
    assert "error" in r and not f.exists(), r
    assert len(m4as()) == 1 and st._query("SELECT COUNT(*) n FROM audio_segments")[0]["n"] == 1


def test_silence_is_kept_but_not_transcribed():
    clear_archive()
    st, d = store(), Path(tempfile.mkdtemp())
    f = Path(d) / f"voice-{int((T0 + 300) * 1000)}.pcm"; f.write_bytes(np.zeros(SR * 3, np.float32).tobytes())
    called = []
    r = voicetx.transcribe_file(st, f, stt=lambda a: called.append(1) or [])
    assert r["transcripts"] == 0 and not called and len(m4as()) == 1
    seg = st._query("SELECT ts_start, ts_end, duration_s FROM audio_segments")[0]
    assert seg["ts_end"] == seg["ts_start"] + 3.0, seg            # no end in the old-style name: start + duration


def test_replay_does_not_duplicate():
    clear_archive()
    st = store()
    a = audiostore.archive(st, tone(2), T0 + 400)
    b = audiostore.archive(st, tone(2), T0 + 400)
    assert a == b and len(m4as()) == 1 and st._query("SELECT COUNT(*) n FROM audio_segments")[0]["n"] == 1


def test_keep_days_setting_and_prune():
    clear_archive()
    st = store()
    old, new = time.time() - 40 * 86400, time.time() - 3600
    ia, ib = audiostore.archive(st, tone(2), old), audiostore.archive(st, tone(2), new)
    outside = Path(tempfile.mkdtemp()) / "voice-1.m4a"; outside.write_bytes(b"x")
    st._exec("INSERT INTO audio_segments (ts_start, ts_end, path) VALUES (?,?,?)", (old, old + 2, str(outside)))
    for v in (None, "", "0", "none", "forever", "keep", "-3", "junk"):
        if v is None: os.environ.pop("VLAB_AUDIO_KEEP_DAYS", None)
        else: os.environ["VLAB_AUDIO_KEEP_DAYS"] = v
        assert audiostore.keep_days() is None, v
        assert audiostore.prune(st) == [] and len(m4as()) == 2, v
    os.environ["VLAB_AUDIO_KEEP_DAYS"] = "30"
    try:
        assert audiostore.prune(st, dry_run=True) == [audiostore.path_for(old)] and len(m4as()) == 2
        gone = audiostore.prune(st)
        assert gone == [audiostore.path_for(old)] and m4as() == [audiostore.path_for(new)], (gone, m4as())
        assert outside.exists(), "never deletes outside the audio dir"
        rows = {r["id"]: r["deleted_at"] for r in st._query("SELECT id, deleted_at FROM audio_segments")}
        assert rows[ia] and rows[ib] is None, rows                  # row stays, marked deleted
        assert retention.policy()["voice_audio"] == 30.0
    finally:
        os.environ.pop("VLAB_AUDIO_KEEP_DAYS")


def test_settings_and_storage_estimate():
    s = audiostore.settings()
    assert s["retain"] is True and s["dir"] == str(ARCHIVE) and s["bitrate"] == "32k" and s["keep_days"] == "forever"
    assert s["gb_per_24h"] == 0.346 and s["gb_per_8h"] == 0.115, s
    os.environ["VLAB_AUDIO_BITRATE"] = "24k"
    try:
        assert audiostore.gb_per_day() == 0.259 and audiostore.bitrate_bps() == 24000
    finally:
        os.environ.pop("VLAB_AUDIO_BITRATE")
    for v in ("0", "off", "no", "false", "OFF"):
        os.environ["VLAB_AUDIO_RETAIN"] = v; assert not audiostore.retain_enabled(), v
    os.environ.pop("VLAB_AUDIO_RETAIN")
    os.environ.pop("VLAB_AUDIO_DIR")
    try:
        assert audiostore.audio_dir() == Path.home() / "Datasets" / "vision-lab-audio"
        assert "Mobile Documents" not in str(audiostore.audio_dir()) and "/Documents/" not in str(audiostore.audio_dir())
    finally:
        os.environ["VLAB_AUDIO_DIR"] = str(ARCHIVE)


# ------------------------------------------------------------------ migration (synthetic copy of the live schema)
LIVE_LIKE = """CREATE TABLE voice_transcripts (
  id INTEGER PRIMARY KEY, ts_start REAL, ts_end REAL, text TEXT, model TEXT, avg_logprob REAL,
  no_speech_prob REAL, source TEXT, source_key TEXT UNIQUE, written_at REAL, ts_start_orig REAL, ts_end_orig REAL, ts_fix TEXT);
CREATE INDEX voice_transcripts_ts ON voice_transcripts(ts_start);"""


def test_migration_is_additive_and_idempotent():
    p = Path(tempfile.mkdtemp()) / "copy.db"
    db = sqlite3.connect(p); db.executescript(LIVE_LIKE)
    db.executemany("INSERT INTO voice_transcripts (ts_start, ts_end, text, source_key, ts_fix) VALUES (?,?,?,?,?)",
                   [(T0 + i, T0 + i + 1, f"synthetic row {i}", f"k{i}", "drift" if i % 2 else None) for i in range(50)])
    db.commit()
    before = db.execute("SELECT * FROM voice_transcripts ORDER BY id").fetchall()
    changed = audiostore.migrate(db)
    assert changed == ["create audio_segments", "voice_transcripts.audio_segment_id", "voice_transcripts.audio_offset_s",
                       "voice_transcripts.words"], changed
    assert audiostore.migrate(db) == [], "second run changes nothing"
    after = db.execute("SELECT id, ts_start, ts_end, text, model, avg_logprob, no_speech_prob, source, source_key, written_at, "
                       "ts_start_orig, ts_end_orig, ts_fix FROM voice_transcripts ORDER BY id").fetchall()
    assert after == before, "existing rows untouched"
    assert db.execute("SELECT COUNT(*) FROM voice_transcripts WHERE audio_segment_id IS NOT NULL OR words IS NOT NULL").fetchone()[0] == 0
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    # the server's own path (voicetx.ensure_schema through Store) on the same file is a no-op now
    st = Store(str(p)); voicetx.ensure_schema(st)
    cols = {r["name"] for r in st._query("PRAGMA table_info(voice_transcripts)")}
    assert {"ts_fix", "audio_segment_id", "audio_offset_s", "words"} <= cols


def test_fresh_db_gets_full_schema():
    st = store(); voicetx.ensure_schema(st)
    cols = {r["name"] for r in st._query("PRAGMA table_info(voice_transcripts)")}
    assert {"audio_segment_id", "audio_offset_s", "words"} <= cols
    assert st._query("SELECT 1 FROM sqlite_master WHERE name='audio_segments'")


# ------------------------------------------------------------------ recorder end to end + mute / privacy
def test_recorder_keeps_contiguous_chunks_and_mute_stops_archiving():
    clear_archive()
    st, d = store(), Path(tempfile.mkdtemp())
    allowed = {"on": True}
    m = recording.MicRecorder(st, voice_tmp=d, source=TONE, chunk_s=1.5, max_chunk_s=2.0,
                              transcribe=lambda f: voicetx.transcribe_file(st, f, stt=fake_stt))
    eng = types.SimpleNamespace(settings={"capture_paused": False, "camera_paused": False})
    rm = recording.RecordingMode(eng, st, enabled=True, mic=m, video=None, screen=None,
                                 mic_allowed=lambda: allowed["on"], poll_s=0.1, restart_s=0.0)
    rm.start()
    try:
        assert wait_for(m.running, 5)
        time.sleep(5)
        allowed["on"] = False                                  # the Mic button (mic_enabled) = mute
        assert wait_for(lambda: not m.running(), 5) and m.drain(30)
        n_muted = len(m4as()); time.sleep(2)
        assert len(m4as()) == n_muted, "nothing kept while the mic is off"
        allowed["on"] = True
        assert wait_for(m.running, 5); time.sleep(2.5)
        eng.settings["capture_paused"] = True                  # Privacy mode
        assert wait_for(lambda: not m.running(), 5) and m.drain(30)
        n_private = len(m4as()); time.sleep(2)
        assert len(m4as()) == n_private and n_private > n_muted, "nothing kept while private"
    finally:
        rm.shutdown(drain_s=30)
    segs = st._query("SELECT ts_start, ts_end, duration_s, path FROM audio_segments ORDER BY ts_start")
    assert len(segs) >= 3 and all(Path(s["path"]).exists() for s in segs), segs
    assert all(s["ts_end"] > s["ts_start"] and 0.3 <= s["duration_s"] <= 2.2 for s in segs), segs
    first_run = [s for s in segs if s["ts_start"] < segs[0]["ts_start"] + 4]
    gaps = [b["ts_start"] - a["ts_end"] for a, b in zip(first_run, first_run[1:])]
    assert all(abs(g) < 0.35 for g in gaps), gaps               # chunks tile the wall clock while recording
    assert not [f for f in d.rglob("*") if f.is_file()], "no temp audio left"
    linked = st._query("SELECT COUNT(*) n FROM voice_transcripts WHERE audio_segment_id IS NOT NULL")[0]["n"]
    assert linked >= len(segs), linked
    assert rm.status()["audio"]["dir"] == str(ARCHIVE)


def wait_for(fn, timeout=8.0, step=0.1):
    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(step)
    return fn()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time(); fn(); print(f"PASS {name} ({time.time() - t:.1f}s)", flush=True)
    print("audio_retain: OK")
