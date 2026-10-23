"""Recording-mode voice: captured audio -> Whisper transcript rows (voice log + Captures) and the audio is deleted.
Also the storage rules: screens kept forever (prune_screens off), video kept forever by default.

    .venv/bin/python tests/voice_transcribe.py          # VLAB_SKIP_WHISPER=1 skips the real-model check
"""
import os, shutil, subprocess, sys, tempfile, time, wave
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["VLAB_AUDIO_DIR"] = tempfile.mkdtemp(prefix="vlab-audio-test-")   # never the real archive
from vision_lab import captures, planloop, retention, voicelog, voicetx
from vision_lab.store import Store

TODAY_10 = datetime.now().replace(hour=0, minute=10, second=0, microsecond=0).timestamp()


def store():
    return Store(str(Path(tempfile.mkdtemp()) / "vlab.db"))


def write_wav(path, audio, sr=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def tone(seconds, freq=440.0, amp=0.3, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def audio_files(d):
    return [f for f in Path(d).rglob("*") if f.suffix.lower() in voicetx.AUDIO_EXT]


def fake_stt(audio):
    n = len(audio) / 16000
    return [{"start": 0.5, "end": 2.0, "text": " Testing the voice log.", "avg_logprob": -0.2, "no_speech_prob": 0.01},
            {"start": 4.0, "end": 5.5, "text": " Second sentence here.", "avg_logprob": -0.3, "no_speech_prob": 0.02},
            {"start": 6.0, "end": n, "text": " Thank you.", "avg_logprob": -1.6, "no_speech_prob": 0.9}]   # hallucination


def test_file_to_transcript_rows_and_deleted():
    st, d = store(), Path(tempfile.mkdtemp())
    f = d / f"voice-{int(TODAY_10 * 1000)}.wav"
    write_wav(f, tone(8))
    ev0 = st._query("SELECT COUNT(*) n FROM events")[0]["n"]
    r = voicetx.transcribe_file(st, f, stt=fake_stt)
    assert r["transcripts"] == 2 and "sounds" not in r, r
    assert st._query("SELECT COUNT(*) n FROM events")[0]["n"] == ev0, "no room-sound events"
    assert not f.exists() and not audio_files(d), "raw audio must be deleted after transcription"
    rows = st._query("SELECT ts_start, ts_end, text FROM voice_transcripts ORDER BY ts_start")
    assert [x["text"] for x in rows] == ["Testing the voice log.", "Second sentence here."]
    assert rows[0]["ts_start"] == TODAY_10 + 0.5 and rows[1]["ts_end"] == TODAY_10 + 5.5
    # an old room-sound row in the DB (10/11-10/12 style) stays in the DB but is not shown
    st.add_event("sound", "Heard speech (Speech, 80%)", ts=TODAY_10 + 3, modality="audio", cls="speech", source_key="old1")
    # voice log: only transcript rows (what was said), newest first
    log = voicelog.build_log(st, before=TODAY_10 + 3600)
    entries = log["days"][0]["entries"]
    assert [e["kind"] for e in entries] == ["transcript", "transcript"], entries
    assert [e["text"] for e in entries] == ["Second sentence here.", "Testing the voice log."]
    assert "sounds" not in log["sources"] and "transcripts" in log["sources"] and log["days"][0]["count"] == 2
    assert st._query("SELECT COUNT(*) n FROM events WHERE modality='audio'")[0]["n"] == 1, "old sound rows are not deleted"
    # captures audio row = the transcript segments only (the sound event at +3 s adds nothing)
    bars = captures.audio_bars(st, TODAY_10 - 60, TODAY_10 + 3600)
    assert bars == [[TODAY_10 + 0.5, TODAY_10 + 2.5], [TODAY_10 + 4.0, TODAY_10 + 6.0]], bars


def test_silence_skips_whisper_and_still_deletes():
    st, d = store(), Path(tempfile.mkdtemp())
    f = d / f"voice-{int(TODAY_10 * 1000)}.wav"; write_wav(f, np.zeros(16000 * 3, np.float32))
    called = []
    r = voicetx.transcribe_file(st, f, stt=lambda a: called.append(1) or [])
    assert r["transcripts"] == 0 and not called and not f.exists()


def test_failure_still_deletes_and_logs():
    st, d = store(), Path(tempfile.mkdtemp())
    f = d / f"voice-{int(TODAY_10 * 1000)}.m4a"; f.write_bytes(b"not audio")
    r = voicetx.transcribe_file(st, f, stt=fake_stt)
    assert "error" in r and not f.exists(), r
    ev = st._query("SELECT kind, text FROM events WHERE kind='transcribe_error'")
    assert len(ev) == 1 and "not audio" not in ev[0]["text"]


def test_crash_recovery_leaves_no_audio():
    st, d = store(), Path(tempfile.mkdtemp())
    for k in range(3):                                         # leftovers from a "crash", one fresh (unsettled)
        write_wav(d / f"voice-{int((TODAY_10 + k * 10) * 1000)}.wav", tone(3, 300 + k * 100))
    (d / "voice-1792000000000.pcm").write_bytes(tone(2).tobytes())
    (d / "notes.txt").write_text("keep me")                    # non-audio files are left alone
    out = voicetx.recover(st, d, stt=fake_stt)
    assert len(out) == 4 and not audio_files(d), (out, audio_files(d))
    assert (d / "notes.txt").exists()
    assert st._query("SELECT COUNT(*) n FROM voice_transcripts")[0]["n"] == 4 * 2


def test_in_memory_path_writes_no_files():
    st = store(); tmp = Path(tempfile.mkdtemp()); before = set(tmp.iterdir())
    r = voicetx.transcribe_pcm(st, tone(6), TODAY_10, stt=fake_stt)
    assert r["transcripts"] == 2 and set(tmp.iterdir()) == before


def test_settle_time_and_name_parsing():
    d = Path(tempfile.mkdtemp()); f = d / "voice-1792531200123.wav"; write_wav(f, tone(1))
    assert voicetx.pending(d, settle_s=60) == [] and voicetx.pending(d, settle_s=0) == [f]
    assert voicetx.start_ts_from_name(f) == 1792531200.123 and voicetx.start_ts_from_name("x.wav") is None
    f.unlink()


def test_real_whisper_on_synthetic_speech():
    if os.environ.get("VLAB_SKIP_WHISPER") or not shutil.which("say"):
        print("skip real whisper"); return
    st, d = store(), Path(tempfile.mkdtemp())
    aiff = d / f"voice-{int(TODAY_10 * 1000)}.aiff"
    subprocess.run(["say", "-o", str(aiff), "The kitchen light is on and the dog is asleep."], check=True)
    r = voicetx.transcribe_file(st, aiff)
    text = " ".join(x["text"] for x in st._query("SELECT text FROM voice_transcripts")).lower()
    assert "kitchen" in text and "dog" in text, (r, text)
    assert not audio_files(d)


def test_screens_kept_forever():
    d = Path(tempfile.mkdtemp())
    old = d / "2020-01-01"; old.mkdir(); (old / "a.jpg").write_bytes(b"x")
    planloop.SCREEN_DIR = d
    assert planloop.DEFAULTS["screen_keep_days"] is None
    assert planloop.prune_screens({"screen_keep_days": 7}) == 0 and old.exists(), "old screen_keep_days must not prune"
    assert planloop.prune_screens(dict(planloop.DEFAULTS)) == 0 and old.exists()
    assert planloop.prune_screens({"screen_keep_days": 7, "screen_prune": True}) == 1 and not old.exists()
    src = (Path(__file__).resolve().parents[1] / "vision_lab" / "screenrec.py").read_text()
    assert "unlink" not in src.replace('(DATA / "recorder.pid").unlink', ""), "screenrec must never delete segments"


def test_video_retention_default():
    """Camera video is kept forever by default, like screens; deletes only with an explicit number."""
    d = Path(tempfile.mkdtemp()); a, b, j = d / "old.mp4", d / "new.mp4", d / "old.jpg"
    for f in (a, b, j): f.write_bytes(b"x")
    t = time.time() - 400 * 86400; os.utime(a, (t, t)); os.utime(j, (t, t))
    for v in (None, "", "0", "none", "forever", "keep", "-3", "junk"):
        if v is None: os.environ.pop("VLAB_VIDEO_KEEP_DAYS", None)
        else: os.environ["VLAB_VIDEO_KEEP_DAYS"] = v
        assert retention.video_keep_days() is None, v
        assert retention.prune_video(d) == [] and a.exists(), f"must keep forever for {v!r}"
    os.environ.pop("VLAB_VIDEO_KEEP_DAYS", None)
    assert retention.policy()["video_days"] == "forever"
    os.environ["VLAB_VIDEO_KEEP_DAYS"] = "3"
    assert retention.video_keep_days() == 3.0
    assert retention.prune_video(d) == [a] and not a.exists() and b.exists() and j.exists()
    os.environ.pop("VLAB_VIDEO_KEEP_DAYS")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("voice_transcribe: OK")
