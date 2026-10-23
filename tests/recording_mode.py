"""recording mode: mic, camera video and screen start and stop together with the RECORDING switch.

Synthetic sources only (ffmpeg lavfi tone / testsrc, numpy frames, `say`), temp dirs and a temp DB.
Never opens the real mic, camera or screen, and never touches the live app, vlab.db or planloop config.

    PYTHONPATH=. .venv/bin/python tests/recording_mode.py      # VLAB_SKIP_WHISPER=1 skips the real-model check
"""
import json, os, signal, shutil, subprocess, sys, tempfile, threading, time, types
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["VLAB_AUDIO_DIR"] = tempfile.mkdtemp(prefix="vlab-audio-test-")   # never the real archive
os.environ["VLAB_NO_CAMERA"] = "1"
from vision_lab import captures, engine, recording, retention, screenrec, voicetx
from vision_lab.store import Store

PY = sys.executable
TONE = ["-re", "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=16000"]
SCREEN_SRC = ["-re", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=5"]


def tmp(name):
    return Path(tempfile.mkdtemp(prefix=f"recmode-{name}-"))


def store(private=False):
    st = Store(str(tmp("db") / "vlab.db"))
    st.set_config("settings", {"capture_paused": private, "camera_paused": private})
    return st


def fake_stt(audio):
    return [{"start": 0.1, "end": min(1.0, len(audio) / 16000), "text": " synthetic chunk", "avg_logprob": -0.2,
             "no_speech_prob": 0.01}]


def files(d, exts=None):
    return [f for f in Path(d).rglob("*") if f.is_file() and (exts is None or f.suffix.lower() in exts)]


def audio_left(d):
    return files(d, voicetx.AUDIO_EXT + (".part",))


def frame(i, w=1920, h=1080):
    f = np.zeros((h, w, 3), np.uint8)
    f[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    x = (i * 23) % (w - 200)
    f[300:500, x:x + 200] = (40, 200, 240)
    return f


class Feeder:
    """Plays the camera thread: pushes synthetic frames into every registered sink at ~15 fps."""
    def __init__(self, sinks):
        self.sinks, self.stop, self.n = sinks, threading.Event(), 0
        self.t = threading.Thread(target=self.run, daemon=True); self.t.start()
    def run(self):
        while not self.stop.is_set():
            for s in list(self.sinks):
                s(frame(self.n), time.time())
            self.n += 1; time.sleep(1 / 15)
    def close(self):
        self.stop.set(); self.t.join(2)


def ffprobe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True)
    return float(out.stdout.strip() or 0)


def fake_engine(st):
    """The real Engine.set_capture_paused / set_camera_paused code on a light object (no models, no camera)."""
    cam = types.SimpleNamespace(paused=False, sinks=[], pause=lambda: setattr(cam, "paused", True) or True,
                                resume=lambda: setattr(cam, "paused", False))
    e = types.SimpleNamespace(store=st, settings={"camera": 0, "capture_paused": False}, camera=cam,
                              capture_listeners=[], event=lambda kind, text, *a, **k: st.add_event(kind, text, ts=k.get("ts")))
    e.set_capture_paused = lambda p: engine.Engine.set_capture_paused(e, p)
    e.set_camera_paused = lambda p: engine.Engine.set_camera_paused(e, p)
    return e


def screen_spawn(env):
    code = ("import json,sys; from vision_lab import screenrec as s; "
            f"print(json.dumps(s.record(cfg={{'segment_s': 2, 'fps': 5}}, source={SCREEN_SRC!r}, poll_s=0.3)))")
    return subprocess.Popen([PY, "-c", code], env={**env, "PYTHONPATH": str(Path(__file__).resolve().parents[1])})


def wait_for(fn, timeout=8.0, step=0.1):
    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(step)
    return fn()


# ------------------------------------------------------------------ mic
def test_mic_chunks_transcribed_then_deleted():
    st, d = store(), tmp("voice")
    m = recording.MicRecorder(st, voice_tmp=d, source=TONE, chunk_s=1.5, max_chunk_s=2.0,
                              transcribe=lambda f: voicetx.transcribe_file(st, f, stt=fake_stt))
    t0 = time.time(); m.start()
    assert m.running()
    time.sleep(5.5)
    m.stop("test"); t1 = time.time()
    assert m.drain(30) and not m.running()
    rows = st._query("SELECT ts_start, ts_end, text FROM voice_transcripts ORDER BY ts_start")
    assert len(rows) >= 2 and all(r["text"] == "synthetic chunk" for r in rows), rows
    assert all(t0 - 1 <= r["ts_start"] <= t1 for r in rows), (t0, t1, rows)
    gaps = [b["ts_start"] - a["ts_start"] for a, b in zip(rows, rows[1:])]
    assert all(1.4 <= g <= 2.6 for g in gaps), gaps                       # chunk starts follow the audio clock
    assert not audio_left(d), audio_left(d)                               # temp chunks gone (an m4a copy is kept in VLAB_AUDIO_DIR)
    kinds = [r["kind"] for r in st._query("SELECT kind FROM events WHERE kind LIKE 'mic_%' ORDER BY id")]
    assert kinds == ["mic_started", "mic_stopped"], kinds
    assert not st._query("SELECT 1 FROM events WHERE modality='audio'"), "no room-sound events"


def test_mic_crash_leftovers_recovered_and_deleted():
    st, d = store(), tmp("voice-crash")
    (d / ".staging").mkdir(parents=True)
    tone = (0.3 * np.sin(np.arange(32000) / 16000 * 2 * np.pi * 300)).astype(np.float32)
    (d / ".staging" / f"voice-{int(time.time() * 1000) - 60000}.part").write_bytes(tone.tobytes())   # half-written
    (d / f"voice-{int(time.time() * 1000) - 30000}.pcm").write_bytes(tone.tobytes())                 # never transcribed
    m = recording.MicRecorder(st, voice_tmp=d, transcribe=lambda f: voicetx.transcribe_file(st, f, stt=fake_stt))
    m.recover()
    assert m.drain(30) and not audio_left(d), audio_left(d)
    assert st._query("SELECT COUNT(*) n FROM voice_transcripts")[0]["n"] == 2


def test_mic_ffmpeg_dies_with_server():
    """SIGKILL the owning process: ffmpeg gets SIGPIPE and stops, so the mic never keeps recording orphaned."""
    d = tmp("voice-orphan")
    code = ("import sys,time; sys.path.insert(0, '.'); from vision_lab import recording as r; from vision_lab.store import Store; "
            f"st=Store({str(tmp('db') / 'o.db')!r}); "
            f"m=r.MicRecorder(st, voice_tmp={str(d)!r}, source={TONE!r}, transcribe=lambda f: None); m.start(); "
            "print(m.proc.pid, flush=True); time.sleep(60)")
    p = subprocess.Popen([PY, "-c", code], stdout=subprocess.PIPE, text=True, cwd=Path(__file__).resolve().parents[1])
    ff = int(p.stdout.readline())
    time.sleep(1.0)
    p.kill(); p.wait()
    assert wait_for(lambda: subprocess.run(["kill", "-0", str(ff)], capture_output=True).returncode != 0, 5), \
        "mic ffmpeg kept running after its owner died"
    shutil.rmtree(d, ignore_errors=True)        # the orphan test leaves a .part (recovered on a real restart)


def test_real_whisper_streamed_speech():
    if os.environ.get("VLAB_SKIP_WHISPER") or not shutil.which("say"):
        print("skip real whisper"); return
    st, d, a = store(), tmp("voice-real"), tmp("say") / "s.aiff"
    subprocess.run(["say", "-o", str(a), "The garden gate is open and the kettle is on."], check=True)
    m = recording.MicRecorder(st, voice_tmp=d, source=["-re", "-i", str(a)], chunk_s=8, max_chunk_s=12)
    m.start(); assert wait_for(lambda: not m.running(), 15)                # the file ends: ffmpeg exits
    m.stop("test"); assert m.drain(120)
    text = " ".join(r["text"] for r in st._query("SELECT text FROM voice_transcripts")).lower()
    assert "garden" in text and "kettle" in text, text
    assert not audio_left(d)


# ------------------------------------------------------------------ camera video
def test_video_segments_index_events_and_timeline():
    st, d = store(), tmp("camrec")
    v = recording.VideoRecorder(st, outdir=d, cfg={"segment_s": 2})
    f = Feeder([v.sink])
    v.start(); t0 = time.time()
    time.sleep(5.5)
    v.stop("test"); t1 = time.time(); f.close()
    mp4 = sorted(files(d, (".mp4",)))
    assert len(mp4) >= 2, mp4
    assert all(ffprobe_duration(p) > 0.5 for p in mp4), [ffprobe_duration(p) for p in mp4]
    db = screenrec.connect(d / "index.db")
    chunks = db.execute("SELECT path, start_ts, end_ts FROM chunks ORDER BY start_ts").fetchall()
    rec = db.execute("SELECT start_ts, end_ts, status FROM recordings").fetchall()
    db.close()
    assert len(chunks) == len(mp4) and rec[0][2] == "test" and rec[0][1] > rec[0][0], (chunks, rec)
    assert abs(chunks[0][1] - t0) < 1.5 and abs(chunks[-1][2] - t1) < 2.5, (t0, t1, chunks)
    ev = st._query("SELECT kind, ts FROM events WHERE kind LIKE 'video_%' ORDER BY id")
    assert [e["kind"] for e in ev] == ["video_started", "video_stopped"], ev
    bars = captures.video_bars(st, t0 - 3600, t1 + 3600)
    assert len(bars) == 1 and abs(bars[0][0] - ev[0]["ts"]) < 1e-6 and abs(bars[0][1] - ev[1]["ts"]) < 1e-6, bars
    s = 1920 / 1280
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height,codec_name",
                          "-of", "json", str(mp4[0])], capture_output=True, text=True)
    info = json.loads(out.stdout)["streams"][0]
    assert info["width"] == 1280 and info["height"] == round(1080 / s) and info["codec_name"] == "hevc", info


def test_video_open_recording_recovered_after_crash():
    st, d = store(), tmp("camrec-crash")
    v = recording.VideoRecorder(st, outdir=d, cfg={"segment_s": 1})
    f = Feeder([v.sink]); v.start(); time.sleep(3)
    with v.lock:                                       # simulate a crash: ffmpeg killed, no clean stop, no event
        v.active = False; v.proc.kill(); v.proc = None; v.rec = None
    f.close()
    db = screenrec.connect(d / "index.db")
    assert db.execute("SELECT COUNT(*) FROM recordings WHERE end_ts IS NULL").fetchone()[0] == 1
    db.close()
    v2 = recording.VideoRecorder(st, outdir=d)
    assert v2.recover() == 1
    db = screenrec.connect(d / "index.db")
    end, status = db.execute("SELECT end_ts, status FROM recordings").fetchone()
    db.close()
    assert status == "crashed" and end is not None
    assert [e["kind"] for e in st._query("SELECT kind FROM events WHERE kind LIKE 'video_%' ORDER BY id")] == \
        ["video_started", "video_stopped"]


def test_video_retention_wired_default_forever():
    st, d = store(), tmp("camrec-ret")
    day = d / "2025-01-01"; day.mkdir(parents=True)
    old = day / "cam-x-000.mp4"; old.write_bytes(b"x")
    t = time.time() - 400 * 86400; os.utime(old, (t, t))
    db = screenrec.connect(d / "index.db")
    db.execute("INSERT INTO chunks(rec_id,path,start_ts,end_ts) VALUES('x',?,?,?)", (str(old), t, t + 300)); db.commit(); db.close()
    v = recording.VideoRecorder(st, outdir=d)
    os.environ.pop("VLAB_VIDEO_KEEP_DAYS", None)
    assert v.prune() == [] and old.exists(), "default keeps video forever"
    os.environ["VLAB_VIDEO_KEEP_DAYS"] = "30"
    try:
        assert v.prune() == [old] and not old.exists()
    finally:
        os.environ.pop("VLAB_VIDEO_KEEP_DAYS")
    db = screenrec.connect(d / "index.db"); assert db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0; db.close()


# ------------------------------------------------------------------ screen
def test_screen_child_starts_and_stops_on_request():
    st, d = store(), tmp("screenrec")
    s = recording.ScreenRecorder(st, cfg={"screenrec_dir": str(d)}, spawn=screen_spawn)
    s.start(); assert s.running()
    assert wait_for(lambda: files(d, (".mp4",)), 10)
    time.sleep(2.5)
    s.stop("test"); assert not s.running()
    db = sqlite3_ro(d / "index.db")
    status = db.execute("SELECT status FROM recordings").fetchall(); n = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    db.close()
    assert status == [("stopped",)] and n >= 1, (status, n)
    assert not (d / "recorder.pid").exists()
    assert [e["kind"] for e in st._query("SELECT kind FROM events WHERE kind LIKE 'screen_%' ORDER BY id")] == \
        ["screen_started", "screen_stopped"]


def sqlite3_ro(p):
    import sqlite3
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def test_planloop_recorder_is_never_doubled():
    d = tmp("screenrec-other")
    holder = subprocess.Popen(["sleep", "30"]); (d / "recorder.pid").write_text(f"{holder.pid} {time.time() + 60}")
    try:
        assert screenrec.recorder_alive(d)
        spawned = []
        s = recording.ScreenRecorder(store(), cfg={"screenrec_dir": str(d)}, spawn=lambda env: spawned.append(env))
        s.start(); assert spawned == [] and not s.running()
        cfg = {"screenrec": True, "screenrec_dir": str(d),
               "screenrec_window": time.strftime("%Y-%m-%dT00:00:00") + "/" + time.strftime("%Y-%m-%dT23:59:59")}
        assert screenrec.ensure_running(cfg, privacy=lambda: False, spawn=lambda: 1 / 0) == "running (other owner)"
    finally:
        holder.kill()


def test_video_and_screen_stop_when_owner_dies():
    """SIGKILL the owner: video ffmpeg sees EOF and finalizes a playable file; the screenrec child stops itself."""
    d, sd = tmp("camrec-orphan"), tmp("screenrec-orphan")
    st = store()
    code = ("import sys,time,json; sys.path.insert(0,'.'); import numpy as np; from vision_lab import recording as r; "
            "from vision_lab.store import Store; "
            f"st=Store({str(tmp('db') / 'o.db')!r}); st.set_config('settings', {{'capture_paused': False}}); "
            f"v=r.VideoRecorder(st, outdir={str(d)!r}, cfg={{'segment_s': 60}}); v.start(); "
            f"s=r.ScreenRecorder(st, cfg={{'screenrec_dir': {str(sd)!r}}}, spawn=lambda env: __import__('subprocess').Popen("
            f"[sys.executable, '-c', {SCREEN_CHILD!r}], env=env)); s.start(); "
            "f=np.zeros((720,1280,3),np.uint8); i=0\n"
            "while True:\n"
            "    f[:, (i*7)%1280] = 255; v.sink(f.copy(), time.time()); i+=1\n"
            "    if i == 30: print(json.dumps([v.proc.pid, s.proc.pid]), flush=True)\n"
            "    time.sleep(0.07)")
    p = subprocess.Popen([PY, "-c", code], stdout=subprocess.PIPE, text=True, cwd=Path(__file__).resolve().parents[1])
    ff, child = json.loads(p.stdout.readline())
    time.sleep(2)
    p.kill(); p.wait()
    gone = lambda pid: subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode != 0
    assert wait_for(lambda: gone(ff), 8), "video ffmpeg kept running after its owner died"
    assert wait_for(lambda: gone(child), 8), "screenrec child kept running after its owner died"
    mp4 = files(d, (".mp4",))
    assert len(mp4) == 1 and ffprobe_duration(mp4[0]) > 1.0, mp4


SCREEN_CHILD = ("from vision_lab import screenrec as s; "
                f"s.record(cfg={{'segment_s': 2, 'fps': 5}}, source={SCREEN_SRC!r}, poll_s=0.3)")


# ------------------------------------------------------------------ the switch
def make_mode(st, root, mic_on=True, enabled=True):
    e = fake_engine(st)
    mic = recording.MicRecorder(st, voice_tmp=root / "voice", source=TONE, chunk_s=1.0, max_chunk_s=1.5,
                                transcribe=lambda f: voicetx.transcribe_file(st, f, stt=fake_stt))
    video = recording.VideoRecorder(st, outdir=root / "camrec", cfg={"segment_s": 2})
    screen = recording.ScreenRecorder(st, cfg={"screenrec_dir": str(root / "screenrec")}, spawn=screen_spawn)
    flag = {"mic": mic_on}
    rm = recording.RecordingMode(e, st, enabled=enabled, mic=mic, video=video, screen=screen,
                                 mic_allowed=lambda: flag["mic"], poll_s=0.2, restart_s=0.5)
    return rm, e, flag


def snapshot(root, st):
    return ({str(f): f.stat().st_size for f in files(root)},
            st._query("SELECT (SELECT COUNT(*) FROM voice_transcripts) t, (SELECT MAX(id) FROM events) e")[0])


def running(rm):
    return {c.name: c.running() for c in rm.components}


def test_one_switch_starts_and_stops_all_three():
    st, root = store(), tmp("switch")
    voicetx.ensure_schema(st)
    rm, e, flag = make_mode(st, root)
    assert e.camera.sinks == [rm.video.sink] and e.capture_listeners == [rm.on_capture_change]
    feeder = Feeder(e.camera.sinks)
    try:
        assert rm.start()
        assert wait_for(lambda: all(running(rm).values()), 5), running(rm)
        assert wait_for(lambda: files(root / "camrec", (".mp4",)) and files(root / "screenrec", (".mp4",)), 10)
        time.sleep(3)
        # Privacy on: all three stop at once (the real Engine.set_capture_paused calls the hook first)
        t = time.time(); e.set_capture_paused(True); took = time.time() - t
        assert not any(running(rm).values()) and took < 12, (running(rm), took)
        assert rm.mic.drain(30)
        files_then, db_then = snapshot(root, st)
        time.sleep(3)                                   # frames keep arriving from the camera thread meanwhile
        files_now, db_now = snapshot(root, st)
        assert files_now == files_then, "something was written while private"
        assert db_now == db_then, "rows were written while private"
        assert not any(running(rm).values())
        assert not audio_left(root / "voice")
        stops = [r["kind"] for r in st._query("SELECT kind FROM events WHERE kind LIKE '%_stopped' AND ts >= ?", (t - 1,))]
        assert sorted(stops) == ["mic_stopped", "screen_stopped", "video_stopped"], stops
        # Privacy off: all three come back
        e.set_capture_paused(False)
        assert wait_for(lambda: all(running(rm).values()), 5), running(rm)
        # Mic button off: only the mic stops
        flag["mic"] = False
        assert wait_for(lambda: running(rm) == {"mic": False, "video": True, "screen": True}, 5), running(rm)
        flag["mic"] = True
        assert wait_for(lambda: all(running(rm).values()), 5)
        # Camera pause stops everything too
        e.set_camera_paused(True)
        assert not any(running(rm).values()), running(rm)
        e.set_camera_paused(False)
        assert wait_for(lambda: all(running(rm).values()), 5)
        time.sleep(1.5)
        # Quit: everything stops, no audio left behind
        rm.shutdown(drain_s=30)
        assert not any(running(rm).values()) and not audio_left(root / "voice")
        time.sleep(1); assert not any(running(rm).values()), "came back after quit"
    finally:
        feeder.close(); rm.closing = True; rm.stop_all("test cleanup")
    assert st._query("SELECT COUNT(*) n FROM voice_transcripts")[0]["n"] >= 2
    bars = captures.video_bars(st, time.time() - 3600, time.time() + 60)
    assert len(bars) >= 3, bars                            # three recording sessions with gaps between them
    assert not st._query("SELECT 1 FROM events WHERE modality='audio'")


def test_disabled_or_private_never_records():
    st, root = store(), tmp("off")
    rm, e, _ = make_mode(st, root, enabled=False)
    assert rm.start() is False
    time.sleep(1); assert not any(running(rm).values()) and not files(root)
    st2, root2 = store(private=True), tmp("private")
    rm2, e2, _ = make_mode(st2, root2)
    e2.settings["capture_paused"] = True
    rm2.start(); time.sleep(1.5)
    assert not any(running(rm2).values()) and not [f for f in files(root2) if f.name != "index.db"], files(root2)
    rm2.shutdown()


def test_env_flags_drop_components():
    st = store()
    e = fake_engine(st)
    os.environ["VLAB_NO_MIC"] = "1"; os.environ["VLAB_NO_SCREEN"] = "1"
    try:
        rm = recording.RecordingMode(e, st, enabled=False)
        assert rm.mic is None and rm.video is None and rm.screen is None   # VLAB_NO_CAMERA is set for this file
        assert rm.status()["components"] == {"mic": None, "video": None, "screen": None}
        assert rm.status()["video_keep_days"] == "forever"
    finally:
        os.environ.pop("VLAB_NO_MIC"); os.environ.pop("VLAB_NO_SCREEN")
    assert not recording.enabled_by_env() or os.environ.get("VLAB_RECORDING_MODE") == "1"


def test_none_means_off_and_self_ended_capture_is_logged():
    """screen=None must never fall back to the real screen recorder (that bug captured the real screen once)."""
    st, root = store(), tmp("none")
    e = fake_engine(st)
    a = tmp("say") / "short.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=1",
                    "-ar", "16000", str(a)], check=True)
    mic = recording.MicRecorder(st, voice_tmp=root / "voice", source=["-re", "-i", str(a)],
                                transcribe=lambda f: voicetx.transcribe_file(st, f, stt=fake_stt))
    rm = recording.RecordingMode(e, st, enabled=True, mic=mic, video=None, screen=None, mic_allowed=lambda: True,
                                 poll_s=0.2, restart_s=60)
    assert rm.components == [mic] and rm.screen is None and rm.video is None
    rm.start()
    time.sleep(3.5)                                    # the 1 s source ends; no restart within restart_s
    rm.shutdown(drain_s=30)
    kinds = [r["kind"] for r in st._query("SELECT kind FROM events WHERE kind LIKE 'mic_%' OR kind LIKE 'screen_%' ORDER BY id")]
    assert kinds == ["mic_started", "mic_stopped"], kinds
    assert not audio_left(root / "voice")


def test_server_wiring_and_permission_tree():
    root = Path(__file__).resolve().parents[1]
    srv, run = (root / "vision_lab/server.py").read_text(), (root / "run.sh").read_text()
    assert "REC = recording.RecordingMode(engine, store)" in srv and "REC.start()" in srv
    assert '@app.on_event("shutdown")' in srv and "REC.shutdown(" in srv and '@app.get("/api/recording")' in srv
    assert 'export VLAB_RECORDING_MODE="${VLAB_RECORDING_MODE:-1}"' in run and run.index("VLAB_RECORDING_MODE") < run.index("exec ")
    sw = (root / "macapp/main.swift").read_text()
    assert 'cmd: "exec ./run.sh' in sw, "the server (and so the mic ffmpeg) must stay a child of Vision Lab.app"
    plist = (root / "macapp/Info.plist").read_text()
    for k in ("NSMicrophoneUsageDescription", "NSCameraUsageDescription", "NSScreenCaptureUsageDescription"):
        assert k in plist, k


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time(); fn(); print(f"PASS {name} ({time.time() - t:.1f}s)", flush=True)
    print("recording_mode: OK")
