""".venv/bin/python -m tests.screenrec_test  (or pytest)  (synthetic ffmpeg testsrc; never touches the real screen or vlab.db)"""
import os, sqlite3, subprocess, tempfile, time
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="screenrec-test-"))
os.environ["VLAB_SCREENREC_DIR"] = str(TMP)
from vision_lab import screenrec as sr

SRC = ["-re", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10"]


def test_window_default_and_midnight_cap():
    s, e = sr.parse_window()
    assert sr.datetime.fromtimestamp(s, sr.TZ).isoformat() == "2026-10-16T10:00:00-07:00"
    assert sr.datetime.fromtimestamp(e, sr.TZ).isoformat() == "2026-10-16T23:59:59-07:00"
    s, e = sr.parse_window("2026-10-16T10:00/2026-10-17T03:00")      # can't run past midnight
    assert sr.datetime.fromtimestamp(e, sr.TZ).isoformat() == "2026-10-17T00:00:00-07:00"
    try: sr.parse_window("2026-10-16T10:00/2026-10-16T09:00"); assert False
    except ValueError: pass


def test_in_window_and_remaining():
    w = (100.0, 200.0)
    assert not sr.in_window(99, w) and sr.in_window(100, w) and not sr.in_window(200, w)
    assert sr.remaining_s(150, w) == 50 and sr.remaining_s(250, w) == 0


def test_record_refuses_outside_window():
    assert sr.record(window=(0, 1), source=SRC)["reason"] == "outside window"


def test_record_autostops_and_indexes():
    now = time.time(); w = (now - 1, now + 7)          # window ends in 7 s
    t0 = time.time()
    r = sr.record(window=w, cfg={"segment_s": 2, "fps": 10}, source=SRC, index_path=TMP / "i.db")
    took = time.time() - t0
    assert r["ok"] and took < 12, (r, took)            # stopped at window end on its own
    db = sr.connect(TMP / "i.db")
    rows = db.execute("SELECT path,start_ts,end_ts FROM chunks ORDER BY start_ts").fetchall()
    assert len(rows) >= 3 and all(Path(p).exists() for p, _, _ in rows)
    for (_, _, e1), (_, s2, _) in zip(rows, rows[1:]):
        assert abs(e1 - s2) < 0.2                      # contiguous
    assert rows[-1][2] <= w[1] + 0.5
    # clip spanning two chunks
    a, b = rows[0][1] + 1.0, rows[1][1] + 1.0
    clip = sr.clip_for_range(db, a, b)
    assert [c["path"] for c in clip] == [rows[0][0], rows[1][0]]
    assert abs(clip[0]["seek_s"] - 1.0) < 1e-6 and abs(clip[1]["seek_s"]) < 1e-6
    assert abs(sum(c["duration_s"] for c in clip) - (b - a)) < 1e-3
    # seek offset actually decodes
    d = subprocess.run([sr.FFMPEG, "-v", "error", "-ss", str(clip[0]["seek_s"]), "-i", clip[0]["path"], "-frames:v", "1", "-f", "null", "-"])
    assert d.returncode == 0


def test_screens_link_by_time():
    db = sr.connect(TMP / "link.db")
    db.execute("INSERT INTO chunks(rec_id,path,start_ts,end_ts) VALUES('r','a.mp4',1000,1300),('r','b.mp4',1300,1600)"); db.commit()
    v = TMP / "vlab.db"; c = sqlite3.connect(v)
    c.execute("CREATE TABLE batch_annotations(id INTEGER PRIMARY KEY, source_type TEXT, source_path TEXT, capture_ts REAL)")
    c.executemany("INSERT INTO batch_annotations VALUES(?,?,?,?)", [(1, "screen", "s1.jpg", 1010), (2, "screen", "s2.jpg", 1450), (3, "screen", "s3.jpg", 1700), (4, "camera", "c.jpg", 1020)])
    c.commit()
    out = sr.screens_with_video(db, v, 0, 2000)
    assert [(o["screen_id"], o["video"] and o["video"]["path"], o["video"] and round(o["video"]["seek_s"])) for o in out] == \
        [(1, "a.mp4", 10), (2, "b.mp4", 150), (3, None, None)]


def test_ensure_running_gates():
    assert sr.ensure_running({"screenrec": False}) == "disabled"
    assert sr.ensure_running({"screenrec": True, "screenrec_window": "2020-01-01T10:00/2020-01-01T11:00"}) == "outside window"


def test_guard_kills_past_end():
    p = subprocess.Popen(["sleep", "60"]); (TMP / "recorder.pid").write_text(f"{p.pid} {time.time() - 1}")
    sr.guard(); p.wait(timeout=5); assert p.returncode != 0


# ---------------- privacy ----------------
def _settings_db(value):
    path = TMP / f"vlab-{time.time_ns()}.db"; c = sqlite3.connect(path)
    c.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    if value is not None:
        c.execute("INSERT INTO config VALUES('settings', ?)", (value,))
    c.commit(); c.close(); return path


def test_privacy_state_reader_fails_closed():
    import json
    assert sr.privacy_blocked(_settings_db(json.dumps({"capture_paused": True}))) is True
    assert sr.privacy_blocked(_settings_db(json.dumps({"capture_paused": False}))) is False
    assert sr.privacy_blocked(TMP / "missing.db") is True                  # unreadable -> blocked
    assert sr.privacy_blocked(_settings_db(None)) is True                  # no settings row -> blocked
    assert sr.privacy_blocked(_settings_db("{not json")) is True           # corrupt -> blocked


def test_privacy_on_means_no_start():
    now = time.time(); w = (now - 1, now + 30)
    r = sr.record(window=w, source=SRC, index_path=TMP / "p1.db", privacy=lambda: True)
    assert r == {"ok": False, "reason": "privacy"}
    spawned = []
    assert sr.ensure_running({"screenrec": True, "screenrec_window": None}, privacy=lambda: True,
                             spawn=lambda: spawned.append(1)) in ("privacy", "outside window")
    assert not spawned


def test_unreadable_state_means_no_record():
    now = time.time(); w = (now - 1, now + 30)
    blocked = lambda: sr.privacy_blocked(TMP / "nope" / "vlab.db")
    assert sr.record(window=w, source=SRC, index_path=TMP / "p2.db", privacy=blocked)["reason"] == "privacy"


def test_privacy_toggled_on_mid_recording_stops():
    now = time.time(); w = (now - 1, now + 60)
    t_on = time.time() + 3
    t0 = time.time()
    r = sr.record(window=w, cfg={"segment_s": 1, "fps": 10}, source=SRC, index_path=TMP / "p3.db",
                  privacy=lambda: time.time() >= t_on, poll_s=0.5)
    took = time.time() - t0
    assert r["reason"] == "privacy" and r["chunks"] >= 1, r
    assert took < 3 + 5 + 1, took                                             # stopped within ~5 s
    db = sr.connect(TMP / "p3.db")
    assert db.execute("SELECT status FROM recordings WHERE rec_id=?", (r["rec_id"],)).fetchone()[0] == "privacy-stop"
    assert all(Path(p).exists() for (p,) in db.execute("SELECT path FROM chunks"))


def test_privacy_off_inside_window_starts():
    class P:
        def poll(self): return None
        def terminate(self): pass
    sr._proc = None
    cfg = {"screenrec": True, "screenrec_window": time.strftime("%Y-%m-%dT00:00:00") + "/" + time.strftime("%Y-%m-%dT23:59:59")}
    assert sr.ensure_running(cfg, privacy=lambda: True, spawn=P) == "privacy" and sr._proc is None
    assert sr.ensure_running(cfg, privacy=lambda: False, spawn=P) == "started"
    assert sr.ensure_running(cfg, privacy=lambda: False, spawn=P) == "running"
    sr._proc = None


def test_ffmpeg_cmd_light_defaults():
    cmd = sr.ffmpeg_cmd(TMP, "x", 10)
    j = " ".join(cmd)
    assert "hevc_videotoolbox" in cmd and cmd[cmd.index("-tag:v") + 1] == "hvc1"
    assert cmd[cmd.index("-vf") + 1] == "fps=5,scale=1920:-2" and "-use_wallclock_as_timestamps" in cmd
    assert cmd[cmd.index("-framerate") + 1] == "5"
    assert cmd[cmd.index("-b:v") + 1] == "800k" and "-maxrate" in cmd
    assert "expr:gte(t,n_forced*300)" in cmd and cmd[cmd.index("-segment_time") + 1] == "300"
    assert "empty_moov" in j and "frag_keyframe" in j
    q = sr.ffmpeg_cmd(TMP, "x", 10, {"quality": 40})
    assert "-q:v" in q and "-b:v" not in q


def test_planloop_keys_reach_child_cfg():
    env = sr.recorder_env({"screenrec_fps": 3, "screenrec_bitrate": "500k", "screenrec_bogus": 1})
    os.environ[sr.CFG_ENV] = env[sr.CFG_ENV]
    try:
        assert sr.env_cfg() == {"fps": 3, "bitrate": "500k"}
    finally:
        os.environ.pop(sr.CFG_ENV)
    assert sr.CFG_ENV not in sr.recorder_env({})


def _probe(path):
    r = subprocess.run([sr.FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries",
                        "stream=codec_name,codec_tag_string,width:format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout


def test_4k_source_is_light_split_and_playable():
    d = TMP / "light"; d.mkdir(exist_ok=True)
    src = ["-f", "lavfi", "-i", "testsrc2=size=3840x2160:rate=5"]
    cmd = sr.ffmpeg_cmd(d, "L", 30, {"segment_s": 10}, src)
    assert subprocess.run(cmd, stdin=subprocess.DEVNULL).returncode == 0
    segs = sorted(d.glob("screen-L-*.mp4"))
    assert len(segs) == 3, segs                                  # really splits
    assert len((d / "L.csv").read_text().splitlines()) == 3
    for f in segs:
        rc, out = _probe(f)
        assert rc == 0 and "hevc,hvc1,1920" in out, out
    mbph = sum(f.stat().st_size for f in segs) / 30 * 3600 / 1e6
    assert mbph < 800, mbph                                      # cap honoured (old setup: ~17000 MB/h)


def test_killed_segment_still_playable():
    d = TMP / "crash"; d.mkdir(exist_ok=True)
    src = ["-re", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=5"]
    p = subprocess.Popen(sr.ffmpeg_cmd(d, "C", 60, {"segment_s": 300}, src), stdin=subprocess.DEVNULL)
    time.sleep(9); p.kill(); p.wait()
    f = d / "screen-C-000.mp4"
    rc, out = _probe(f)
    assert rc == 0 and "hevc" in out, out                        # fragmented mp4 (5 s fragments) survives SIGKILL


if __name__ == "__main__":
    fails = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try: f(); print("PASS", n)
            except Exception as e: fails += 1; print("FAIL", n, repr(e))
    raise SystemExit(fails)


def test_screenrec_dir_config_sets_child_env():
    env = sr.recorder_env({"screenrec_dir": "/Volumes/Studio Scratch/screenrec"})
    assert env["VLAB_SCREENREC_DIR"] == "/Volumes/Studio Scratch/screenrec"
    assert "VLAB_SCREENREC_WINDOW" in env
