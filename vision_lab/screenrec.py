"""Continuous full-screen video capture for a scheduled window (local-only, screens are sensitive).

Design
- Recorder: ffmpeg avfoundation screen grab -> h264_videotoolbox, rolling segment files
  data/screenrec/YYYY-MM-DD/screen-<rec_id>-NNN.mp4 (data/ is gitignored).
- Permission: must be spawned from a process the Vision Lab.app owns (planloop), so TCC Screen
  Recording is attributed to com.houghtonlabs.visionlab. A bare launchd/Terminal launch would need a new grant.
- Index: data/screenrec/index.db (separate from vlab.db, so no vlab.db schema change). Each chunk row has
  wall-clock [start_ts, end_ts). Screens link to chunks by time (batch_annotations.capture_ts).
- Auto-stop, three layers: (1) ffmpeg -t <seconds until end>; (2) watchdog thread SIGTERM/SIGKILL at end;
  (3) launchd guard (macapp/launchd) that pkills anything tagged vlab-screenrec at 00:00:30 and after.
"""
import csv, json, os, signal, sqlite3, subprocess, sys, threading, time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Los_Angeles")
ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("VLAB_SCREENREC_DIR", ROOT / "data" / "screenrec"))
FFMPEG = os.environ.get("VLAB_FFMPEG", "/opt/homebrew/bin/ffmpeg")
TAG = "vlab-screenrec"
VLAB_DB = os.environ.get("VLAB_DB") or str(ROOT / "data" / "vlab.db")
PRIVACY_POLL_S = 2.0


# ---------------- privacy ----------------
def privacy_blocked(db_path=None):
    """True if recording must NOT happen. Source of truth = what /api/privacy reports:
    vlab.db config 'settings'.capture_paused. Read-only. Fails closed: any read error -> True."""
    try:
        path = db_path or os.environ.get("VLAB_DB") or VLAB_DB
        if not Path(path).exists():
            return True
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=3)
        try:
            row = con.execute("SELECT value FROM config WHERE key='settings'").fetchone()
        finally:
            con.close()
        if not row:
            return True
        st = json.loads(row[0])
        if not isinstance(st, dict):
            return True
        return bool(st.get("capture_paused") or st.get("camera_paused") and "capture_paused" not in st)
    except Exception:
        return True


def _log(msg):
    print(f"screenrec: {datetime.now(TZ).isoformat(timespec='seconds')} {msg}", flush=True)

# Single config value for the window. Override with VLAB_SCREENREC_WINDOW="2026-10-17T10:00/2026-10-17T23:59:59".
DEFAULT_WINDOW = "2026-10-16T10:00:00/2026-10-16T23:59:59"
# 4K h264 at "3M" really wrote ~17 GB/h (VideoToolbox ignored -b:v without maxrate) and never split.
# Now HEVC (hvc1), scaled to 1920 wide, 5 fps, capped bitrate, forced keyframes at each segment boundary,
# fragmented mp4 so a crash/kill still leaves playable files. Override via planloop.json "screenrec_<key>".
DEFAULTS = {"fps": 5, "segment_s": 300, "codec": "hevc_videotoolbox", "bitrate": "800k", "maxrate": "1200k",
            "width": 1920, "quality": None, "screen_device": "Capture screen 0", "capture_cursor": 1}
CFG_ENV = "VLAB_SCREENREC_CFG"   # JSON of DEFAULTS overrides passed planloop -> recorder child


def parse_window(spec=None):
    spec = spec or os.environ.get("VLAB_SCREENREC_WINDOW") or DEFAULT_WINDOW
    a, b = spec.split("/")
    s = datetime.fromisoformat(a); e = datetime.fromisoformat(b)
    s = s if s.tzinfo else s.replace(tzinfo=TZ); e = e if e.tzinfo else e.replace(tzinfo=TZ)
    if e <= s:
        raise ValueError("window end must be after start")
    # hard cap: never past the midnight that follows the start day
    midnight = (s.astimezone(TZ) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return s.timestamp(), min(e.timestamp(), midnight.timestamp())


def in_window(now, window):
    return window[0] <= now < window[1]


def remaining_s(now, window):
    return max(0.0, window[1] - now)


# ---------------- index ----------------
DDL = """CREATE TABLE IF NOT EXISTS recordings (rec_id TEXT PRIMARY KEY, start_ts REAL, end_ts REAL,
  window_end REAL, fps REAL, codec TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, rec_id TEXT, path TEXT UNIQUE, start_ts REAL, end_ts REAL);
CREATE INDEX IF NOT EXISTS chunks_time ON chunks(start_ts, end_ts);"""


def connect(path=None):
    path = Path(path or DATA / "index.db"); path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path); db.executescript(DDL); return db


def ingest_segment_list(db, rec_id, rec_start, list_csv):
    """ffmpeg segment_list CSV rows: filename,start,end (seconds from recording start)."""
    p = Path(list_csv)
    if not p.exists():
        return 0
    n = 0
    for row in csv.reader(p.read_text().splitlines()):
        if len(row) < 3:
            continue
        f = p.parent / row[0]
        db.execute("INSERT OR REPLACE INTO chunks(rec_id,path,start_ts,end_ts) VALUES(?,?,?,?)",
                   (rec_id, str(f), rec_start + float(row[1]), rec_start + float(row[2])))
        n += 1
    db.commit(); return n


def clip_for_range(db, t0, t1):
    """Return [{path, seek_s, duration_s, start_ts, end_ts}] covering [t0, t1) across chunks, in order."""
    out = []
    for path, s, e in db.execute("SELECT path,start_ts,end_ts FROM chunks WHERE end_ts>? AND start_ts<? ORDER BY start_ts", (t0, t1)):
        a, b = max(t0, s), min(t1, e)
        out.append({"path": path, "seek_s": round(a - s, 3), "duration_s": round(b - a, 3), "start_ts": a, "end_ts": b})
    return out


def chunk_at(db, ts):
    r = clip_for_range(db, ts, ts + 1e-3)
    return r[0] if r else None


def screens_with_video(db, vlab_db, t0, t1):
    """Link existing screenshots (batch_annotations, source_type='screen') to video chunk + seek offset by time."""
    v = sqlite3.connect(vlab_db)
    rows = v.execute("SELECT id, source_path, capture_ts FROM batch_annotations WHERE source_type='screen' "
                     "AND capture_ts>=? AND capture_ts<? ORDER BY capture_ts", (t0, t1)).fetchall()
    return [{"screen_id": i, "screen_path": p, "capture_ts": ts, "video": chunk_at(db, ts)} for i, p, ts in rows]


# ---------------- recorder ----------------
def ffmpeg_cmd(out_dir, rec_id, seconds, cfg=None, source=None):
    c = {**DEFAULTS, **(cfg or {})}
    # wallclock pts: avfoundation screen pts came through as ~1 us apart (hundreds of "frames"/s), so -t,
    # keyframe expr, segment_time and bitrate cap never applied. The fps filter then forces true CFR.
    src = source or ["-use_wallclock_as_timestamps", "1", "-f", "avfoundation", "-capture_cursor", str(c["capture_cursor"]), "-framerate", str(c["fps"]),
                     "-i", f"{c['screen_device']}:none"]
    seg = int(c["segment_s"])
    filt = [f"fps={c['fps']}"] + ([f"scale={int(c['width'])}:-2"] if c.get("width") else [])
    vf = ["-vf", ",".join(filt), "-fps_mode", "cfr"]
    if c.get("quality"):   # constant quality fallback if the encoder ignores bitrate
        rate = ["-q:v", str(c["quality"])]
    else:
        rate = ["-b:v", str(c["bitrate"]), "-maxrate", str(c.get("maxrate") or c["bitrate"]),
                "-bufsize", str(c.get("maxrate") or c["bitrate"])]
    tag = ["-tag:v", "hvc1"] if "hevc" in str(c["codec"]) else []
    return [FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", *src, "-t", f"{seconds:.3f}", *vf,
            "-c:v", c["codec"], *tag, *rate, "-pix_fmt", "yuv420p",
            "-force_key_frames", f"expr:gte(t,n_forced*{seg})", "-g", str(int(c["fps"]) * seg),
            "-f", "segment", "-segment_time", str(seg), "-reset_timestamps", "1", "-segment_format", "mp4",
            "-segment_format_options", "movflags=+frag_keyframe+empty_moov+default_base_moof:frag_duration=5000000",
            "-segment_list", str(out_dir / f"{rec_id}.csv"), "-segment_list_type", "csv",
            "-metadata", f"comment={TAG}", str(out_dir / f"screen-{rec_id}-%03d.mp4")]


def _stop(p):
    """Clean stop: 'q' lets ffmpeg finalize the current segment; escalate only if needed."""
    try: p.stdin.write(b"q"); p.stdin.flush()
    except Exception: pass
    for sig in (None, p.terminate, p.kill):
        if sig: sig()
        try: p.wait(timeout=2.5 if sig is None else 2); return
        except subprocess.TimeoutExpired: pass


def record(window=None, cfg=None, source=None, index_path=None, now=time.time, privacy=None, poll_s=None, stop=None):
    """Blocking. Records only inside the window and only while privacy mode is off; returns the index summary.
    Never runs past window end. privacy: callable -> True when blocked (default privacy_blocked).
    stop: threading.Event; set it (or send SIGTERM/SIGINT to this process) to stop at once, finalizing the
    current segment (recording mode stops all captures together on privacy, pause or quit)."""
    stop = stop or threading.Event()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop.set())
    window = window or parse_window()
    cfg = {**env_cfg(), **(cfg or {})}
    privacy = privacy or privacy_blocked
    poll_s = PRIVACY_POLL_S if poll_s is None else poll_s
    t = now()
    if not in_window(t, window):
        return {"ok": False, "reason": "outside window"}
    if privacy():
        _log("privacy mode on (or unreadable): not starting")
        return {"ok": False, "reason": "privacy"}
    secs = remaining_s(t, window)
    rec_id = datetime.fromtimestamp(t, TZ).strftime("%Y%m%d-%H%M%S")
    out_dir = DATA / datetime.fromtimestamp(t, TZ).strftime("%Y-%m-%d"); out_dir.mkdir(parents=True, exist_ok=True)
    db = connect(index_path)
    db.execute("INSERT OR REPLACE INTO recordings VALUES(?,?,?,?,?,?,?)",
               (rec_id, t, None, window[1], cfg.get("fps", DEFAULTS["fps"]), cfg.get("codec", DEFAULTS["codec"]), "recording"))
    db.commit()
    p = subprocess.Popen(ffmpeg_cmd(out_dir, rec_id, secs, cfg, source), stdin=subprocess.PIPE)
    (DATA / "recorder.pid").write_text(f"{p.pid} {window[1]}")

    def watchdog():   # layer 2: kill at window end even if ffmpeg ignores -t
        while p.poll() is None:
            if time.time() >= window[1] + 2:
                try: p.stdin.write(b"q"); p.stdin.flush()
                except Exception: pass
                time.sleep(3)
                if p.poll() is None: p.terminate(); time.sleep(3)
                if p.poll() is None: p.kill()
            time.sleep(1)
    threading.Thread(target=watchdog, daemon=True).start()
    stopped_for_privacy = stopped_by_owner = False
    last_ingest = 0.0
    parent = int(os.environ.get("VLAB_SCREENREC_PARENT") or 0)   # recording mode: stop if the server goes away
    while p.poll() is None:
        if parent and os.getppid() != parent:
            _log("owner process is gone: stopping"); stop.set()
        if stop.is_set():
            _log(f"stop requested: stopping {rec_id}")
            _stop(p); stopped_by_owner = True; break
        if privacy():
            _log(f"privacy mode on (or unreadable): stopping {rec_id}")
            _stop(p); stopped_for_privacy = True; break
        if time.time() - last_ingest >= 5:
            ingest_segment_list(db, rec_id, t, out_dir / f"{rec_id}.csv"); last_ingest = time.time()
        stop.wait(poll_s)
    n = ingest_segment_list(db, rec_id, t, out_dir / f"{rec_id}.csv")
    status = "privacy-stop" if stopped_for_privacy else "stopped" if stopped_by_owner else f"exit {p.returncode}"
    db.execute("UPDATE recordings SET end_ts=?, status=? WHERE rec_id=?", (time.time(), status, rec_id)); db.commit()
    (DATA / "recorder.pid").unlink(missing_ok=True)
    if stopped_for_privacy:
        _log(f"stopped {rec_id} for privacy; finalized {n} chunk(s)")
        return {"ok": True, "reason": "privacy", "rec_id": rec_id, "chunks": n, "dir": str(out_dir)}
    if stopped_by_owner:
        return {"ok": True, "reason": "stopped", "rec_id": rec_id, "chunks": n, "dir": str(out_dir)}
    return {"ok": p.returncode == 0, "rec_id": rec_id, "chunks": n, "dir": str(out_dir)}


_proc = None


def env_cfg():
    try:
        d = json.loads(os.environ.get(CFG_ENV) or "{}")
        return {k: v for k, v in d.items() if k in DEFAULTS}
    except Exception:
        return {}


def cfg_overrides(cfg=None):
    """planloop.json keys screenrec_fps, screenrec_bitrate, ... -> DEFAULTS overrides."""
    return {k: (cfg or {})[f"screenrec_{k}"] for k in DEFAULTS if (cfg or {}).get(f"screenrec_{k}") is not None}


def recorder_env(cfg=None):
    """Env for the recorder child; cfg["screenrec_dir"] overrides the output dir (VLAB_SCREENREC_DIR)."""
    env = {**os.environ, "VLAB_SCREENREC_WINDOW": (cfg or {}).get("screenrec_window") or DEFAULT_WINDOW}
    if (cfg or {}).get("screenrec_dir"):
        env["VLAB_SCREENREC_DIR"] = str((cfg or {})["screenrec_dir"])
    ov = cfg_overrides(cfg)
    if ov:
        env[CFG_ENV] = json.dumps(ov)
    else:
        env.pop(CFG_ENV, None)
    return env


def recorder_alive(data=None):
    """True if a recorder (any owner: planloop or recording mode) is running, per recorder.pid."""
    try:
        pid = int((Path(data or DATA) / "recorder.pid").read_text().split()[0])
        os.kill(pid, 0); return True
    except (OSError, ValueError, IndexError):
        return False


def ensure_running(cfg=None, privacy=None, spawn=None):
    """Called each planloop tick (planloop owns TCC). Starts a detached recorder child when in window and
    privacy mode is off; the child also polls privacy and exits, so the next tick after privacy goes off restarts it."""
    global _proc
    privacy = privacy or privacy_blocked
    w = parse_window((cfg or {}).get("screenrec_window"))
    if not (cfg or {}).get("screenrec", False):
        return "disabled"
    if _proc is not None and _proc.poll() is None:
        if privacy():   # belt and braces: the child stops itself too
            _proc.terminate(); return "privacy"
        return "running"
    if not in_window(time.time(), w):
        return "outside window"
    if privacy():
        return "privacy"
    if recorder_alive(recorder_env(cfg).get("VLAB_SCREENREC_DIR")):
        return "running (other owner)"   # recording mode (server) already records the screen: never two recorders
    if spawn:
        _proc = spawn(); return "started"
    env = recorder_env(cfg)
    _proc = subprocess.Popen([sys.executable, "-m", "vision_lab.screenrec", "record", TAG], cwd=ROOT, env=env)
    return "started"


def guard(now=None):
    """launchd guard: kill any recorder past its window end."""
    pidf = DATA / "recorder.pid"; now = now or time.time()
    if pidf.exists():
        pid, end = pidf.read_text().split()
        if now >= float(end):
            try: os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError: pass
    subprocess.run(["pkill", "-f", TAG], capture_output=True)  # anything tagged, at/after midnight
    return "guarded"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "record":
        print(json.dumps(record()))
    elif cmd == "guard":
        print(guard())
    elif cmd == "window":
        w = parse_window(); print(datetime.fromtimestamp(w[0], TZ), "->", datetime.fromtimestamp(w[1], TZ))
