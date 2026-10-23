"""Recording mode: one switch runs camera video, mic and screen capture together.

The switch is the header RECORDING / PRIVATE button. While it shows RECORDING (Privacy mode off and the
camera not paused) and recording mode is enabled for this server (VLAB_RECORDING_MODE=1, which run.sh sets
for the app), three captures run together. Privacy on, a camera pause, or quitting stops all three at
once, and nothing is written while private.

  mic     ffmpeg (avfoundation) -> raw 16 kHz mono float32 on a pipe -> short chunks (MIC_CHUNK_S, cut at a
          quiet moment, never longer than MIC_MAX_CHUNK_S) written to VLAB_VOICE_TMP/.staging/*.part, then
          renamed to VLAB_VOICE_TMP/voice-<start ms>-<end ms>.pcm and handed to voicetx, which keeps it as AAC
          in the audio archive (vision_lab/audiostore.py, VLAB_AUDIO_RETAIN=0 = old
          transcribe-then-delete), transcribes it and deletes the temp file. Each chunk's <start ms> is stamped from the wall
          clock when its first audio arrives (<end ms> = wall clock at close), NOT from a running sample count: the mic delivered only ~80%
          of the nominal 16 kHz, so sample-count time fell ~11 min/h behind (drift fix). The
          sample-count vs wall-clock gap is still measured and logged as a `mic_drift` event (rate-limited)
          plus a server log line; ffmpeg's own mic warnings go to the server log (rate-limited) instead of
          /dev/null. A crash leaves at most a chunk or two; they are transcribed and deleted on the next
          start (archived + transcribed). If the server dies, ffmpeg gets SIGPIPE and exits. The Mic button
          (mic_enabled) and Privacy mode stop this capture; nothing is written while either is off.
          ffmpeg runs inside the server's process tree (a child of Vision Lab.app), so macOS attributes the
          mic permission to com.houghtonlabs.visionlab. Skipped with VLAB_NO_MIC or when the Mic button is off.
  video   Frames the engine already grabs (Camera.sinks) -> VIDEO_FPS, VIDEO_WIDTH wide -> ffmpeg HEVC
          (VideoToolbox) segment files in VLAB_CAMREC_DIR/YYYY-MM-DD/cam-<rec>-NNN.mp4, indexed in
          VLAB_CAMREC_DIR/index.db (same schema as screenrec). Logs video_started (first real frame) and
          video_stopped events for the Timeline. Kept forever unless VLAB_VIDEO_KEEP_DAYS is set
          (retention.py). If the server dies, ffmpeg sees EOF on stdin and closes the file. Skipped with
          VLAB_NO_CAMERA.
  screen  vision_lab.screenrec child (window now..midnight, restarted after midnight), kept forever. The child
          also stops on its own if Privacy mode turns on or this server goes away. If planloop's recorder
          is already running, it is left alone (never two recorders). Skipped with VLAB_NO_SCREEN.
"""
import json
import os
import queue
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import audiostore, retention, screenrec, voicetx

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = os.environ.get("VLAB_FFMPEG", "/opt/homebrew/bin/ffmpeg")
TAG = "vlab-recmode"
SR = 16000
MIC_CHUNK_S, MIC_MAX_CHUNK_S, MIC_QUIET_DBFS = 10.0, 20.0, -45.0
MIC_DRIFT_LOG_S = 2.0             # log when sample-count time and wall-clock time differ by more than this
MIC_DRIFT_LOG_EVERY_S = 300.0     # ...at most once per this many seconds while it stays over the threshold
MIC_STDERR_LINES_PER_MIN = 5      # ffmpeg mic warnings copied to the server log, per minute
VIDEO = {"fps": 10, "width": 1280, "segment_s": 300, "codec": "hevc_videotoolbox", "bitrate": "700k", "maxrate": "1000k"}
COLOR_ON, COLOR_OFF = "#FF6B6B", "#A3AFBD"


def enabled_by_env():
    return os.environ.get("VLAB_RECORDING_MODE") == "1"


def camrec_dir():
    return Path(os.environ.get("VLAB_CAMREC_DIR") or ROOT / "data" / "camrec")


def db_path(store):
    try:
        return store.db.execute("PRAGMA database_list").fetchone()[2]
    except Exception:
        return os.environ.get("VLAB_DB") or str(ROOT / "data" / "vlab.db")


def _event(store, kind, text, color=COLOR_OFF, ts=None, data=None):
    try:
        store.add_event(kind, text, color, ts=ts or time.time(), modality="system", data=data)
    except Exception as e:
        print(f"[recording] could not log {kind}: {e}", flush=True)


def _dbfs(a):
    return 20 * np.log10(float(np.sqrt(np.mean(a ** 2))) + 1e-9) if len(a) else -120.0


# ---------------------------------------------------------------- mic
class MicRecorder:
    name = "mic"

    def __init__(self, store, voice_tmp=None, source=None, device=None, chunk_s=MIC_CHUNK_S,
                 max_chunk_s=MIC_MAX_CHUNK_S, transcribe=None, clock=None):
        self.store = store
        self.clock = clock or time.time                # injectable wall clock (tests)
        self.drift_s = 0.0                             # wall-clock elapsed minus sample-count elapsed
        self.drift_events = 0
        self.dir = Path(voice_tmp or voicetx.VOICE_TMP)
        self.staging = self.dir / ".staging"
        self.source, self.device = source, device      # source: ffmpeg input args (tests); device: () -> index
        self.chunk_s, self.max_chunk_s = chunk_s, max_chunk_s
        self.transcribe = transcribe or (lambda f: voicetx.transcribe_file(self.store, f))
        self.proc = self.reader = self.worker = None
        self.stop_flag = threading.Event()
        self.work = queue.Queue()
        self.chunks = 0
        self.started_at = None

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def recover(self):
        """Crash leftovers: finished-but-untranscribed chunks and half-written .part files -> archive+transcribe."""
        self.staging.mkdir(parents=True, exist_ok=True)
        for f in sorted(self.staging.glob("*.part")):
            os.replace(f, self.dir / (f.stem + ".pcm"))
        self._ensure_worker()
        for f in voicetx.pending(self.dir, settle_s=0):
            self.work.put(f)

    def _ensure_worker(self):
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(target=self._work, daemon=True, name="rec-mic-tx")
            self.worker.start()

    def _work(self):
        while True:
            f = self.work.get()
            try:
                if Path(f).exists():
                    self.transcribe(f)
            except Exception as e:
                print(f"[recording] transcribe failed: {e}", flush=True)
            finally:
                Path(f).unlink(missing_ok=True)       # belt and braces: raw audio never stays
                self.work.task_done()   # (the kept copy, if retention is on, is the archived m4a)

    def cmd(self):
        dev = self.device() if self.device else "default"
        src = self.source or ["-f", "avfoundation", "-i", f":{dev}"]
        return [FFMPEG, "-hide_banner", "-loglevel", "warning", "-nostdin", *src,
                "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]

    def start(self):
        self.recover()
        self.stop_flag.clear()
        self.proc = subprocess.Popen(self.cmd(), stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)
        self.started_at = time.time()
        self.reader = threading.Thread(target=self._read, args=(self.proc,), daemon=True, name="rec-mic")
        self.reader.start()
        threading.Thread(target=self._stderr, args=(self.proc,), daemon=True, name="rec-mic-err").start()
        _event(self.store, "mic_started", "Mic capture started (recording mode)", COLOR_ON)

    def _stderr(self, proc):
        """Copy ffmpeg's mic warnings to the server log, at most MIC_STDERR_LINES_PER_MIN per minute."""
        window, shown, dropped = 0.0, 0, 0
        try:
            for raw in iter(proc.stderr.readline, b""):
                now = time.time()
                if now - window >= 60:
                    if dropped:
                        print(f"[recording] mic ffmpeg: {dropped} more warning line(s) not shown", flush=True)
                    window, shown, dropped = now, 0, 0
                if shown < MIC_STDERR_LINES_PER_MIN:
                    shown += 1
                    print(f"[recording] mic ffmpeg: {raw.decode(errors='replace').rstrip()[:300]}", flush=True)
                else:
                    dropped += 1
        except Exception:
            pass

    def _check_drift(self, t0, total, now, state):
        """Sample-count elapsed vs wall-clock elapsed; log `mic_drift` (rate-limited) when over the threshold."""
        self.drift_s = drift = (now - t0) - total / SR
        if abs(drift) <= MIC_DRIFT_LOG_S:
            state["over"] = False
            return
        if state.get("over") and now - state.get("last", 0) < MIC_DRIFT_LOG_EVERY_S:
            return
        state["over"], state["last"] = True, now
        elapsed = max(now - t0, 1e-9)
        ratio = total / SR / elapsed
        self.drift_events += 1
        msg = (f"Mic clock drift {drift:+.1f} s after {elapsed / 60:.1f} min "
               f"(mic delivered {ratio * 100:.1f}% of {SR} Hz); chunks are stamped from the wall clock")
        print(f"[recording] {msg}", flush=True)
        _event(self.store, "mic_drift", msg, COLOR_OFF, ts=now,
               data={"drift_s": round(drift, 2), "elapsed_s": round(elapsed, 1), "delivered_ratio": round(ratio, 4)})

    def _read(self, proc):
        block = SR // 10 * 4                           # 100 ms of float32
        t0, total, cur = None, 0, None                 # cur = [file, path, start_ts, n_samples]
        drift_state = {}

        def close(publish=True):
            nonlocal cur
            if cur is None:
                return
            cur[0].close()
            if publish and cur[3] >= SR * 0.3:
                end_ms = max(int(self.clock() * 1000), int(cur[2] * 1000) + 1)      # wall clock at close
                final = self.dir / f"{cur[1].stem}-{end_ms}.pcm"
                os.replace(cur[1], final)
                self.chunks += 1
                self.work.put(final)
            else:
                cur[1].unlink(missing_ok=True)
            cur = None

        try:
            while True:
                b = proc.stdout.read(block)
                if not b or self.stop_flag.is_set():
                    break
                b = b[: len(b) // 4 * 4]
                a = np.frombuffer(b, np.float32)
                now = self.clock()
                if t0 is None:
                    t0 = now - len(a) / SR             # first audio actually arrived now
                if cur is None:
                    start = now - len(a) / SR          # wall clock, not t0 + total/SR (that drifted ~11 min/h)
                    p = self.staging / f"voice-{int(start * 1000)}.part"
                    cur = [open(p, "wb"), p, start, 0]
                cur[0].write(b)
                cur[3] += len(a); total += len(a)
                self._check_drift(t0, total, now, drift_state)
                dur = cur[3] / SR
                if dur >= self.max_chunk_s or (dur >= self.chunk_s and _dbfs(a) < MIC_QUIET_DBFS):
                    close()
        finally:
            close()                                    # audio from before the stop is still transcribed

    def stop(self, reason="stop"):
        self.stop_flag.set()
        p = self.proc
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(2)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(2)
        if self.reader is not None:
            self.reader.join(3)
        if p is not None:
            _event(self.store, "mic_stopped", f"Mic capture stopped ({reason})", COLOR_OFF, data={"chunks": self.chunks})
        self.proc = None

    def reap(self):
        if self.proc is not None and self.proc.poll() is not None:
            self.stop("ended")

    def drain(self, timeout=30.0):
        end = time.time() + timeout
        while self.work.unfinished_tasks and time.time() < end:
            time.sleep(0.1)
        return self.work.unfinished_tasks == 0


# ---------------------------------------------------------------- camera video
class VideoRecorder:
    name = "video"

    def __init__(self, store, outdir=None, cfg=None):
        self.store = store
        self.dir = Path(outdir or camrec_dir())
        self.cfg = {**VIDEO, **(cfg or {})}
        self.q = queue.Queue(maxsize=int(self.cfg["fps"]) * 2)
        self.lock = threading.Lock()
        self.active = False
        self.proc = self.thread = None
        self.rec = None                                # {rec_id, start, dir, shape}
        self.last_ts = 0.0
        self.frames = self.dropped = 0

    @property
    def index(self):
        return self.dir / "index.db"

    def running(self):
        return self.active

    # camera thread: must never block
    def sink(self, frame, ts):
        if not self.active or frame is None:
            return
        if ts - self.last_ts < 0.9 / self.cfg["fps"]:
            return
        self.last_ts = ts
        try:
            self.q.put_nowait((frame, ts))
        except queue.Full:
            self.dropped += 1

    def start(self):
        self.recover()
        with self.lock:
            while not self.q.empty():
                self.q.get_nowait()
            self.active = True
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run, daemon=True, name="rec-video")
            self.thread.start()

    def _scale(self, frame):
        import cv2
        h, w = frame.shape[:2]
        W = int(self.cfg["width"]) if self.cfg.get("width") and w > int(self.cfg["width"]) else w
        H = int(round(h * W / w / 2)) * 2
        W -= W % 2
        return frame if (W, H) == (w, h) else cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

    def _ffmpeg(self, out_dir, rec_id, w, h):
        c = self.cfg; seg = int(c["segment_s"])
        return [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-use_wallclock_as_timestamps", "1", "-i", "-",
                "-vf", f"fps={c['fps']}", "-fps_mode", "cfr", "-c:v", c["codec"],
                *(["-tag:v", "hvc1"] if "hevc" in c["codec"] else []),
                "-b:v", c["bitrate"], "-maxrate", c["maxrate"], "-bufsize", c["maxrate"], "-pix_fmt", "yuv420p",
                "-force_key_frames", f"expr:gte(t,n_forced*{seg})", "-g", str(int(c["fps"]) * seg),
                "-f", "segment", "-segment_time", str(seg), "-reset_timestamps", "1", "-segment_format", "mp4",
                "-segment_format_options", "movflags=+frag_keyframe+empty_moov+default_base_moof:frag_duration=5000000",
                "-segment_list", str(out_dir / f"{rec_id}.csv"), "-segment_list_type", "csv",
                "-metadata", f"comment={TAG}", str(out_dir / f"cam-{rec_id}-%03d.mp4")]

    def _open(self, shape, ts):
        h, w = shape[:2]
        rec_id = datetime.fromtimestamp(ts).strftime("%Y%m%d-%H%M%S-") + f"{int(ts * 1000) % 1000:03d}"
        out_dir = self.dir / datetime.fromtimestamp(ts).strftime("%Y-%m-%d"); out_dir.mkdir(parents=True, exist_ok=True)
        db = screenrec.connect(self.index)
        try:
            db.execute("INSERT OR REPLACE INTO recordings VALUES(?,?,?,?,?,?,?)",
                       (rec_id, ts, None, None, self.cfg["fps"], self.cfg["codec"], "recording")); db.commit()
        finally:
            db.close()
        self.proc = subprocess.Popen(self._ffmpeg(out_dir, rec_id, w, h), stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.rec = {"rec_id": rec_id, "start": ts, "dir": out_dir, "shape": shape, "last": ts}
        _event(self.store, "video_started", "Camera video recording started", COLOR_ON, ts=ts,
               data={"rec_id": rec_id, "size": f"{w}x{h}", "fps": self.cfg["fps"]})

    def _ingest(self):
        r = self.rec
        if not r:
            return 0
        db = screenrec.connect(self.index)
        try:
            return screenrec.ingest_segment_list(db, r["rec_id"], r["start"], r["dir"] / f"{r['rec_id']}.csv")
        finally:
            db.close()

    def _close(self, reason):
        """Caller holds self.lock. Finalize the current file, index it, log video_stopped."""
        p, r = self.proc, self.rec
        self.proc = self.rec = None
        if p is None:
            return
        try:
            p.stdin.close()                            # EOF: ffmpeg finishes the segment and exits
        except Exception:
            pass
        try:
            p.wait(8)
        except subprocess.TimeoutExpired:
            p.terminate()
            try: p.wait(3)
            except subprocess.TimeoutExpired: p.kill()
        self.rec = r; n = self._ingest(); self.rec = None
        end = r["last"]
        db = screenrec.connect(self.index)
        try:
            db.execute("UPDATE recordings SET end_ts=?, status=? WHERE rec_id=?", (end, reason, r["rec_id"])); db.commit()
        finally:
            db.close()
        _event(self.store, "video_stopped", f"Camera video recording stopped ({reason})", COLOR_OFF, ts=end,
               data={"rec_id": r["rec_id"], "chunks": n})

    def _run(self):
        last_ingest = time.time()
        while True:
            try:
                frame, ts = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            with self.lock:
                if not self.active:
                    continue                           # stopped: drop queued frames, write nothing
                small = self._scale(frame)
                if self.proc is not None and (self.proc.poll() is not None or small.shape != self.rec["shape"]
                                              or datetime.fromtimestamp(ts).date() != datetime.fromtimestamp(self.rec["start"]).date()):
                    self._close("new segment")         # ffmpeg died, camera size changed, or midnight
                if self.proc is None:
                    self._open(small.shape, ts)
                try:
                    self.proc.stdin.write(np.ascontiguousarray(small).tobytes())
                    self.rec["last"] = ts; self.frames += 1
                except (BrokenPipeError, OSError):
                    self._close("encoder error")
                if time.time() - last_ingest >= 5:
                    self._ingest(); last_ingest = time.time()

    def stop(self, reason="stop"):
        with self.lock:
            self.active = False
            self._close(reason)

    def reap(self):
        pass                                           # an ffmpeg exit is handled on the next frame

    def recover(self):
        """Close recordings left open by a crash: index their segments and log the missing video_stopped."""
        if not self.index.exists():
            return 0
        db = screenrec.connect(self.index)
        try:
            rows = db.execute("SELECT rec_id, start_ts FROM recordings WHERE end_ts IS NULL").fetchall()
        finally:
            db.close()
        for rec_id, start in rows:
            day = self.dir / datetime.fromtimestamp(start).strftime("%Y-%m-%d")
            db = screenrec.connect(self.index)
            try:
                n = screenrec.ingest_segment_list(db, rec_id, start, day / f"{rec_id}.csv")
                end = db.execute("SELECT MAX(end_ts) FROM chunks WHERE rec_id=?", (rec_id,)).fetchone()[0] or start
                db.execute("UPDATE recordings SET end_ts=?, status='crashed' WHERE rec_id=?", (end, rec_id)); db.commit()
            finally:
                db.close()
            _event(self.store, "video_stopped", "Camera video recording stopped (server ended unexpectedly)", COLOR_OFF,
                   ts=end, data={"rec_id": rec_id, "chunks": n, "recovered": True})
        return len(rows)

    def prune(self, now=None):
        """Retention (default: keep forever). Removes index rows only for files it deleted."""
        gone = retention.prune_video(self.dir, now=now)
        if gone and self.index.exists():
            db = screenrec.connect(self.index)
            try:
                db.executemany("DELETE FROM chunks WHERE path=?", [(str(f),) for f in gone]); db.commit()
            finally:
                db.close()
        return gone


# ---------------------------------------------------------------- screen
class ScreenRecorder:
    name = "screen"

    def __init__(self, store, cfg=None, spawn=None):
        self.store, self.spawn = store, spawn
        self.cfg = cfg                                  # planloop.json dict (screenrec_dir, screenrec_fps, ...)
        self.proc = None

    def _cfg(self):
        if self.cfg is not None:
            return self.cfg
        try:
            return json.loads(Path(os.environ.get("PLANLOOP_CONFIG") or ROOT / "data" / "planloop.json").read_text())
        except (OSError, ValueError):
            return {}

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        cfg = self._cfg()
        env = screenrec.recorder_env(cfg)
        out = env.get("VLAB_SCREENREC_DIR") or str(screenrec.DATA)
        if screenrec.recorder_alive(out):
            return                                      # planloop's recorder is already running: leave it
        now = time.time()
        iso = lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%dT%H:%M:%S")
        env.update({"VLAB_SCREENREC_WINDOW": f"{iso(now)}/{iso(now + 86400)}",   # capped at midnight by screenrec
                    "VLAB_SCREENREC_PARENT": str(os.getpid()), "VLAB_DB": db_path(self.store)})
        self.proc = self.spawn(env) if self.spawn else subprocess.Popen(
            [sys.executable, "-m", "vision_lab.screenrec", "record", screenrec.TAG, TAG], cwd=ROOT, env=env)
        _event(self.store, "screen_started", "Screen recording started (recording mode)", COLOR_ON, data={"dir": out})

    def reap(self):
        if self.proc is not None and self.proc.poll() is not None:
            self.stop("ended")

    def stop(self, reason="stop"):
        p = self.proc
        if p is None:
            return
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)               # screenrec finalizes the segment and exits
            try:
                p.wait(8)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(3)
        self.proc = None
        _event(self.store, "screen_stopped", f"Screen recording stopped ({reason})", COLOR_OFF)


# ---------------------------------------------------------------- the switch
DEFAULT = object()   # "build the real component"; None always means "this capture is off"


class RecordingMode:
    def __init__(self, engine, store, enabled=None, mic=DEFAULT, video=DEFAULT, screen=DEFAULT, mic_allowed=None,
                 poll_s=0.5, restart_s=30.0, prune_every_s=3600.0):
        self.engine, self.store = engine, store
        self.enabled = enabled_by_env() if enabled is None else bool(enabled)
        # None = that capture is off. VLAB_NO_MIC / VLAB_NO_SCREEN always win (even over a passed-in component);
        # VLAB_NO_CAMERA only stops the real default (tests feed synthetic frames into a passed-in VideoRecorder).
        self.mic = None if os.environ.get("VLAB_NO_MIC") else MicRecorder(store, device=_mic_index) if mic is DEFAULT else mic
        self.video = None if os.environ.get("VLAB_NO_CAMERA") and video is DEFAULT else VideoRecorder(store) if video is DEFAULT else video
        self.screen = None if os.environ.get("VLAB_NO_SCREEN") else ScreenRecorder(store) if screen is DEFAULT else screen
        self.mic_allowed = mic_allowed or _mic_enabled
        self.poll_s, self.restart_s, self.prune_every_s = poll_s, restart_s, prune_every_s
        self.lock = threading.RLock()
        self.closing = False
        self.hold = False          # set by the engine hook the moment privacy / pause starts turning on
        self.last_try = {}
        self.thread = None
        self.last_prune = 0.0
        if self.video is not None and hasattr(engine, "camera"):
            engine.camera.sinks.append(self.video.sink)
        if hasattr(engine, "capture_listeners"):
            engine.capture_listeners.append(self.on_capture_change)

    @property
    def components(self):
        return [c for c in (self.mic, self.video, self.screen) if c is not None]

    def private(self):
        s = self.engine.settings
        return bool(s.get("capture_paused") or s.get("camera_paused"))

    def wanted(self):
        return self.enabled and not self.closing and not self.hold and not self.private()

    def _allowed(self, c):
        if c is self.mic:
            try:
                return bool(self.mic_allowed())
            except Exception:
                return False                            # unknown mic state: do not record
        return True

    def sync(self):
        with self.lock:
            want = self.wanted()
            for c in self.components:
                c.reap()                                 # ended on its own (device gone, midnight): log the stop
                on = want and self._allowed(c)
                if on and not c.running():
                    if time.time() - self.last_try.get(c.name, 0) < self.restart_s:
                        continue                         # don't hot-loop on a failing device
                    self.last_try[c.name] = time.time()
                    try:
                        c.start()
                    except Exception as e:
                        _event(self.store, "recording_error", f"{c.name} capture could not start ({type(e).__name__})",
                               "#FF6B6B", data={"component": c.name, "error": str(e)[:300]})
                elif not on and c.running():
                    c.stop("privacy" if (self.hold or self.private()) else "quit" if self.closing else "off")
                    self.last_try.pop(c.name, None)
            if time.time() - self.last_prune >= self.prune_every_s:
                self.last_prune = time.time()
                try:
                    if self.video is not None:
                        self.video.prune()
                    if self.mic is not None:
                        audiostore.prune(self.store)     # default keep forever: deletes nothing
                except Exception as e:
                    print(f"[recording] prune failed: {e}", flush=True)

    def on_capture_change(self, stopping):
        """Engine hook: called BEFORE the camera is released when privacy/pause turns on, and after resume."""
        if stopping:
            self.hold = True        # before the settings flip, so the sync loop cannot restart anything
            self.stop_all("privacy")
        else:
            self.hold = False
            self.last_try.clear()
            self.sync()

    def stop_all(self, reason):
        with self.lock:
            for c in self.components:
                if c.running():
                    c.stop(reason)

    def start(self):
        if not self.enabled:
            print("[recording] recording mode is off for this server (set VLAB_RECORDING_MODE=1)", flush=True)
            return False
        if self.mic is not None:
            self.mic.recover()                          # transcribe+delete audio left by a crash
        if self.video is not None:
            self.video.recover()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="recording-mode")
        self.thread.start()
        return True

    def _loop(self):
        while not self.closing:
            try:
                self.sync()
            except Exception as e:
                print(f"[recording] sync failed: {e}", flush=True)
            time.sleep(self.poll_s)

    def shutdown(self, drain_s=15.0):
        self.closing = True
        self.stop_all("quit")
        if self.mic is not None:
            self.mic.drain(drain_s)                     # anything left is recovered on the next start

    def status(self):
        return {"enabled": self.enabled, "active": self.wanted() and any(c.running() for c in self.components),
                "privacy": self.private(),
                "components": {n: (None if c is None else {"running": c.running()})
                               for n, c in (("mic", self.mic), ("video", self.video), ("screen", self.screen))},
                "video_dir": str(self.video.dir) if self.video else None,
                "video_keep_days": retention.video_keep_days() or "forever",
                "audio": audiostore.settings()}


def _mic_enabled():
    from . import mic
    return mic.enabled()


def _mic_index():
    from . import mic
    dev = mic.state()["device"]
    if dev is None:
        return "default"
    idx = next((d["index"] for d in mic.list_devices() if d["name"] == dev), None)
    if idx is None:
        raise RuntimeError(f"selected mic {dev!r} is not connected")
    return idx
