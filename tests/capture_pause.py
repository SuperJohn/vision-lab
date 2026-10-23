"""Privacy mode stops camera, mic and screen captures together.

    .venv/bin/python tests/capture_pause.py
"""
import json, os, sqlite3, sys, tempfile, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["VLAB_NO_CAMERA"] = "1"
from vision_lab import engine, planloop


class FakeStore:
    def __init__(self): self.cfg = {}
    def get_config(self, k, d): return self.cfg.get(k, d)
    def set_config(self, k, v): self.cfg[k] = json.loads(json.dumps(v))


class FakeCam:
    def __init__(self): self.paused = False
    def pause(self): self.paused = True; return True
    def resume(self): self.paused = False


def fake_engine(mic_on=True):
    st = FakeStore(); st.cfg["mic_enabled"] = mic_on
    ev = []
    e = types.SimpleNamespace(store=st, settings={"camera": 0}, camera=FakeCam(),
                              event=lambda kind, text, *a, **k: ev.append((kind, text)))
    e.set_capture_paused = lambda p: engine.Engine.set_capture_paused(e, p)
    return e, st, ev


def test_flag_and_mic_restore():
    e, st, ev = fake_engine(mic_on=True)
    r = e.set_capture_paused(True)
    assert r["paused"] and r["privacy_mode"] and e.camera.paused, r
    assert st.cfg["settings"]["capture_paused"] is True and st.cfg["settings"]["camera_paused"] is True
    assert st.cfg["mic_enabled"] is False, "mic not disabled on pause"
    e.set_capture_paused(True)                       # double pause must not overwrite remembered mic state
    assert st.cfg["settings"]["capture_mic_prev"] is True
    r = e.set_capture_paused(False)
    assert not r["paused"] and not e.camera.paused
    assert st.cfg["settings"]["capture_paused"] is False
    assert st.cfg["mic_enabled"] is True, "mic not restored on resume"
    assert [k for k, _ in ev] == ["capture_paused", "capture_resumed"], ev
    assert ev[0][1].startswith("Privacy mode on") and ev[1][1].startswith("Privacy mode off"), ev
    assert r["privacy_mode"] is False

    e, st, ev = fake_engine(mic_on=False)            # mic that was off stays off after resume
    e.set_capture_paused(True); e.set_capture_paused(False)
    assert st.cfg["mic_enabled"] is False


def test_ui_wiring():
    root = Path(__file__).resolve().parents[1] / "vision_lab"
    html = (root / "static/index.html").read_text(); js = (root / "static/app.js").read_text()
    srv = (root / "server.py").read_text()
    assert 'id="privacy-label">SWITCHING' in html and "Pause all capture" not in html and "Privacy mode: OFF" not in html
    assert "'PRIVATE'" in js and "'RECORDING'" in js and "/api/privacy" in js and "Privacy mode: ON" not in js
    css = (root / "static/style.css").read_text()
    assert ".btn.privacy.on { background: #C62828" in css and ".btn.privacy.off { background: #2E7D32" in css
    assert "#6D3FD1" not in css, "purple badge styling should be gone"
    assert '@app.post("/api/privacy")' in srv and '@app.post("/api/power")' in srv


def test_screen_worker_skips():
    db = Path(tempfile.mkdtemp()) / "vlab.db"
    con = sqlite3.connect(db); con.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    def setflag(v):
        con.execute("INSERT OR REPLACE INTO config VALUES ('settings', ?)", (json.dumps({"capture_paused": v}),)); con.commit()
    planloop.DB_PATH = str(db)
    calls = []
    planloop.screen_capture = lambda cfg, ts: calls.append(ts) or {"ok": True, "path": None}
    planloop.is_quiet = lambda cfg, now: False
    cfg = {"screen_capture": True, "screen_fast_caption": False}
    setflag(True)
    assert planloop.screen_tick(cfg) is None and not calls, "screencapture ran while paused"
    setflag(False)                                   # picked up next tick, no restart
    planloop.screen_tick(cfg)
    assert len(calls) == 1, "screen capture did not resume"
    con.execute("DELETE FROM config"); con.commit()
    planloop.screen_tick(cfg)
    assert len(calls) == 2, "missing flag must not block capture"


if __name__ == "__main__":
    test_flag_and_mic_restore(); test_ui_wiring(); test_screen_worker_skips()
    print("capture_pause: OK")
