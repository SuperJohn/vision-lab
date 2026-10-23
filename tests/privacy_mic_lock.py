"""Follow-up: Privacy mode locks the mic (server refuses with 409, UI disables the button).

    .venv/bin/python tests/privacy_mic_lock.py
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vision_lab import mic


class FakeStore:
    def __init__(self, **cfg): self.cfg = dict(cfg)
    def get_config(self, k, d): return self.cfg.get(k, d)
    def set_config(self, k, v): self.cfg[k] = json.loads(json.dumps(v))


def client(mic_on=False, private=False):
    st = FakeStore(mic_enabled=mic_on)
    flag = {"on": private}
    mic.STORE["store"] = st
    mic.PRIVACY["on"] = lambda: flag["on"]
    mic.list_devices = lambda: [{"index": 0, "name": "Test Mic", "manufacturer": None}]
    app = FastAPI(); app.include_router(mic.router)
    return TestClient(app), st, flag


def test_private_refuses_turning_mic_on():
    c, st, _ = client(mic_on=False, private=True)
    r = c.post("/api/mic/toggle")
    assert r.status_code == 409 and "Privacy mode is on" in r.json()["detail"], r.text
    assert st.cfg["mic_enabled"] is False
    r = c.post("/api/mic", json={"enabled": True})
    assert r.status_code == 409 and st.cfg["mic_enabled"] is False
    r = c.post("/api/mic/test")
    assert r.status_code == 409, "mic test must not record while private"
    s = c.get("/api/mic/state").json()
    assert s == {"enabled": False, "device": None, "privacy": True}, s


def test_private_masks_stale_enabled_flag():
    c, st, _ = client(mic_on=True, private=True)     # stored flag on, but privacy wins
    assert mic.enabled() is False, "voice gate must drop audio while private"
    assert c.get("/api/mic").json()["enabled"] is False
    assert c.post("/api/mic/toggle").status_code == 409 and st.cfg["mic_enabled"] is True


def test_private_allows_off_and_device_change():
    c, st, _ = client(mic_on=False, private=True)
    assert c.post("/api/mic", json={"enabled": False}).status_code == 200
    r = c.post("/api/mic", json={"device": "Test Mic"})
    assert r.status_code == 200 and st.cfg["mic_device"] == "Test Mic"
    assert r.json()["enabled"] is False and st.cfg["mic_enabled"] is False, "device change must not turn the mic on"


def test_not_private_toggles_normally():
    c, st, flag = client(mic_on=False, private=False)
    r = c.post("/api/mic/toggle")
    assert r.status_code == 200 and r.json()["enabled"] is True and r.json()["privacy"] is False
    assert c.post("/api/mic/toggle").json()["enabled"] is False
    assert c.post("/api/mic", json={"enabled": True}).json()["enabled"] is True
    flag["on"] = True                                # privacy turns on: state reads off immediately
    assert c.get("/api/mic/state").json()["enabled"] is False


def test_privacy_read_error_fails_closed():
    c, st, _ = client(mic_on=False)
    mic.PRIVACY["on"] = lambda: 1 / 0
    assert c.post("/api/mic/toggle").status_code == 409


def test_no_mic_env_blocks_server_recording():
    import os
    c, _, _ = client(mic_on=True, private=False)
    os.environ["VLAB_NO_MIC"] = "1"
    try:
        r = c.post("/api/mic/test")
        assert r.status_code == 409 and "VLAB_NO_MIC" in r.json()["detail"]
    finally:
        os.environ.pop("VLAB_NO_MIC")


def test_ui_wiring():
    root = Path(__file__).resolve().parents[1] / "vision_lab"
    js = (root / "static/mic.js").read_text(); app = (root / "static/app.js").read_text()
    html = (root / "static/index.html").read_text(); css = (root / "static/style.css").read_text()
    srv = (root / "server.py").read_text()
    assert "'Mic off (private)'" in js and "b.disabled = priv" in js and "mic-locked" in js
    assert "/api/mic/state" in js and "window.vlabRefreshMic = refresh" in js and "r.status === 409" in js
    assert "window.vlabStopMic()" in js, "turning off / going private must close a browser mic stream"
    assert app.count("privacyChanged(") >= 4 and "window.vlabRefreshMic()" in app
    assert "Turn microphone listening on or off" not in html and "Shows whether the microphone is on" in html
    assert ".btn.mic-locked:disabled" in css and "grayscale(1)" in css
    assert 'mic.PRIVACY["on"] = lambda: bool(engine.settings.get("capture_paused"))' in srv


if __name__ == "__main__":
    test_private_refuses_turning_mic_on(); test_private_masks_stale_enabled_flag()
    test_private_allows_off_and_device_change(); test_not_private_toggles_normally()
    test_privacy_read_error_fails_closed(); test_no_mic_env_blocks_server_recording(); test_ui_wiring()
    print("privacy_mic_lock: OK")
