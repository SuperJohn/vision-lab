"""feat/voice-log: GET /api/voice/log groups captured voice by local day, newest first, paginated by day.

    .venv/bin/python tests/voice_log.py
"""
import sys, tempfile, time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vision_lab import voicelog
from vision_lab.store import Store


def at(days_ago, h, m, s=0):
    d = datetime.now().replace(hour=h, minute=m, second=s, microsecond=0) - timedelta(days=days_ago)
    return d.timestamp()


END = at(0, 23, 59, 59) + 1     # fixtures use fixed clock times today; read up to the end of today


def get(c, **p):
    return c.get("/api/voice/log", params={"before": END, **p})


def make(fill=True, private=False):
    st = Store(str(Path(tempfile.mkdtemp()) / "vlab.db"))
    if fill:
        n = 0
        def say(ts, role, text, **k):
            nonlocal n; n += 1
            st.add_utterance(f"t{n}", role, text, ts, ts + 2, source_key=f"k{n}", **k)
        say(at(0, 9, 5, 7), "user", "today one")
        say(at(0, 9, 5, 9), "assistant", "reply one")
        say(at(0, 11, 30, 1), "user", "today two", interrupted=True)
        say(at(0, 11, 31), "user", "   ")                      # blank: skipped
        say(at(2, 8, 0), "user", "two days ago")
        say(at(5, 23, 59, 59), "user", "five days ago late")
        say(at(9, 7, 0), "user", "nine days ago")
        st.add_event("sound", "Heard music (Music, 40%)", ts=at(0, 10, 0, 3), modality="audio", cls="music",
                     data={"label": "music", "audioset_label": "Music", "confidence": 0.4}, source_key="s1")
        st.add_event("track", "Person came into view", ts=at(0, 10, 1), modality="vision", source_key="v1")
    voicelog.STORE["store"] = st
    voicelog.CAPTURE["state"] = lambda: {"privacy": private, "mic_enabled": not private}
    app = FastAPI(); app.include_router(voicelog.router)
    return TestClient(app), st


def test_grouping_and_order():
    c, _ = make()
    r = get(c).json()
    assert r["order"] == "newest first"
    dates = [d["date"] for d in r["days"]]
    assert dates == sorted(dates, reverse=True) and len(dates) == 4, dates      # "days" = days that have entries
    today = r["days"][0]
    assert today["today"] and not r["days"][1]["today"]
    kinds = [e["kind"] for e in today["entries"]]
    assert kinds == ["speech", "reply", "speech"], kinds                       # newest first, blank skipped, no sounds
    ts = [e["ts"] for e in today["entries"]]
    assert ts == sorted(ts, reverse=True)
    assert today["count"] == 2 and today["replies"] == 1, today               # replies are not "captured voice"
    assert today["first_ts"] == at(0, 9, 5, 7) and today["last_ts"] == at(0, 11, 30, 1)
    assert today["entries"][0]["interrupted"] is True
    assert all(e["source"] == "voice chat" for d in r["days"] for e in d["entries"]), "vision leaked in"
    late = r["days"][2]["entries"][0]
    assert late["text"] == "five days ago late" and r["days"][2]["date"] == datetime.fromtimestamp(at(5, 12, 0)).strftime("%Y-%m-%d")


def test_filters():
    c, _ = make()
    r = get(c, replies=0).json()
    kinds = {e["kind"] for d in r["days"] for e in d["entries"]}
    assert kinds == {"speech"}, kinds
    assert r["sources"] == ["utterances", "transcripts"]          # room sounds are not a default source
    assert "sound" not in {e["kind"] for d in get(c).json()["days"] for e in d["entries"]}
    assert len(c.get("/api/voice/log", params={"before": END}).json()["days"][0]["entries"]) == 3


def test_pagination_by_day():
    c, _ = make()
    p1 = get(c, days=2).json()
    assert [d["date"] for d in p1["days"]] == [datetime.fromtimestamp(at(i, 12, 0)).strftime("%Y-%m-%d") for i in (0, 2)]
    assert p1["next_before"] == p1["days"][-1]["day_start"]
    p2 = c.get("/api/voice/log", params={"days": 2, "before": p1["next_before"]}).json()
    assert [d["entries"][0]["text"] for d in p2["days"]] == ["five days ago late", "nine days ago"]
    assert p2["next_before"] is None, "nothing older: no next page"
    seen = [e["id"] for p in (p1, p2) for d in p["days"] for e in d["entries"]]
    assert len(seen) == len(set(seen)) == 6, seen                            # no gaps, no repeats
    assert c.get("/api/voice/log", params={"days": 0}).status_code == 422
    assert c.get("/api/voice/log", params={"days": 61}).status_code == 422


def test_limit_caps_entries():
    c, _ = make()
    r = get(c, limit=2).json()
    assert sum(len(d["entries"]) for d in r["days"]) == 2 and r["next_before"] is not None


def test_empty_db():
    c, _ = make(fill=False)
    r = get(c).json()
    assert r["days"] == [] and r["next_before"] is None
    assert r["capture"]["recording_mode"] is False


def test_privacy_is_read_only_and_leaks_nothing_new():
    c, st = make(private=True)
    before = st._query("SELECT (SELECT COUNT(*) FROM utterances) u, (SELECT COUNT(*) FROM events) e, "
                       "(SELECT COUNT(*) FROM config) c")[0]
    changes = st.db.total_changes
    r = get(c).json()
    assert r["capture"] == {"privacy": True, "mic_enabled": False, "recording_mode": False}
    assert st.db.total_changes == changes, "the log endpoint wrote to the DB"
    after = st._query("SELECT (SELECT COUNT(*) FROM utterances) u, (SELECT COUNT(*) FROM events) e, "
                      "(SELECT COUNT(*) FROM config) c")[0]
    assert before == after
    keys = {k for d in r["days"] for e in d["entries"] for k in e}
    assert keys <= {"id", "ts", "ts_end", "kind", "source", "text", "label", "confidence", "interrupted", "turn_id"}, keys
    assert "snapshot" not in str(r) and "latencies" not in str(r), "camera snapshot / internals must not leak"


def test_pluggable_source():
    c, _ = make()
    t = at(0, 12, 0, 30)
    voicelog.SOURCES["voice_segments"] = lambda store, t0, t1: (
        [{"id": "seg:1", "ts": t, "ts_end": t + 300, "kind": "transcript", "source": "recording mode",
          "text": "chunk text", "label": None, "confidence": 0.9, "interrupted": False}] if t0 <= t < t1 else [])
    try:
        today = get(c).json()["days"][0]
        assert today["entries"][0]["kind"] == "transcript" and today["count"] == 3
    finally:
        voicelog.SOURCES.pop("voice_segments")


def test_ui_wiring():
    root = Path(__file__).resolve().parents[1] / "vision_lab"
    html = (root / "static/index.html").read_text(); js = (root / "static/voicelog.js").read_text()
    app = (root / "static/app.js").read_text(); srv = (root / "server.py").read_text()
    assert not (root / "static/voice.js").exists() and not (root / "static/mic-worklet.js").exists()
    for gone in ('id="v-mic"', 'id="v-stop"', "Start talking", "Stop reply", "/static/voice.js", 'id="vl-sounds"',
                 "room sound", "Show room sounds"):
        assert gone not in html, gone
    assert "getUserMedia" not in js and "WebSocket" not in js, "the voice log must not capture audio"
    for need in ('id="vl-days"', 'id="vl-expand"', 'id="vl-collapse"', 'id="vl-replies"', 'id="vl-empty"',
                 "Voice is captured in recording mode", "/static/voicelog.js", "/static/mic.js"):
        assert need in html, need
    assert "/api/voice/log" in js and "d.today" in js and "next_before" in js
    assert "window.vlabLoadVoiceLog()" in app
    assert "app.include_router(voicelog.router)" in srv and '@router.websocket("/ws/voice")' in (root / "voice.py").read_text()
    # nav order and device pickers
    tabs = [l.split(">")[1].split("<")[0] for l in html.splitlines() if 'class="tab' in l and "data-view" in l]
    assert tabs == ["Video", "Voice", "Screens", "Zones &amp; rules", "Timeline", "Model lab"], tabs
    header = html[html.index("<header"):html.index("</header>")]
    assert 'id="mic-device"' not in header and 'id="camera"' not in header
    assert 'id="mic-toggle"' in header and 'id="power"' in header
    voice = html[html.index('id="view-voice"'):html.index('id="view-screens"')]
    video = html[html.index('id="view-live"'):html.index("</section>", html.index('id="view-live"'))]
    assert 'id="mic-device"' in voice and 'id="camera"' in video
    # Snapshot / Record video UI removed; endpoints stay; rule "Save the frame" option untouched
    for gone in ('id="snapshot"', 'id="record"', 'id="hud-rec"'):
        assert gone not in html, gone
    for ref in ("$('#snapshot')", "$('#record')", "$('#hud-rec')"):
        assert ref not in app, ref
    assert '@app.post("/api/record")' in srv and '@app.post("/api/snapshot")' in srv and "engine.camera.stop_recording()" in srv
    assert "Save the frame with the event" in app
    # every #id the scripts look up with $()/querySelector exists in the page
    import re
    ids = set(re.findall(r'id="([^"]+)"', html)) | set(re.findall(r"id=\"([^\"]+)\"", app))
    for f in ("app.js", "mic.js", "voicelog.js"):
        src = (root / "static" / f).read_text()
        for ref in re.findall(r"""\$[lm]?\(['"]#([\w-]+)['"]\)""", src):
            assert ref in ids or "${" in ref, f"{f} looks up missing #{ref}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("voice_log: OK")
