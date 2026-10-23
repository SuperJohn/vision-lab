"""Timeline Captures box: GET /api/timeline/captures rows for Video, Audio and Screen capture.

    .venv/bin/python tests/timeline_captures.py
"""
import os, sqlite3, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vision_lab import captures as cap, screenrec
from vision_lab.store import Store

T0 = 1_800_000_000.0          # fixed window [T0, T0 + 1 day)


def store():
    return Store(str(Path(tempfile.mkdtemp()) / "vlab.db"))


def test_merge_and_subtract():
    assert cap.merge([[10, 20], [0, 5], [21, 30], [40, 41]], gap=1) == [[0, 5], [10, 30], [40, 41]]
    assert cap.merge([[0, 10], [2, 3]]) == [[0, 10]], "contained bar"
    assert cap.subtract([[0, 100]], [[10, 20], [50, 60]]) == [[0, 10], [20, 50], [60, 100]]
    assert cap.subtract([[0, 10]], [[-5, 50]]) == []
    assert cap.clip([[-10, 10], [90, 200]], 0, 100) == [[0, 10], [90, 100]]


def test_video_cut_by_privacy_and_camera_pause():
    st = store()
    for i in range(0, 3600, 30):                                  # frames every 30 s for the first hour
        st.add_event("frame_probe", "probe", ts=T0 + i, source_key=f"f{i}")
    for i in range(7200, 7800, 40):                               # second session after a long gap
        st.add_keyframe(T0 + i, "/dev/null")
    st.add_event("capture_paused", "Privacy mode on", ts=T0 + 600)
    st.add_event("capture_resumed", "Privacy mode off", ts=T0 + 1200)
    st.add_event("camera_paused", "Camera paused", ts=T0 + 2000)
    st.add_event("camera_resumed", "Camera resumed", ts=T0 + 2100)
    r = cap.captures(st, T0, T0 + 86400, index="/nonexistent")
    video = next(x for x in r["rows"] if x["key"] == "video")["bars"]
    assert video == [[T0, T0 + 600], [T0 + 1200, T0 + 2000], [T0 + 2100, T0 + 3570], [T0 + 7200, T0 + 7760]], video
    assert r["privacy"] == [[T0 + 600, T0 + 1200], [T0 + 2000, T0 + 2100]]


def test_privacy_still_on_cuts_to_window_end():
    st = store()
    for i in range(0, 1800, 20):
        st.add_event("scene", "s", ts=T0 + i, source_key=f"s{i}")
    st.add_event("capture_paused", "Privacy mode on", ts=T0 + 900)      # never resumed
    video = cap.video_bars(st, T0, T0 + 86400)
    assert video == [[T0, T0 + 900]], video


def test_audio_only_actual_segments():
    st = store()
    st.add_utterance("a", "user", "hello", T0 + 100, T0 + 103.5, source_key="u1")
    st.add_utterance("a", "assistant", "hi there", T0 + 104, T0 + 110, source_key="u2")      # reply: not captured voice
    st.add_utterance("b", "user", "no end", T0 + 500, None, source_key="u3")
    st.add_utterance("c", "user", "  ", T0 + 600, T0 + 605, source_key="u4")                # blank: ignored
    st.add_event("sound", "Heard music", ts=T0 + 1000, modality="audio", cls="music", source_key="a1")
    for i in range(0, 3600, 30):                                  # a whole camera session must not leak into audio
        st.add_event("frame_probe", "probe", ts=T0 + i, source_key=f"f{i}")
    audio = cap.audio_bars(st, T0, T0 + 86400)
    assert audio == [[T0 + 100, T0 + 103.5], [T0 + 500, T0 + 502]], audio      # room-sound events are not audio bars
    cap.AUDIO_SOURCES["segments"] = lambda s, a, b: [[T0 + 2000, T0 + 2300]]   # future recording-mode segments plug in
    try:
        assert cap.audio_bars(st, T0, T0 + 86400)[-1] == [T0 + 2000, T0 + 2300]
    finally:
        cap.AUDIO_SOURCES.pop("segments")


def test_screen_segments_merge_into_sessions():
    d = Path(tempfile.mkdtemp()); db = screenrec.connect(d / "index.db")
    for k in range(4):                                            # 4 x 300 s back-to-back (1 s gaps) -> one session
        db.execute("INSERT INTO chunks(rec_id,path,start_ts,end_ts) VALUES(?,?,?,?)",
                   ("r1", f"/x/{k}.mp4", T0 + k * 301, T0 + k * 301 + 300))
    db.execute("INSERT INTO chunks(rec_id,path,start_ts,end_ts) VALUES('r2','/x/9.mp4',?,?)", (T0 + 5000, T0 + 5300))
    db.execute("INSERT INTO recordings VALUES('r3', ?, ?, NULL, 5, 'hevc', 'exit -9')", (T0 + 8000, T0 + 8270))   # no chunks
    db.execute("INSERT INTO recordings VALUES('r4', ?, NULL, NULL, 5, 'hevc', 'recording')", (T0 + 9000,))    # no end: skipped
    db.commit(); db.close()
    before = (d / "index.db").stat().st_mtime_ns
    bars = cap.screen_bars(T0, T0 + 86400, d / "index.db")
    assert bars == [[T0, T0 + 3 * 301 + 300], [T0 + 5000, T0 + 5300], [T0 + 8000, T0 + 8270]], bars
    assert (d / "index.db").stat().st_mtime_ns == before, "screenrec index must be opened read-only"
    assert cap.screen_bars(T0, T0 + 86400, d / "missing.db") == []


def test_endpoint_read_only_and_shape():
    st = store()
    st.add_utterance("a", "user", "hello", T0 + 100, T0 + 103, source_key="u1")
    cap.STORE["store"] = st
    os.environ["VLAB_SCREENREC_INDEX"] = "/nonexistent/index.db"
    app = FastAPI(); app.include_router(cap.router)
    ch = st.db.total_changes
    r = TestClient(app).get("/api/timeline/captures", params={"start": T0, "end": T0 + 86400}).json()
    assert st.db.total_changes == ch
    assert [x["label"] for x in r["rows"]] == ["Video", "Audio", "Screen capture"]
    assert r["rows"][1]["bars"] == [[T0 + 100, T0 + 103]] and r["rows"][1]["seconds"] == 3
    d = TestClient(app).get("/api/timeline/captures").json()                 # default: today
    assert d["end"] - d["start"] in (82800, 86400, 90000)


def test_ui_wiring():
    root = Path(__file__).resolve().parents[1] / "vision_lab"
    html = (root / "static/index.html").read_text(); js = (root / "static/app.js").read_text()
    tl = html[html.index('id="view-timeline"'):html.index("<!-- ============ MODELS")]
    assert tl.index('aria-label="Captures"') < tl.index('aria-label="Object activity"'), "Captures must sit above"
    assert ">Object activity<" in tl and "Activity today" not in tl
    assert tl.count('class="scroll-x tl-scroll"') == 2, "both boxes share the synced scroller"
    for need in ("/api/timeline/captures?", "/api/timeline?${q}", "const TL = {", "tlPct(", "tlApplyZoom", "tl-zoom"):
        assert need in js, need
    assert "renderCaptures" in js and "'Hatched = Privacy mode'" in js


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("timeline_captures: OK")
