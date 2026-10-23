"""`vlab find` section search + Yap export on synthetic fixtures.

The narration below is invented for the test (no real transcripts). Temp DB, temp screenshots, temp screenrec index
with lavfi test video, temp audio archive. The search must open the DB read-only.

    PYTHONPATH=. .venv/bin/python tests/vlab_find.py
"""
import contextlib, hashlib, io, json, os, sqlite3, subprocess, sys, tempfile, time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TMP = Path(tempfile.mkdtemp(prefix="vlab-find-test-"))
os.environ["VLAB_AUDIO_DIR"] = str(TMP / "audio")
os.environ["VLAB_SCREENS_DIR"] = str(TMP / "screens")
os.environ["VLAB_SCREENREC_DIR"] = str(TMP / "screenrec")
os.environ["HF_HUB_OFFLINE"] = "1"
for k in ("VLAB_AUDIO_RETAIN", "VLAB_AUDIO_KEEP_DAYS"):
    os.environ.pop(k, None)
from vision_lab import audiostore, find, voicetx
from vision_lab.store import Store

DAY = datetime(2026, 10, 1)


def at(h, m, s=0, day=DAY):
    return day.replace(hour=h, minute=m, second=s).timestamp()


SCRIPT = [  # (start, seconds, synthetic narration)
    (at(10, 0, 0), 6, "Let me check my email real quick."),
    (at(10, 0, 20), 5, "Archiving the newsletters, nothing urgent here."),
    (at(10, 5, 0), 5, "Ok, I'm going to walk through how I set up my open-claw."),
    (at(10, 5, 20), 6, "First you clone the repo and install the dependencies."),
    (at(10, 5, 45), 7, "The OpenClaw gateway needs a config file in the home folder."),
    (at(10, 6, 30), 8, "Then you paste the API key into the environment, never into the file."),
    (at(10, 7, 10), 5, "Now restart the gateway and check the logs."),
    (at(10, 8, 0), 5, "Open Claw should show the agent as connected."),
    (at(10, 8, 30), 3, "That's it for the setup."),
    (at(10, 9, 0), 5, "Ok, now I'm going to look at the invoices for this month."),
    (at(10, 9, 30), 4, "Three invoices are still unpaid."),
    (at(14, 0, 0), 6, "I've got to update my openclaw configuration to accommodate the new workflow."),
    (at(14, 0, 25), 5, "Adding a second agent profile for the review step."),
    (at(14, 1, 10), 4, "Saving the config and reloading."),
    (at(16, 0, 0), 4, "Ok so the weather widget is broken again."),
]


def make_db(script=SCRIPT, name="vlab.db"):
    p = TMP / "db" / name; p.parent.mkdir(parents=True, exist_ok=True)
    st = Store(str(p)); voicetx.ensure_schema(st)
    for i, (t, d, text) in enumerate(script):
        st._exec("INSERT INTO voice_transcripts (ts_start, ts_end, text, model, avg_logprob, no_speech_prob, source, "
                 "source_key, written_at) VALUES (?,?,?,?,?,?,?,?,?)",
                 (t, t + d, text, "synthetic", -0.2, 0.01, "test", f"{name}:{i}", t + d))
    return st, p


def rows_for(p, t0=None, t1=None):
    db = find.connect_ro(p)
    return db, find.load_rows(db, t0 or at(0, 0), t1 or at(23, 59))


def idx(rows, t):
    return next(i for i, r in enumerate(rows) if r["ts_start"] == t)


ST, DB = make_db()


# ------------------------------------------------------------------ matching
def test_open_claw_spellings_all_match():
    assert [t for t, _ in find.query_terms("walk-through of open-claw setup")] == ["walkthrough", "openclaw", "setup"]
    assert [t for t, _ in find.query_terms("Open_Claw set-up")] == ["openclaw", "setup"]
    terms = find.query_terms("openclaw")
    for text in ("my open-claw", "the OpenClaw gateway", "Open Claw should", "openclaw configuration", "open claw"):
        kw, hit = find.keyword_score(terms, text, find.term_weights(terms, [text]))
        assert kw > 0.9 and hit, (text, kw, hit)
    for text in ("open the claw machine later", "the closet door", "clause four"):
        kw, _ = find.keyword_score(terms, text, find.term_weights(terms, [text]))
        assert kw < 0.34, (text, kw)


def test_walkthrough_query_finds_the_setup_section_with_cue_boundaries():
    db, rows = rows_for(DB)
    for q in ("find the section where I was doing a walk-through of open-claw setup", "open-claw setup walkthrough",
              "how I set up openclaw"):
        secs = find.find_sections(q, rows, embed=None)
        assert secs, q
        top = secs[0]
        assert top["start"] == at(10, 5, 0) and top["start_cue"], (q, find.fmt(top["start"]))
        assert top["end"] == at(10, 8, 30) + 3, (q, find.fmt(top["end"]))          # "That's it" closes it
        assert idx(rows, at(10, 0, 0)) not in top["rows"] and idx(rows, at(10, 9, 0)) not in top["rows"]
        assert any(s["start"] == at(14, 0, 0) for s in secs[1:]), [find.fmt(s["start"]) for s in secs]


def test_update_config_query_prefers_the_afternoon_section():
    db, rows = rows_for(DB)
    secs = find.find_sections("update my openclaw configuration", rows)
    top = secs[0]
    assert top["start"] == at(14, 0, 0) and top["start_cue"], find.fmt(top["start"])
    assert top["end"] == at(14, 1, 10) + 4, find.fmt(top["end"])                    # silence ends it; 4 PM not pulled in
    assert idx(rows, at(16, 0, 0)) not in top["rows"]


def test_new_topic_cue_ends_a_section_and_unrelated_queries_stay_apart():
    db, rows = rows_for(DB)
    top = find.find_sections("unpaid invoices", rows)[0]
    assert top["start"] == at(10, 9, 0) and top["end"] == at(10, 9, 30) + 4, (find.fmt(top["start"]), find.fmt(top["end"]))
    assert find.find_sections("kubernetes cluster upgrade", rows) == []


def test_max_len_caps_a_section():
    db, rows = rows_for(DB)
    top = find.find_sections("openclaw setup", rows, max_len_s=120)[0]
    assert top["start"] == at(10, 5, 0) and top["end"] - top["start"] <= 120, find.fmt(top["end"])


def test_semantic_auto_is_optional_and_agrees():
    emb = find.get_embedder("auto")
    assert find.get_embedder("off") is None
    db, rows = rows_for(DB)
    secs = find.find_sections("walk-through of open-claw setup", rows, embed=emb)
    assert secs[0]["start"] == at(10, 5, 0)
    if emb is None:
        print("   (no local embedding model: keyword + fuzzy only)")
        return
    sc = find.score_rows("walk-through of open-claw setup", rows, emb)
    assert all(s["sem"] is not None for s in sc)
    # meaning without the keyword: the setup narration beats the invoices/weather rows
    sc = find.score_rows("installing and configuring the agent gateway", rows, emb)
    setup = max(sc[idx(rows, at(10, 5, 45))]["sem"], sc[idx(rows, at(10, 7, 10))]["sem"])
    assert setup > sc[idx(rows, at(10, 9, 30))]["sem"] and setup > sc[idx(rows, at(16, 0, 0))]["sem"], sc


def test_junk_rows_are_ignored():
    st, p = make_db([(at(9, 0), 2, "Thank you."), (at(9, 0, 10), 2, "Thank you."), (at(9, 0, 20), 5, "Ok, I'm going to set up openclaw.")],
                    name="junk.db")
    db, rows = rows_for(p)
    assert [r["text"] for r in rows][-1].endswith("openclaw.") and len(find.find_sections("openclaw", rows)) == 1


# ------------------------------------------------------------------ linkage + CLI + export
def jpg(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([audiostore.ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x36", "-frames:v", "1",
                    str(path)], check=True)


def make_screens(t0, t1):
    root = Path(os.environ["VLAB_SCREENS_DIR"])
    first = None
    for t in np.arange(t0, t1, 10.0):
        f = root / datetime.fromtimestamp(t).strftime("%Y-%m-%d") / (datetime.fromtimestamp(t).strftime("%H%M%S") + ".jpg")
        if first is None:
            jpg(f); first = f
        else:
            f.write_bytes(first.read_bytes())


def make_screenrec(spans):
    root = Path(os.environ["VLAB_SCREENREC_DIR"]); root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "index.db")
    db.execute("CREATE TABLE IF NOT EXISTS chunks (rec_id INTEGER, path TEXT, start_ts REAL, end_ts REAL)")
    for t0, t1 in spans:
        f = root / datetime.fromtimestamp(t0).strftime("%Y-%m-%d") / f"seg-{int(t0)}.mp4"; f.parent.mkdir(exist_ok=True)
        subprocess.run([audiostore.ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=5",
                        "-t", f"{t1 - t0:.2f}", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(f)],
                       check=True)
        db.execute("INSERT INTO chunks VALUES (1, ?, ?, ?)", (str(f), t0, t1))
    db.commit(); db.close()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_cli(*args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = find.main(list(args))
    return rc, buf.getvalue()


def test_cli_reports_frames_screen_and_audio_and_is_read_only():
    make_screens(at(10, 4, 0), at(10, 10, 0))
    make_screenrec([(at(10, 4, 30), at(10, 7, 0))])
    for t in np.arange(at(10, 5, 0), at(10, 7, 0), 20.0):
        audiostore.archive(ST, np.zeros(16000 * 20, np.float32), float(t), float(t) + 20)
    before = sha(DB)
    rc, out = run_cli("open-claw setup walkthrough", "--db", str(DB), "--since", "2026-10-01", "--until", "2026-10-01",
                      "--semantic", "off", "--json")
    assert rc in (0, None) and sha(DB) == before, "find must not write to the DB"
    res = json.loads(out)
    s = res["sections"][0]
    assert "10:05:00 AM" in s["start_local"] and "10:08:33 AM" in s["end_local"], s
    assert s["start_cue"].startswith("Ok, I'm going to walk through")
    assert s["quotes"] and all(q["text"] for q in s["quotes"])
    assert s["frames"]["start"].endswith("2026-10-01/100500.jpg") and s["frames"]["before_start"].endswith("100440.jpg")
    assert s["frames"]["end"].endswith("100830.jpg") and s["frames"]["after_end"].endswith("100850.jpg"), s["frames"]
    assert s["screen"]["start"]["offset_s"] == 30.0 and s["screen"]["end"] is None      # recording stopped at 10:07
    assert s["audio"]["segments"] == 6 and 0.55 < s["audio"]["coverage"] < 0.65, s["audio"]
    rc, human = run_cli("open-claw setup walkthrough", "--db", str(DB), "--since", "2026-10-01", "--until", "2026-10-01",
                        "--semantic", "off")
    assert "#1" in human and "starts:" in human and "frame start" in human and "audio kept" in human, human


def tone(seconds, freq=440.0):
    return (0.4 * np.sin(2 * np.pi * freq * np.arange(int(seconds * 16000)) / 16000)).astype(np.float32)


def test_export_builds_a_yap_ready_folder_in_sync():
    day = datetime(2026, 9, 15)
    script = [(at(11, 0, 0, day), 4, "Ok, I'm going to walk through the openclaw setup."),
              (at(11, 0, 10, day), 5, "The openclaw config lives in the home folder."),
              (at(11, 0, 22, day), 4, "That's it.")]
    st, p = make_db(script, name="export.db")
    # word timestamps on the first row
    st._exec("UPDATE voice_transcripts SET words=? WHERE ts_start=?",
             (json.dumps([[at(11, 0, 0, day) + 0.2, at(11, 0, 0, day) + 0.5, "Ok,", 0.9]]), at(11, 0, 0, day)))
    t_sec0, t_sec1 = at(11, 0, 0, day), at(11, 0, 26, day)
    make_screenrec([(t_sec0 - 10, t_sec0 + 12), (t_sec0 + 14, t_sec1 + 20)])          # 2 s gap in the screen recording
    audiostore.archive(st, tone(10), t_sec0 - 2, t_sec0 + 8)                          # audio 10:59:58 - 11:00:08
    audiostore.archive(st, tone(8), t_sec0 + 15, t_sec0 + 24)                         # silence 11:00:08 - 11:00:15
    make_screens(t_sec0 - 10, t_sec1 + 10)
    out = TMP / "export"
    rc, js = run_cli("openclaw setup", "--db", str(p), "--since", "2026-09-15", "--until", "2026-09-15", "--semantic",
                     "off", "--json", "--export", "1", "--out", str(out), "--pad", "5")
    res = json.loads(js)
    man = json.loads((out / "manifest.json").read_text())
    assert man["start"] == t_sec0 - 5 and man["end"] == t_sec1 + 5, man
    assert set(man["files"]) >= {"audio", "screen", "clip", "frames", "transcript"}, man
    assert man["screen_gaps"] and man["screen_gaps"][0]["gap_s"] == 2.0, man
    want = t_sec1 + 5 - (t_sec0 - 5)
    for f in ("audio.m4a", "screen.mp4", "clip.mp4"):
        d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                  str(out / f)], capture_output=True, text=True).stdout)
        assert abs(d - want) < 0.6, (f, d, want)          # the 2 s screen gap is held, not dropped
    # a late-starting recording is padded at the front, not shifted: export a range that begins before the screen
    late = TMP / "late"
    sec = {"start": t_sec0 - 20, "end": t_sec0 - 2}
    find.build_screen(find.screen_chunks(sec["start"], sec["end"]), sec["start"], sec["end"], late.with_suffix(".mp4"), TMP)
    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              str(late.with_suffix(".mp4"))], capture_output=True, text=True).stdout)
    assert abs(d - 18) < 0.6, d
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                              str(out / "clip.mp4")], capture_output=True, text=True).stdout.split()
    assert sorted(streams) == ["audio", "video"], streams
    # audio on the wall clock: tone at clip 3-13 s, silence 13-20 s, tone 20-28 s
    pcm = np.frombuffer(subprocess.run([audiostore.ffmpeg_bin(), "-v", "error", "-i", str(out / "audio.m4a"), "-f", "f32le",
                                        "-ac", "1", "-ar", "16000", "-"], capture_output=True).stdout, np.float32)
    rms = lambda a, b: float(np.sqrt(np.mean(pcm[int(a * 16000):int(b * 16000)] ** 2)))
    assert rms(0.5, 2.5) < 0.01 and rms(4, 12) > 0.2 and rms(14, 19) < 0.01 and rms(21, 27) > 0.2 and rms(29, 35) < 0.01, \
        [round(rms(a, a + 1), 3) for a in range(0, 36, 2)]
    srt = (out / "transcript.srt").read_text()
    assert srt.startswith("1\n00:00:05,000 --> 00:00:09,000\nOk, I'm going to walk through the openclaw setup.\n"), srt
    tj = json.loads((out / "transcript.json").read_text())
    assert tj[0]["words"] == [{"start": 5.2, "end": 5.5, "word": "Ok,"}] and len(tj) == 3, tj
    assert len(list((out / "frames").glob("*.jpg"))) == 4, sorted((out / "frames").iterdir())
    steps = (out / "yap-steps.md").read_text()
    assert "asset.import" in steps and "transcript.import" in steps and str(out / "clip.mp4") in steps
    assert res["export"]["dir"] == str(out)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time(); fn(); print(f"PASS {name} ({time.time() - t:.1f}s)", flush=True)
    print("vlab_find: OK")
