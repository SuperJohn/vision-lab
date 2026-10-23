"""Voice log junk filter: Whisper noise/hallucination segments are never saved, and old ones are
hidden from the voice log and Captures bars without touching the DB rows. All strings here are synthetic.

    .venv/bin/python tests/voice_filter.py
"""
import sys, tempfile, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vision_lab import captures, voicelog, voicetx
from vision_lab.store import Store
from vision_lab.voicefilter import ALLOW, is_junk_transcript, junk_reason, normalize


def test_empty():
    for t in ("", "   ", None, "...", " - ", "!?"):
        assert junk_reason(t) == "empty", t


def test_no_latin():
    for t in ("\u01a2", "\u01a5", "\u01a2\u01a2\u01a2", "\u266a", "\u266a\u266a", "\u3002", "\u0416\u0416", "* * *", "\u2014"):
        assert junk_reason(t) in ("no_latin", "empty"), t
    assert junk_reason("\u01a2") == "no_latin"
    assert junk_reason("caf\u00e9 time") is None, "Latin-1 letters are real text"


def test_short():
    for t in ("t", "s", "x", "Q.", "ab", "7", " k "):
        assert junk_reason(t) == "short", t


def test_stopword():
    for t in ("the", "The.", "and", "Umm.", "uhm", "Hmm", "you", "You.", "mhm", "she"):
        assert junk_reason(t) == "stopword", t
    for t in ("Uh.", "um", "oh"):
        assert junk_reason(t) in ("short", "stopword"), t


def test_hallucination():
    for t in ("Thank you.", "thank you", "Thanks for watching!", "Thank you for watching.", "Please subscribe.",
              "Like and subscribe", "Subtitles by the Amara.org community", "See you next time."):
        assert junk_reason(t) == "hallucination", t
    assert junk_reason("Thank you for the coffee this morning.") is None, "only the whole segment is junk"


def test_repeated():
    for t in ("the the the", "la la la la", "aaaaaa", "Zzzzz", "ha ha ha ha ha", "word word word word other"):
        assert junk_reason(t) == "repeated", t
    assert junk_reason("no no no") is None, "allowlisted word repeated is real"
    assert junk_reason("we need to go to the store") is None


def test_allowlist_and_real_speech_kept():
    for t in ("yes", "No.", "ok", "Okay.", "Hi!", "hey", "stop", "Wait.", "yeah", "Sure.", "bye"):
        assert junk_reason(t) is None, t
    for t in ("Bob", "Priya", "Sam!", "Call the plumber at five.", "Turn left at the light.", "Hello there"):
        assert junk_reason(t) is None, t
    assert {"yes", "no", "okay", "ok", "hi", "stop"} <= ALLOW


def test_meta_rules():
    assert junk_reason("Pick up the package.", {"no_speech_prob": 0.9, "avg_logprob": -1.5}) == "no_speech"
    assert junk_reason("Pick up the package.", {"no_speech_prob": 0.9, "avg_logprob": -0.3}) is None
    assert junk_reason("Pick up the package.", {"no_speech_prob": 0.1, "avg_logprob": -1.5}) is None
    assert junk_reason("Pick up the package.", {"compression_ratio": 3.1}) == "compression"
    assert junk_reason("Pick up the package.", {"compression_ratio": 1.4}) is None
    assert junk_reason("Pick up the package.", {"no_speech_prob": None, "avg_logprob": None}) is None
    assert is_junk_transcript("t") and not is_junk_transcript("yes")
    assert normalize("  Hello,   World!! ") == "hello, world"


def _seg(text, start=0.5, **m):
    return {"start": start, "end": start + 0.5, "text": text, "avg_logprob": -0.2, "no_speech_prob": 0.01, **m}


def test_insert_drops_junk_and_logs_only_counts():
    st = Store(str(Path(tempfile.mkdtemp()) / "vlab.db"))
    segs = [_seg(" Synthetic sentence one."), _seg(" \u01a2"), _seg(" the"), _seg(" t"), _seg(" Thank you."),
            _seg(" yes"), _seg(" Synthetic sentence two.", compression_ratio=3.0)]
    audio = (0.3 * np.sin(np.arange(16000 * 3) / 16000 * 2 * np.pi * 440)).astype(np.float32)
    r = voicetx.transcribe_pcm(st, audio, 1_700_000_000.0, stt=lambda a: segs)
    assert r["transcripts"] == 2 and r["dropped"] == 5, r
    assert sorted(x["text"] for x in st._query("SELECT text FROM voice_transcripts")) == ["Synthetic sentence one.", "yes"]
    assert set(r) == {"transcripts", "dropped", "seconds"}, "result carries counts only, no dropped text"


def test_display_hides_old_junk_but_keeps_rows():
    st = Store(str(Path(tempfile.mkdtemp()) / "vlab.db"))
    voicetx.ensure_schema(st)
    now = time.time() - 60
    rows = [("Synthetic real line.", -0.2, 0.01), ("\u01a2", -0.2, 0.01), ("the", -0.2, 0.01), ("s", -0.2, 0.01),
            ("Thank you.", -0.2, 0.01), ("okay", -0.2, 0.01), ("Noise guess.", -1.6, 0.95)]
    for i, (t, lp, nsp) in enumerate(rows):
        st._exec("INSERT INTO voice_transcripts (ts_start, ts_end, text, model, avg_logprob, no_speech_prob, source, "
                 "source_key, written_at) VALUES (?,?,?,?,?,?,?,?,?)",
                 (now + i, now + i + 0.5, t, "m", lp, nsp, "recording mode", f"k:{i}", now))
    voicelog.STORE["store"] = st
    app = FastAPI(); app.include_router(voicelog.router)
    j = TestClient(app).get("/api/voice/log", params={"before": time.time() + 10}).json()
    shown = [e["text"] for d in j["days"] for e in d["entries"] if e["kind"] == "transcript"]
    assert sorted(shown) == ["Synthetic real line.", "okay"], shown
    assert st._query("SELECT COUNT(*) AS n FROM voice_transcripts")[0]["n"] == len(rows), "rows stay in the DB"
    bars = captures._transcripts(st, now - 5, now + 30)
    assert len(bars) == 2, bars


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("voice_filter: OK")
