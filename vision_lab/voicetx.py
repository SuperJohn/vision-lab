"""Voice log transcription: captured mic audio -> Whisper text (+ word timings); the audio is retained.

Retention rule: mic audio is KEPT as compressed AAC (vision_lab/audiostore.py: location,
bitrate, VLAB_AUDIO_KEEP_DAYS) and linked to its transcript rows. VLAB_AUDIO_RETAIN=0 restores the 10/20 rule
(transcribe, then delete; only transcript text is kept). No room-sound classifier runs and no sound events are
written (the AudioSet/AST classifier is out of the voice log and of persistent capture).

Flow for recording mode (vision_lab/recording.py is the recorder):
  recorder -> short segment files in VOICE_TMP named voice-<start epoch ms>[-<end epoch ms>].<ext>
  for each finished file: decode to 16 kHz in memory, archive it (audiostore.archive -> m4a + audio_segments
  row) BEFORE transcribing, so a Whisper failure never loses audio; transcribe (mlx-whisper large-v3-turbo,
  local, word timestamps), write transcript rows linked to the segment, then delete the temp file. The temp
  file is deleted whether transcription worked or not (a failure is logged as a text event), and `recover()`
  (run at start) processes anything left by a crash. transcribe_pcm() is the in-memory path (no archive).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import audiostore
from . import voice
from . import voicefilter
from .voicefilter import junk_reason

ROOT = Path(__file__).resolve().parent.parent
VOICE_TMP = Path(os.environ.get("VLAB_VOICE_TMP", ROOT / "data" / "voice-tmp"))
AUDIO_EXT = (".wav", ".m4a", ".mp3", ".webm", ".ogg", ".flac", ".caf", ".pcm", ".aiff", ".aif")
NAME = re.compile(r"^voice-(\d{10,13})(?:-(\d+))?\.[a-z0-9]+$")
SR = 16000
MIN_RMS_DBFS = -55.0          # quieter than this is silence: skip Whisper (it hallucinates on silence)
MAX_NO_SPEECH, MIN_LOGPROB = voicefilter.MAX_NO_SPEECH, voicefilter.MIN_LOGPROB   # kept for callers
SETTLE_S = 2.0                # a file must be this old (unmodified) before we treat it as finished
DDL = """CREATE TABLE IF NOT EXISTS voice_transcripts (
  id INTEGER PRIMARY KEY, ts_start REAL, ts_end REAL, text TEXT, model TEXT, avg_logprob REAL,
  no_speech_prob REAL, source TEXT, source_key TEXT UNIQUE, written_at REAL);
CREATE INDEX IF NOT EXISTS voice_transcripts_ts ON voice_transcripts(ts_start);"""


def ensure_schema(store):
    """Additive only (new table + index). Back up vlab.db before this first runs on the live DB."""
    if getattr(store, "_voicetx_schema", False):
        return
    with store.lock:
        store.db.executescript(DDL)
        store.db.commit()
        audiostore.migrate(store.db)          # audio_segments + link/word columns (additive)
    store._voicetx_schema = True


def has_table(store):
    return bool(store._query("SELECT 1 FROM sqlite_master WHERE type='table' AND name='voice_transcripts'"))


def start_ts_from_name(path):
    m = NAME.match(Path(path).name)
    if not m:
        return None
    v = int(m.group(1))
    return v / 1000 if v > 10**11 else float(v)


def end_ts_from_name(path):
    """voice-<start ms>-<end ms>.pcm (recorder): wall-clock close time. Older names have no end."""
    m = NAME.match(Path(path).name)
    if not m or not m.group(2) or len(m.group(1)) != 13 or len(m.group(2)) != 13:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return b / 1000 if 0 < b - a < 3_600_000 else None


def decode(path):
    """Any audio file -> mono float32 at 16 kHz, in memory (ffmpeg writes to a pipe, never to disk)."""
    p = Path(path)
    if p.suffix == ".pcm":                       # raw float32 16 kHz mono
        return np.fromfile(p, np.float32)
    out = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(p),
                          "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode()[-300:] or "ffmpeg failed")
    return np.frombuffer(out.stdout, np.float32).copy()


def whisper_segments(audio):
    """Local mlx-whisper; returns [{start, end, text, avg_logprob, no_speech_prob, compression_ratio}] in seconds
    from clip start."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # on-device only: use the cached model, never the network
    import mlx_whisper
    r = mlx_whisper.transcribe(audio, path_or_hf_repo=voice.SETTINGS["stt_model"], language="en",
                               condition_on_previous_text=False, word_timestamps=True)
    return [{**{k: s.get(k) for k in ("start", "end", "text", "avg_logprob", "no_speech_prob", "compression_ratio")},
             "words": [{k: w.get(k) for k in ("word", "start", "end", "probability")} for w in s.get("words") or []]}
            for s in r.get("segments", [])]


def _dbfs(a):
    return 20 * np.log10(float(np.sqrt(np.mean(a ** 2))) + 1e-9) if len(a) else -120.0


def transcribe_pcm(store, audio, start_ts, stt=whisper_segments, source="recording mode", key=None, audio_segment_id=None):
    """In-memory audio (float32 16 kHz mono) -> transcript rows only (no room-sound classifier).
    audio_segment_id links each row to its retained file (audio_offset_s = seconds into that file)."""
    ensure_schema(store)
    key = key or f"pcm:{start_ts:.3f}"
    n_txt, n_junk = 0, 0
    dur = len(audio) / SR
    if _dbfs(audio) >= MIN_RMS_DBFS and stt is not None:
        for i, s in enumerate(stt(audio)):
            text = (s.get("text") or "").strip()
            if junk_reason(text, s):             # noise/hallucination (voicefilter rules); only counted, never logged
                n_junk += 1
                continue
            off = float(s.get("start") or 0)
            a, b = start_ts + off, start_ts + float(s.get("end") or dur)
            words = [(start_ts + float(w["start"]), start_ts + float(w["end"]), (w.get("word") or "").strip(),
                      w.get("probability")) for w in s.get("words") or [] if w.get("start") is not None]
            store._exec("INSERT OR IGNORE INTO voice_transcripts (ts_start, ts_end, text, model, avg_logprob, "
                        "no_speech_prob, source, source_key, written_at, audio_segment_id, audio_offset_s, words) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (a, max(b, a), text, voice.SETTINGS["stt_model"], s.get("avg_logprob"),
                         s.get("no_speech_prob"), source, f"{key}:{i}", time.time(), audio_segment_id,
                         round(off, 3) if audio_segment_id else None,
                         audiostore.words_json(words) if words else None))
            n_txt += 1
    return {"transcripts": n_txt, "dropped": n_junk, "seconds": round(dur, 2)}


def transcribe_file(store, path, start_ts=None, stt=whisper_segments, end_ts=None):
    """Archive (unless VLAB_AUDIO_RETAIN=0) and transcribe one captured segment file, then delete the temp file
    (always, success or failure). The archived m4a is written first, so audio survives a Whisper failure."""
    p = Path(path)
    start_ts = start_ts if start_ts is not None else (start_ts_from_name(p) or p.stat().st_mtime)
    end_ts = end_ts if end_ts is not None else end_ts_from_name(p)
    try:
        audio = decode(p)
        seg_id, extra = None, {}
        if audiostore.retain_enabled() and len(audio):
            try:
                ensure_schema(store)
                seg_id = audiostore.archive(store, audio, start_ts, end_ts)
                extra = {"audio_segment_id": seg_id}
            except Exception as e:      # archiving must never block transcription
                store.add_event("audio_archive_error", f"Voice audio could not be kept ({type(e).__name__})",
                                "#FF6B6B", ts=start_ts, modality="system", data={"file": p.name, "error": str(e)[:300]})
                extra = {"archive_error": str(e)[:300]}
        return {"file": p.name, **extra,
                **transcribe_pcm(store, audio, start_ts, stt, key=f"file:{p.name}", audio_segment_id=seg_id)}
    except Exception as e:
        store.add_event("transcribe_error", f"Voice segment could not be transcribed ({type(e).__name__})",
                        "#FF6B6B", ts=start_ts, modality="system", data={"file": p.name, "error": str(e)[:300]})
        return {"file": p.name, "error": str(e)[:300]}
    finally:
        p.unlink(missing_ok=True)          # the temp file; the kept copy (if any) is the archived m4a


def pending(dirpath=None, settle_s=SETTLE_S, now=None):
    d = Path(dirpath or VOICE_TMP)
    if not d.exists():
        return []
    now = now or time.time()
    return sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXT
                  and now - f.stat().st_mtime >= settle_s)


def recover(store, dirpath=None, **kw):
    """Crash/restart cleanup: archive + transcribe (or, failing that, still delete) every leftover temp file."""
    return [transcribe_file(store, f, **kw) for f in pending(dirpath, settle_s=0)]


def run(store, dirpath=None, poll_s=2.0, once=False, **kw):
    recover(store, dirpath, **kw)
    while True:
        for f in pending(dirpath):
            r = transcribe_file(store, f, **kw)
            print(json.dumps(r), flush=True)
        if once:
            return
        time.sleep(poll_s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["run", "once", "recover", "file"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--start", type=float, default=None, help="segment start (epoch s) for `file`")
    a = ap.parse_args(argv)
    from .store import Store
    st = Store()
    kw = {}
    if a.cmd == "file":
        print(json.dumps(transcribe_file(st, a.path, a.start, **kw)))
    elif a.cmd == "recover":
        print(json.dumps(recover(st, a.dir, **kw)))
    else:
        run(st, a.dir, once=a.cmd == "once", **kw)


if __name__ == "__main__":
    sys.exit(main())
