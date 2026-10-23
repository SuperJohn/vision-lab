"""drift fix: voice chunk timestamps follow the wall clock even when the mic under-delivers samples.

The live mic delivered ~80% of the nominal 16 kHz, so stamping chunks as t0 + samples/16000 fell ~11 min/h
behind. These tests feed MicRecorder._read a fake ffmpeg pipe plus a fake clock: no real mic, no ffmpeg, a
temp DB and temp dirs only. Never touches the live app or vlab.db.

    PYTHONPATH=. .venv/bin/python tests/mic_clock_drift.py
"""
import shutil, sys, tempfile, types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
os.environ["VLAB_AUDIO_DIR"] = tempfile.mkdtemp(prefix="vlab-audio-test-")   # never the real archive
from vision_lab import recording, voicetx
from vision_lab.store import Store

SR = recording.SR
BLOCK = SR // 10                                       # MicRecorder reads 100 ms (1600 samples) per read


class FakeClock:
    def __init__(self, t=1_792_600_000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakePipe:
    """ffmpeg stdout stand-in: every read returns 100 ms of audio while `wall_per_block` s of wall time pass."""

    def __init__(self, clock, wall_s, delivered=0.8):
        self.clock, self.left = clock, int(wall_s / (0.1 / delivered))
        self.wall_per_block = 0.1 / delivered
        self.firsts = []                               # wall time at which each block arrived
        tone = 0.3 * np.sin(2 * np.pi * 330 * np.arange(BLOCK) / SR)
        self.bytes = tone.astype(np.float32).tobytes()

    def read(self, n):
        if self.left <= 0:
            return b""
        self.left -= 1
        self.clock.t += self.wall_per_block            # this block's audio finished arriving now
        self.firsts.append(self.clock.t)
        return self.bytes[:n]


def run(wall_s, delivered):
    tmp = Path(tempfile.mkdtemp(prefix="micdrift-"))
    st = Store(str(tmp / "vlab.db"))
    clock = FakeClock()
    m = recording.MicRecorder(st, voice_tmp=tmp / "voice", chunk_s=2.0, max_chunk_s=3.0,
                              transcribe=lambda f: None, clock=clock)
    m.staging.mkdir(parents=True, exist_ok=True)
    pipe = FakePipe(clock, wall_s, delivered)
    m._read(types.SimpleNamespace(stdout=pipe))        # no worker: published chunks stay queued on disk
    chunks = [Path(f.name) for f in sorted((tmp / "voice").glob("voice-*.pcm"), key=voicetx.start_ts_from_name)]
    drift = st._query("SELECT ts, data FROM events WHERE kind = 'mic_drift' ORDER BY ts")
    shutil.rmtree(tmp, ignore_errors=True)             # temp audio only (this test's own files)
    return m, pipe, clock, chunks, drift


def test_chunks_track_wall_clock_at_80_percent():
    m, pipe, clock, chunks, _ = run(wall_s=1200, delivered=0.8)
    assert len(chunks) > 300, len(chunks)
    # each chunk = 3 s of audio = 30 blocks; chunk k starts with block 30k, whose audio began one block earlier
    worst = 0.0
    for k, f in enumerate(chunks):
        expected = pipe.firsts[30 * k] - 0.1
        worst = max(worst, abs(voicetx.start_ts_from_name(f) - expected))
    assert worst < 0.01, f"chunk stamp off the wall clock by {worst:.3f} s"
    last = voicetx.start_ts_from_name(chunks[-1])
    assert clock.t - last < 5, f"newest chunk {clock.t - last:.1f} s behind the wall clock"
    # the old t0 + samples/SR stamp would have been ~4 min behind after 20 min at 80% (~12 min/h)
    assert 235 < m.drift_s < 245, m.drift_s


def test_drift_event_logged_and_rate_limited():
    m, _, clock, _, drift = run(wall_s=900, delivered=0.8)
    # crosses 2 s at ~10 s of wall time, then at most once per MIC_DRIFT_LOG_EVERY_S (300 s): ~10, ~310, ~610
    assert len(drift) == 3, len(drift)
    assert m.drift_events == 3
    gaps = [b["ts"] - a["ts"] for a, b in zip(drift, drift[1:])]
    assert all(g >= recording.MIC_DRIFT_LOG_EVERY_S for g in gaps), gaps
    import json
    d = json.loads(drift[0]["data"])
    assert d["drift_s"] > recording.MIC_DRIFT_LOG_S and abs(d["delivered_ratio"] - 0.8) < 0.01, d


def test_no_drift_event_when_mic_is_on_time():
    m, pipe, _, chunks, drift = run(wall_s=600, delivered=1.0)
    assert not drift, drift
    assert abs(m.drift_s) < 0.01, m.drift_s
    for k, f in enumerate(chunks):
        assert abs(voicetx.start_ts_from_name(f) - (pipe.firsts[30 * k] - 0.1)) < 0.01


def test_file_name_carries_wall_clock_start_and_end():
    """voice-<start ms>-<end ms>.pcm; end = wall clock at close, so a 3 s audio chunk at 80% spans 3.75 s."""
    _, _, _, chunks, _ = run(wall_s=10, delivered=0.8)
    f = chunks[0]
    a, b = f.stem[6:].split("-")
    assert f.name.startswith("voice-") and f.suffix == ".pcm" and len(a) == 13 and len(b) == 13 and a.isdigit() and b.isdigit()
    s, e = voicetx.start_ts_from_name(f), voicetx.end_ts_from_name(f)
    assert s is not None and e is not None and abs((e - s) - 3.0 / 0.8) < 0.05, (s, e)
    assert voicetx.end_ts_from_name("voice-1792600000000.pcm") is None            # older names: no end
    assert voicetx.end_ts_from_name("voice-1792600000000-2.pcm") is None          # dedupe suffix is not an end


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time(); fn(); print(f"PASS {name} ({time.time() - t:.1f}s)", flush=True)
    print("mic_clock_drift: OK")
