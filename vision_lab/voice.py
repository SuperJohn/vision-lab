"""Local real-time voice chat: browser mic -> Silero VAD -> mlx-whisper -> Ollama -> Kokoro -> browser.

Wire protocol on /ws/voice
  client -> server  binary: float32 mono PCM at 16 kHz (any chunk size)
                    text:   {"type": "config", ...} | {"type": "text", "text": "..."} | {"type": "stop"}
  server -> client  text:   state / partial / transcript / token / reply / audio / interrupt / metrics / error
                    binary: int16 mono PCM at 24 kHz, always right after an {"type":"audio"} header
Every finished turn becomes one event dict (kind="voice_turn") kept in memory and appended to
data/voice_events.jsonl so step 2 can load it straight into vlab.db.
"""
import asyncio
import collections
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
from fastapi import APIRouter, Body, WebSocket, WebSocketDisconnect

ROOT = Path(__file__).resolve().parent.parent
KOKORO_DIR = ROOT / "weights" / "kokoro"
LOG = Path(os.environ.get("VLAB_VOICE_LOG", ROOT / "data" / "voice_events.jsonl"))
OLLAMA = "http://localhost:11434"
SR_IN, SR_OUT, VAD_CHUNK = 16000, 24000, 512

SETTINGS = {
    "llm_model": os.environ.get("VLAB_MODEL", "gemma4:26b"),  # VLAB_MODEL=gemma3:27b to switch back
    "stt_model": "mlx-community/whisper-large-v3-turbo",
    "tts_engine": "kokoro",          # "kokoro" (local) | "grok" (cloud, not wired yet)
    "voice": "af_heart",
    "speed": 1.05,
    "vad_start": 0.5,                # speech probability to open a turn
    "vad_end": 0.35,                 # below this counts as silence
    "silence_ms": 600,               # silence that ends a turn
    "min_speech_ms": 250,
    "partial_every_ms": 700,
    "grok_voice_enabled": False,     # cloud toggle: stub only, deliberately not wired
    "ambient": True,                 # classify room sounds while the mic is open
    "ambient_model": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "ambient_every_ms": 1000,
    "ambient_min_conf": 0.25,
    "loud_dbfs": -12.0,
    "vision_context": True,          # give the voice brain a compact summary of recent vision data
}
# AudioSet labels -> the few household sounds we log. Anything else is ignored.
AMBIENT_GROUPS = {
    "dog": {"Dog", "Bark", "Bow-wow", "Yip", "Growling", "Whimper (dog)", "Howl", "Canidae, dogs, wolves"},
    "doorbell": {"Doorbell", "Ding-dong"},
    "knock": {"Knock", "Tap"},
    "speech": {"Speech", "Conversation", "Narration, monologue", "Male speech, man speaking",
               "Female speech, woman speaking", "Child speech, kid speaking", "Shout", "Yell"},
    "alarm": {"Smoke detector, smoke alarm", "Alarm", "Beep, bleep", "Siren", "Buzzer"},
    "music": {"Music"},
    "water": {"Water tap, faucet", "Sink (filling or washing)", "Dishes, pots, and pans"},
}
AMBIENT_COOLDOWN = {"speech": 30, "music": 60, "water": 30}   # seconds between repeat events; default 8
# Set by the server: sink(event) stores events in vlab.db; context(question) returns recent vision context.
HOOKS = {"sink": None, "context": None, "gpu_lock": None, "mic_enabled": None}
SYSTEM = ("You are a voice assistant running locally on John's Mac Studio, talking with John out loud. "
          "Answer directly in one or two short, natural spoken sentences. Do not editorialize or talk "
          "about John in the third person. No markdown, lists, emoji or code. If unsure, say so briefly.")

router = APIRouter()
EVENTS = collections.deque(maxlen=500)
STATS = {"turns": 0, "interruptions": 0}
voice_busy_until = 0.0    # the vision caption worker yields while this is in the future

_models, _load_lock, _stt_lock, _tts_lock = {}, threading.Lock(), threading.Lock(), threading.Lock()


def _get(name):
    with _load_lock:
        if name not in _models:
            if name == "vad":
                from silero_vad import load_silero_vad
                _models[name] = load_silero_vad(onnx=True)
            elif name == "tts":
                from kokoro_onnx import Kokoro
                _models[name] = Kokoro(str(KOKORO_DIR / "kokoro-v1.0.onnx"), str(KOKORO_DIR / "voices-v1.0.bin"))
            elif name == "stt":
                import mlx_whisper
                _models[name] = mlx_whisper
        return _models[name]


def warmup():
    """Load and exercise every model once so the first real turn is fast."""
    t0 = time.time()
    audio = tts("Ready.")
    stt(_resample(audio, SR_OUT, SR_IN))
    _get("vad")
    if SETTINGS["ambient"]:
        threading.Thread(target=load_ambient, daemon=True).start()
    return round((time.time() - t0) * 1000)


def _resample(a, sr_from, sr_to):
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr_from, sr_to)
    return resample_poly(a, sr_to // g, sr_from // g).astype(np.float32)


def stt(audio16k):
    with _stt_lock:
        r = _get("stt").transcribe(audio16k, path_or_hf_repo=SETTINGS["stt_model"], language="en",
                                  condition_on_previous_text=False)
    return r["text"].strip()


def tts(text):
    if SETTINGS["tts_engine"] != "kokoro":
        raise RuntimeError("Grok Voice (cloud) is not wired up yet; switch back to Kokoro (local).")
    with _tts_lock:
        audio, _ = _get("tts").create(text, voice=SETTINGS["voice"], speed=SETTINGS["speed"], lang="en-us")
    return audio.astype(np.float32)


_amb_load_lock = threading.Lock()


def load_ambient():
    """Loaded in the background (separate lock) so it never delays speech recognition."""
    with _amb_load_lock:
        if "ambient" not in _models:
            from transformers import pipeline
            import torch
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            clf = pipeline("audio-classification", model=SETTINGS["ambient_model"], device=dev)
            clf({"raw": np.zeros(SR_IN, np.float32), "sampling_rate": SR_IN}, top_k=1)
            _models["ambient"] = clf


def classify_ambient(audio16k):
    """Top AudioSet labels for a short clip, folded into our household groups."""
    gpu = HOOKS.get("gpu_lock")   # the vision pipeline's MPS lock; skip this window rather than stall vision
    if gpu is not None and not gpu.acquire(timeout=0.2):
        return {}, []
    try:
        with _amb_lock:
            preds = _get("ambient")({"raw": audio16k, "sampling_rate": SR_IN}, top_k=10)
    finally:
        if gpu is not None:
            gpu.release()
    found = {}
    for p in preds:
        for group, labels in AMBIENT_GROUPS.items():
            if p["label"] in labels and p["score"] > found.get(group, (0, ""))[0]:
                found[group] = (p["score"], p["label"])
    return found, preds[:3]


_amb_lock = threading.Lock()


def _log(event):
    EVENTS.append(event)
    if HOOKS["sink"]:
        try:
            HOOKS["sink"](event)
        except Exception as e:  # never let storage break a conversation
            print("voice sink error:", e)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(json.dumps(event) + "\n")
    except OSError:
        pass


_SENT_END = re.compile(r"(.+?[.!?;:])(\s+|$)", re.S)


class Session:
    def __init__(self, sock):
        self.sock = sock
        self.send_lock = asyncio.Lock()
        self.history = []
        self.buf = np.zeros(0, np.float32)      # unprocessed mic samples
        self.speech = []                         # samples of the current utterance
        self.in_speech, self.silence_ms, self.speech_ms = False, 0, 0
        self.last_partial = 0.0
        self.partial_task = None
        self.turn_task = None
        self.speaking_until = 0.0                # when queued playback should finish (server clock)
        self.vad = _get("vad")
        self.amb_buf = collections.deque(maxlen=SR_IN * 2 // VAD_CHUNK)   # last ~2 s of mic
        self.amb_last, self.amb_task, self.amb_fired = 0.0, None, {}

    async def send(self, obj=None, data=None):
        async with self.send_lock:
            if obj is not None:
                await self.sock.send_text(json.dumps(obj))
            if data is not None:
                await self.sock.send_bytes(data)

    def assistant_active(self):
        return (self.turn_task and not self.turn_task.done()) or time.time() < self.speaking_until

    async def interrupt(self, reason="barge-in"):
        if not self.assistant_active():
            return
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
        self.speaking_until = 0.0
        STATS["interruptions"] += 1
        _log({"kind": "voice_interrupt", "ts": time.time(), "reason": reason})
        await self.send({"type": "interrupt", "reason": reason, "interruptions": STATS["interruptions"]})

    async def feed(self, pcm):
        global voice_busy_until
        self.buf = np.concatenate([self.buf, pcm])
        while len(self.buf) >= VAD_CHUNK:
            chunk, self.buf = self.buf[:VAD_CHUNK], self.buf[VAD_CHUNK:]
            self.amb_buf.append(chunk)
            self.maybe_ambient()
            prob = float(self.vad(__import__("torch").from_numpy(chunk.copy()), SR_IN).item())
            ms = VAD_CHUNK * 1000 / SR_IN
            if not self.in_speech:
                self.speech = (self.speech + [chunk])[-6:]   # ~190 ms pre-roll
                if prob >= SETTINGS["vad_start"]:
                    self.in_speech, self.silence_ms, self.speech_ms = True, 0, ms
                    self.t_speech_start = time.time()
                    voice_busy_until = time.time() + 30
                    await self.send({"type": "state", "state": "listening"})
                    await self.interrupt()
                continue
            self.speech.append(chunk)
            self.speech_ms += ms
            self.silence_ms = self.silence_ms + ms if prob < SETTINGS["vad_end"] else 0
            now = time.time()
            if (now - self.last_partial) * 1000 > SETTINGS["partial_every_ms"] and \
                    (self.partial_task is None or self.partial_task.done()):
                self.last_partial = now
                self.partial_task = asyncio.create_task(self._partial(np.concatenate(self.speech)))
            if self.silence_ms >= SETTINGS["silence_ms"]:
                audio = np.concatenate(self.speech)
                self.in_speech, self.speech = False, []
                if self.speech_ms - self.silence_ms < SETTINGS["min_speech_ms"]:
                    await self.send({"type": "state", "state": "idle"})
                    continue
                self.start_turn(audio=audio, t_vad_end=now)

    def maybe_ambient(self):
        now = time.time()
        if not SETTINGS["ambient"] or (now - self.amb_last) * 1000 < SETTINGS["ambient_every_ms"] \
                or (self.amb_task and not self.amb_task.done()) or len(self.amb_buf) < 16:
            return
        if "ambient" not in _models:
            if not _amb_load_lock.locked():
                threading.Thread(target=load_ambient, daemon=True).start()
            return
        self.amb_last = now
        self.amb_task = asyncio.create_task(self._ambient(np.concatenate(list(self.amb_buf)[-31:]), now))

    async def _ambient(self, clip, ts):
        rms = float(np.sqrt(np.mean(clip ** 2)) + 1e-9)
        dbfs = round(20 * np.log10(rms), 1)
        hits = {}
        if dbfs >= SETTINGS["loud_dbfs"]:
            hits["loud noise"] = (1.0, f"{dbfs} dBFS")
        try:
            t0 = time.time()
            found, top = (await asyncio.to_thread(classify_ambient, clip)) if "ambient" in _models else ({}, [])
            if top:
                STATS["ambient_ms"] = round((time.time() - t0) * 1000)
            hits.update({g: v for g, v in found.items() if v[0] >= SETTINGS["ambient_min_conf"]})
        except Exception as e:
            STATS["ambient_error"] = str(e)
            top = []
        if self.in_speech or self.assistant_active():
            hits.pop("speech", None)    # that's the conversation itself, already logged as utterances
        for group, (conf, label) in hits.items():
            if ts - self.amb_fired.get(group, 0) < AMBIENT_COOLDOWN.get(group, 8):
                continue
            self.amb_fired[group] = ts
            ev = {"kind": "audio_event", "ts": ts, "label": group, "audioset_label": label,
                  "confidence": round(float(conf), 3), "dbfs": dbfs, "model": SETTINGS["ambient_model"]}
            _log(ev)
            await self.send({"type": "ambient", **ev})

    async def _partial(self, audio):
        try:
            text = await asyncio.to_thread(stt, audio)
            if self.in_speech and re.search(r"[A-Za-z0-9]", text):
                await self.send({"type": "partial", "text": text})
        except Exception:
            pass

    def start_turn(self, audio=None, text=None, t_vad_end=None):
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
        self.turn_task = asyncio.create_task(self.turn(audio, text, t_vad_end or time.time()))

    async def turn(self, audio, text, t_end):
        global voice_busy_until
        voice_busy_until = time.time() + 60
        m = {"kind": "voice_turn", "turn_id": uuid.uuid4().hex[:12], "ts": t_end,
             "ts_speech_start": getattr(self, "t_speech_start", None) if audio is not None else t_end, "llm": SETTINGS["llm_model"], "tts": SETTINGS["tts_engine"],
             "voice": SETTINGS["voice"], "interrupted": False}
        try:
            if audio is not None:
                m["speech_ms"] = round(len(audio) / SR_IN * 1000)
                if self.partial_task and not self.partial_task.done():
                    await self.partial_task
                await self.send({"type": "state", "state": "transcribing"})
                text = await asyncio.to_thread(stt, audio)
                m["stt_ms"] = round((time.time() - t_end) * 1000)
            m["user"] = text
            await self.send({"type": "transcript", "text": text, "stt_ms": m.get("stt_ms")})
            if not text or not re.search(r"\w", text):
                await self.send({"type": "state", "state": "idle"})
                return
            await self.send({"type": "state", "state": "thinking"})
            reply = await self._think_and_speak(text, t_end, m)
            self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
            self.history = self.history[-12:]
            m["reply"] = reply
            m["total_ms"] = round((time.time() - t_end) * 1000)
            m["ts_reply_end"] = max(self.speaking_until, time.time())
            STATS["turns"] += 1
        except asyncio.CancelledError:
            m["interrupted"] = True
            raise
        except Exception as e:
            m["error"] = str(e)
            await self.send({"type": "error", "error": str(e)})
        finally:
            _log(m)
            voice_busy_until = max(self.speaking_until, time.time()) + 5
            try:
                await self.send({"type": "metrics", "turn": m, "stats": STATS})
            except Exception:
                pass

    async def _think_and_speak(self, text, t_end, m):
        system = SYSTEM + f" Local time: {time.strftime('%A %-I:%M %p')}."
        if SETTINGS["vision_context"] and HOOKS["context"]:
            t0 = time.time()
            try:
                ctx = await asyncio.to_thread(HOOKS["context"], text)
            except Exception as e:
                ctx = f"(home camera data unavailable: {e})"
            m["context_ms"] = round((time.time() - t0) * 1000)
            m["context_chars"] = len(ctx)
            system += ("\nYou can also see John's apartment through the home camera. Use this log only when "
                       "relevant; if it doesn't say, say you didn't see it. The dog is John's dog.\n" + ctx)
        msgs = [{"role": "system", "content": system}] \
            + self.history + [{"role": "user", "content": text}]
        body = {"model": SETTINGS["llm_model"], "messages": msgs, "stream": True, "keep_alive": "30m", "think": False,
                "options": {"temperature": 0.6, "num_predict": 160}}
        reply, pending, seq = "", "", 0
        tts_q = asyncio.Queue()

        async def speaker():
            nonlocal seq
            while True:
                sentence = await tts_q.get()
                if sentence is None:
                    return
                audio = await asyncio.to_thread(tts, sentence)
                if "tts_first_audio_ms" not in m:
                    m["tts_first_audio_ms"] = round((time.time() - t_end) * 1000)
                    m["tts_first_chunk_ms"] = round((time.time() - m["_t_first_sentence"]) * 1000)
                pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
                dur = len(audio) / SR_OUT
                self.speaking_until = max(self.speaking_until, time.time()) + dur
                await self.send({"type": "audio", "seq": seq, "text": sentence, "sr": SR_OUT,
                                 "duration_ms": round(dur * 1000)}, pcm)
                seq += 1

        spk = asyncio.create_task(speaker())
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", f"{OLLAMA}/api/chat", json=body) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        tok = json.loads(line).get("message", {}).get("content", "")
                        if tok and "llm_first_token_ms" not in m:
                            m["llm_first_token_ms"] = round((time.time() - t_end) * 1000)
                        reply += tok
                        pending += tok
                        await self.send({"type": "token", "text": tok})
                        while (mt := _SENT_END.match(pending)) and len(mt.group(1)) > 3:
                            m.setdefault("_t_first_sentence", time.time())
                            await tts_q.put(mt.group(1).strip())
                            pending = pending[mt.end():]
            m["llm_done_ms"] = round((time.time() - t_end) * 1000)
            if pending.strip():
                m.setdefault("_t_first_sentence", time.time())
                await tts_q.put(pending.strip())
            await tts_q.put(None)
            await spk
        finally:
            if not spk.done():
                spk.cancel()
            m.pop("_t_first_sentence", None)
        reply = re.sub(r"[*_#`]", "", reply).strip()
        await self.send({"type": "reply", "text": reply})
        await self.send({"type": "state", "state": "speaking"})
        return reply


@router.websocket("/ws/voice")
async def ws_voice(sock: WebSocket):
    await sock.accept()
    await sock.send_text(json.dumps({"type": "state", "state": "loading"}))
    try:
        ms = await asyncio.to_thread(warmup) if "stt" not in _models else 0
    except Exception as e:
        await sock.send_text(json.dumps({"type": "error", "error": f"model load failed: {e}"}))
        await sock.close()
        return
    s = Session(sock)
    await s.send({"type": "ready", "warmup_ms": ms, "settings": SETTINGS, "stats": STATS})
    try:
        while True:
            msg = await sock.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if HOOKS["mic_enabled"] and not HOOKS["mic_enabled"]():
                    continue   # mic toggled off: drop audio, no turns, no ambient events
                await s.feed(np.frombuffer(msg["bytes"], np.float32))
            elif msg.get("text"):
                cmd = json.loads(msg["text"])
                if cmd.get("type") == "text":
                    await s.interrupt("new text")
                    s.start_turn(text=cmd.get("text", ""))
                elif cmd.get("type") == "stop":
                    await s.interrupt("stop button")
                elif cmd.get("type") == "playback_done":
                    s.speaking_until = 0.0
    except WebSocketDisconnect:
        pass
    finally:
        if s.turn_task and not s.turn_task.done():
            s.turn_task.cancel()


@router.get("/api/voice/state")
def voice_state():
    return {"settings": SETTINGS, "stats": STATS, "events": list(EVENTS)[-50:], "loaded": sorted(_models)}


@router.post("/api/voice/settings")
def voice_settings(patch: dict = Body(...)):
    # Grok Voice (cloud) is a visible stub: it stays off until keys are wired on purpose.
    patch.pop("grok_voice_enabled", None)
    if patch.get("tts_engine") == "grok":
        patch["tts_engine"] = "kokoro"
    SETTINGS.update({k: v for k, v in patch.items() if k in SETTINGS})
    return SETTINGS
