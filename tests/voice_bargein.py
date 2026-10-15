"""Barge-in test: ask a long question, start talking again as soon as reply audio arrives,
and expect an interrupt event followed by a fresh turn."""
import asyncio, json, sys
import numpy as np, websockets
sys.path.insert(0, ".")
from vision_lab import voice

def speech(text):
    return voice._resample(voice.tts(text), voice.SR_OUT, voice.SR_IN)

async def stream(ws, pcm):
    for i in range(0, len(pcm), 512):
        await ws.send(pcm[i:i + 512].tobytes()); await asyncio.sleep(512 / 16000)

async def main(port):
    a = np.concatenate([np.zeros(8000, np.float32), speech("Tell me a long story about a dog."), np.zeros(32000, np.float32)])
    b = np.concatenate([speech("Stop. What day is it?"), np.zeros(32000, np.float32)])
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws/voice", max_size=None) as ws:
        while json.loads(await ws.recv())["type"] != "ready": pass
        sender = asyncio.create_task(stream(ws, a))
        seen, turns, barged = [], [], False
        while len(turns) < 2:
            m = await asyncio.wait_for(ws.recv(), 120)
            if isinstance(m, bytes):
                if not barged:
                    barged = True; await sender; sender = asyncio.create_task(stream(ws, b))
                continue
            m = json.loads(m)
            if m["type"] in ("interrupt", "transcript"): seen.append((m["type"], m.get("text") or m.get("reason")))
            if m["type"] == "metrics": turns.append({k: m["turn"].get(k) for k in ("user", "interrupted", "stt_ms", "llm_first_token_ms", "tts_first_audio_ms", "reply")}); stats = m["stats"]
        print(json.dumps({"events": seen, "turns": turns, "stats": stats}, indent=1))

asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8766))
