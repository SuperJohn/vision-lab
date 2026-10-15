"""End-to-end voice test: Kokoro speaks a question, it is streamed in real time over /ws/voice,
and we check for a transcript plus spoken reply audio. Usage: python tests/voice_e2e.py [port]"""
import asyncio, json, sys, time
import numpy as np, websockets
sys.path.insert(0, ".")
from vision_lab import voice

async def main(port=8766, question="What time is it?"):
    q = voice._resample(voice.tts(question), voice.SR_OUT, voice.SR_IN)
    pcm = np.concatenate([np.zeros(8000, np.float32), q, np.zeros(16000 * 2, np.float32)])
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws/voice", max_size=None) as ws:
        while json.loads(await ws.recv())["type"] != "ready":
            pass
        t0 = time.time()
        for i in range(0, len(pcm), 512):   # real-time pacing, 32 ms chunks
            await ws.send(pcm[i:i + 512].tobytes())
            await asyncio.sleep(512 / 16000)
        out, audio_bytes, partials = {}, 0, []
        while True:
            m = await asyncio.wait_for(ws.recv(), 120)
            if isinstance(m, bytes):
                audio_bytes += len(m); continue
            m = json.loads(m)
            if m["type"] == "partial": partials.append(m["text"])
            if m["type"] in ("transcript", "reply", "error"): out[m["type"]] = m.get("text") or m.get("error")
            if m["type"] == "metrics":
                out["metrics"] = m["turn"]; break
        out["partials"] = partials
        out["reply_audio_s"] = round(audio_bytes / 2 / 24000, 2)
        print(json.dumps(out, indent=1))

asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8766))
