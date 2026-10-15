"""Step 2 checks against a running server: spoken + typed turns land in vlab.db with keyframe links,
the brain uses vision context, ambient sounds are logged, and the merged APIs/export work.
Usage: python tests/multimodal_e2e.py [port]"""
import asyncio, json, sys, time, urllib.request
import numpy as np, websockets
sys.path.insert(0, ".")
from vision_lab import voice

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
BASE = f"http://127.0.0.1:{PORT}"
get = lambda p: urllib.request.urlopen(BASE + p, timeout=60).read()

async def turn(ws, text=None, pcm=None):
    if text:
        await ws.send(json.dumps({"type": "text", "text": text}))
    else:
        for i in range(0, len(pcm), 512):
            await ws.send(pcm[i:i + 512].tobytes()); await asyncio.sleep(512 / 16000)
    amb = []
    while True:
        m = await asyncio.wait_for(ws.recv(), 120)
        if isinstance(m, bytes): continue
        m = json.loads(m)
        if m["type"] == "ambient": amb.append((m["label"], m["audioset_label"], m["confidence"]))
        if m["type"] == "metrics": return m["turn"], amb

async def main():
    t_start = time.time()
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws/voice", max_size=None) as ws:
        while json.loads(await ws.recv())["type"] != "ready": pass
        q = voice._resample(voice.tts("Is the dog here right now?"), 24000, 16000)
        sr = 16000
        # 1 s of knocking-like clicks and a loud tone before the question, to exercise the ambient classifier
        t = np.arange(sr) / sr
        knocks = np.zeros(sr, np.float32)
        for k in (0.1, 0.35, 0.6):
            i = int(k * sr); knocks[i:i + 800] = (np.random.randn(800) * np.exp(-np.arange(800) / 120)).astype(np.float32) * 0.8
        tone = (0.9 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        pcm = np.concatenate([knocks, np.zeros(sr, np.float32), tone, np.zeros(sr * 2, np.float32), q, np.zeros(sr * 2, np.float32)])
        results = []
        m, amb = await turn(ws, pcm=pcm); results.append(m)
        m2, _ = await turn(ws, text="What was I doing at 3pm today?"); results.append(m2)
    await asyncio.sleep(1)
    keys = ("user", "reply", "stt_ms", "context_ms", "context_chars", "llm_first_token_ms", "tts_first_audio_ms", "total_ms")
    print("TURNS", json.dumps([{k: r.get(k) for k in keys} for r in results], indent=1))
    print("AMBIENT", amb)
    tl = json.loads(get(f"/api/timeline/events?since={t_start - 5}&limit=50"))
    print("TIMELINE since test start:", tl["count"], [(i["modality"], i["kind"], i["label"], (i["text"] or "")[:50], i["keyframe_id"],
          round((i["written_at"] - i["ts"]) * 1000) if i["written_at"] else None) for i in tl["items"]])
    u = [i for i in tl["items"] if i["source"] == "utterance"][0]
    print("SNAPSHOT keys", list((u["data"] or {}).keys()))
    print("SEARCH dog:", json.loads(get("/api/search/all?q=dog&limit=5"))["count"])
    print("SEARCH time (voice):", [(i["label"], i["text"][:40]) for i in json.loads(get("/api/search/all?q=time&modality=voice&limit=5"))["items"]])
    for fmt in ("jsonl", "parquet"):
        t0 = time.time(); b = get(f"/api/export?format={fmt}&since={time.time() - 86400}")
        print(f"EXPORT {fmt}: {len(b)} bytes in {round((time.time() - t0) * 1000)} ms")
    print("STATE", json.loads(get("/api/voice/state"))["stats"])

asyncio.run(main())
