"""FastAPI app: live WebSocket stream plus the REST API behind the four screens."""
import asyncio
import json
import os
import signal
import datetime as dt
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import llm, mic, multimodal, voice
from .bootinfo import BOOT_ID, install_no_cache, render_index, with_boot_id
from .engine import DEPTH_MODEL, Engine, color_for
from .store import Store

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Vision Lab")
app.include_router(voice.router)
app.include_router(mic.router)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


install_no_cache(app)   # HOU-146: browsers re-check UI files (no-cache, must-revalidate)

store = Store()
engine = Engine(store)
voice.HOOKS["sink"] = lambda ev: multimodal.store_voice_event(store, engine, ev)
voice.HOOKS["context"] = lambda q: multimodal.vision_context(store, engine, q)
voice.HOOKS["gpu_lock"] = engine.models.lock
mic.STORE["store"] = store
voice.HOOKS["mic_enabled"] = mic.enabled
try:
    mic.ensure_default()
except Exception as e:
    print(f"[mic] device scan failed: {e}")
multimodal.import_jsonl(store)   # idempotent: rows already in the DB are skipped

CATALOG = [
    {"key": "yolo11n", "task": "Detect", "name": "YOLO11n", "gives": "80 everyday classes, fastest", "kind": "yolo"},
    {"key": "yolo11s", "task": "Detect", "name": "YOLO11s", "gives": "80 everyday classes, more accurate", "kind": "yolo"},
    {"key": "yolov8s-worldv2", "task": "Detect", "name": "YOLO-World v2", "gives": "Detect anything you type", "kind": "yolo"},
    {"key": "yolo11s-seg", "task": "Segment", "name": "YOLO11s-seg", "gives": "Outline of each object", "kind": "yolo"},
    {"key": "sam2.1_b", "task": "Segment", "name": "SAM 2.1 base", "gives": "Click an object to outline it", "kind": "sam"},
    {"key": "bytetrack", "task": "Track", "name": "ByteTrack", "gives": "Stable IDs across frames", "kind": "builtin"},
    {"key": "yolo11s-pose", "task": "Pose", "name": "YOLO11s-pose", "gives": "17 body keypoints per person", "kind": "yolo"},
    {"key": "depth", "task": "Depth", "name": "Depth Anything V2 Small", "gives": "Near and far, per pixel", "kind": "depth"},
    {"key": "siglip", "task": "Search", "name": "SigLIP base", "gives": "Find keyframes by text", "kind": "siglip"},
]
SUGGESTED_VLM = "qwen3-vl:8b"
bench_results = {}


def day_start(rng="today"):
    start = dt.datetime.combine(dt.date.today(), dt.time()).timestamp()
    return start - 6 * 86400 if rng == "week" else start


def current_frame():
    frame, _, _ = engine.camera.latest()
    if frame is None:
        raise HTTPException(503, engine.camera.error or "No camera frame yet")
    return frame


def jpeg(frame, width=1280):
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    return Response(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(),
                    media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/")
def index():
    return render_index(STATIC)   # asset URLs carry ?v=BOOT_ID


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    last = -1
    try:
        while True:
            if engine.out_seq != last:
                with engine.out_lock:
                    last, meta, img = engine.out_seq, engine.out_json, engine.out_jpeg
                await sock.send_text(with_boot_id(meta))
                if img:
                    await sock.send_bytes(img)
            await asyncio.sleep(0.005)
    except (WebSocketDisconnect, RuntimeError):
        pass


# ---------- live view ----------
@app.get("/api/state")
def state():
    vlms = [m["name"] for m in llm.list_models() if "vision" in llm.capabilities(m["name"])]
    texts = [m["name"] for m in llm.list_models()]
    return {"settings": engine.settings, "cameras": engine.cameras, "zones": engine.zones,
            "rules": engine.rules, "classes": engine.class_names(), "vlms": vlms, "llms": texts,
            "embedder_error": engine.embedder.error, "boot_id": BOOT_ID, "paused": bool(engine.settings.get("capture_paused"))}


@app.post("/api/settings")
def settings(patch: dict = Body(...)):
    engine.update_settings(patch)
    return engine.settings


@app.get("/api/events")
def events(since: float = 0, limit: int = 100):
    return store.events(since, limit)


@app.post("/api/ask")
def ask(body: dict = Body(...)):
    frame = engine.paused_frame if body.get("paused") and engine.paused_frame is not None else current_frame()
    t0 = time.time()
    try:
        answer = llm.chat(engine.settings["describe_model"], body["question"], frame)
    except Exception as e:
        raise HTTPException(502, f"Vision model error: {e}")
    return {"answer": answer, "latency": round(time.time() - t0, 1), "model": engine.settings["describe_model"]}


@app.post("/api/snapshot")
def snapshot():
    frame, ts = current_frame(), time.time()
    kf = engine.save_keyframe(frame, ts)
    engine.event("snapshot", "Snapshot saved", "#E8EDF2", keyframe_id=kf, ts=ts)
    return {"keyframe": kf}


@app.post("/api/power")
def power(body: dict = Body(...)):
    """HOU-146: the Privacy mode toggle; on = camera, screen and mic all stopped."""
    return engine.set_capture_paused(not body.get("on"))


@app.get("/api/privacy")
def privacy_state():
    on = bool(engine.settings.get("capture_paused"))
    return {"privacy_mode": on, "paused": on, "camera_paused": engine.camera.paused}


@app.post("/api/privacy")
def privacy(body: dict = Body(...)):
    """Body {"on": bool}: turn Privacy mode on (stop all capture) or off."""
    return engine.set_capture_paused(bool(body.get("on")))


@app.post("/api/shutdown")
def shutdown():
    """Stop the server process, closing any recording cleanly first."""
    engine.camera.stop_recording()
    threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    return {"stopping": True}


@app.post("/api/record")
def record(body: dict = Body(...)):
    if body.get("on"):
        path = engine.camera.start_recording()
        if path:
            engine.event("record", f"Recording started → {path.name}", "#FF6B6B")
    else:
        path = engine.camera.stop_recording()
        if path:
            engine.event("record", f"Recording saved: {path.name}", "#FF6B6B")
    return {"recording": engine.camera.recording_path is not None, "path": str(path) if path else None}


# ---------- zones & rules ----------
@app.get("/api/frame.jpg")
def frame_jpg(pause: bool = False):
    frame = current_frame()
    if pause:
        engine.paused_frame = frame.copy()
    return jpeg(frame)


@app.get("/api/paused.jpg")
def paused_jpg():
    if engine.paused_frame is None:
        raise HTTPException(404, "No paused frame")
    return jpeg(engine.paused_frame)


@app.put("/api/zones")
def put_zones(zones: list = Body(...)):
    engine.set_zones(zones)
    return zones


@app.put("/api/rules")
def put_rules(rules: list = Body(...)):
    engine.set_rules(rules)
    return rules


@app.post("/api/segment")
def segment(body: dict = Body(...)):
    frame = engine.paused_frame if engine.paused_frame is not None else current_frame()
    h, w = frame.shape[:2]
    with engine.models.lock:
        sam = engine.models.yolo("sam2.1_b")
        r = sam(frame, points=[[body["x"] * w, body["y"] * h]], labels=[1],
                device=engine.models.device, verbose=False)[0]
    if r.masks is None or not len(r.masks.xyn):
        raise HTTPException(404, "Nothing segmented at that point")
    poly = max(r.masks.xyn, key=len)
    poly = poly[:: max(1, len(poly) // 60)]
    return {"points": [[round(float(x), 4), round(float(y), 4)] for x, y in poly]}


@app.post("/api/rule_test")
def rule_test(body: dict = Body(...)):
    return ask({"question": body["prompt"], "paused": True})


# ---------- timeline & search ----------
@app.get("/api/keyframe/{kid}.jpg")
def keyframe(kid: int):
    k = store.keyframe(kid)
    if not k or not Path(k["path"]).exists():
        raise HTTPException(404)
    return FileResponse(k["path"], media_type="image/jpeg")


@app.get("/api/search")
def search(q: str, range: str = "today"):
    since = day_start(range)
    total = store.count_keyframes(since)
    hits = engine.embedder.search(q, store.keyframe_embeddings(since))
    method = "siglip"
    if not hits:
        method = "caption"
        hits = [(r["id"], r["ts"], r["caption"], None) for r in store.keyframes_text_search(q, since)]
    return {"total": total, "method": method, "error": engine.embedder.error,
            "results": [{"id": i, "ts": ts, "caption": c, "score": None if s is None else round(s, 3)}
                        for i, ts, c, s in hits]}


@app.get("/api/timeline")
def timeline(range: str = "today"):
    since = day_start(range)
    lanes = {}
    for s in store.spans(since):
        if s["end"] - s["start"] < 3:
            continue
        segs = lanes.setdefault(s["cls"], [])
        start = max(s["start"], since)
        if segs and start <= segs[-1][1] + 30:
            segs[-1][1] = max(segs[-1][1], s["end"])
        else:
            segs.append([start, s["end"]])
    rules = [e["ts"] for e in store.events(since, 1000) if e["kind"] == "rule"]
    out = [{"cls": c, "color": color_for(c), "spans": v,
            "seconds": round(sum(b - a for a, b in v))} for c, v in lanes.items()]
    out.sort(key=lambda l: -l["seconds"])
    marks = store.merged(since, modalities=["audio", "voice"], limit=2000)
    return {"start": since, "end": since + (7 if range == "week" else 1) * 86400,
            "lanes": out[:8], "rules": rules,
            "audio": [r["ts"] for r in marks if r["modality"] == "audio"],
            "voice": [r["ts"] for r in marks if r["modality"] == "voice"]}


# ---------- multimodal record (step 2) ----------
def _mods(modality):
    return [m.strip() for m in modality.split(",") if m.strip()] or None


@app.get("/api/timeline/events")
def timeline_events(since: float = 0, until: float = 0, range: str = "", modality: str = "",
                    q: str = "", limit: int = 500):
    """Merged, newest-first stream across vision / audio / voice / system. Filters: since/until (epoch s)
    or range=today|week, modality=comma list, q=text match."""
    since = day_start(range) if range else since
    rows = store.merged(since, until or None, _mods(modality), q or None, min(limit, 5000))
    return {"since": since, "count": len(rows), "items": rows}


@app.get("/api/search/all")
def search_all(q: str, range: str = "week", modality: str = "", limit: int = 50):
    """Text search over camera captions, event text and voice transcripts/replies."""
    rows = store.merged(day_start(range), None, _mods(modality), q, min(limit, 500))
    return {"q": q, "count": len(rows), "items": rows}


def _screens_db():
    import sqlite3
    from vision_lab import batch
    db = sqlite3.connect(batch.DB_PATH, timeout=10); db.row_factory = sqlite3.Row
    return db, batch.DATA / "screens"


@app.get("/api/screens")
def screens(date: str = "", limit: int = 200):
    """Local-only (HOU-68): captioned screenshots from batch_annotations, newest first. date=YYYY-MM-DD (local)."""
    db, _ = _screens_db()
    try:
        if date:
            t0 = dt.datetime.strptime(date, "%Y-%m-%d").timestamp()
            rows = db.execute("SELECT id, capture_ts, caption, label, model, gemma_ms, status FROM batch_annotations WHERE source_type='screen' "
                              "AND capture_ts >= ? AND capture_ts < ? ORDER BY capture_ts DESC LIMIT ?", (t0, t0 + 86400, min(limit, 1000))).fetchall()
        else:
            rows = db.execute("SELECT id, capture_ts, caption, label, model, gemma_ms, status FROM batch_annotations WHERE source_type='screen' "
                              "ORDER BY capture_ts DESC LIMIT ?", (min(limit, 1000),)).fetchall()
        days = [r[0] for r in db.execute("SELECT DISTINCT date(capture_ts,'unixepoch','localtime') d FROM batch_annotations "
                                          "WHERE source_type='screen' ORDER BY d DESC")]
    finally:
        db.close()
    return {"days": days, "items": [dict(r) for r in rows]}


@app.get("/api/screens/{sid}.jpg")
def screen_image(sid: int):
    db, root = _screens_db()
    try:
        r = db.execute("SELECT source_path FROM batch_annotations WHERE id=? AND source_type='screen'", (sid,)).fetchone()
    finally:
        db.close()
    if not r:
        raise HTTPException(404)
    p = Path(r[0]).resolve()
    if root.resolve() not in p.parents or not p.exists():
        raise HTTPException(404, "screen pruned or outside screens dir")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/export")
def export(format: str = "jsonl", since: float = 0, until: float = 0, modality: str = "", limit: int = 1_000_000):
    """Everything in one table for ingestion: JSONL (default) or Parquet."""
    rows = store.merged(since, until or None, _mods(modality), None, limit)
    rows.reverse()
    for r in rows:   # HOU-68: screen captures/captions are local-only, never exported
        d = r.get("data")
        if isinstance(d, dict) and "screen" in d:
            r["data"] = {k: v for k, v in d.items() if k != "screen"}
    if format == "parquet":
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq
        flat = [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in rows]
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(flat), buf)
        return Response(buf.getvalue(), media_type="application/vnd.apache.parquet",
                        headers={"Content-Disposition": "attachment; filename=vlab-export.parquet"})
    body = "\n".join(json.dumps(r) for r in rows) + "\n"
    return Response(body, media_type="application/x-ndjson",
                    headers={"Content-Disposition": "attachment; filename=vlab-export.jsonl"})


@app.get("/api/context")
def context(q: str = ""):
    """The exact vision summary the voice brain would get for this question."""
    t0 = time.time()
    text = multimodal.vision_context(store, engine, q)
    return {"ms": round((time.time() - t0) * 1000), "chars": len(text), "text": text}


@app.post("/api/summary")
def summary(body: dict = Body(default={})):
    since = day_start("today")
    evs = list(reversed(store.events(since, 400)))
    lines = []
    for e in evs:
        line = f"{dt.datetime.fromtimestamp(e['ts']):%H:%M:%S} {e['text']}"
        if e["vlm_text"]:
            line += f" — {e['vlm_text']}"
        lines.append(line)
    tl = timeline("today")
    presence = ", ".join(f"{l['cls']}: {l['seconds'] / 60:.0f} min in view" for l in tl["lanes"])
    question = body.get("question") or (
        "Summarize the day in 3-4 plain sentences: who and what was around, for how long, "
        "and anything notable. Use times.")
    prompt = (f"Activity log from a home camera for today (local time now "
              f"{dt.datetime.now():%H:%M}).\nTime in view: {presence or 'none'}\n"
              f"Events:\n" + ("\n".join(lines) or "(no events)") + f"\n\nQuestion: {question}\n"
              "Answer only from the log. If the log doesn't say, say so.")
    model = engine.settings["summary_model"]
    try:
        text = llm.chat(model, prompt, timeout=300)
    except Exception as e:
        raise HTTPException(502, f"{model}: {e}")
    return {"text": text, "model": model, "events": len(lines)}


# ---------- model lab ----------
def model_status(entry):
    k = entry["kind"]
    if k in ("yolo", "sam"):
        return engine.models.weights_present(entry["key"])
    if k == "depth":
        return engine.models.hf_present(DEPTH_MODEL)
    if k == "siglip":
        return engine.models.hf_present(llm.Embedder.MODEL)
    return True


@app.get("/api/models")
def models():
    rows = [{**c, "installed": model_status(c), "hw": "CPU" if c["kind"] == "builtin" else "GPU",
             "bench": bench_results.get(c["key"])} for c in CATALOG]
    names = [m["name"] for m in llm.list_models()]
    vlms = [n for n in names if "vision" in llm.capabilities(n)]
    for n in vlms + ([SUGGESTED_VLM] if SUGGESTED_VLM not in names else []):
        rows.append({"key": n, "task": "Describe", "name": n, "kind": "ollama", "hw": "GPU",
                     "gives": "Sentences about the scene" if n in vlms else "Newer vision model (Ollama)",
                     "installed": n in names, "pulling": n in engine.pulling,
                     "bench": bench_results.get(n)})
    return rows


@app.get("/api/memory")
def memory():
    vm = psutil.virtual_memory()
    proc = psutil.Process().memory_info().rss
    try:
        import torch
        gpu = torch.mps.driver_allocated_memory()
    except Exception:
        gpu = 0
    ollama = [{"name": m["name"], "bytes": m.get("size", 0)} for m in llm.loaded_models()]
    return {"total": vm.total, "used": vm.total - vm.available, "lab_process": proc,
            "lab_gpu": gpu, "ollama": ollama}


def _bench_yolo(key, frame, n=20):
    with engine.models.lock:
        m = engine.models.yolo(key, aux=True)
        if "world" in key:
            m.set_classes(["person", "cup", "laptop"])
        kw = {"device": engine.models.device, "verbose": False}
        if key.startswith("sam"):
            h, w = frame.shape[:2]
            kw.update(points=[[w / 2, h / 2]], labels=[1])
            n = 5
        else:
            kw["imgsz"] = 640
        for _ in range(3):
            m(frame, **kw)
        t0 = time.perf_counter()
        for _ in range(n):
            m(frame, **kw)
        return n / (time.perf_counter() - t0)


@app.post("/api/benchmark")
def benchmark(body: dict = Body(...)):
    key, frame = body["key"], current_frame()
    entry = next((c for c in CATALOG if c["key"] == key), {"kind": "ollama"})
    try:
        if entry["kind"] in ("yolo", "sam"):
            fps = _bench_yolo(key, frame)
            res = {"fps": round(fps, 1), "label": f"{fps:.0f} fps"}
        elif entry["kind"] == "depth":
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(cv2.resize(frame, (518, 292)), cv2.COLOR_BGR2RGB))
            with engine.models.lock:
                p = engine.models.depth()
                p(img)
                t0 = time.perf_counter()
                for _ in range(10):
                    p(img)
            fps = 10 / (time.perf_counter() - t0)
            res = {"fps": round(fps, 1), "label": f"{fps:.0f} fps"}
        elif entry["kind"] == "siglip":
            engine.embedder.image(frame)
            t0 = time.perf_counter()
            for _ in range(10):
                engine.embedder.image(frame)
            fps = 10 / (time.perf_counter() - t0)
            res = {"fps": round(fps, 1), "label": f"{fps:.0f} img/s"}
        elif entry["kind"] == "builtin":
            res = {"fps": None, "label": "< 1 ms"}
        else:
            llm.chat(key, "Say OK.", timeout=600)  # load into memory first
            t0 = time.time()
            llm.describe_scene(key, frame)
            secs = time.time() - t0
            res = {"fps": round(1 / secs, 2), "label": f"{secs:.1f} s"}
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    bench_results[key] = res
    return res


@app.post("/api/download")
def download(body: dict = Body(...)):
    key = body["key"]
    entry = next((c for c in CATALOG if c["key"] == key), {"kind": "ollama"})
    if entry["kind"] == "ollama":
        def run():
            engine.pulling.add(key)
            try:
                llm.pull(key)
            finally:
                engine.pulling.discard(key)
        threading.Thread(target=run, daemon=True).start()
        return {"started": True}
    try:
        if entry["kind"] in ("yolo", "sam"):
            engine.models.yolo(key, aux=True)
        elif entry["kind"] == "depth":
            engine.models.depth()
        elif entry["kind"] == "siglip":
            engine.embedder.error = None
            engine.embedder.text("test")
            if engine.embedder.error:
                raise RuntimeError(engine.embedder.error)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    return {"installed": True}


@app.post("/api/compare")
def compare(body: dict = Body(...)):
    frame = current_frame()
    classes = [c.strip() for c in body.get("classes", "").split(",") if c.strip()]
    out = []
    for key in (body["a"], body["b"]):
        try:
            with engine.models.lock:
                m = engine.models.yolo(key, aux=True)
                if "world" in key:
                    m.set_classes(classes or ["person"])
                m(frame, device=engine.models.device, imgsz=640, verbose=False)  # warm-up
                t0 = time.perf_counter()
                r = m(frame, device=engine.models.device, imgsz=640, conf=0.3, verbose=False)[0]
                ms = (time.perf_counter() - t0) * 1000
            objs = [{"cls": r.names[int(c)], "conf": round(float(cf), 2),
                     "box": [round(float(v), 4) for v in b], "color": color_for(r.names[int(c)])}
                    for c, cf, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxyn)]
            out.append({"key": key, "ms": round(ms, 1), "objects": objs})
        except Exception as e:
            out.append({"key": key, "error": f"{type(e).__name__}: {e}", "objects": []})
    engine.paused_frame = frame.copy()
    return {"results": out}
