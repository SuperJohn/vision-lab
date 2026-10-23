"""Batch annotation of stored frames (separate process from the live loop).

Processes, oldest first, idempotently (UNIQUE source_type+source_path):
  screen    data/screens/YYYY-MM-DD/*.jpg   (only frames kept after dHash dedup)
  keyframe  keyframes table / data/keyframes/*.jpg
  recording data/recordings/*.mp4 sampled at 1 frame / 10 s   (off by default: "recordings": true)
Each item: one gemma3:27b caption via Ollama (one request at a time) + optional YOLO11s-seg detections.

Runs only when HID idle > idle_min_s OR inside the nightly window, and backs off when Vision Lab's
live caption cadence slows or Ollama is serving a live request.

  python -m vision_lab.batch loop      # every 15 min (screen session 'vlabbatch', taskpolicy -b + nice)
  python -m vision_lab.batch run [--force] [--limit N]
  python -m vision_lab.batch stats     # backlog, throughput, latency median/p90, capture->result delay
"""
import base64, json, os, statistics, sqlite3, subprocess, sys, time, urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = os.environ.get("VLAB_DB") or str(ROOT / "data" / "vlab.db")
DATA = Path(DB_PATH).parent
CFG_PATH = os.environ.get("BATCH_CONFIG") or str(DATA / "batch.json")
DEFAULTS = {
    "every_s": 900, "batch_size": 20, "max_run_s": 600,
    "idle_min_s": 600, "night_window": ["01:00", "06:00"],
    "model": os.environ.get("VLAB_MODEL", "gemma4:26b"), "ollama_url": "http://127.0.0.1:11434",
    "detect": True, "detect_model": "weights/yolo11s-seg.pt", "detect_imgsz": 960,
    "recordings": False, "recording_every_s": 10,
    "order": ["screen", "keyframe", "recording"],
    "backoff_caption_gap_factor": 2.0,   # back off if live scene-caption gap > factor x its 1-hour median
}
DDL = """CREATE TABLE IF NOT EXISTS batch_annotations (
  id INTEGER PRIMARY KEY, source_type TEXT, source_path TEXT, keyframe_id INTEGER, capture_ts REAL,
  processed_ts REAL, created_at REAL, model TEXT, caption TEXT, label TEXT, detections TEXT,
  gemma_ms INTEGER, detect_ms INTEGER, status TEXT, error TEXT, UNIQUE(source_type, source_path));
CREATE INDEX IF NOT EXISTS batch_ann_capture ON batch_annotations(source_type, capture_ts);"""
PROMPTS = {
    "screen": "No preamble. Reply exactly as '<label> - <sentence>'. Screenshot of the user's Mac. Start with one label from: work, email_admin, chat, browsing, video, social, "
              "game, idle_screen. Then ' - ' and one sentence on what is on screen (apps, document/site, task).",
    "keyframe": "No preamble. Reply exactly as '<label> - <sentence>'. Home camera frame (camera sits on the desk, looking at the room: couch left, kitchen right, front "
                "door center). Start with one label from: at_desk, couch, kitchen, door, away, dog_only, empty. Then ' - ' "
                "and one sentence: who is where, doing what.",
}
PROMPTS["recording"] = PROMPTS["keyframe"]


LABELS = {"screen": ["work", "email_admin", "chat", "browsing", "video", "social", "game", "idle_screen"],
          "keyframe": ["at_desk", "couch", "kitchen", "door", "away", "dog_only", "empty"]}
LABELS["recording"] = LABELS["keyframe"]


def find_label(text, kind):
    import re
    m = re.search(r"\b(" + "|".join(LABELS[kind]) + r")\b", (text or "").lower())
    return m.group(1) if m else None


def cfg_load():
    c = dict(DEFAULTS)
    if os.path.exists(CFG_PATH):
        c.update(json.load(open(CFG_PATH)))
    else:
        json.dump(DEFAULTS, open(CFG_PATH, "w"), indent=1)
    return c


def connect():
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript(DDL)
    have = {r[1] for r in db.execute("PRAGMA table_info(batch_annotations)")}
    for col, typ in (("sensitive", "INTEGER NOT NULL DEFAULT 0"), ("exportable", "INTEGER NOT NULL DEFAULT 1")):
        if col not in have:   # screens are sensitive, local-only, never exported
            db.execute(f"ALTER TABLE batch_annotations ADD COLUMN {col} {typ}")
    db.execute("UPDATE batch_annotations SET sensitive=1, exportable=0 WHERE source_type='screen' AND exportable!=0")
    db.commit()
    return db


def ingest_tags(kind):
    """tags applied at ingest. Screens can show anything (mail, banking, chats): local-only."""
    return {"sensitive": 1, "exportable": 0} if kind == "screen" else {"sensitive": 0, "exportable": 1}


def caption_model(db, c):
    """Vision Lab's configured caption model (engine settings.describe_model), else batch.json model."""
    try:
        row = db.execute("SELECT value FROM config WHERE key='settings'").fetchone()
        return (json.loads(row[0]) if row else {}).get("describe_model") or c["model"]
    except Exception:
        return c["model"]


def hid_idle():
    out = subprocess.run(["ioreg", "-c", "IOHIDSystem"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if '"HIDIdleTime"' in line:
            return int(line.split("=")[-1]) / 1e9
    return 0


def in_window(c, now=None):
    hm = datetime.fromtimestamp(now or time.time()).strftime("%H:%M")
    a, b = c["night_window"]
    return (a <= hm < b) if a <= b else (hm >= a or hm < b)


def live_busy(db, c):
    """(busy, reason). Live caption cadence from Vision Lab 'scene' events + Ollama in-flight check."""
    ts = [r[0] for r in db.execute("SELECT ts FROM events WHERE kind='scene' AND ts > ? ORDER BY ts", (time.time() - 3600,))]
    if len(ts) >= 8:
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        base, recent = statistics.median(gaps[:-4] or gaps), statistics.median(gaps[-4:])
        since_last = time.time() - ts[-1]
        if recent > c["backoff_caption_gap_factor"] * base:
            return True, f"live caption gap {recent:.0f}s vs normal {base:.0f}s"
        if since_last > max(60, 4 * base) and since_last < 600:
            return True, f"live caption overdue ({since_last:.0f}s)"
    try:   # Ollama exposes no queue; requests serialize per model, so cadence above is the busy signal
        urllib.request.urlopen(c["ollama_url"] + "/api/ps", timeout=3).read()
    except Exception as e:
        return True, f"ollama unreachable: {e}"
    return False, ""


def backlog(db, c):
    done = {(r[0], r[1]) for r in db.execute("SELECT source_type, source_path FROM batch_annotations")}
    items = []
    for f in sorted((DATA / "screens").glob("20??-??-??/*.jpg")):
        if ("screen", str(f)) not in done:
            d, t = f.parent.name, f.stem
            items.append(("screen", str(f), None, datetime.strptime(d + t, "%Y-%m-%d%H%M%S").timestamp()))
    for r in db.execute("SELECT id, ts, path FROM keyframes ORDER BY ts"):
        if ("keyframe", r["path"]) not in done and os.path.exists(r["path"]):
            items.append(("keyframe", r["path"], r["id"], r["ts"]))
    if c["recordings"]:
        import cv2
        for f in sorted((DATA / "recordings").glob("*.mp4")):
            cap = cv2.VideoCapture(str(f)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            t0 = f.stat().st_mtime - n / fps
            for k in range(0, int(n / fps), c["recording_every_s"]):
                key = f"{f}#t={k}"
                if ("recording", key) not in done:
                    items.append(("recording", key, None, t0 + k))
    rank = {s: i for i, s in enumerate(c["order"])}
    items.sort(key=lambda x: (rank.get(x[0], 9), x[3]))
    return items


def load_image(kind, path):
    import cv2
    if kind == "recording":
        f, t = path.split("#t=")
        cap = cv2.VideoCapture(f); cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, img = cap.read(); cap.release()
        return img if ok else None
    return cv2.imread(path)


_det = None


def detect(c, img):
    global _det
    if _det is None:
        from ultralytics import YOLO
        _det = YOLO(str(ROOT / c["detect_model"]))
    r = _det.predict(img, imgsz=c["detect_imgsz"], conf=0.3, verbose=False)[0]
    return [{"cls": r.names[int(k)], "conf": round(p, 2), "box": [round(v, 3) for v in b]}
            for b, k, p in zip(r.boxes.xyxyn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]


def gemma(c, img, prompt, model=None):
    import cv2
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    body = {"model": model or c["model"], "stream": False, "think": False, "keep_alive": "30m", "options": {"num_predict": 80, "temperature": 0},
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(jpg.tobytes()).decode()]}]}
    req = urllib.request.Request(c["ollama_url"] + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())["message"]["content"].strip()


def run(force=False, limit=None):
    c, db = cfg_load(), connect()
    idle = hid_idle()
    if not force and idle < c["idle_min_s"] and not in_window(c):
        return print(f"skip: user active (idle {idle:.0f}s) and outside night window {c['night_window']}")
    items = backlog(db, c)
    print(f"backlog {len(items)} items", flush=True)
    t_start, n = time.time(), 0
    for kind, path, kid, cap_ts in items[: limit or c["batch_size"]]:
        if time.time() - t_start > c["max_run_s"]:
            break
        busy, why = live_busy(db, c)
        if busy and not force:
            print("back off:", why, flush=True); break
        if not force and not in_window(c) and hid_idle() < c["idle_min_s"]:
            print("back off: user came back", flush=True); break
        model = caption_model(db, c)
        rec = dict(source_type=kind, source_path=path, keyframe_id=kid, capture_ts=cap_ts, model=model, **ingest_tags(kind))
        try:
            img = load_image(kind, path)
            if img is None:
                raise RuntimeError("unreadable image")
            dets, dms = None, None
            if c["detect"]:
                t = time.time(); dets = detect(c, img); dms = int((time.time() - t) * 1000)
            t = time.time(); cap = gemma(c, img, PROMPTS[kind], model); gms = int((time.time() - t) * 1000)
            rec.update(caption=cap, label=find_label(cap, kind), detections=json.dumps(dets),
                       gemma_ms=gms, detect_ms=dms, status="ok")
        except Exception as e:
            rec.update(status="error", error=str(e)[:300])
        now = time.time(); rec.update(processed_ts=now, created_at=now)
        db.execute(f"INSERT OR IGNORE INTO batch_annotations ({','.join(rec)}) VALUES ({','.join('?' * len(rec))})", tuple(rec.values()))
        db.commit(); n += 1
        print(kind, os.path.basename(path), rec.get("gemma_ms"), "ms", (rec.get("caption") or rec.get("error"))[:90], flush=True)
    print(f"done {n} items in {time.time() - t_start:.0f}s", flush=True)


_fast_busy_until = 0.0
_fast_backoff = 10


def fast_screen(path, cap_ts, db=None, force=False):
    """Fast lane: caption one fresh screen right away (no YOLO), one request at a time.
    Returns {"status": ok|error|backoff|skip, ...}. Never queues: if Vision Lab's live captioner looks
    busy we back off (exponentially, up to 5 min) and the nightly/idle batch run picks the frame up later."""
    global _fast_busy_until, _fast_backoff
    c = cfg_load()
    own = db is None
    db = db or connect()
    try:
        if db.execute("SELECT 1 FROM batch_annotations WHERE source_type='screen' AND source_path=?", (path,)).fetchone():
            return {"status": "skip", "why": "already annotated"}
        if not force and time.time() < _fast_busy_until:
            return {"status": "backoff", "why": f"cooling down {(_fast_busy_until - time.time()):.0f}s"}
        busy, why = live_busy(db, c)
        if busy and not force:
            _fast_backoff = min(300, _fast_backoff * 2); _fast_busy_until = time.time() + _fast_backoff
            return {"status": "backoff", "why": why}
        import cv2
        model = caption_model(db, c)
        rec = dict(source_type="screen", source_path=path, capture_ts=cap_ts, model=model, **ingest_tags("screen"))
        try:
            img = cv2.imread(path)
            if img is None:
                raise RuntimeError("unreadable image")
            t = time.time(); cap = gemma(c, img, PROMPTS["screen"], model); gms = int((time.time() - t) * 1000)
            rec.update(caption=cap, label=find_label(cap, "screen"), gemma_ms=gms, status="ok")
            _fast_backoff = 10
            # self-throttle: keep the GPU duty cycle for screens <= ~35% even if each caption runs long
            _fast_busy_until = time.time() + max(0, 2 * gms / 1000 - 10)
        except Exception as e:
            rec.update(status="error", error=str(e)[:300])
        now = time.time(); rec.update(processed_ts=now, created_at=now)
        db.execute(f"INSERT OR IGNORE INTO batch_annotations ({','.join(rec)}) VALUES ({','.join('?' * len(rec))})", tuple(rec.values()))
        db.commit()
        return {"status": rec["status"], "caption": rec.get("caption"), "error": rec.get("error"), "ms": rec.get("gemma_ms")}
    finally:
        if own:
            db.close()


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None


def stats():
    c, db = cfg_load(), connect()
    items = backlog(db, c)
    by = {}
    for k, *_ in items:
        by[k] = by.get(k, 0) + 1
    print("backlog:", by or "empty")
    for k in ("screen", "keyframe", "recording"):
        rows = db.execute("SELECT gemma_ms, detect_ms, capture_ts, processed_ts FROM batch_annotations WHERE source_type=? AND status='ok'", (k,)).fetchall()
        if not rows:
            continue
        g = [r[0] for r in rows]; d = [r[1] for r in rows if r[1] is not None]
        delay = [(r[3] - r[2]) / 3600 for r in rows if r[2]]
        print(f"{k:9} done={len(rows)} gemma median={statistics.median(g)/1000:.1f}s p90={pct(g,.9)/1000:.1f}s"
              + (f" detect median={statistics.median(d)}ms" if d else "")
              + (f" capture->result median={statistics.median(delay):.1f}h p90={pct(delay,.9):.1f}h" if delay else ""))
    err = db.execute("SELECT COUNT(*) FROM batch_annotations WHERE status='error'").fetchone()[0]
    print("errors:", err, "| last run:", db.execute("SELECT datetime(MAX(processed_ts),'unixepoch','localtime') FROM batch_annotations").fetchone()[0])


def loop():
    os.nice(15)
    while True:
        try:
            run()
        except Exception as e:
            print("error:", repr(e), flush=True)
        time.sleep(cfg_load()["every_s"])


if __name__ == "__main__":
    a = sys.argv[1:] or ["stats"]
    if a[0] == "run":
        run(force="--force" in a, limit=int(a[a.index("--limit") + 1]) if "--limit" in a else None)
    elif a[0] == "screen":   # python -m vision_lab.batch screen <path> [--force]
        p = a[1]; st = Path(p).stem; d = Path(p).parent.name
        print(json.dumps(fast_screen(p, datetime.strptime(d + st, "%Y-%m-%d%H%M%S").timestamp(), force="--force" in a), indent=1))
    elif a[0] == "loop":
        loop()
    else:
        stats()
