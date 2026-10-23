"""Planned-vs-actual feedback loop, v1 (rules only, no ML).

One process, two cadences:
  * every 30 s  -> desktop sample (frontmost app/window + HID idle) and a camera probe
                   (person/dog boxes from Vision Lab's /api/frame.jpg; never opens the camera)
  * every 5 min -> classify the slice, match it to the plan block, tag voice check-ins,
                   maybe speak one drift nudge (Kokoro + afplay).

Everything lands in vlab.db (or $VLAB_DB):
  events(modality='desktop', kind='desktop_sample' | modality='vision', kind='frame_probe' |
         modality='activity', kind='activity_slice' | modality='voice', kind='checkin' |
         modality='activity', kind='nudge')
  plan_blocks / plan_changes (also written by the Week Planner on every save).

  python -m vision_lab.planloop run            # the daemon (runs in screen session 'planloop')
  python -m vision_lab.planloop status         # last slice, last sample, nudges today
  python -m vision_lab.planloop once [--dry-run] [--at EPOCH]   # one classify pass
  python -m vision_lab.planloop seed-plan PLAN.json             # load planner blocks
"""
import json, os, re, sqlite3, subprocess, sys, tempfile, time, urllib.request
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = os.environ.get("VLAB_DB") or str(ROOT / "data" / "vlab.db")
CONFIG_PATH = os.environ.get("PLANLOOP_CONFIG") or str(ROOT / "data" / "planloop.json")
VLAB_URL = os.environ.get("VLAB_URL", "http://127.0.0.1:8765")

DEFAULTS = {
    "sample_every_s": 30,
    "slice_every_s": 300,
    "quiet_hours": ["22:00", "08:00"],
    "kid_windows": [],              # e.g. [{"days": ["Sat","Sun"], "start": "09:00", "end": "17:00"}]
    "kid_keywords": [],             # plan blocks whose task mentions these are excluded from nudges
    "max_nudges_per_hour": 3,
    "drift_slices": 2,              # consecutive off-plan slices before a nudge (~10 min)
    "nudge_cooldown_s": 600,
    "dry_run": False,               # true = log the nudge but don't speak
    "voice": "af_heart", "speed": 1.0,
    "idle_active_s": 60,            # HID idle below this counts as 'at the keyboard'
    "desk_box_area": 0.12,          # person box covering >= this share of frame = at the desk (camera is at the desk)
    "camera_probe": True,
    "screen_capture": True,         # full-screen JPEG every screen_every_s into data/screens/YYYY-MM-DD/
    "screenrec": False,             # continuous full-screen video in screenrec_window (vision_lab/screenrec.py)
    "screenrec_dir": None,          # recorder output dir; None = data/screenrec (or $VLAB_SCREENREC_DIR)
    "screenrec_window": None,       # "YYYY-MM-DDTHH:MM/YYYY-MM-DDTHH:MM:SS" PT; None = screenrec.DEFAULT_WINDOW
    "screen_every_s": 10,           # screen capture cadence (desktop/camera samples stay on sample_every_s)
    "screen_fast_caption": True,    # caption each new (non-duplicate) screen via vision_lab.batch fast lane
    "screen_dedup_bits": 4,         # skip a frame whose 64-bit dHash is within this many bits of the last kept one
    "screen_width": 1280, "screen_quality": 60, "screen_keep_days": None,   # screens are kept forever
    "screen_caption": True,         # one gemma3 caption of the latest screen per 5-min slice
    "caption_model": os.environ.get("VLAB_MODEL", "gemma4:26b"), "ollama_url": "http://127.0.0.1:11434",
    "probe_model": "weights/yolo11n.pt",
    "work_apps": ["Code", "Cursor", "Terminal", "iTerm2", "Xcode", "GitHub Desktop", "Pages", "Numbers", "Microsoft Word",
                  "Microsoft Excel", "Preview", "Mail", "Notes", "Linear", "Finder", "Calendar"],
    "call_apps": ["zoom.us", "FaceTime", "Microsoft Teams", "Webex", "Discord", "Slack Huddle", "Phone"],
    "distraction_title_words": ["youtube", "netflix", "reddit", "twitter", " x.com", "instagram", "facebook", "tiktok",
                                "hulu", "twitch", "prime video"],
    # PROVISIONAL zones (normalized x,y; NOT saved in Vision Lab). Replace once real zones are saved via /api/zones.
    "zones_provisional": True,
    "zones": {
        "couch": [[0, .55], [.33, .55], [.33, 1], [0, 1]],
        "island": [[.48, .55], [.67, .55], [.67, .7], [.48, .7]],
        "front door": [[.46, .33], [.53, .33], [.53, .6], [.46, .6]],
        "kitchen": [[.74, .5], [.97, .5], [.97, .84], [.74, .84]],
        "dog bowls": [[.87, .84], [1, .84], [1, .96], [.87, .96]],
    },
    "zone_activity": {"couch": "couch", "island": "kitchen", "kitchen": "kitchen", "dog bowls": "kitchen",
                      "front door": "away"},
    # which guesses count as on-plan for a block's zone
    "zone_accepts": {"desk": ["at_desk_work", "on_call", "talking_to_assistant"], "kitchen": ["kitchen"],
                     "couch": ["couch"], "out": ["away"], "phone": ["on_call", "at_desk_work", "talking_to_assistant", "away"],
                     "any": ["at_desk_work", "at_desk_other", "couch", "kitchen", "away", "on_call", "talking_to_assistant"]},
}

PLAN_DDL = """
CREATE TABLE IF NOT EXISTS plan_blocks (
  id TEXT PRIMARY KEY, day TEXT, task TEXT, task_id TEXT, task_rank INTEGER, linear_issue TEXT, kind TEXT,
  planned_start REAL, planned_end REAL, start_local TEXT, end_local TEXT, budget_min INTEGER,
  zone TEXT DEFAULT 'desk', status TEXT DEFAULT 'planned', source TEXT, tags TEXT, updated_at REAL);
CREATE INDEX IF NOT EXISTS plan_blocks_day ON plan_blocks(day, planned_start);
CREATE TABLE IF NOT EXISTS plan_changes (
  id INTEGER PRIMARY KEY, ts REAL, block_id TEXT, change TEXT, before TEXT, after TEXT, source TEXT);
CREATE INDEX IF NOT EXISTS plan_changes_block ON plan_changes(block_id, ts);
"""
# Same additive migration as store.py (af13ee9) minus its presence cleanup, so it is safe while Vision Lab runs.
EVENT_COLS = [("modality", "TEXT DEFAULT 'vision'"), ("written_at", "REAL"), ("data", "TEXT"), ("source_key", "TEXT")]
UTT_DDL = """CREATE TABLE IF NOT EXISTS utterances (
  id INTEGER PRIMARY KEY, turn_id TEXT, role TEXT, text TEXT, ts_start REAL, ts_end REAL,
  voice TEXT, model TEXT, latencies TEXT, interrupted INTEGER DEFAULT 0, error TEXT,
  keyframe_id INTEGER, snapshot TEXT, written_at REAL, source_key TEXT UNIQUE);
CREATE INDEX IF NOT EXISTS utterances_ts ON utterances(ts_start);"""


def load_config():
    cfg = json.loads(json.dumps(DEFAULTS))
    if os.path.exists(CONFIG_PATH):
        user = json.load(open(CONFIG_PATH))
        for suffix in ("_windows", "_keywords"):  # backward compat: legacy <name>_windows/_keywords keys
            new = "kid" + suffix
            legacy = [k for k in user if k.endswith(suffix) and k != new and k not in DEFAULTS]
            if new not in user and legacy:
                user[new] = user[legacy[0]]
            for k in legacy:
                user.pop(k)
        cfg.update(user)
    else:
        json.dump(DEFAULTS, open(CONFIG_PATH, "w"), indent=1)
    return cfg


def connect(path=None):
    db = sqlite3.connect(path or DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    ensure_schema(db)
    return db


def ensure_schema(db):
    have = {r[1] for r in db.execute("PRAGMA table_info(events)")}
    if not have:
        raise SystemExit("events table missing - is this a Vision Lab database?")
    for col, decl in EVENT_COLS:
        if col not in have:
            db.execute(f"ALTER TABLE events ADD COLUMN {col} {decl}")
    db.execute("CREATE INDEX IF NOT EXISTS events_modality_ts ON events(modality, ts)")
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS events_source_key ON events(source_key)")
    db.executescript(UTT_DDL + PLAN_DDL)
    db.commit()


def add_event(db, kind, modality, text, data, ts=None, cls=None, color="#8E7CC3"):
    now = time.time()
    cur = db.execute("INSERT INTO events (ts, kind, cls, text, color, modality, written_at, data) VALUES (?,?,?,?,?,?,?,?)",
                     (ts or now, kind, cls, text, color, modality, now, json.dumps(data)))
    db.commit()
    return cur.lastrowid


# ---------------- 30 s sampling ----------------
FRONT_SCRIPT = '''tell application "System Events"
  set p to first application process whose frontmost is true
  set n to name of p
  set w to ""
  try
    set w to name of front window of p
  end try
end tell
return n & tab & w'''


def desktop_sample():
    out = {"app": None, "title": None, "idle_s": None, "error": None}
    try:
        r = subprocess.run(["osascript", "-e", FRONT_SCRIPT], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            app, _, title = r.stdout.rstrip("\n").partition("\t")
            out["app"], out["title"] = app or None, title or None
        else:
            out["error"] = r.stderr.strip()[:300]
    except Exception as e:
        out["error"] = str(e)[:300]
    try:
        r = subprocess.run(["ioreg", "-c", "IOHIDSystem"], capture_output=True, text=True, timeout=10)
        m = re.search(r'"HIDIdleTime" = (\d+)', r.stdout)
        if m:
            out["idle_s"] = round(int(m.group(1)) / 1e9, 1)
    except Exception as e:
        out["error"] = (out["error"] or "") + " ioreg: " + str(e)[:100]
    return out


_yolo = None


def frame_probe(cfg):
    """Person/dog boxes on Vision Lab's latest frame. Reads /api/frame.jpg only; never touches the camera."""
    global _yolo
    try:
        raw = urllib.request.urlopen(VLAB_URL + "/api/frame.jpg", timeout=5).read()
    except Exception as e:
        return {"ok": False, "error": f"frame: {e}"[:200]}
    try:
        import cv2, numpy as np
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"ok": False, "error": "frame not decodable (camera paused?)"}
        if _yolo is None:
            from ultralytics import YOLO
            _yolo = YOLO(str(ROOT / cfg["probe_model"]))
        h, w = img.shape[:2]
        res = _yolo.predict(img, classes=[0, 15, 16], conf=0.35, verbose=False, device="cpu")[0]
        boxes = []
        for b, c, p in zip(res.boxes.xyxyn.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            cls = "person" if int(c) == 0 else "dog"   # cat relabels to dog in this home, same as Vision Lab
            x1, y1, x2, y2 = b
            foot = ((x1 + x2) / 2, y2)
            boxes.append({"cls": cls, "conf": round(p, 2), "box": [round(v, 3) for v in b],
                          "area": round((x2 - x1) * (y2 - y1), 3), "zones": zones_for(foot, cfg)})
        return {"ok": True, "w": w, "h": h, "boxes": boxes}
    except Exception as e:
        return {"ok": False, "error": f"probe: {e}"[:200]}


def _inside(pt, poly):
    x, y = pt
    n, inside = len(poly), False
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-9) + x1:
            inside = not inside
    return inside


def zones_for(pt, cfg):
    return [name for name, poly in cfg["zones"].items() if _inside(pt, poly)]


SCREEN_DIR = Path(DB_PATH).parent / "screens"


def screen_capture(cfg, ts):
    """screencapture -x (silent). Needs Screen Recording permission for the app that runs this process."""
    import cv2
    day = SCREEN_DIR / datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    import tempfile   # screencapture itself can't write under ~/Documents (TCC), so grab to $TMPDIR then re-encode
    raw = Path(tempfile.gettempdir()) / f"planloop-{int(ts)}.raw.jpg"
    r = subprocess.run(["screencapture", "-x", "-t", "jpg", str(raw)], capture_output=True, text=True, timeout=20)
    if r.returncode != 0 or not raw.exists():
        return {"ok": False, "error": (r.stderr or "screencapture failed").strip()[:200]}
    img = cv2.imread(str(raw)); raw.unlink()
    if img is None:
        return {"ok": False, "error": "unreadable capture"}
    global _last_hash
    hsh = dhash(img)
    if _last_hash is not None and bin(hsh ^ _last_hash[0]).count("1") <= cfg["screen_dedup_bits"]:
        return {"ok": True, "dup_of": _last_hash[1], "path": None, "dhash": f"{hsh:016x}"}
    h, w = img.shape[:2]
    if w > cfg["screen_width"]:
        img = cv2.resize(img, (cfg["screen_width"], int(h * cfg["screen_width"] / w)), interpolation=cv2.INTER_AREA)
    out = day / (datetime.fromtimestamp(ts).strftime("%H%M%S") + ".jpg")
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, cfg["screen_quality"]])
    _last_hash = (hsh, str(out))
    return {"ok": True, "path": str(out), "dhash": f"{hsh:016x}", "bytes": out.stat().st_size, "src_w": w, "src_h": h}


_last_hash = None


def dhash(img):
    import cv2
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def prune_screens(cfg):
    """Retention rule: screen captures are kept forever. Pruning only runs if it is explicitly
    re-enabled with screen_prune: true AND screen_keep_days > 0; an old screen_keep_days alone does nothing."""
    import shutil
    days = cfg.get("screen_keep_days")
    if cfg.get("screen_prune") is not True or not days or days <= 0:
        return 0
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    for d in SCREEN_DIR.glob("20??-??-??"):
        if d.name < cutoff:
            shutil.rmtree(d, ignore_errors=True)
    return 1


def caption_screen(cfg, path):
    import base64
    body = {"model": cfg["caption_model"], "stream": False, "think": False, "keep_alive": "30m",
            "options": {"num_predict": 60, "temperature": 0},
            "messages": [{"role": "user", "images": [base64.b64encode(open(path, "rb").read()).decode()],
                          "content": "No preamble. Reply exactly as '<label> - <description>'. This is a screenshot of the user's Mac. Reply with one label from: work, email_admin, "
                                     "chat, browsing, video, social, game, idle_screen, then ' - ' and a short description."}]}
    req = urllib.request.Request(cfg["ollama_url"] + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["message"]["content"].strip()


_last_screen = None
_cap_thread = None


def _fast_caption(path, ts):
    try:
        from vision_lab import batch
        r = batch.fast_screen(path, ts)
        if r["status"] in ("ok", "error"):
            print("screen caption:", os.path.basename(path), r.get("ms"), "ms", (r.get("caption") or r.get("error") or "")[:90], flush=True)
    except Exception as e:
        print("screen caption error:", repr(e), flush=True)


def capture_paused(db_path=None):
    """Vision Lab's Privacy mode flag (config 'settings'.capture_paused), read fresh each tick."""
    try:
        con = sqlite3.connect(db_path or DB_PATH, timeout=5)
        try:
            row = con.execute("SELECT value FROM config WHERE key='settings'").fetchone()
        finally:
            con.close()
        return bool(row and json.loads(row[0]).get("capture_paused"))
    except Exception:
        return False


def screen_tick(cfg):
    """Capture one screen (dedup + quiet hours), and hand a new frame to the caption fast lane.
    At most one caption in flight; a frame arriving while one runs is left for the batch backlog."""
    global _last_screen, _cap_thread
    now = time.time()
    if not cfg.get("screen_capture") or is_quiet(cfg, now):
        return None
    if capture_paused():
        _last_screen = {"ok": False, "paused": True, "error": "capture paused"}
        return None
    try:
        s = screen_capture(cfg, now)
    except Exception as e:
        s = {"ok": False, "error": str(e)[:200]}
    s.update(sensitive=True, exportable=False)   # local-only
    _last_screen = s
    if s.get("path") and cfg.get("screen_fast_caption", True) and cfg.get("screen_caption", True):
        if _cap_thread is None or not _cap_thread.is_alive():
            import threading
            _cap_thread = threading.Thread(target=_fast_caption, args=(s["path"], now), daemon=True)
            _cap_thread.start()
    return s


def sample_once(db, cfg, screen=True):
    d = desktop_sample()
    if screen:
        screen_tick(cfg)
    if _last_screen is not None and cfg.get("screen_capture"):
        d["screen"] = _last_screen
    add_event(db, "desktop_sample", "desktop", f"{d['app'] or '?'} - {d['title'] or ''} (idle {d['idle_s']}s)", d,
              cls=d["app"], color="#5B8DEF")
    if cfg.get("camera_probe"):
        p = frame_probe(cfg)
        add_event(db, "frame_probe", "vision", "probe: " + (", ".join(
            f"{b['cls']}@{'/'.join(b['zones']) or 'none'}({b['area']})" for b in p.get("boxes", [])) or p.get("error") or "nobody"),
            p, cls="probe", color="#A3AFBD")
    return d


# ---------------- plan ----------------
def local_day(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def block_at(db, ts):
    r = db.execute("SELECT * FROM plan_blocks WHERE status != 'removed' AND planned_start <= ? AND planned_end > ? "
                   "ORDER BY planned_start DESC LIMIT 1", (ts, ts)).fetchone()
    return dict(r) if r else None


def set_block_status(db, block, status, source, note=None):
    before = {"status": block["status"]}
    db.execute("UPDATE plan_blocks SET status=?, updated_at=? WHERE id=?", (status, time.time(), block["id"]))
    db.execute("INSERT INTO plan_changes (ts, block_id, change, before, after, source) VALUES (?,?,?,?,?,?)",
               (time.time(), block["id"], "status", json.dumps(before), json.dumps({"status": status, "note": note}), source))
    db.commit()


def seed_plan(db, blocks, source="seed"):
    """blocks: [{id, day, task, task_id, task_rank, kind, start_min, dur, ...}] in LA wall-clock."""
    n = 0
    for b in blocks:
        start = datetime.strptime(b["day"], "%Y-%m-%d") + timedelta(minutes=b["start_min"])
        end = start + timedelta(minutes=b["dur"])
        row = dict(id=b["id"], day=b["day"], task=b["task"], task_id=b.get("task_id"), task_rank=b.get("task_rank"),
                   linear_issue=b.get("linear_issue"), kind=b.get("kind"), planned_start=start.timestamp(),
                   planned_end=end.timestamp(), start_local=start.strftime("%H:%M"), end_local=end.strftime("%H:%M"),
                   budget_min=b["dur"], zone=b.get("zone", "desk"), status="planned", source=source,
                   tags=json.dumps(b.get("tags", [])), updated_at=time.time())
        old = db.execute("SELECT * FROM plan_blocks WHERE id=?", (b["id"],)).fetchone()
        if old:
            continue
        db.execute(f"INSERT INTO plan_blocks ({','.join(row)}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))
        db.execute("INSERT INTO plan_changes (ts, block_id, change, before, after, source) VALUES (?,?,?,?,?,?)",
                   (time.time(), b["id"], "insert", None, json.dumps(row), source))
        n += 1
    db.commit()
    return n


# ---------------- voice check-ins ----------------
CHECKIN_PATTERNS = [
    ("start", re.compile(r"\b(?:i'?m |i am |okay,? |ok,? )?(?:starting|start|working on|beginning|getting on|on to)\s+(?:the |my )?(.+)", re.I)),
    ("pulled", re.compile(r"\b(?:got|get|getting|been) (?:pulled|sucked|dragged) into\s+(.+)|\bdistracted by\s+(.+)", re.I)),
    ("done", re.compile(r"^\W*(?:i'?m |i am )?(?:all )?(?:done|finished|wrapped up|completed?)\b(?:\s+with\s+(.+))?", re.I)),
    ("break", re.compile(r"\b(?:taking a break|break time|going on a break)\b", re.I)),
]
STATUS_FOR = {"start": "in_progress", "done": "done", "pulled": "interrupted", "break": "paused"}


def _tokens(s):
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w) > 2}


def match_block(db, ts, phrase):
    """The block whose task best overlaps the phrase (today, nearest in time); else the current block."""
    cur = block_at(db, ts)
    if phrase:
        want = _tokens(phrase)
        best, score = None, 0
        for r in db.execute("SELECT * FROM plan_blocks WHERE day=? AND status!='removed'", (local_day(ts),)):
            s = len(want & _tokens(r["task"])) - abs(r["planned_start"] - ts) / 86400
            if s > score:
                best, score = dict(r), s
        if best and score > 0.5:
            return best
    return cur


def tag_checkins(db, since, until):
    out = []
    rows = db.execute("SELECT * FROM utterances WHERE role='user' AND ts_start > ? AND ts_start <= ? ORDER BY ts_start",
                      (since, until)).fetchall()
    for u in rows:
        text = (u["text"] or "").strip()
        for intent, rx in CHECKIN_PATTERNS:
            m = rx.search(text)
            if not m:
                continue
            what = next((g for g in m.groups() if g), None) if m.groups() else None
            what = what.strip(" .!?") if what else None
            key = f"checkin:{u['id']}"
            if db.execute("SELECT 1 FROM events WHERE source_key=?", (key,)).fetchone():
                break
            blk = match_block(db, u["ts_start"], what if intent == "start" else (what if intent == "done" else None))
            data = {"utterance_id": u["id"], "intent": intent, "what": what, "text": text,
                    "block_id": blk["id"] if blk else None, "block_task": blk["task"] if blk else None}
            now = time.time()
            db.execute("INSERT OR IGNORE INTO events (ts, kind, cls, text, color, modality, written_at, data, source_key) "
                       "VALUES (?,?,?,?,?,?,?,?,?)", (u["ts_start"], "checkin", intent,
                       f"Check-in ({intent}): {text}" + (f" -> {blk['task']}" if blk else ""), "#E59E3B", "voice", now,
                       json.dumps(data), key))
            db.commit()
            if blk:
                set_block_status(db, blk, STATUS_FOR[intent], "voice_checkin", note=text)
            out.append(data)
            break
    return out


def latest_checkin(db, before, window=3600):
    r = db.execute("SELECT data, ts FROM events WHERE kind='checkin' AND ts <= ? AND ts > ? ORDER BY ts DESC LIMIT 1",
                   (before, before - window)).fetchone()
    return (json.loads(r["data"]), r["ts"]) if r else (None, None)


# ---------------- 5 min classification ----------------
def classify(db, cfg, t0, t1):
    ev = {"window": [t0, t1]}
    desk = [json.loads(r["data"]) for r in db.execute(
        "SELECT data FROM events WHERE kind='desktop_sample' AND ts>=? AND ts<? ORDER BY ts", (t0, t1))]
    probes = [json.loads(r["data"]) for r in db.execute(
        "SELECT data FROM events WHERE kind='frame_probe' AND ts>=? AND ts<? ORDER BY ts", (t0, t1))]
    probes = [p for p in probes if p.get("ok")]
    utt = db.execute("SELECT COUNT(*) FROM utterances WHERE role='user' AND ts_start>=? AND ts_start<?", (t0, t1)).fetchone()[0]
    sounds = [r[0] for r in db.execute("SELECT text FROM events WHERE modality='audio' AND ts>=? AND ts<?", (t0, t1))]
    pres = [r[0] for r in db.execute("SELECT cls FROM presence WHERE cls IN ('person','dog') AND start < ? "
                                     "AND (end IS NULL OR end >= ?)", (t1, t0))]

    active = [d for d in desk if d.get("idle_s") is not None and d["idle_s"] < cfg["idle_active_s"]]
    active_frac = len(active) / len(desk) if desk else 0.0
    apps = {}
    for d in active or desk:
        apps[d.get("app") or "?"] = apps.get(d.get("app") or "?", 0) + 1
    top_app = max(apps, key=apps.get) if apps else None
    titles = " ".join((d.get("title") or "") for d in (active or desk)).lower()
    distracted = any(w.strip() in titles for w in cfg["distraction_title_words"])
    shots = [d["screen"].get("path") or d["screen"].get("dup_of") for d in desk if (d.get("screen") or {}).get("ok")]
    shots = [x for x in shots if x]
    caption, label = None, ""
    if shots and cfg.get("screen_caption"):
        try:
            caption = caption_screen(cfg, shots[-1])
        except Exception as e:
            caption = f"error: {e}"[:120]
        m = re.search(r"\b(work|email_admin|chat|browsing|video|social|game|idle_screen)\b", (caption or "").lower())
        label = m.group(1) if m else ""
        if label in ("browsing", "video", "social", "game"):
            distracted = True
        elif label in ("work", "email_admin"):
            distracted = False
    on_call_app = any(d.get("app") in cfg["call_apps"] for d in desk)

    zone_votes, desk_votes, person_seen = {}, 0, 0
    for p in probes:
        people = [b for b in p["boxes"] if b["cls"] == "person"]
        if people:
            person_seen += 1
        for b in people:
            if b["area"] >= cfg["desk_box_area"]:
                desk_votes += 1
                continue
            for z in b["zones"]:
                a = cfg["zone_activity"].get(z)
                if a:
                    zone_votes[a] = zone_votes.get(a, 0) + 1
    dogs = sum(1 for p in probes for b in p["boxes"] if b["cls"] == "dog")
    ev.update(desktop_samples=len(desk), active_frac=round(active_frac, 2), top_app=top_app, distracted_title=distracted,
              probes=len(probes), person_probes=person_seen, desk_votes=desk_votes, zone_votes=zone_votes,
              dog_boxes=dogs, screens=len(shots), screen_caption=caption, last_screen=shots[-1] if shots else None, user_utterances=utt, sounds=sounds[-5:], presence=pres[:20],
              zones_provisional=cfg.get("zones_provisional", True))

    at_desk = active_frac >= 0.3 or (probes and desk_votes >= max(1, len(probes) // 3))
    if on_call_app and active_frac > 0:
        guess, why = "on_call", "call app frontmost"
    elif utt >= 2:
        guess, why = "talking_to_assistant", f"{utt} voice turns"
    elif at_desk:
        work = top_app in cfg["work_apps"] and not distracted
        if not work and top_app and not distracted and top_app not in ("Safari", "Google Chrome", "Firefox", "Arc", "Music", "TV"):
            work = True   # unknown non-browser app: give benefit of the doubt
        if label in ("work", "email_admin"):
            work = True
        if distracted:
            work = False
        guess, why = ("at_desk_work" if work else "at_desk_other"), f"active {active_frac:.0%}, app {top_app}"
    elif zone_votes:
        guess = max(zone_votes, key=zone_votes.get)
        why = f"person in {guess} zone ({'provisional zones' if cfg.get('zones_provisional') else 'zones'})"
    elif probes and person_seen == 0 and active_frac == 0:
        guess, why = "away", "nobody on camera, no input"
    elif not probes and not desk:
        guess, why = "unknown", "no samples"
    elif person_seen:
        guess, why = "at_desk_other", "person visible, outside known zones, no input"
    else:
        guess, why = "away", "no input"
    ev["why"] = why
    return guess, ev


def is_quiet(cfg, ts):
    a, b = cfg["quiet_hours"]
    hm = datetime.fromtimestamp(ts).strftime("%H:%M")
    return (a <= hm or hm < b) if a > b else (a <= hm < b)


def is_kid_time(cfg, ts, block):
    if block and any(k.lower() in (block.get("task") or "").lower() or k.lower() in (block.get("tags") or "").lower()
                     for k in cfg["kid_keywords"]):
        return True
    dt = datetime.fromtimestamp(ts)
    hm, dow = dt.strftime("%H:%M"), dt.strftime("%a")
    for w in cfg["kid_windows"]:
        if (not w.get("days") or dow in w["days"]) and (not w.get("date") or w["date"] == dt.strftime("%Y-%m-%d")) \
                and w["start"] <= hm < w["end"]:
            return True
    return False


def run_slice(db, cfg, t1=None, dry_run=None):
    t1 = t1 or time.time()
    t0 = t1 - cfg["slice_every_s"]
    last = db.execute("SELECT MAX(ts) FROM events WHERE kind='activity_slice'").fetchone()[0] or t0
    checkins = tag_checkins(db, min(last, t0) - 1, t1)
    guess, ev = classify(db, cfg, t0, t1)
    blk = block_at(db, t1)
    ck, ck_ts = latest_checkin(db, t1)
    override = None
    if ck and (not blk or ck.get("block_id") in (None, blk["id"]) or ck["intent"] == "pulled"):
        if ck["intent"] == "start" and blk and ck.get("block_id") == blk["id"]:
            override = "on_plan"
        elif ck["intent"] == "pulled":
            override = "off_plan"
    accepts = cfg["zone_accepts"].get((blk or {}).get("zone") or "desk", cfg["zone_accepts"]["desk"]) if blk else None
    on_plan = None
    if blk and blk["status"] not in ("done", "removed"):
        on_plan = guess in accepts
        if override == "on_plan" and guess != "away":
            on_plan = True
        elif override == "off_plan":
            on_plan = False
    if blk and blk["status"] == "done":
        on_plan = None
    data = {"guess": guess, "evidence": ev, "block_id": blk["id"] if blk else None,
            "block_task": blk["task"] if blk else None, "block_zone": blk["zone"] if blk else None,
            "on_plan": on_plan, "checkin_override": override, "checkins_tagged": checkins}
    sid = add_event(db, "activity_slice", "activity",
                    f"{guess}" + (f" during '{blk['task'][:60]}' - {'on plan' if on_plan else 'off plan'}" if blk and on_plan is not None else ""),
                    data, ts=t1, cls=guess, color="#3BB273" if on_plan else ("#E5484D" if on_plan is False else "#A3AFBD"))
    data["slice_id"] = sid
    data["nudge"] = maybe_nudge(db, cfg, t1, blk, guess, dry_run=dry_run)
    return data


# ---------------- nudges ----------------
def maybe_nudge(db, cfg, ts, blk, guess, dry_run=None):
    dry = cfg.get("dry_run") if dry_run is None else dry_run
    if not blk:
        return {"sent": False, "reason": "no plan block now"}
    recent = [json.loads(r["data"]) for r in db.execute(
        "SELECT data FROM events WHERE kind='activity_slice' AND ts<=? ORDER BY ts DESC LIMIT ?", (ts, cfg["drift_slices"]))]
    if len(recent) < cfg["drift_slices"] or not all(r.get("on_plan") is False and r.get("block_id") == blk["id"] for r in recent):
        return {"sent": False, "reason": "not drifting long enough"}
    if not guess.startswith("at_desk"):
        return {"sent": False, "reason": f"not at the desk ({guess})"}
    if is_quiet(cfg, ts):
        return {"sent": False, "reason": "quiet hours"}
    if is_kid_time(cfg, ts, blk):
        return {"sent": False, "reason": "kid time"}
    n_hour = db.execute("SELECT COUNT(*) FROM events WHERE kind='nudge' AND ts > ?", (ts - 3600,)).fetchone()[0]
    if n_hour >= cfg["max_nudges_per_hour"]:
        return {"sent": False, "reason": "hourly cap"}
    last = db.execute("SELECT MAX(ts) FROM events WHERE kind='nudge' AND json_extract(data,'$.block_id')=?", (blk["id"],)).fetchone()[0]
    if last and ts - last < cfg["nudge_cooldown_s"]:
        return {"sent": False, "reason": "cooldown"}
    task = re.sub(r"\s*[(/:].*$", "", blk["task"])[:70]
    mins = max(0, int((blk["planned_end"] - ts) / 60))
    text = f"Hey, quick check. This block is for {task}, with about {mins} minutes left. Want to get back to it, or should I move it?"
    spoke, err = False, None
    if not dry:
        try:
            speak(cfg, text)
            spoke = True
        except Exception as e:
            err = str(e)[:300]
    data = {"block_id": blk["id"], "task": blk["task"], "guess": guess, "text": text, "dry_run": bool(dry),
            "spoken": spoke, "error": err, "ack": None}
    add_event(db, "nudge", "activity", ("[dry-run] " if dry else "") + text, data, ts=ts, cls="drift", color="#F5A524")
    return {"sent": True, **data}


def speak(cfg, text):
    import numpy as np, soundfile as sf
    from kokoro_onnx import Kokoro
    k = ROOT / "weights" / "kokoro"
    audio, sr = Kokoro(str(k / "kokoro-v1.0.onnx"), str(k / "voices-v1.0.bin")).create(
        text, voice=cfg["voice"], speed=cfg["speed"], lang="en-us")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, np.asarray(audio, dtype=np.float32), sr)
    subprocess.run(["afplay", f.name], timeout=60)
    os.unlink(f.name)


# ---------------- status / daemon ----------------
def status(db):
    def one(sql, *a):
        r = db.execute(sql, a).fetchone()
        return dict(r) if r else None
    fmt = lambda t: datetime.fromtimestamp(t).strftime("%a %H:%M:%S") if t else "-"
    today0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    s = one("SELECT ts, text, data FROM events WHERE kind='activity_slice' ORDER BY ts DESC LIMIT 1")
    d = one("SELECT ts, text FROM events WHERE kind='desktop_sample' ORDER BY ts DESC LIMIT 1")
    p = one("SELECT ts, text FROM events WHERE kind='frame_probe' ORDER BY ts DESC LIMIT 1")
    n = db.execute("SELECT ts, text FROM events WHERE kind='nudge' AND ts>=? ORDER BY ts", (today0,)).fetchall()
    b = block_at(db, time.time())
    hb = Path(DB_PATH).with_name("planloop.heartbeat")
    print(f"db:            {DB_PATH}")
    print(f"daemon beat:   {fmt(float(hb.read_text())) if hb.exists() else 'never'}")
    print(f"last sample:   {fmt(d['ts']) if d else '-'}  {d['text'] if d else ''}")
    print(f"last probe:    {fmt(p['ts']) if p else '-'}  {p['text'] if p else ''}")
    print(f"last slice:    {fmt(s['ts']) if s else '-'}  {s['text'] if s else ''}")
    if s:
        print(f"   why:        {json.loads(s['data'])['evidence'].get('why')}")
    print(f"current block: {(b['start_local'] + '-' + b['end_local'] + ' ' + b['task'] + ' [' + b['status'] + ']') if b else 'none'}")
    print(f"plan today:    {db.execute('SELECT COUNT(*) FROM plan_blocks WHERE day=? AND status!=?', (local_day(time.time()), 'removed')).fetchone()[0]} blocks")
    sd = SCREEN_DIR / datetime.now().strftime("%Y-%m-%d")
    files = list(sd.glob("*.jpg")) if sd.exists() else []
    print(f"screens today: {len(files)} frames, {sum(f.stat().st_size for f in files)/1e6:.1f} MB in {sd}")
    print(f"nudges today:  {len(n)}")
    for r in n:
        print(f"   {fmt(r['ts'])}  {r['text']}")


def run():
    cfg = load_config()
    db = connect()
    os.nice(10)   # low priority: never starve Vision Lab
    hb = Path(DB_PATH).with_name("planloop.heartbeat")
    print(f"planloop: db={DB_PATH} sample={cfg['sample_every_s']}s screen={cfg.get('screen_every_s', cfg['sample_every_s'])}s slice={cfg['slice_every_s']}s", flush=True)
    next_slice = (time.time() // cfg["slice_every_s"] + 1) * cfg["slice_every_s"]
    next_sample = 0
    while True:
        t = time.time()
        try:
            cfg = load_config()
            screen_tick(cfg)
            if cfg.get("screenrec"):
                from vision_lab import screenrec
                screenrec.ensure_running(cfg)
            if t >= next_sample:
                sample_once(db, cfg, screen=False)
                next_sample = t + cfg["sample_every_s"]
            if t >= next_slice:
                prune_screens(cfg)
                r = run_slice(db, cfg, t1=next_slice)
                print(datetime.now().strftime("%H:%M"), r["guess"], r["on_plan"], r["nudge"].get("reason", "NUDGED"), flush=True)
                next_slice += cfg["slice_every_s"]
                while next_slice <= time.time():
                    next_slice += cfg["slice_every_s"]
            hb.write_text(str(time.time()))
        except Exception as e:
            print("error:", repr(e), flush=True)
        tick = min(cfg["sample_every_s"], cfg.get("screen_every_s", cfg["sample_every_s"]))
        time.sleep(max(1, tick - (time.time() - t)))


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "run":
        return run()
    db = connect()
    if cmd == "status":
        return status(db)
    if cmd == "sample":
        print(json.dumps(sample_once(db, load_config()), indent=1))
    elif cmd == "once":
        at = float(argv[argv.index("--at") + 1]) if "--at" in argv else None
        print(json.dumps(run_slice(db, load_config(), t1=at, dry_run=True if "--dry-run" in argv else None), indent=1, default=str))
    elif cmd == "seed-plan":
        print("inserted", seed_plan(db, json.load(open(argv[2])), source="seed"))
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
