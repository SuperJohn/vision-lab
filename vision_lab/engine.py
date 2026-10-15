"""Camera capture, the per-frame model pipeline, tracking, zones/rules and the VLM worker."""
import base64
import collections
import itertools
import json
import math
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import llm, voice
from .store import KEYFRAME_DIR, RECORDING_DIR

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

DEFAULT_SETTINGS = {
    "camera": 0,
    "stages": {"detect": True, "segment": True, "pose": True, "depth": False, "describe": True},
    "detector": "yolo11s",
    "world_classes": "person, cup, laptop, phone, headphones, book, cat, dog",
    "conf": 0.35,
    "describe_model": os.environ.get("VLAB_MODEL", "gemma4:26b"),
    "summary_model": os.environ.get("VLAB_MODEL", "gemma4:26b"),
    "describe_every": 10,
    # Home tuning: only these COCO classes are detected (empty = all 80)
    "home_classes": "person, dog, couch, bed, chair, dining table, tv, laptop, cell phone, remote, book, cup, bottle, bowl, wine glass, refrigerator, oven, microwave, sink, backpack, handbag, potted plant, clock, keyboard, mouse",
    # Common mislabels in this apartment get folded into the right class
    "relabel": {"cat": "dog", "teddy bear": "dog", "sheep": "dog", "bear": "dog"},
    # Stricter per-class thresholds for flaky small objects
    "class_conf": {"remote": 0.45, "cell phone": 0.4, "book": 0.4, "bottle": 0.4, "wine glass": 0.45, "clock": 0.45},
}

DEFAULT_RULES = [
    {"id": "r-arrive", "name": "Someone comes in", "enabled": True, "object": "person",
     "event": "appears", "target": None, "duration_s": 0, "posture": "any", "direction": "any",
     "cooldown_s": 120,
     "actions": {"notify": False, "snapshot": True,
                 "vlm_prompt": "Describe the person who just came into view and what they are carrying. One sentence."}},
    {"id": "r-floor", "name": "Person lying down", "enabled": True, "object": "person",
     "event": "posture_change", "target": None, "duration_s": 0, "posture": "lying", "direction": "any",
     "cooldown_s": 300, "actions": {"notify": True, "snapshot": True, "vlm_prompt": ""}},
]

FIXED_COLORS = {"person": "#F5A524", "laptop": "#5CC8FF", "cup": "#FF8CC0", "cat": "#9BE15D",
                "dog": "#9BE15D", "potted plant": "#C4AEFF", "cell phone": "#FFD166",
                "tv": "#7EE0C3", "book": "#FF9F7A", "chair": "#9DB4FF"}
PALETTE = ["#FFD166", "#7EE0C3", "#FF9F7A", "#9DB4FF", "#E0E0E0", "#F28DF0", "#B5E48C"]


def color_for(cls):
    return FIXED_COLORS.get(cls) or PALETTE[sum(map(ord, cls)) % len(PALETTE)]


def list_cameras():
    """AVFoundation video devices in OpenCV's index order, minus screen capture."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices",
                              "true", "-i", ""], capture_output=True, text=True, timeout=10).stderr
    except (OSError, subprocess.TimeoutExpired):
        return [{"index": 0, "name": "Camera 0"}]
    cams, in_video = [], False
    for line in out.splitlines():
        if "video devices" in line:
            in_video = True
            continue
        if "audio devices" in line:
            break
        m = re.search(r"\] \[(\d+)\] (.+)$", line)
        if in_video and m and not m.group(2).startswith("Capture screen"):
            cams.append({"index": int(m.group(1)), "name": m.group(2).strip()})
    return cams or [{"index": 0, "name": "Camera 0"}]


def notify(title, text):
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.Popen(["osascript", "-e",
                      f'display notification "{esc(text)}" with title "{esc(title)}"'])


def classify_posture(k):
    """k: (17, 3) COCO keypoints in pixels. Returns standing / sitting / lying / unknown."""
    def pt(i):
        return k[i, :2] if k[i, 2] > 0.4 else None

    def mid(a, b):
        pa, pb = pt(a), pt(b)
        if pa is not None and pb is not None:
            return (pa + pb) / 2
        return pa if pa is not None else pb

    sh, hip, knee = mid(5, 6), mid(11, 12), mid(13, 14)
    if sh is None or hip is None:
        return "unknown"
    torso = hip - sh
    torso_len = float(np.linalg.norm(torso)) + 1e-6
    if math.degrees(math.atan2(abs(torso[0]), abs(torso[1]) + 1e-6)) > 60:
        return "lying"
    if knee is None:
        return "unknown"
    thigh = knee - hip
    thigh_angle = math.degrees(math.atan2(abs(thigh[0]), thigh[1]))
    if thigh_angle > 45 or np.linalg.norm(thigh) < 0.45 * torso_len:
        return "sitting"
    return "standing"


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def segments_cross(p1, p2, q1, q2):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (orient(p1, p2, q1) * orient(p1, p2, q2) < 0
            and orient(q1, q2, p1) * orient(q1, q2, p2) < 0)


class Camera:
    def __init__(self, paused=False):
        self.lock = threading.Lock()
        self.frame, self.seq, self.ts, self.fps = None, 0, 0.0, 0.0
        self.index, self.cap, self.error = None, None, None
        self.writer, self.recording_path, self.recording_since = None, None, None
        self._want, self.paused = None, bool(paused)
        self.cap_lock = threading.Lock()      # guards every open/read/release of self.cap
        self.released = threading.Event()     # set while no capture handle is open
        self.released.set()
        # macOS only shows the camera-permission prompt for a request made on the main thread,
        # so probe once here before the capture thread takes over (skipped when starting paused).
        if (threading.current_thread() is threading.main_thread() and not self.paused
                and not os.environ.get("VLAB_NO_CAMERA")):
            cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION).release()
        threading.Thread(target=self._loop, daemon=True, name="camera").start()

    def open(self, index):
        """Select a camera. While paused this only records the choice; resume() opens it."""
        if self.paused:
            self.index = index
        else:
            self._want = index

    def _release(self):
        with self.cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.released.set()
        with self.lock:
            self.frame = None
        self.fps = 0.0

    def pause(self, timeout=3.0):
        """Release every capture handle so the device (and its green light) turns off.

        Returns True once the handle is actually closed. The capture thread may be inside a
        blocking read(), so release under cap_lock (which read() also holds) instead of just
        flagging the loop and hoping it notices.
        """
        self.stop_recording()
        self.paused = True
        self._want = None
        if self.cap_lock.acquire(timeout=timeout):
            self.cap_lock.release()
            self._release()
        return self.released.wait(timeout)

    def resume(self):
        if self.paused:
            self.paused = False
            self._want = self.index

    def _loop(self):
        times, fails = collections.deque(maxlen=30), 0
        while True:
            if self.paused:
                if self.cap is not None or not self.released.is_set():
                    self._release()
                time.sleep(0.1)
                continue
            if self._want is not None:
                idx, self._want = self._want, None
                if self.cap:
                    self._release()
                if isinstance(idx, str):  # a video file (VLAB_SOURCE), looped in real time
                    cap = cv2.VideoCapture(idx)
                    file_dt = 1 / (cap.get(cv2.CAP_PROP_FPS) or 30)
                else:
                    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    file_dt = None
                with self.cap_lock:
                    if self.paused:          # paused while we were opening: close it right away
                        cap.release()
                        continue
                    self.cap, self.index, fails = cap, idx, 0
                    self.released.clear()
                times.clear()
                self.error = None if cap.isOpened() else (
                    "Could not open the camera. Allow camera access for the app running "
                    "Python (System Settings › Privacy & Security › Camera), then restart.")
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.2)
                continue
            with self.cap_lock:
                if self.paused or self.cap is None:
                    continue
                ok, frame = self.cap.read()
            if not ok and file_dt:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if file_dt:
                time.sleep(file_dt)
            if not ok:
                fails += 1
                if fails > 60:
                    self.error = "The camera returned no frames (no permission, or in use by another app)."
                time.sleep(0.01)
                continue
            fails, self.error = 0, None
            now = time.time()
            times.append(now)
            if len(times) > 1:
                self.fps = (len(times) - 1) / (times[-1] - times[0])
            with self.lock:
                self.frame, self.ts = frame, now
                self.seq += 1
                if self.writer is not None:
                    self.writer.write(frame)

    def latest(self):
        with self.lock:
            return self.frame, self.seq, self.ts

    def start_recording(self):
        with self.lock:
            if self.frame is None or self.writer is not None:
                return self.recording_path
            h, w = self.frame.shape[:2]
            path = RECORDING_DIR / time.strftime("Vision Lab %Y-%m-%d at %H.%M.%S.mp4")
            fps = round(self.fps) if self.fps >= 5 else 30
            # H.264 plays in QuickTime/Photos; fall back to MPEG-4 Part 2 if unavailable
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
            if not self.writer.isOpened():
                self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            self.recording_path, self.recording_since = path, time.time()
            return path

    def stop_recording(self):
        with self.lock:
            if self.writer is not None:
                self.writer.release()
            path, self.writer, self.recording_path, self.recording_since = self.recording_path, None, None, None
            return path


class Models:
    """Lazy model cache. `lock` serialises all GPU (MPS) work across threads."""

    def __init__(self):
        import torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.cache = {}
        self.lock = threading.RLock()
        WEIGHTS.mkdir(exist_ok=True)

    def yolo(self, name, aux=False):
        # aux instances serve benchmarks/A-B so the live tracker state isn't disturbed
        key = name + ("#aux" if aux else "")
        with self.lock:
            if key not in self.cache:
                from ultralytics import SAM, YOLO
                path = str(WEIGHTS / f"{name}.pt")
                self.cache[key] = SAM(path) if name.startswith("sam") else YOLO(path)
            return self.cache[key]

    def depth(self):
        with self.lock:
            if "depth" not in self.cache:
                from transformers import pipeline
                self.cache["depth"] = pipeline("depth-estimation", model=DEPTH_MODEL, device=self.device)
            return self.cache["depth"]

    @staticmethod
    def weights_present(name):
        return (WEIGHTS / f"{name}.pt").exists()

    @staticmethod
    def hf_present(repo):
        return (HF_CACHE / ("models--" + repo.replace("/", "--"))).exists()


class Track:
    def __init__(self, tid, cls, ts):
        self.id, self.cls = tid, cls
        self.first = self.last = ts
        self.frames, self.confirmed, self.span = 0, False, None
        self.trail = collections.deque(maxlen=40)
        self.zone_in = {}        # zone id -> entry time
        self.zone_flip = {}      # zone id -> consecutive frames disagreeing with zone_in
        self.side = {}           # line id -> last side sign
        self.posture, self.cand, self.cand_n = "unknown", None, 0
        self.fired = set()       # once-per-visit keys for dwell rules


class Engine:
    def __init__(self, store):
        self.store = store
        saved = store.get_config("settings", {})
        self.settings = {**DEFAULT_SETTINGS, **saved,
                         "stages": {**DEFAULT_SETTINGS["stages"], **saved.get("stages", {})}}
        self.zones = store.get_config("zones", [])
        self.rules = store.get_config("rules", DEFAULT_RULES)
        if self.settings.get("camera_paused") and "capture_paused" not in saved:
            self.settings["capture_paused"] = True   # HOU-146: migrate the old camera-only pause
        self.camera = Camera(paused=bool(self.settings.get("capture_paused") or self.settings.get("camera_paused")))
        self.models = Models()
        self.embedder = llm.Embedder()
        self.embedder.lock = self.models.lock
        self.cameras = list_cameras()
        source = os.environ.get("VLAB_SOURCE")
        if source:
            self.cameras.insert(0, {"index": source, "name": f"File: {Path(source).name}"})
            self.settings["camera"] = source
        if self.settings["camera"] not in [c["index"] for c in self.cameras]:
            self.settings["camera"] = self.cameras[0]["index"]
        self.camera.index = self.settings["camera"]
        if not os.environ.get("VLAB_NO_CAMERA"):   # test servers run without touching the camera
            self.camera.open(self.settings["camera"])   # no-op while paused (state survives restarts)

        self.tracks, self.rule_last = {}, {}
        self.active_model, self.world_set = None, None
        self.out_lock = threading.Lock()
        self.out_seq, self.out_json, self.out_jpeg = 0, None, None
        self.event_rev = 0
        self.scene, self.scene_error, self.last_activity = None, None, None
        self.last_describe = 0.0
        self.depth_jpeg, self.pipeline_error = None, None
        self.paused_frame = None
        self.last_kf_ts, self.last_kf_thumb = 0.0, None
        self.pulling = set()
        self.vlm_q = queue.PriorityQueue()
        self._qn = itertools.count()
        for target, name in ((self._pipeline, "pipeline"), (self._vlm_loop, "vlm"),
                             (self._embed_loop, "embed")):
            threading.Thread(target=target, daemon=True, name=name).start()

    # ---------- settings, zones, rules ----------
    def update_settings(self, patch):
        if "stages" in patch:
            patch = {**patch, "stages": {**self.settings["stages"], **patch["stages"]}}
        self.settings = {**self.settings, **patch}
        self.store.set_config("settings", self.settings)
        if "camera" in patch and patch["camera"] != self.camera.index:
            self.camera.open(patch["camera"])
            self._reset_tracks()

    def set_camera_paused(self, paused):
        """Pause releases the camera device; resume reopens it. Persisted and logged to the timeline."""
        if paused:
            released = self.camera.pause()
            self.event("camera_paused", "Camera paused (device released)" if released
                       else "Camera paused (release still pending)", "#A3AFBD")
        else:
            released = False
            if self.camera.paused:
                self.camera.resume()
                self.event("camera_resumed", "Camera resumed", "#5BD38A")
        if self.settings.get("camera_paused") != bool(paused):
            self.settings = {**self.settings, "camera_paused": bool(paused)}
            self.store.set_config("settings", self.settings)
        return {"paused": self.camera.paused, "released": released if paused else False}

    def set_capture_paused(self, paused):
        """HOU-146 Privacy mode: pause ALL capture: camera (device released), mic (remembering its prior state)
        and screen captures (planloop reads settings.capture_paused before every screencapture)."""
        paused = bool(paused)
        was = bool(self.settings.get("capture_paused"))
        mic_on = bool(self.store.get_config("mic_enabled", True))
        patch = {"capture_paused": paused, "camera_paused": paused}
        if paused:
            released = self.camera.pause()
            if not was:
                patch["capture_mic_prev"] = mic_on
            self.store.set_config("mic_enabled", False)
        else:
            released = False
            if self.camera.paused:
                self.camera.resume()
            if was:
                self.store.set_config("mic_enabled", bool(self.settings.get("capture_mic_prev", True)))
        self.settings = {**self.settings, **patch}
        self.store.set_config("settings", self.settings)
        if paused and not was:
            self.event("capture_paused", "Privacy mode on (camera, screen and mic stopped)" if released
                       else "Privacy mode on (camera release still pending)", "#A3AFBD")
        elif was and not paused:
            self.event("capture_resumed", "Privacy mode off (capture resumed)", "#5BD38A")
        return {"paused": paused, "capture_paused": paused, "privacy_mode": paused, "camera_paused": self.camera.paused,
                "released": released if paused else False,
                "mic_enabled": bool(self.store.get_config("mic_enabled", True))}

    def set_zones(self, zones):
        self.zones = zones
        self.store.set_config("zones", zones)

    def set_rules(self, rules):
        self.rules = rules
        self.store.set_config("rules", rules)

    def class_names(self):
        try:
            return list(self.models.yolo("yolo11s").names.values())
        except Exception:
            return ["person", "cat", "dog", "cup", "laptop"]

    # ---------- events & keyframes ----------
    def event(self, kind, text, color="#A3AFBD", track=None, keyframe_id=None, rule_id=None, ts=None):
        eid = self.store.add_event(kind, text, color, track.id if track else None,
                                   track.cls if track else None, keyframe_id, rule_id, ts)
        self.event_rev += 1
        return eid

    def snapshot(self):
        """What vision sees right now: confirmed tracks with their zones and posture, plus the last caption."""
        names = {z["id"]: z["name"] for z in self.zones}
        tracks = [{"id": t.id, "cls": t.cls, "zones": [names.get(z, z) for z in t.zone_in],
                   "posture": t.posture if t.cls == "person" else None, "since": round(t.first, 1)}
                  for t in list(self.tracks.values()) if t.confirmed]
        scene = self.scene or {}
        return {"tracks": tracks, "scene": scene.get("description"), "activity": scene.get("activity"),
                "scene_ts": scene.get("ts"), "camera_fps": round(self.camera.fps, 1), "paused": self.camera.paused,
                "capture_paused": bool(self.settings.get("capture_paused"))}

    def save_keyframe(self, frame, ts, caption=None):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (960, int(h * 960 / w)), interpolation=cv2.INTER_AREA) if w > 960 else frame
        path = KEYFRAME_DIR / f"{int(ts * 1000)}.jpg"
        cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.last_kf_ts = ts
        self.last_kf_thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
        return self.store.add_keyframe(ts, path, caption)

    def maybe_keyframe(self, frame, ts, caption=None):
        """Keep a keyframe when the scene changed (or a minute passed), at most every 10 s."""
        if ts - self.last_kf_ts < 10:
            return None
        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
        changed = self.last_kf_thumb is None or float(np.abs(thumb - self.last_kf_thumb).mean()) > 6
        if changed or ts - self.last_kf_ts > 60:
            return self.save_keyframe(frame, ts, caption)
        return None

    # ---------- per-frame pipeline ----------
    def _reset_tracks(self):
        now = time.time()
        for t in self.tracks.values():
            if t.span:
                self.store.close_span(t.span, now)
        self.tracks = {}

    def _detect(self, frame, timing):
        s = self.settings
        world = "world" in s["detector"]
        seg = s["stages"]["segment"] and not world
        name = s["detector"] + ("-seg" if seg else "")
        if name != self.active_model:
            self._reset_tracks()
            self.active_model = name
        with self.models.lock:
            m = self.models.yolo(name)
            if world:
                classes = [c.strip() for c in s["world_classes"].split(",") if c.strip()]
                if classes != self.world_set:
                    m.set_classes(classes)
                    self.world_set = classes
            kw = {}
            if not world:
                keep = {c.strip() for c in s.get("home_classes", "").split(",") if c.strip()}
                keep |= {k for k, v in s.get("relabel", {}).items() if v in keep}
                ids = [i for i, n in m.names.items() if n in keep]
                if ids:
                    kw["classes"] = ids
            t0 = time.perf_counter()
            r = m.track(frame, persist=True, tracker="bytetrack.yaml", device=self.models.device,
                        conf=s["conf"], imgsz=640, verbose=False, **kw)[0]
            wall = (time.perf_counter() - t0) * 1000
        sp = r.speed
        timing["detect"] = sp.get("preprocess", 0) + sp.get("inference", 0)
        if seg:
            timing["segment"] = sp.get("postprocess", 0)
        else:
            timing["detect"] += sp.get("postprocess", 0)
        timing["track"] = max(0.0, wall - sum(sp.values()))
        if r.boxes is None or r.boxes.id is None:
            return []
        polys = r.masks.xyn if seg and r.masks is not None else None
        objs = []
        for i, (tid, c, cf, b) in enumerate(zip(r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(),
                                                r.boxes.conf.tolist(), r.boxes.xyxyn.tolist())):
            cls = s.get("relabel", {}).get(r.names[c], r.names[c])
            if cf < s.get("class_conf", {}).get(cls, 0):
                continue
            o = {"id": tid, "cls": cls, "conf": round(cf, 2), "box": [round(v, 4) for v in b],
                 "color": color_for(cls)}
            if polys is not None and i < len(polys) and len(polys[i]) > 2:
                p = polys[i][:: max(1, len(polys[i]) // 80)]
                o["poly"] = [[round(float(x), 4), round(float(y), 4)] for x, y in p]
            objs.append(o)
        return objs

    def _pose(self, frame, objs, timing):
        with self.models.lock:
            m = self.models.yolo("yolo11s-pose")
            t0 = time.perf_counter()
            r = m.predict(frame, device=self.models.device, conf=0.4, imgsz=640, verbose=False)[0]
            timing["pose"] = (time.perf_counter() - t0) * 1000
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return
        h, w = frame.shape[:2]
        kps = r.keypoints.data.cpu().numpy()
        boxes = r.boxes.xyxyn.cpu().numpy()
        for o in objs:
            if o["cls"] != "person":
                continue
            ious = [box_iou(o["box"], b) for b in boxes]
            j = int(np.argmax(ious))
            if ious[j] > 0.3:
                k = kps[j]
                o["kp"] = [[round(float(x / w), 4), round(float(y / h), 4), round(float(c), 2)] for x, y, c in k]
                o["posture_raw"] = classify_posture(k)

    def _depth(self, frame, timing):
        from PIL import Image
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (518, int(h * 518 / w)), interpolation=cv2.INTER_AREA)
        with self.models.lock:
            p = self.models.depth()
            t0 = time.perf_counter()
            d = np.array(p(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))["depth"])
            timing["depth"] = (time.perf_counter() - t0) * 1000
        colored = cv2.applyColorMap(cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                                    cv2.COLORMAP_INFERNO)
        ok, buf = cv2.imencode(".jpg", colored, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self.depth_jpeg = base64.b64encode(buf.tobytes()).decode() if ok else None

    def _pipeline(self):
        last_seq, n = -1, 0
        done = collections.deque(maxlen=30)
        while True:
            frame, seq, ts = self.camera.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.003)
                if frame is None:
                    self._publish_idle()
                continue
            last_seq, n = seq, n + 1
            # Models see a 960 px copy: outputs are normalised, and mask/keypoint
            # post-processing at full resolution costs more than the inference itself.
            fh, fw = frame.shape[:2]
            small = cv2.resize(frame, (960, int(fh * 960 / fw)), interpolation=cv2.INTER_AREA) if fw > 960 else frame
            st = self.settings["stages"]
            timing, objs = {}, []
            try:
                if st["detect"]:
                    objs = self._detect(small, timing)
                if st["pose"] and any(o["cls"] == "person" for o in objs):
                    self._pose(small, objs, timing)
                if st["depth"]:
                    if n % 3 == 0 or self.depth_jpeg is None:
                        self._depth(small, timing)
                else:
                    self.depth_jpeg = None
                self.pipeline_error = None
            except Exception as e:
                self.pipeline_error = f"{type(e).__name__}: {e}"
                time.sleep(0.5)
            self._update_tracks(objs, ts, frame)
            if not st["describe"]:
                self.maybe_keyframe(frame, ts)

            t0 = time.perf_counter()
            h, w = frame.shape[:2]
            view = cv2.resize(frame, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_AREA) if w > 1280 else frame
            ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 72])
            timing["encode"] = (time.perf_counter() - t0) * 1000
            done.append(time.time())
            fps = (len(done) - 1) / (done[-1] - done[0]) if len(done) > 1 and done[-1] > done[0] else 0
            counts = collections.Counter(o["cls"] for o in objs)
            payload = {
                "ts": ts, "w": w, "h": h, "objects": objs,
                "timing": {k: round(v, 1) for k, v in timing.items()},
                "fps": {"camera": round(self.camera.fps, 1), "pipeline": round(fps, 1)},
                "counts": dict(counts.most_common()),
                "depth": self.depth_jpeg,
                **self._status(),
            }
            with self.out_lock:
                self.out_json, self.out_jpeg = json.dumps(payload), buf.tobytes() if ok else None
                self.out_seq += 1

    def _status(self):
        return {"scene": self.scene, "scene_error": self.scene_error, "events_rev": self.event_rev,
                "recording": self.camera.recording_path is not None,
                "recording_since": self.camera.recording_since, "camera": self.camera.index,
                "paused": self.camera.paused, "capture_paused": bool(self.settings.get("capture_paused")), "error": self.camera.error or self.pipeline_error}

    def _publish_idle(self):
        if time.time() - getattr(self, "_idle_ts", 0) < 0.5:
            return
        self._idle_ts = time.time()
        with self.out_lock:
            self.out_json = json.dumps({"objects": [], "timing": {}, "fps": {"camera": 0, "pipeline": 0},
                                        "counts": {}, "depth": None, **self._status(),
                                        "error": None if self.camera.paused
                                        else self.camera.error or "Waiting for the camera…"})
            self.out_jpeg = None
            self.out_seq += 1

    # ---------- tracking, zones, rules ----------
    def _rules(self, event, track, target=None):
        for r in self.rules:
            if (r.get("enabled") and r.get("event") == event
                    and r.get("object") in ("any", track.cls)
                    and (not r.get("target") or r.get("target") == target)):
                yield r

    def _fire(self, rule, t, ts, frame, detail):
        key = rule["id"]  # cooldown per rule: tracker IDs churn, so per-track would spam
        if ts - self.rule_last.get(key, 0) < rule.get("cooldown_s", 60):
            return
        self.rule_last[key] = ts
        actions = rule.get("actions", {})
        prompt = (actions.get("vlm_prompt") or "").strip()
        kf = self.save_keyframe(frame, ts) if actions.get("snapshot") or prompt else None
        text = f"{rule['name']}: {t.cls} #{t.id} {detail}"
        eid = self.event("rule", text, "#5CC8FF", t, kf, rule["id"], ts)
        if actions.get("notify"):
            notify("Vision Lab", text)
        if prompt:
            self.vlm_q.put((0, next(self._qn), {"kind": "rule", "event_id": eid, "prompt": prompt,
                                                "frame": frame.copy(), "keyframe": kf}))

    def _update_tracks(self, objs, ts, frame):
        zones = [z for z in self.zones if z.get("kind") == "zone"]
        lines = [z for z in self.zones if z.get("kind") == "line"]
        zone_name = {z["id"]: z["name"] for z in self.zones}
        for o in objs:
            t = self.tracks.get(o["id"])
            if t is None:
                t = self.tracks[o["id"]] = Track(o["id"], o["cls"], ts)
            t.cls, t.last = o["cls"], ts
            t.frames += 1
            b = o["box"]
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            t.trail.append([round(c[0], 4), round(c[1], 4)])
            o["trail"] = list(t.trail)[::2]
            label = f"{t.cls.capitalize()} #{t.id}"

            if not t.confirmed and t.frames >= 8:
                t.confirmed = True
                t.span = self.store.open_span(t.id, t.cls, t.first)
                self.event("track", f"{label} came into view", color_for(t.cls), t, ts=ts)
                for r in self._rules("appears", t):
                    self._fire(r, t, ts, frame, "appeared")

            if t.confirmed:
                for z in zones:
                    inside = cv2.pointPolygonTest(np.array(z["points"], np.float32), c, False) >= 0
                    if inside == (z["id"] in t.zone_in):
                        t.zone_flip[z["id"]] = 0
                        continue
                    t.zone_flip[z["id"]] = t.zone_flip.get(z["id"], 0) + 1
                    if t.zone_flip[z["id"]] < 10:
                        continue
                    t.zone_flip[z["id"]] = 0
                    if inside:
                        t.zone_in[z["id"]] = ts
                        self.event("zone", f"{label} entered {z['name']}", color_for(t.cls), t, ts=ts)
                        for r in self._rules("enters_zone", t, z["id"]):
                            self._fire(r, t, ts, frame, f"entered {z['name']}")
                    else:
                        entered = t.zone_in.pop(z["id"])
                        mins = (ts - entered) / 60
                        self.event("zone", f"{label} left {z['name']} after {mins:.0f} min",
                                   color_for(t.cls), t, ts=ts)
                        for r in self._rules("leaves_zone", t, z["id"]):
                            self._fire(r, t, ts, frame, f"left {z['name']}")

                for ln in lines:
                    (x1, y1), (x2, y2) = ln["points"][:2]
                    side = 1 if (x2 - x1) * (c[1] - y1) - (y2 - y1) * (c[0] - x1) > 0 else -1
                    prev = t.side.get(ln["id"])
                    t.side[ln["id"]] = side
                    if prev is not None and prev != side and len(t.trail) >= 2 and \
                            segments_cross(t.trail[-2], t.trail[-1], (x1, y1), (x2, y2)):
                        direction = "in" if side > 0 else "out"
                        self.event("line", f"{label} crossed {ln['name']} ({direction})",
                                   color_for(t.cls), t, ts=ts)
                        for r in self._rules("crosses_line", t, ln["id"]):
                            if r.get("direction", "any") in ("any", direction):
                                self._fire(r, t, ts, frame, f"crossed {ln['name']} ({direction})")

                raw = o.get("posture_raw")
                if raw and raw != "unknown":
                    if raw == t.cand:
                        t.cand_n += 1
                    else:
                        t.cand, t.cand_n = raw, 1
                    if t.cand_n >= 15 and t.cand != t.posture:
                        before, t.posture = t.posture, t.cand
                        if before != "unknown":
                            self.event("posture", f"{label} went from {before} to {t.posture}",
                                       color_for(t.cls), t, ts=ts)
                        for r in self._rules("posture_change", t):
                            if r.get("posture", "any") in ("any", t.posture):
                                self._fire(r, t, ts, frame, f"is now {t.posture}")

                for zid, entered in t.zone_in.items():
                    for r in self._rules("in_zone_for", t, zid):
                        key = (r["id"], zid, entered)
                        if key in t.fired or ts - entered < r.get("duration_s", 0):
                            continue
                        if r.get("posture", "any") not in ("any", t.posture):
                            continue
                        t.fired.add(key)
                        self._fire(r, t, ts, frame,
                                   f"in {zone_name.get(zid, 'zone')} for {(ts - entered) / 60:.0f} min")

            o["posture"] = t.posture if t.cls == "person" else None
            o["zones"] = [zone_name.get(z, "") for z in t.zone_in]
            o["dwell"] = {zone_name.get(z, ""): round(ts - e) for z, e in t.zone_in.items()}

        for tid, t in list(self.tracks.items()):
            if ts - t.last > 3:
                if t.confirmed:
                    self.store.close_span(t.span, t.last)
                    secs = t.last - t.first
                    dur = f"{secs / 60:.0f} min" if secs >= 90 else f"{secs:.0f} s"
                    self.event("track", f"{t.cls.capitalize()} #{t.id} left view after {dur}",
                               color_for(t.cls), t, ts=ts)
                    for zid in list(t.zone_in):
                        for r in self._rules("leaves_zone", t, zid):
                            self._fire(r, t, ts, frame, f"left {zone_name.get(zid, 'zone')}")
                del self.tracks[tid]

    # ---------- vision-language worker ----------
    def _vlm_loop(self):
        while True:
            try:
                _, _, job = self.vlm_q.get(timeout=0.25)
            except queue.Empty:
                job = None
            if job is None:
                s = self.settings
                if not s["stages"]["describe"] or time.time() - self.last_describe < s["describe_every"]:
                    continue
                if voice.voice_busy_until > time.time():
                    continue  # a voice turn is live: leave Ollama and the GPU to it
                frame, _, ts = self.camera.latest()
                if frame is None:
                    continue
                job = {"kind": "describe", "frame": frame.copy(), "ts": ts}
            model = self.settings["describe_model"]
            try:
                if job["kind"] == "describe":
                    t0 = time.time()
                    d = llm.describe_scene(model, job["frame"])
                    self.scene = {**d, "ts": time.time(), "model": model, "latency": round(time.time() - t0, 1)}
                    self.scene_error = None
                    self.maybe_keyframe(job["frame"], job["ts"], d["description"])
                    act = (d.get("activity") or "").lower()
                    if act and act != self.last_activity:
                        if self.last_activity is not None:
                            self.event("scene", f"Activity: {act} — {d['description']}", "#C4AEFF")
                        self.last_activity = act
                else:
                    answer = llm.chat(model, job["prompt"], job["frame"])
                    self.store.set_event_vlm(job["event_id"], answer)
                    if job.get("keyframe"):
                        self.store.set_keyframe_caption(job["keyframe"], answer)
                    self.event_rev += 1
            except Exception as e:
                self.scene_error = f"{model}: {e}"
            finally:
                if job["kind"] == "describe":
                    self.last_describe = time.time()

    def _embed_loop(self):
        while True:
            time.sleep(2)
            for row in self.store.keyframes_missing_embedding():
                img = cv2.imread(row["path"])
                if img is None:
                    self.store.set_keyframe_embedding(row["id"], np.zeros(1, np.float32))
                    continue
                emb = self.embedder.image(img)
                if emb is None:
                    break  # embedder unavailable; search falls back to captions
                self.store.set_keyframe_embedding(row["id"], emb)
