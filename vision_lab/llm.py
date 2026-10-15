"""Local language/vision models: Ollama for text and image questions, SigLIP for search."""
import base64
import os
import json
import re
import threading

import cv2
import httpx
import numpy as np

OLLAMA = "http://localhost:11434"
# Default local model for captions / Q&A / voice. Switch back with VLAB_MODEL=gemma3:27b.
DEFAULT_MODEL = os.environ.get("VLAB_MODEL", "gemma4:26b")


def _jpeg_b64(frame, max_side=896):
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def chat(model, prompt, frame=None, json_mode=False, system=None, timeout=180):
    """One-shot chat with an Ollama model; returns the reply text."""
    msg = {"role": "user", "content": prompt}
    if frame is not None:
        msg["images"] = [_jpeg_b64(frame)]
    messages = ([{"role": "system", "content": system}] if system else []) + [msg]
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0.2}}
    if json_mode:
        body["format"] = "json"
    r = httpx.post(f"{OLLAMA}/api/chat", json=body, timeout=timeout)
    if r.status_code == 400 and "think" in r.text:
        body.pop("think")  # models without a thinking switch reject the flag
        r = httpx.post(f"{OLLAMA}/api/chat", json=body, timeout=timeout)
    if r.status_code == 501 and "format" in body:
        body.pop("format")  # e.g. gemma4: "structured output is unavailable"; parse JSON from text instead
        r = httpx.post(f"{OLLAMA}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    text = r.json()["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def extract_json(text):
    """Parse JSON from model text: strips code fences, falls back to the first {...} block."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def describe_scene(model, frame):
    prompt = (
        "You are watching a home camera. Describe this frame for an activity log. "
        'Reply as JSON: {"description": "<2 short sentences: who/what is present and what '
        'they are doing>", "activity": "<one or two words>", "people": <int>, "pets": <int>}')
    raw = chat(model, prompt, frame, json_mode=True)
    try:
        d = extract_json(raw)
        return {"description": str(d.get("description", "")).strip(),
                "activity": str(d.get("activity", "")).strip(),
                "people": d.get("people"), "pets": d.get("pets")}
    except (json.JSONDecodeError, AttributeError):
        return {"description": raw, "activity": "", "people": None, "pets": None}


def list_models():
    try:
        r = httpx.get(f"{OLLAMA}/api/tags", timeout=5)
        return r.json().get("models", [])
    except httpx.HTTPError:
        return []


def loaded_models():
    try:
        return httpx.get(f"{OLLAMA}/api/ps", timeout=5).json().get("models", [])
    except httpx.HTTPError:
        return []


def capabilities(model):
    try:
        r = httpx.post(f"{OLLAMA}/api/show", json={"model": model}, timeout=10)
        return r.json().get("capabilities", [])
    except httpx.HTTPError:
        return []


def pull(model):
    r = httpx.post(f"{OLLAMA}/api/pull", json={"model": model, "stream": False}, timeout=None)
    r.raise_for_status()


def _features(out):
    """transformers 5 returns a model output (pooler_output); older versions a tensor."""
    t = getattr(out, "pooler_output", out)
    return t[0]


class Embedder:
    """SigLIP image/text embeddings for searching keyframes with a phrase."""
    MODEL = "google/siglip-base-patch16-224"

    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.processor = None
        self.error = None

    def _load(self):
        if self.model is None and self.error is None:
            try:
                import torch
                from transformers import AutoModel, AutoProcessor
                self.device = "mps" if torch.backends.mps.is_available() else "cpu"
                self.processor = AutoProcessor.from_pretrained(self.MODEL)
                self.model = AutoModel.from_pretrained(self.MODEL).to(self.device).eval()
            except Exception as e:  # missing weights or packages: search falls back to captions
                self.error = str(e)
        return self.model is not None

    def image(self, bgr):
        import torch
        with self.lock:
            if not self._load():
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                v = _features(self.model.get_image_features(**inputs))
            v = v / v.norm()
            return v.float().cpu().numpy()

    def text(self, q):
        import torch
        with self.lock:
            if not self._load():
                return None
            inputs = self.processor(text=[f"a photo of {q}"], padding="max_length",
                                    return_tensors="pt").to(self.device)
            with torch.no_grad():
                v = _features(self.model.get_text_features(**inputs))
            v = v / v.norm()
            return v.float().cpu().numpy()

    def search(self, q, items, k=12):
        """items: [(id, ts, caption, emb)]; returns [(id, ts, caption, score)]."""
        t = self.text(q)
        if t is None or not items:
            return []
        items = [it for it in items if it[3].shape == t.shape]  # skip stale/placeholder vectors
        if not items:
            return []
        mat = np.stack([it[3] for it in items])
        sims = mat @ t
        order = np.argsort(-sims)[:k]
        return [(items[i][0], items[i][1], items[i][2], float(sims[i])) for i in order]
