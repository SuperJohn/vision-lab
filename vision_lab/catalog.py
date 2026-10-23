"""Model catalog for the Model lab tab: what each model does in Vision Lab, whether it's in use, and benchmarks.

Three kinds of facts, kept apart on purpose:
  * ENTRIES  - static: plain-language task / "what you get", the feature that would use the model, and PUBLISHED
               benchmark numbers with a source link (never measured here, never estimated).
  * usage()  - derived at request time from the live config (engine settings, voice settings, batch / plan-loop
               config) plus what is loaded right now. Pure function of a context dict, so tests can feed fakes.
  * MEASURED - numbers measured on this Mac by tools/bench_catalog.py (synthetic inputs only), stored in
               catalog_measured.json with the date and machine. Missing = "not measured"; nothing is guessed.
"""
import json
from pathlib import Path

MEASURED_PATH = Path(__file__).resolve().parent / "catalog_measured.json"

# in-use states shown in the catalog
ACTIVE, ON_DEMAND, IDLE = "active", "on_demand", "idle"
STATUS_LABEL = {ACTIVE: "In use", ON_DEMAND: "On demand", IDLE: "Not in use"}

U_YOLO = "https://docs.ultralytics.com/models/yolo11/"
U_WORLD = "https://docs.ultralytics.com/models/yolo-world/"


def pub(label, value, source, url, better="higher", note=None):
    return {"label": label, "value": value, "kind": "published", "source": source, "url": url,
            "better": better, "note": note}


ENTRIES = [
    # ---------------- camera ----------------
    {"key": "yolo11n", "group": "Camera", "name": "YOLO11n", "kind": "yolo", "hw": "GPU",
     "task": "Spots people and everyday objects on the camera, fastest option",
     "gives": "Boxes and labels for 80 kinds of things (person, dog, cup, laptop…) at full frame rate; "
              "the plan loop also uses it to check whether you're at your desk",
     "published": [pub("COCO accuracy (mAP50-95)", "39.5", "Ultralytics", U_YOLO),
                   pub("Size", "2.6M params · 6.5 GFLOPs", "Ultralytics", U_YOLO, better=None)]},
    {"key": "yolo11s", "group": "Camera", "name": "YOLO11s", "kind": "yolo", "hw": "GPU",
     "task": "Spots people and everyday objects on the camera, more accurate",
     "gives": "Live boxes and labels for the 80 everyday classes when outlines are off; also supplies the class "
              "list used by rules and the home-class filter",
     "published": [pub("COCO accuracy (mAP50-95)", "47.0", "Ultralytics", U_YOLO),
                   pub("Size", "9.4M params · 21.6 GFLOPs", "Ultralytics", U_YOLO, better=None)]},
    {"key": "yolov8s-worldv2", "group": "Camera", "name": "YOLO-World v2 (s)", "kind": "yolo", "hw": "GPU",
     "task": "Finds the things you type, not just the 80 built-in classes",
     "gives": "Live boxes for your own list, e.g. \"headphones, framed print\" (Detector → YOLO-World)",
     "published": [pub("COCO zero-shot (mAP50-95)", "37.7", "Ultralytics", U_WORLD)]},
    {"key": "yolo11s-seg", "group": "Camera", "name": "YOLO11s-seg", "kind": "yolo", "hw": "GPU",
     "task": "Outlines each object on the camera instead of just boxing it",
     "gives": "Live object outlines (the Segment stage); the overnight batch also uses it to tag what's in "
              "saved frames",
     "published": [pub("COCO mask accuracy (mAP50-95)", "37.8", "Ultralytics", U_YOLO),
                   pub("COCO box accuracy (mAP50-95)", "46.6", "Ultralytics", U_YOLO),
                   pub("Size", "10.1M params · 33.1 GFLOPs", "Ultralytics", U_YOLO, better=None)]},
    {"key": "sam2.1_b", "group": "Camera", "name": "SAM 2.1 base+", "kind": "sam", "hw": "GPU",
     "task": "Outlines whatever you click when drawing zones",
     "gives": "A precise outline of one clicked object, turned into a zone (Zones → Click to segment)",
     "published": [pub("SA-V video segmentation (J&F)", "78.2", "SAM 2 repo",
                       "https://github.com/facebookresearch/sam2#sam-21-checkpoints"),
                   pub("Size", "80.8M params", "SAM 2 repo",
                       "https://github.com/facebookresearch/sam2#sam-21-checkpoints", better=None)]},
    {"key": "bytetrack", "group": "Camera", "name": "ByteTrack", "kind": "builtin", "hw": "CPU",
     "task": "Keeps the same ID on each person or object from frame to frame",
     "gives": "Stable IDs, so zone entries, time-in-view and rules count one person once, not every frame",
     "published": [pub("MOT17 tracking (MOTA / IDF1)", "80.3 / 77.3", "ByteTrack paper",
                       "https://arxiv.org/abs/2110.06864", note="with the paper's YOLOX-X detector")]},
    {"key": "yolo11s-pose", "group": "Camera", "name": "YOLO11s-pose", "kind": "yolo", "hw": "GPU",
     "task": "Reads body position: sitting, standing, lying down",
     "gives": "17 body keypoints per person, used for posture in events and rules like \"standing at the door\"",
     "published": [pub("COCO keypoints (mAP50-95)", "58.9", "Ultralytics", U_YOLO),
                   pub("Size", "9.9M params", "Ultralytics", U_YOLO, better=None)]},
    {"key": "depth", "group": "Camera", "name": "Depth Anything V2 Small", "kind": "depth", "hw": "GPU",
     "task": "Estimates how far away each part of the room is",
     "gives": "A near/far heat map over the camera view (the Depth stage, off by default)",
     "published": [pub("Size", "24.8M params", "Depth Anything V2",
                       "https://github.com/DepthAnything/Depth-Anything-V2#pre-trained-models", better=None)]},
    # ---------------- search ----------------
    {"key": "siglip", "group": "Search", "name": "SigLIP base", "kind": "siglip", "hw": "GPU",
     "task": "Lets you search saved camera keyframes by describing them",
     "gives": "Timeline search like \"red mug on the desk\"; every new keyframe is indexed in the background",
     "published": [pub("ImageNet zero-shot accuracy", "76.2%", "SigLIP paper, Table 3",
                       "https://arxiv.org/abs/2303.15343"),
                   pub("COCO text→image recall@1", "47.2%", "SigLIP paper, Table 3",
                       "https://arxiv.org/abs/2303.15343")]},
    {"key": "minilm", "group": "Search", "name": "all-MiniLM-L6-v2", "kind": "info", "hw": "CPU",
     "task": "Finds moments in your narration by meaning, not just exact words",
     "gives": "`vlab find \"fixing the printer\"` also matches \"repairing the printer\"",
     "published": [pub("Sentence-embedding score (14 sets)", "68.06", "SBERT",
                       "https://www.sbert.net/docs/sentence_transformer/pretrained_models.html"),
                   pub("Semantic search score (6 sets)", "49.54", "SBERT",
                       "https://www.sbert.net/docs/sentence_transformer/pretrained_models.html")]},
    # ---------------- voice ----------------
    {"key": "whisper-large-v3-turbo", "group": "Voice", "name": "Whisper large-v3-turbo (MLX)", "kind": "info",
     "hw": "GPU",
     "task": "Turns your speech into text",
     "gives": "The Voice log transcript with timestamps (recording mode) and what you say in voice chat",
     "published": [pub("Word error rate (Open ASR avg)", "7.83%", "Hugging Face model card",
                       "https://huggingface.co/openai/whisper-large-v3-turbo", better="lower"),
                   pub("Size", "809M params", "Hugging Face model card",
                       "https://huggingface.co/openai/whisper-large-v3-turbo", better=None)]},
    {"key": "silero-vad", "group": "Voice", "name": "Silero VAD", "kind": "info", "hw": "CPU",
     "task": "Notices when you start and stop talking in voice chat",
     "gives": "Hands-free turn taking: replies start after a short pause, and you can talk over a reply",
     "published": [pub("Speed (published)", "< 1 ms per 30 ms chunk, 1 CPU thread", "Silero VAD README",
                       "https://github.com/snakers4/silero-vad", better=None)]},
    {"key": "kokoro", "group": "Voice", "name": "Kokoro 82M", "kind": "info", "hw": "CPU",
     "task": "Speaks voice-chat replies out loud",
     "gives": "The spoken reply in voice chat, on this Mac (voice: af_heart by default)",
     "published": [pub("Size", "82M params", "Kokoro model card", "https://huggingface.co/hexgrad/Kokoro-82M",
                       better=None)]},
    {"key": "ast-audioset", "group": "Voice", "name": "AST (AudioSet)", "kind": "info", "hw": "GPU",
     "task": "Recognizes background sounds during voice chat",
     "gives": "Ambient events like doorbell, dog bark or music (off unless Ambient sounds is on)",
     "published": [pub("AudioSet accuracy (mAP)", "0.459", "AST repo", "https://github.com/YuanGongND/ast")]},
]

# Published numbers for Ollama models we know about (looked up by Ollama tag prefix).
LLM_PUBLISHED = {
    "gemma4:26b": [pub("MMMU Pro (image reasoning)", "73.8%", "Gemma 4 model card",
                       "https://ai.google.dev/gemma/docs/core/model_card_4"),
                   pub("MMLU Pro (knowledge)", "82.6%", "Gemma 4 model card",
                       "https://ai.google.dev/gemma/docs/core/model_card_4"),
                   pub("Size", "25.2B params, 3.8B active (MoE)", "Gemma 4 model card",
                       "https://ai.google.dev/gemma/docs/core/model_card_4", better=None)],
}

LLM_FEATURES = [  # (context key that names the model, feature label, status when it matches)
    ("describe_model", "Scene captions on the Video tab", "describe"),
    ("summary_model", "Today, summarized + Ask about today", ON_DEMAND),
    ("voice_llm", "Voice chat replies", ON_DEMAND),
    ("batch_model", "Screen and keyframe captions (background batch)", "batch"),
    ("planloop_model", "Plan-loop screen captions", "planloop"),
]


def _stages(ctx):
    return ctx.get("settings", {}).get("stages", {})


def usage(key, ctx):
    """-> {"status": active|on_demand|idle, "label", "where": [feature strings], "loaded": bool|None,
           "loaded_where": [..]}. ctx is built by server.catalog_context(); every field is optional."""
    s = ctx.get("settings", {})
    st = _stages(ctx)
    det = s.get("detector", "")
    detect_on = bool(st.get("detect"))
    world = "world" in det
    seg_on = bool(st.get("segment")) and not world
    vs = ctx.get("voice", {})
    procs = ctx.get("processes", {})
    cache = set(ctx.get("engine_cache", []))
    voice_loaded = set(ctx.get("voice_loaded", []))
    where, loaded_where, status = [], [], IDLE

    def use(feature, how=ACTIVE):
        nonlocal status
        where.append(feature)
        if how == ACTIVE:
            status = ACTIVE
        elif status == IDLE:
            status = how

    if key == "yolo11n":
        if detect_on and det == "yolo11n":
            use("Live camera detector" + (" (with outlines: yolo11n-seg)" if seg_on else ""))
        pl = ctx.get("planloop", {})
        if pl.get("camera_probe") and str(pl.get("probe_model", "")).endswith("yolo11n.pt"):
            use("Plan loop: at-desk check on camera samples", ACTIVE if procs.get("planloop") else ON_DEMAND)
            if procs.get("planloop"):
                loaded_where.append("plan-loop process")
    elif key == "yolo11s":
        if detect_on and det == "yolo11s" and not seg_on:
            use("Live camera detector")
        use("Class list for rules and the home-class filter (names only)", ON_DEMAND)
    elif key == "yolo11s-seg":
        if detect_on and det == "yolo11s" and seg_on:
            use("Live camera detector + outlines (Segment stage)")
        b = ctx.get("batch", {})
        if b.get("detect") and str(b.get("detect_model", "")).endswith("yolo11s-seg.pt"):
            use("Background batch: tags objects in saved frames", ACTIVE if procs.get("batch") else ON_DEMAND)
            if procs.get("batch"):
                loaded_where.append("batch process (while a batch runs)")
    elif key == "yolov8s-worldv2":
        if detect_on and world:
            use("Live camera detector (your own classes)")
    elif key == "sam2.1_b":
        use("Zones → Click to segment", ON_DEMAND)
    elif key == "bytetrack":
        if detect_on:
            use("Live tracking of every detection")
    elif key == "yolo11s-pose":
        if detect_on and st.get("pose"):
            use("Posture for people on the camera (runs when a person is in view)")
    elif key == "depth":
        if st.get("depth"):
            use("Depth stage on the Video tab")
    elif key == "siglip":
        use("Indexes each new keyframe in the background")
        use("Timeline search", ON_DEMAND)
        if ctx.get("embedder_loaded"):
            loaded_where.append("Vision Lab server")
    elif key == "minilm":
        use("`vlab find` narration search (Terminal)", ON_DEMAND)
    elif key == "whisper-large-v3-turbo":
        stt = vs.get("stt_model", "")
        if stt.endswith("whisper-large-v3-turbo"):
            if ctx.get("voice_log_recording"):
                use("Voice log transcription (recording mode, mic on)")
            else:
                use("Voice log transcription (when recording with the mic on)", ON_DEMAND)
            use("Voice chat speech-to-text", ON_DEMAND)
        if ctx.get("whisper_loaded") or "stt" in voice_loaded:
            loaded_where.append("Vision Lab server")
    elif key == "silero-vad":
        use("Voice chat turn detection", ON_DEMAND)
        if "vad" in voice_loaded:
            loaded_where.append("Vision Lab server")
    elif key == "kokoro":
        if vs.get("tts_engine", "kokoro") == "kokoro":
            use("Voice chat spoken replies", ON_DEMAND)
        if "tts" in voice_loaded:
            loaded_where.append("Vision Lab server")
    elif key == "ast-audioset":
        if vs.get("ambient"):
            use("Ambient sounds during voice chat", ON_DEMAND)
        if "ambient" in voice_loaded:
            loaded_where.append("Vision Lab server")
    else:  # an Ollama model
        for field, label, how in LLM_FEATURES:
            if ctx.get(field) != key:
                continue
            if how == "describe":
                how = ACTIVE if st.get("describe") else IDLE
            elif how == "batch":
                how = ACTIVE if procs.get("batch") else ON_DEMAND
            elif how == "planloop":
                how = ACTIVE if procs.get("planloop") and ctx.get("planloop", {}).get("screen_caption") else ON_DEMAND
            if how != IDLE:
                use(label, how)
        if key in set(ctx.get("ollama_loaded", [])):
            loaded_where.append("Ollama (in memory)")

    if key in cache or f"{key}#aux" in cache or (key == "depth" and "depth" in cache):
        loaded_where.insert(0, "Vision Lab server")
    if key == "yolo11n" and detect_on and det == "yolo11n" and seg_on and "yolo11n-seg" in cache:
        loaded_where.insert(0, "Vision Lab server")
    loaded = bool(loaded_where) if (ctx.get("loaded_known", True)) else None
    return {"status": status, "label": STATUS_LABEL[status], "where": where,
            "loaded": loaded, "loaded_where": sorted(set(loaded_where), key=loaded_where.index)}


def load_measured(path=MEASURED_PATH):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {"models": {}}


def benchmarks(key, measured=None, live=None):
    """Published + measured-here metrics for one model. `live` is a fresh result from the Benchmark button."""
    measured = measured if measured is not None else load_measured()
    rows = []
    m = measured.get("models", {}).get(key, {})
    for r in m.get("metrics", []):
        rows.append({**r, "kind": "measured", "date": r.get("date") or measured.get("date"),
                     "machine": measured.get("machine")})
    if live:
        rows.insert(0, {"label": "Speed on the live camera frame", "value": live.get("label"), "kind": "measured",
                        "date": "just now", "machine": measured.get("machine"),
                        "note": "Benchmark button, real camera frame"})
    entry = next((e for e in ENTRIES if e["key"] == key), None)
    published = entry["published"] if entry else LLM_PUBLISHED.get(key, [])
    rows += [dict(p) for p in published]
    return rows


def measure_note(key, measured=None):
    """Why there is no measured-here number (e.g. "not installed"), or None."""
    measured = measured if measured is not None else load_measured()
    m = measured.get("models", {}).get(key)
    if m is None:
        return None
    return None if m.get("metrics") else (m.get("error") or None)


def llm_entry(name, is_vision, installed=True):
    """Catalog row for an Ollama model."""
    return {"key": name, "group": "Language", "name": name, "kind": "ollama", "hw": "GPU",
            "task": "Describes the scene and answers questions in plain sentences" if is_vision
            else "Writes summaries and answers questions about your logs",
            "gives": "Scene captions, \"Ask about this frame\", day summaries, screen captions and voice-chat "
                     "replies, whichever features are set to use it" if is_vision
            else "Text answers for summaries and voice chat (can't look at images)",
            "installed": installed}


def benchable(entry):
    return entry.get("kind") in ("yolo", "sam", "depth", "siglip", "builtin", "ollama")
