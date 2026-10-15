"""Step 2: one multimodal record. Voice turns, interrupts and room sounds go into vlab.db next to
vision events; voice turns are linked to the nearest keyframe and a snapshot of what the camera sees;
and the voice brain gets a compact summary of recent vision data."""
import datetime as dt
import json
import re
import time

from . import voice

AUDIO_COLOR, VOICE_COLOR = "#5DC8E1", "#C4AEFF"


def store_voice_event(store, engine, ev):
    """Write one voice.py event dict to the DB. Keys make it idempotent (live writes and the JSONL import agree)."""
    kind, ts = ev.get("kind"), ev.get("ts") or time.time()
    key = f"{kind}:{ts:.6f}"
    if kind == "voice_turn":
        lat = {k: ev[k] for k in ev if k.endswith("_ms")}
        kf = store.nearest_keyframe(ts)
        snap = ev.get("snapshot")
        if snap is None and engine is not None and time.time() - ts < 30:   # live turn: take it now
            snap = engine.snapshot()
        common = dict(voice=ev.get("voice"), model=ev.get("llm"), latencies=lat,
                      interrupted=ev.get("interrupted"), error=ev.get("error"),
                      keyframe_id=kf["id"] if kf else None, snapshot=snap)
        tid = ev.get("turn_id") or key
        store.add_utterance(tid, "user", ev.get("user"), ev.get("ts_speech_start") or ts, ts,
                            source_key=key + ":user", **common)
        if ev.get("reply") or ev.get("interrupted") or ev.get("error"):
            start = ts + (ev.get("tts_first_audio_ms") or 0) / 1000
            store.add_utterance(tid, "assistant", ev.get("reply"), start, ev.get("ts_reply_end") or start,
                                source_key=key + ":assistant", **common)
    elif kind == "voice_interrupt":
        store.add_event("interrupt", f"Voice reply interrupted ({ev.get('reason')})", VOICE_COLOR, ts=ts,
                        modality="voice", data=ev, source_key=key)
    elif kind == "audio_event":
        store.add_event("sound", f"Heard {ev['label']} ({ev.get('audioset_label')}, {ev['confidence']:.0%})",
                        AUDIO_COLOR, cls=ev["label"], ts=ts, modality="audio", data=ev,
                        source_key=f"{key}:{ev['label']}")


def import_jsonl(store, path=voice.LOG):
    """One-time (and safe to repeat) import of turns logged before the DB integration."""
    n = 0
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            store_voice_event(store, None, ev)
            n += 1
    return n


_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)", re.I)


def _hm(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%-I:%M %p")


_DWELL = re.compile(r"left view after (\d+) (s|min)")


def _useful(r):
    """Keep what a person would mention; drop detector churn (short came/left-view tracks, line crossings)."""
    if r["modality"] != "vision" or r["source"] == "keyframe":
        return True
    if r["kind"] in ("scene", "zone", "rule", "posture", "power", "camera_paused", "camera_resumed", "capture_paused", "capture_resumed"):
        return r["kind"] != "scene" or r["text"].startswith("Activity")
    if r["kind"] == "track" and (r.get("label") in ("person", "dog")):
        d = _DWELL.search(r["text"] or "")
        return bool(d) and (d.group(2) == "min")
    return False


def vision_context(store, engine, question="", max_chars=2400):
    """Compact plain-text summary of what the camera knows, for the voice brain's system prompt."""
    now = time.time()
    day0 = dt.datetime.combine(dt.date.today(), dt.time()).timestamp()
    out = []
    if engine is not None:
        snap = engine.snapshot()
        seen = ", ".join(f"{t['cls']}" + (f" in {'/'.join(t['zones'])}" if t["zones"] else "")
                         + (f" ({t['posture']})" if t.get("posture") and t["posture"] != "unknown" else "")
                         for t in snap["tracks"]) or "nothing confirmed"
        out.append(f"In view right now: {seen}." + (" Camera is paused." if snap["paused"] else ""))
        if snap["scene"]:
            out.append(f"Latest caption ({_hm(snap['scene_ts'])}): {snap['scene']}")
    spans = {}
    for s in store.spans(day0):
        if s["end"] - s["start"] >= 3:
            spans.setdefault(s["cls"], []).append((max(s["start"], day0), s["end"]))
    mins = {}
    for c, iv in spans.items():   # union of overlapping tracks, so minutes can't exceed wall time
        iv.sort()
        total, (a, b) = 0.0, iv[0]
        for x, y in iv[1:]:
            if x > b:
                total, a, b = total + b - a, x, y
            else:
                b = max(b, y)
        mins[c] = (total + b - a) / 60
    if mins:
        out.append("Time in view today: " + ", ".join(f"{c} {m:.0f} min" for c, m in
                                                      sorted(mins.items(), key=lambda x: -x[1])[:6]) + ".")
        last = {}
        for s in store.spans(now - 6 * 3600):
            last[s["cls"]] = max(last.get(s["cls"], 0), s["end"])
        out.append("Last seen: " + ", ".join(f"{c} {_hm(t)}" for c, t in last.items() if c in ("person", "dog")) + ".")
    m = _TIME.search(question or "")
    if m:   # "what was I doing at 3pm" -> the log around that time today
        h = int(m.group(1)) % 12 + (12 if m.group(3).lower().startswith("p") else 0)
        t = dt.datetime.combine(dt.date.today(), dt.time(h, int(m.group(2) or 0))).timestamp()
        if t > now:
            t -= 86400
        rows = list(reversed(store.merged(t - 1200, t + 1200, limit=40)))
        out.append(f"Log around {_hm(t)}:")
    else:
        rows = list(reversed(store.merged(now - 1800, now, limit=25)))
        out.append("Last 30 minutes:")
    lines = []
    for r in rows:
        if r["source"] == "utterance" or (r["source"] == "keyframe" and not m) or not _useful(r):
            continue
        lines.append(f"- {_hm(r['ts'])} [{r['modality']}] {r['text']}")
    out += lines[-20:] or ["- nothing notable"]
    if not m:   # coarse day outline from keyframe captions, one per half hour
        buckets = {}
        for r in reversed(store.merged(day0, now - 1800, modalities=["vision"], limit=2000)):
            if r["source"] == "keyframe":
                buckets.setdefault(int(r["ts"] // 1800), r)
        if buckets:
            out.append("Earlier today (one caption per half hour):")
            out += [f"- {_hm(r['ts'])} {r['text']}" for r in list(buckets.values())[-16:]]
    text = "\n".join(out)
    return text if len(text) <= max_chars else text[:max_chars] + "…"
