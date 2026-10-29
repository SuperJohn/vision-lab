#!/usr/bin/env python3
"""Generate the Vision Lab synthetic demo dataset under demo/data (seeded, deterministic).

A fictional one-week timeline for a made-up apartment ("Resident A" and a small dog), in the same shapes as the
local event model (docs/event-model.md): events, presence spans, keyframes, batch annotations, voice turns,
zones/rules config and daily roll-ups. Nothing here is read from a real camera, database or home.

    python3 demo/generate_data.py               # -> demo/data/*.json (byte-identical for the same seed)

demo/data/data.json (the public showcase aggregates used by site/) is not touched by this script.
"""
import hashlib, json
from pathlib import Path


def _enc(o, level=0):
    """Pretty JSON where arrays of objects/arrays put one compact record per line (small, diff-friendly)."""
    pad, pad1 = " " * level, " " * (level + 1)
    if isinstance(o, dict):
        if not o:
            return "{}"
        items = [f'{pad1}{json.dumps(k, ensure_ascii=False)}: {_enc(v, level + 1)}' for k, v in o.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(o, list) and o and all(isinstance(x, (dict, list)) for x in o):
        rows = [pad1 + json.dumps(x, ensure_ascii=False, separators=(", ", ": ")) for x in o]
        return "[\n" + ",\n".join(rows) + "\n" + pad + "]"
    return json.dumps(o, ensure_ascii=False)


def write(path: Path, obj) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _enc(obj) + "\n"
    json.loads(text)  # must round-trip
    path.write_text(text, encoding="utf-8")
    b = text.encode()
    return {"name": path.name, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
# ---------------------------------------------------------------------------------------------------------------
import argparse
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

SEED = 42
TZ = ZoneInfo("America/Los_Angeles")
DAYS = 7
FIRST_DAY = datetime(2026, 10, 19, tzinfo=TZ)  # Mon 19 Oct .. Sun 25 Oct 2026
HOME = "Demo Apartment (fictional)"

ZONES = [
    {"id": "z-couch", "name": "Couch", "kind": "zone", "points": [[0.05, 0.55], [0.42, 0.55], [0.42, 0.92], [0.05, 0.92]]},
    {"id": "z-desk", "name": "Desk", "kind": "zone", "points": [[0.62, 0.40], [0.96, 0.40], [0.96, 0.85], [0.62, 0.85]]},
    {"id": "z-kitchen", "name": "Kitchen", "kind": "zone", "points": [[0.40, 0.18], [0.70, 0.18], [0.70, 0.50], [0.40, 0.50]]},
    {"id": "z-dogbed", "name": "Dog bed", "kind": "zone", "points": [[0.20, 0.88], [0.34, 0.88], [0.34, 0.99], [0.20, 0.99]]},
    {"id": "l-door", "name": "Front door", "kind": "line", "points": [[0.02, 0.20], [0.02, 0.60]]},
]
RULES = [
    {"id": "r-arrive", "name": "Someone comes in", "enabled": True, "object": "person", "event": "appears", "target": None,
     "duration_s": 0, "posture": "any", "direction": "any", "cooldown_s": 120,
     "actions": {"notify": False, "snapshot": True, "vlm_prompt": "Describe the person who just came into view and what they are carrying. One sentence."}},
    {"id": "r-floor", "name": "Person lying down", "enabled": True, "object": "person", "event": "posture_change", "target": None,
     "duration_s": 0, "posture": "lying", "direction": "any", "cooldown_s": 300, "actions": {"notify": True, "snapshot": True, "vlm_prompt": ""}},
    {"id": "r-desk-long", "name": "Desk stretch over 90 min", "enabled": True, "object": "person", "event": "in_zone_for", "target": "z-desk",
     "duration_s": 5400, "posture": "sitting", "direction": "any", "cooldown_s": 1800, "actions": {"notify": True, "snapshot": False, "vlm_prompt": ""}},
    {"id": "r-dog-kitchen", "name": "Dog in the kitchen", "enabled": True, "object": "dog", "event": "enters_zone", "target": "z-kitchen",
     "duration_s": 0, "posture": "any", "direction": "any", "cooldown_s": 600, "actions": {"notify": False, "snapshot": True, "vlm_prompt": "Is the dog near any food? One sentence."}},
    {"id": "r-door", "name": "Front door crossing", "enabled": True, "object": "person", "event": "crosses_line", "target": "l-door",
     "duration_s": 0, "posture": "any", "direction": "any", "cooldown_s": 60, "actions": {"notify": False, "snapshot": True, "vlm_prompt": ""}},
]
COLORS = {"person": "#F5A524", "laptop": "#5CC8FF", "cup": "#FF8CC0", "dog": "#9BE15D", "chair": "#9DB4FF", "couch": "#E0E0E0",
          "tv": "#7EE0C3", "book": "#FF9F7A", "cell phone": "#FFD166", "refrigerator": "#B5E48C", "potted plant": "#C4AEFF"}

# Daily routine for the fictional resident ("Resident A") and a small dog. (start_h, end_h, label, zone, posture)
ROUTINE = [
    (6.8, 7.4, "kitchen", "z-kitchen", "standing"), (7.4, 8.1, "couch", "z-couch", "sitting"),
    (8.6, 10.4, "work", "z-desk", "sitting"), (10.4, 10.6, "kitchen", "z-kitchen", "standing"),
    (10.6, 12.2, "at_desk", "z-desk", "sitting"), (12.2, 12.8, "kitchen", "z-kitchen", "standing"),
    (12.8, 13.5, "away", None, None), (13.5, 15.4, "work", "z-desk", "sitting"), (15.4, 15.8, "chat", "z-desk", "sitting"),
    (15.8, 17.3, "browsing", "z-desk", "sitting"), (17.3, 18.4, "away", None, None), (18.4, 19.1, "kitchen", "z-kitchen", "standing"),
    (19.1, 21.6, "couch", "z-couch", "sitting"), (21.6, 22.3, "video", "z-couch", "lying"),
]
WEEKEND = [
    (8.0, 8.8, "kitchen", "z-kitchen", "standing"), (8.8, 11.0, "couch", "z-couch", "sitting"), (11.0, 15.5, "away", None, None),
    (15.5, 16.6, "social", "z-couch", "sitting"), (16.6, 17.4, "email_admin", "z-desk", "sitting"),
    (18.0, 18.9, "kitchen", "z-kitchen", "standing"), (18.9, 22.5, "couch", "z-couch", "sitting"),
]
CAPTIONS = {
    "empty": ["An empty open-plan living room and kitchen in soft daylight.", "The living area is empty; the couch cushions are neatly arranged.",
              "No one is in view; late-afternoon light falls across the wood floor."],
    "kitchen": ["A person stands at the kitchen counter making coffee.", "A person is rinsing dishes at the kitchen sink.",
                "Someone is preparing a meal at the counter while a small dog watches."],
    "couch": ["A person sits on the grey couch reading a paperback.", "A person relaxes on the couch with a mug in hand.",
              "A person sits on the couch with a laptop on their knees."],
    "at_desk": ["A person sits at the desk facing a monitor.", "A person at the desk is writing in a notebook next to a laptop."],
    "work": ["A person types at a desk with two monitors.", "A person at the desk is on a video call wearing headphones."],
    "browsing": ["A person leans back at the desk scrolling on a laptop.", "A person at the desk scrolls through a web page."],
    "chat": ["A person at the desk talks into a headset.", "A person waves at the laptop camera during a call."],
    "idle_screen": ["The desk is unattended; the monitor shows a screensaver."],
    "away": ["The room is empty and the front door is closed.", "The apartment is quiet; a small dog sleeps on its bed."],
    "dog_only": ["A small dog stands on the floor between the couch and the kitchen.", "A small dog is curled up on its bed near the couch."],
    "social": ["Two people sit on the couch talking and laughing.", "Two people share snacks at the coffee table."],
    "video": ["A person lies on the couch watching TV under a blanket."],
    "email_admin": ["A person at the desk sorts papers next to a laptop."],
    "door": ["A person walks through the front door carrying a grocery bag.", "A person steps out of the front door with a backpack."],
}
DETS = {"kitchen": ["person", "refrigerator", "cup"], "couch": ["person", "couch", "book"], "at_desk": ["person", "chair", "laptop"],
        "work": ["person", "chair", "laptop", "cup"], "browsing": ["person", "laptop", "chair"], "chat": ["person", "laptop", "chair"],
        "video": ["person", "couch", "tv"], "social": ["person", "person", "couch", "cup"], "email_admin": ["person", "chair", "laptop", "book"],
        "away": ["couch", "chair"], "empty": ["couch", "chair", "refrigerator"], "dog_only": ["dog", "couch"], "idle_screen": ["chair", "laptop"],
        "door": ["person"]}
VOICE = [
    ("What am I looking at right now?", "You're at your desk; I can see a laptop, a mug and a notebook. The dog is on its bed."),
    ("Is the dog in the kitchen?", "No, the dog is on its bed by the couch. Last time it went into the kitchen was about 40 minutes ago."),
    ("How long have I been at the desk?", "About an hour and twenty minutes since you sat down after lunch."),
    ("Did anyone come in while I was out?", "No one came in. The only motion was the dog moving between its bed and the kitchen."),
    ("Remind me what I was doing at three.", "At 3 PM you were on a call at the desk wearing headphones."),
    ("Pause the camera for a bit.", "Okay, the camera is paused. Say 'resume camera' when you want it back."),
    ("Resume camera.", "Camera's back on. Nothing changed in the room while it was paused."),
    ("What did I eat for lunch?", "I saw you making something at the counter around 12:15; it looked like a sandwich and a salad."),
    ("Is the kitchen light still on?", "Yes, the light above the counter looks on, and the kitchen is empty."),
    ("Summarize my afternoon.", "Mostly desk work: two long focus stretches, one call, a kitchen break at 3:30 and you headed out around 5:15."),
    ("Where did I leave my book?", "The last frame with a book shows it on the couch armrest, around 8 this morning."),
    ("Was I sitting too long today?", "You had one 95-minute desk stretch this morning; the reminder fired at 10:05."),
]


def _ts(day, hour):
    return (day + timedelta(hours=hour)).timestamp()


def build(seed=SEED):
    rng = random.Random(seed)
    events, presence, keyframes, utter, batch, daily, plan_blocks = [], [], [], [], [], [], []
    eid = kfid = pid = tid = 0

    def ev(ts, kind, text, modality="vision", cls=None, track=None, kf=None, rule=None, data=None, vlm=None):
        nonlocal eid
        eid += 1
        events.append({"id": eid, "ts": round(ts, 2), "kind": kind, "modality": modality, "written_at": round(ts + rng.uniform(0.02, 0.4), 2),
                       "track_id": track, "cls": cls, "keyframe_id": kf, "vlm_text": vlm, "rule_id": rule, "data": data or {},
                       "text": text, "color": COLORS.get(cls or "", "#E0E0E0")})
        return eid

    def keyframe(ts, label):
        nonlocal kfid
        kfid += 1
        cap = rng.choice(CAPTIONS[label])
        keyframes.append({"id": kfid, "ts": round(ts, 2), "path": f"keyframes/demo_{kfid:05d}.jpg", "caption": cap, "label": label,
                          "embedding": None})
        dets = []
        for c in DETS[label]:
            x1, y1 = rng.uniform(0.05, 0.7), rng.uniform(0.1, 0.6)
            dets.append({"cls": c, "conf": round(rng.uniform(0.52, 0.97), 2),
                         "box": [round(x1, 3), round(y1, 3), round(min(0.99, x1 + rng.uniform(0.08, 0.3)), 3), round(min(0.99, y1 + rng.uniform(0.1, 0.35)), 3)]})
        batch.append({"id": kfid, "source_type": "keyframe", "source_path": f"keyframes/demo_{kfid:05d}.jpg", "keyframe_id": kfid,
                      "capture_ts": round(ts, 2), "processed_ts": round(ts + rng.uniform(30, 600), 2), "model": "gemma4:26b",
                      "caption": cap, "label": label, "detections": dets, "gemma_ms": int(rng.gauss(905, 60)), "detect_ms": int(rng.gauss(423, 40)),
                      "status": "ok", "error": None, "sensitive": 0, "exportable": 1})
        return kfid

    for d in range(DAYS):
        day = FIRST_DAY + timedelta(days=d)
        dstr = day.strftime("%Y-%m-%d")
        weekend = day.weekday() >= 5
        sched = WEEKEND if weekend else ROUTINE
        counts, labels, vlm_ms, desk_s, couch_s = {}, {}, [], 0.0, 0.0
        bump = lambda k: counts.__setitem__(k, counts.get(k, 0) + 1)
        paused = None
        if d == 2:
            paused = (14.2, 14.7)  # privacy pause mid-afternoon on Wednesday
        for (h0, h1, label, zone, posture) in sched:
            j0, j1 = h0 + rng.uniform(-0.12, 0.12), h1 + rng.uniform(-0.12, 0.12)
            t0, t1 = _ts(day, j0), _ts(day, j1)
            labels[label] = labels.get(label, 0) + int((j1 - j0) * 12)  # one activity slice per 5 min
            if label == "away":
                ev(t0, "line", "person #%d crossed Front door (out)" % (tid + 1), cls="person", track=tid + 1, data={"line": "l-door", "direction": "out"}); bump("line")
                k = keyframe(t0 + 3, "door")
                ev(t0 + 3, "rule", "Front door crossing: person out", cls="person", track=tid + 1, kf=k, rule="r-door"); bump("rule")
                # dog wanders while resident is away
                for _ in range(rng.randint(1, 3)):
                    tid += 1
                    s = rng.uniform(t0, t1 - 300)
                    pid += 1; presence.append({"id": pid, "track_id": tid, "cls": "dog", "start": round(s, 2), "end": round(s + rng.uniform(60, 400), 2)})
                    ev(s, "track", f"dog #{tid} appeared", cls="dog", track=tid); bump("track")
                    if rng.random() < 0.5:
                        k = keyframe(s + 20, "dog_only")
                        ev(s + 20, "zone", f"dog #{tid} entered Kitchen", cls="dog", track=tid, data={"zone": "z-kitchen", "dir": "enter"}); bump("zone")
                        ev(s + 21, "rule", f"Dog in the kitchen: dog #{tid} entered", cls="dog", track=tid, kf=k, rule="r-dog-kitchen",
                           vlm="The dog is sniffing the floor; no food is within reach."); bump("rule")
                ev(t1, "line", "person crossed Front door (in)", cls="person", data={"line": "l-door", "direction": "in"}); bump("line")
                k = keyframe(t1 + 2, "door")
                ev(t1 + 2, "rule", "Someone comes in: person appeared", cls="person", kf=k, rule="r-arrive",
                   vlm=rng.choice(["A person comes in carrying a tote bag.", "A person walks in holding a coffee cup and keys."])); bump("rule")
                continue
            tid += 1
            pid += 1; presence.append({"id": pid, "track_id": tid, "cls": "person", "start": round(t0, 2), "end": round(t1, 2)})
            ev(t0, "track", f"person #{tid} appeared", cls="person", track=tid); bump("track")
            if zone:
                zn = next(z["name"] for z in ZONES if z["id"] == zone)
                ev(t0 + 4, "zone", f"person #{tid} entered {zn}", cls="person", track=tid, data={"zone": zone, "dir": "enter"}); bump("zone")
            if posture:
                ev(t0 + 6, "posture", f"person #{tid} is {posture}", cls="person", track=tid, data={"from": "unknown", "to": posture}); bump("posture")
                if posture == "lying":
                    k = keyframe(t0 + 7, label)
                    ev(t0 + 7, "rule", f"Person lying down: person #{tid} on Couch", cls="person", track=tid, kf=k, rule="r-floor"); bump("rule")
            if zone == "z-desk":
                desk_s += t1 - t0
                if t1 - t0 >= 5400:
                    ev(t0 + 5400, "rule", f"Desk stretch over 90 min: person #{tid}", cls="person", track=tid, rule="r-desk-long"); bump("rule")
                    plan_blocks.append({"id": f"pb-{dstr}-{len(plan_blocks)}", "day": dstr, "task": "Stand up and stretch", "linear_issue": None,
                                        "planned_start": round(t0 + 5400, 0), "planned_end": round(t0 + 5700, 0), "status": "nudged"})
            if zone == "z-couch":
                couch_s += t1 - t0
            # scene captions every ~20 min, keyframes ~ every 40 min
            t = t0 + rng.uniform(60, 300)
            while t < t1 - 60:
                if paused and _ts(day, paused[0]) <= t <= _ts(day, paused[1]):
                    t += 600; continue
                k = keyframe(t, label) if rng.random() < 0.5 else None
                cap = keyframes[-1]["caption"] if k else rng.choice(CAPTIONS[label])
                ms = int(rng.gauss(905, 70)); vlm_ms.append(ms)
                ev(t, "scene", cap, kf=k, data={"label": label, "vlm_ms": ms, "model": "gemma4:26b"}, vlm=cap); bump("scene")
                if rng.random() < 0.35:
                    obj = rng.choice([c for c in DETS[label] if c != "person"] or ["cup"])
                    tid += 1
                    ev(t + 5, "track", f"{obj} #{tid} appeared", cls=obj, track=tid); bump("track")
                    pid += 1; presence.append({"id": pid, "track_id": tid, "cls": obj, "start": round(t + 5, 2), "end": round(t + rng.uniform(120, 1500), 2)})
                ev(t + 10, "activity_slice", f"activity: {label}", modality="activity", data={"label": label, "minutes": 5}); bump("activity_slice")
                t += rng.uniform(900, 1500)
            if zone:
                ev(t1 - 2, "zone", f"person #{tid} left {zn}", cls="person", track=tid, data={"zone": zone, "dir": "leave"}); bump("zone")
        if paused:
            ev(_ts(day, paused[0]), "capture_paused", "Privacy mode on: camera paused", modality="vision", data={"by": "voice"}); bump("capture_paused")
            ev(_ts(day, paused[1]), "capture_resumed", "Privacy mode off: camera resumed", modality="vision", data={"by": "voice"}); bump("capture_resumed")
        # night: empty room, a few sounds
        for h in (0.5, 2.0, 3.5, 5.0):
            t = _ts(day, h + rng.uniform(0, 1))
            ev(t, "scene", rng.choice(CAPTIONS["empty"] + CAPTIONS["dog_only"][1:]), data={"label": "empty", "vlm_ms": int(rng.gauss(905, 70))}); bump("scene")
            labels["empty"] = labels.get("empty", 0) + 18
        if rng.random() < 0.6:
            ev(_ts(day, rng.uniform(1, 4)), "sound", "sound: dog barking", modality="audio", data={"label": "Dog bark", "confidence": round(rng.uniform(0.6, 0.9), 2)}); bump("sound")
        vs = sorted(vlm_ms) or [0]
        daily.append({"day": dstr, "weekday": day.strftime("%a"), "events_by_kind": dict(sorted(counts.items())),
                      "activity_minutes_by_label": {k: v * 5 for k, v in sorted(labels.items(), key=lambda x: -x[1])},
                      "desk_hours": round(desk_s / 3600, 2), "couch_hours": round(couch_s / 3600, 2),
                      "vlm_p50_ms": vs[len(vs) // 2], "vlm_p95_ms": vs[int(len(vs) * 0.95) - 1 if len(vs) > 1 else 0], "vlm_calls": len(vlm_ms),
                      "privacy_pauses": 1 if paused else 0, "uptime_pct": round(rng.uniform(98.6, 99.9), 1)})

    # voice turns, each linked to the keyframe nearest in time
    kf_ts = [(k["ts"], k["id"]) for k in keyframes]
    for i, (q, a) in enumerate(VOICE):
        day = FIRST_DAY + timedelta(days=i % DAYS)
        t = _ts(day, rng.choice([9.2, 10.1, 11.4, 14.6, 15.1, 16.2, 20.3]) + rng.uniform(0, 0.4))
        near = min(kf_ts, key=lambda x: abs(x[0] - t))[1]
        turn = f"turn-{i + 1:03d}"
        lat = {"stt_ms": int(rng.gauss(310, 40)), "llm_ttft_ms": int(rng.gauss(180, 30)), "tts_first_audio_ms": int(rng.gauss(240, 35))}
        utter.append({"id": 2 * i + 1, "turn_id": turn, "role": "user", "text": q, "ts_start": round(t, 2), "ts_end": round(t + 2.4, 2),
                      "voice": None, "model": "whisper-large-v3-turbo", "latencies": None, "interrupted": 0, "error": None, "keyframe_id": near})
        utter.append({"id": 2 * i + 2, "turn_id": turn, "role": "assistant", "text": a, "ts_start": round(t + 3.0, 2), "ts_end": round(t + 7.5, 2),
                      "voice": "demo-voice-1", "model": "gemma4:26b", "latencies": lat, "interrupted": int(i == 5), "error": None, "keyframe_id": near})
        ev(t, "voice", q, modality="voice", kf=near, data={"turn_id": turn})
    events.sort(key=lambda e: (e["ts"], e["id"]))
    for i, e in enumerate(events, 1):
        e["id"] = i  # rowid order == time order, like the live table
    return {"events": events, "presence": presence, "keyframes": keyframes, "utterances": utter, "batch": batch,
            "daily": daily, "plan_blocks": plan_blocks}


def main():
    ap = argparse.ArgumentParser(description="Generate the Vision Lab synthetic demo dataset (seeded, deterministic).")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    out = Path(a.out)
    d = build(a.seed)
    meta = {"synthetic": True, "home": HOME, "seed": a.seed, "first_day": FIRST_DAY.strftime("%Y-%m-%d"),
            "last_day": (FIRST_DAY + timedelta(days=DAYS - 1)).strftime("%Y-%m-%d"), "tz": "America/Los_Angeles",
            "note": "All rows are generated from a fixed seed. No camera frames, voices or people from real homes."}
    files = [
        write(out / "config.json", {**meta, "zones": ZONES, "rules": RULES}),
        write(out / "events.json", {**meta, "count": len(d["events"]), "events": d["events"]}),
        write(out / "presence.json", {**meta, "count": len(d["presence"]), "presence": d["presence"]}),
        write(out / "keyframes.json", {**meta, "count": len(d["keyframes"]), "keyframes": d["keyframes"]}),
        write(out / "batch_annotations.json", {**meta, "count": len(d["batch"]), "batch_annotations": d["batch"]}),
        write(out / "utterances.json", {**meta, "count": len(d["utterances"]), "utterances": d["utterances"]}),
        write(out / "daily_metrics.json", {**meta, "days": d["daily"]}),
        write(out / "plan_blocks.json", {**meta, "plan_blocks": d["plan_blocks"]}),
    ]
    counts = {"events": len(d["events"]), "presence": len(d["presence"]), "keyframes": len(d["keyframes"]),
              "batch_annotations": len(d["batch"]), "utterances": len(d["utterances"]), "days": len(d["daily"]),
              "zones": len(ZONES), "rules": len(RULES), "plan_blocks": len(d["plan_blocks"])}
    write(out / "manifest.json", {"app": "vision-lab", **meta, "generator": "demo/generate_data.py", "rows": counts,
                                  "unchanged": ["data.json (public showcase aggregates; served as site/assets/data.json)"], "files": files})
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
