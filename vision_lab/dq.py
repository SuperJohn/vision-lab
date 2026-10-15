"""Data-quality checks over vlab.db (HOU-85). Read-only; prints aggregate metrics, never row content.

    python -m vision_lab.dq --db data/vlab.db [--conf 0.5] [--json]

Checks
  couch_fp     couch false positives in slow-loop detections (batch_annotations):
                 * on_screen  couch on a screen capture (a desktop screenshot can't contain a real couch)
                 * on_bed     couch box overlapping a bed box in the same frame (bed/couch confusion)
                 * low_conf   couch below the confidence threshold
  dup_person   duplicate person IDs:
                 * frame      two person boxes in one frame with IoU >= --iou (NMS miss)
                 * track      two person tracks starting within --window s of each other and overlapping;
                              confirmed when both tracks' keyframe embeddings have cosine >= --sim,
                              otherwise reported as 'unverified' (embedding missing -> degrade, don't guess)
"Before" = all detections; "after" = only detections with conf >= --conf.
"""
import argparse, json, sqlite3, sys
from itertools import combinations

import numpy as np


def connect_ro(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _frames(con):
    for r in con.execute("SELECT source_type, detections FROM batch_annotations WHERE status='ok' AND detections IS NOT NULL"):
        try:
            dets = json.loads(r["detections"]) or []
        except ValueError:
            continue
        yield r["source_type"], [d for d in dets if isinstance(d, dict) and len(d.get("box") or []) == 4]


def _rate(n, d):
    return round(n / d, 4) if d else None


def couch_fp(con, conf=0.5, bed_iou=0.3):
    out = {}
    for label, keep in (("before", lambda d: True), ("after", lambda d: d.get("conf", 0) >= conf)):
        total = on_screen = on_bed = low = flagged = 0
        for st, dets in _frames(con):
            dets = [d for d in dets if keep(d)]
            beds = [d["box"] for d in dets if d.get("cls") == "bed"]
            for d in dets:
                if d.get("cls") != "couch":
                    continue
                total += 1
                f_screen = st == "screen"
                f_bed = any(iou(d["box"], b) >= bed_iou for b in beds)
                f_low = d.get("conf", 0) < conf
                on_screen += f_screen; on_bed += f_bed; low += f_low
                flagged += f_screen or f_bed or f_low
        out[label] = {"couch_detections": total, "on_screen": on_screen, "on_bed": on_bed, "low_conf": low,
                      "flagged": flagged, "fp_rate": _rate(flagged, total)}
    return out


def _person_frame_dups(con, thr, conf):
    frames = dups = pairs = 0
    for _, dets in _frames(con):
        ppl = [d["box"] for d in dets if d.get("cls") == "person" and d.get("conf", 0) >= conf]
        if len(ppl) < 2:
            frames += bool(ppl)
            continue
        frames += 1
        n = sum(iou(a, b) >= thr for a, b in combinations(ppl, 2))
        pairs += n; dups += bool(n)
    return {"frames_with_person": frames, "frames_with_dup": dups, "dup_pairs": pairs, "dup_frame_rate": _rate(dups, frames)}


def _track_embeddings(con):
    """track_id -> unit embedding of a keyframe linked to that track (events.keyframe_id), if any."""
    emb = {}
    try:
        rows = con.execute("SELECT e.track_id, k.embedding FROM events e JOIN keyframes k ON k.id = e.keyframe_id "
                           "WHERE e.track_id IS NOT NULL AND e.cls='person' AND k.embedding IS NOT NULL")
        for r in rows:
            if r["track_id"] in emb:
                continue
            v = np.frombuffer(r["embedding"], dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n > 0:
                emb[r["track_id"]] = v / n
    except sqlite3.Error:
        pass
    return emb


def _person_track_dups(con, window, sim):
    spans = con.execute("SELECT track_id, start, COALESCE(end, start) AS end FROM presence "
                        "WHERE cls='person' AND start IS NOT NULL ORDER BY start").fetchall()
    emb = _track_embeddings(con)
    cand = confirmed = rejected = unverified = 0
    j0 = 0
    for i, a in enumerate(spans):
        while spans[j0]["start"] < a["start"] - window:
            j0 += 1
        for b in spans[i + 1:]:
            if b["start"] - a["start"] > window:
                break
            if b["track_id"] == a["track_id"] or b["start"] > a["end"]:
                continue
            cand += 1
            ea, eb = emb.get(a["track_id"]), emb.get(b["track_id"])
            if ea is None or eb is None:
                unverified += 1
            elif float(ea @ eb) >= sim:
                confirmed += 1
            else:
                rejected += 1
    return {"person_tracks": len(spans), "candidate_pairs": cand, "confirmed": confirmed, "rejected": rejected,
            "unverified": unverified, "candidate_rate": _rate(cand, len(spans)), "confirmed_rate": _rate(confirmed, len(spans)),
            "embedding_coverage": _rate(sum(1 for s in spans if s["track_id"] in emb), len(spans))}


def dup_person(con, iou_thr=0.7, conf=0.5, window=2.0, sim=0.95):
    return {"frame": {"before": _person_frame_dups(con, iou_thr, 0.0), "after": _person_frame_dups(con, iou_thr, conf)},
            "track": _person_track_dups(con, window, sim)}


def run(path, conf=0.5, iou_thr=0.7, window=2.0, sim=0.95):
    con = connect_ro(path)
    try:
        return {"params": {"conf": conf, "iou": iou_thr, "window_s": window, "sim": sim},
                "couch_fp": couch_fp(con, conf), "dup_person": dup_person(con, iou_thr, conf, window, sim)}
    finally:
        con.close()


def metrics_table(res):
    c, f, t = res["couch_fp"], res["dup_person"]["frame"], res["dup_person"]["track"]
    rows = [("couch_fp", "couch detections", c["before"]["couch_detections"], c["after"]["couch_detections"]),
            ("couch_fp", "flagged (screen/bed/low-conf)", c["before"]["flagged"], c["after"]["flagged"]),
            ("couch_fp", "  on screen capture", c["before"]["on_screen"], c["after"]["on_screen"]),
            ("couch_fp", "  overlaps bed", c["before"]["on_bed"], c["after"]["on_bed"]),
            ("couch_fp", "  below conf", c["before"]["low_conf"], c["after"]["low_conf"]),
            ("couch_fp", "FP rate", c["before"]["fp_rate"], c["after"]["fp_rate"]),
            ("dup_person", "frames with a person", f["before"]["frames_with_person"], f["after"]["frames_with_person"]),
            ("dup_person", "frames with duplicate box", f["before"]["frames_with_dup"], f["after"]["frames_with_dup"]),
            ("dup_person", "duplicate-frame rate", f["before"]["dup_frame_rate"], f["after"]["dup_frame_rate"]),
            ("dup_person", "person tracks", t["person_tracks"], "-"),
            ("dup_person", "candidate duplicate track pairs", t["candidate_pairs"], "-"),
            ("dup_person", "  confirmed / rejected / unverified", f"{t['confirmed']} / {t['rejected']} / {t['unverified']}", "-"),
            ("dup_person", "candidate rate (per track)", t["candidate_rate"], "-"),
            ("dup_person", "embedding coverage", t["embedding_coverage"], "-")]
    p = res["params"]
    lines = [f"conf threshold {p['conf']} · person IoU {p['iou']} · track window {p['window_s']}s · embedding cos {p['sim']}", "",
             "| check | metric | before threshold | after threshold |", "|---|---|---|---|"]
    lines += [f"| {a} | {b} | {x} | {y} |" for a, b, x, y in rows]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vision_lab.dq", description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--sim", type=float, default=0.95)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = run(a.db, a.conf, a.iou, a.window, a.sim)
    print(json.dumps(res, indent=1) if a.json else metrics_table(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
