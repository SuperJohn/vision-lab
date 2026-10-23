"""Re-time voice_transcripts rows written by the drifting mic recorder.

Before commit 48c3ca4 each mic chunk was named voice-<t0 + samples/16000 ms>.pcm. The mic delivered fewer
samples than 16 kHz, so names (and every transcript segment derived from them) fell behind the wall clock.
The only wall-clock truth left per chunk is written_at (when its transcript rows were stored, i.e. chunk
end + Whisper latency L). This tool rebuilds wall times per chunk:

  name n_k          chunk start in drifted sample time (s), from source_key file:voice-<ms>.pcm:<seg>
  d_k               chunk length in sample seconds = n_{k+1} - n_k when that is < MAX_CHUNK_S (so exactly
                    one chunk), else max(MIN_CHUNK_S, last segment end + 0.1) capped at MAX_CHUNK_S
  r_k               local mic rate (sample s per wall s): least-squares slope of n vs written_at over
                    +-WINDOW_S of the same recorder session
  true_end_k        min(written_at) - L
  true_start_k      true_end of the previous chunk with speech when that chunk is certainly adjacent
                    (name gap < 20 s: two chunks are >= 20 s), else true_end_k - d_k / r_k
  segment ts        true_start_k + (ts - n_k) / r_k          (ts_end clamped to <= true_end_k)

Originals are kept in ts_start_orig / ts_end_orig and rows are tagged ts_fix (default 'rewrite-v1'), so a
rerun skips them and the change can be undone:

  UPDATE voice_transcripts SET ts_start = ts_start_orig, ts_end = ts_end_orig, ts_fix = NULL
   WHERE ts_fix = 'rewrite-v1';

Usage (dry run is the default; it never writes):
  python tools/fix_voice_ts_drift.py --db /tmp/copy.db --t-fix 1792604543.571
  python tools/fix_voice_ts_drift.py --db data/vlab.db --t-fix 1792604543.571 --apply
"""
import argparse, json, re, sqlite3, statistics, sys
from collections import defaultdict

MAX_CHUNK_S, MIN_CHUNK_S, WINDOW_S = 20.0, 10.0, 1800.0
UNAMBIG_S = 0.05                  # gap < 20 - this => exactly one chunk between two names
KEY = re.compile(r"^file:voice-(\d{10,13})\.pcm:(\d+)$")
TAG = "rewrite-v1"


def _name_s(ms):
    v = int(ms)
    return v / 1000 if v > 10**11 else float(v)


def columns(db):
    return {r[1] for r in db.execute("PRAGMA table_info(voice_transcripts)")}


def load(db, t_fix):
    cols = columns(db)
    skip = " AND ts_fix IS NULL" if "ts_fix" in cols else ""
    rows = db.execute("SELECT id, ts_start, ts_end, written_at, source_key FROM voice_transcripts "
                      f"WHERE written_at < ? AND source_key LIKE 'file:voice-%'{skip} ORDER BY id", (t_fix,)).fetchall()
    starts = [r[0] for r in db.execute("SELECT ts FROM events WHERE kind = 'mic_started' AND ts < ? ORDER BY ts", (t_fix,))]
    return [dict(id=r[0], ts_start=r[1], ts_end=r[2], written_at=r[3], key=r[4]) for r in rows], starts


def estimate_latency(db, t_fix, max_rows=2000):
    """Median of min(written_at of chunk k) - name of chunk k+1 for chunks recorded AFTER the fix (wall-clock names)."""
    rows = db.execute("SELECT written_at, source_key FROM voice_transcripts WHERE written_at >= ? "
                      "AND source_key LIKE 'file:voice-%' ORDER BY id LIMIT ?", (t_fix, max_rows)).fetchall()
    chunks = {}
    for w, k in rows:
        m = KEY.match(k or "")
        if m:
            n = _name_s(m.group(1))
            chunks[n] = min(w, chunks.get(n, w))
    names = sorted(chunks)
    lat = [chunks[a] - b for a, b in zip(names, names[1:]) if 0 < b - a <= MAX_CHUNK_S + 5 and chunks[a] >= b]
    return (statistics.median(lat), len(lat)) if lat else (None, 0)


def _slope(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else None


def plan(rows, starts, latency, window_s=WINDOW_S):
    """-> (updates [(id, new_start, new_end)], chunks info list). Pure function; no DB access."""
    by_chunk = defaultdict(list)
    for r in rows:
        m = KEY.match(r["key"] or "")
        if not m:
            continue
        by_chunk[_name_s(m.group(1))].append(r)
    chunks = []
    for n, rs in by_chunk.items():
        w = min(r["written_at"] for r in rs)
        sess = max([s for s in starts if s <= w] or [0.0])
        chunks.append(dict(n=n, w=w, session=sess, rows=rs, seg_end=max(r["ts_end"] - n for r in rs)))
    sessions = defaultdict(list)
    for c in chunks:
        sessions[c["session"]].append(c)
    updates = []
    for sess, cs in sessions.items():
        cs.sort(key=lambda c: c["n"])
        ws, ns = [c["w"] for c in cs], [c["n"] for c in cs]
        r_sess = _slope(ws, ns) if len(cs) >= 2 else 1.0
        r_sess = min(max(r_sess or 1.0, 0.5), 1.05)
        for i, c in enumerate(cs):
            gap = ns[i + 1] - c["n"] if i + 1 < len(cs) else None
            prev_gap = c["n"] - ns[i - 1] if i > 0 else None
            if gap is not None and gap < MAX_CHUNK_S - UNAMBIG_S:
                d = gap                                # exactly one chunk until the next name
            else:
                # recorder rule: a chunk closes at the first quiet 100 ms block once it is >= 10 s, so its length
                # is ~ max(10 s, end of the last speech + one block), capped at 20 s
                d = min(MAX_CHUNK_S, max(MIN_CHUNK_S if gap is not None else 0.0, c["seg_end"] + 0.1))
            idx = [j for j in range(len(cs)) if abs(ws[j] - c["w"]) <= window_s]
            r = _slope([ws[j] for j in idx], [ns[j] for j in idx]) if len(idx) >= 5 else None
            r = min(max(r, 0.5), 1.05) if r else r_sess
            end = c["w"] - latency
            if prev_gap is not None and prev_gap < MAX_CHUNK_S - UNAMBIG_S:
                # previous chunk with speech is certainly the one right before this one (two chunks are >= 20 s):
                # this chunk starts when that one ended, a wall-clock anchor that needs no length guess
                start, how = ws[i - 1] - latency, "prev-end"
            else:
                start, how = end - d / r, ("len" if gap is not None and gap < MAX_CHUNK_S - UNAMBIG_S else "len-guess")
            if start >= end:                           # never invert a chunk
                start, how = end - d / r, "len"
            sure_len = gap is not None and gap < MAX_CHUNK_S - UNAMBIG_S
            if how == "prev-end" and sure_len and end - start > 0:
                r = d / (end - start)                  # both ends anchored and the length known: exact local rate
            c.update(d=d, r=r, true_start=start, true_end=end, how=how)
            for row in c["rows"]:
                a = min(start + (row["ts_start"] - c["n"]) / r, end)
                b = min(start + (row["ts_end"] - c["n"]) / r, end)
                updates.append((row["id"], a, max(a, b)))
    updates.sort()
    return updates, chunks


def checks(db, t_fix, tag, starts):
    """Sanity checks on rows tagged `tag` (or would-be values in a dry-run temp table)."""
    q = lambda s, a=(): db.execute(s, a).fetchall()
    rows = q("SELECT id, ts_start, ts_end, written_at FROM voice_transcripts WHERE ts_fix = ? ORDER BY id", (tag,))
    out = {"rows": len(rows)}
    if not rows:
        return out
    lag = sorted(w - e for _, _, e, w in rows)
    out["ts_start_lt_written_at"] = all(s < w for _, s, _, w in rows)
    out["lag_end_to_written_s"] = {"median": round(statistics.median(lag), 2), "min": round(lag[0], 2), "max": round(lag[-1], 2)}
    # order: by session, ts_start should not go backwards along id order (allow 0.5 s jitter)
    sess_of = lambda w: max([s for s in starts if s <= w] or [0.0])
    back, prev = 0, {}
    for i, s, _, w in rows:
        k = sess_of(w)
        if k in prev and s < prev[k] - 0.5:
            back += 1
        prev[k] = max(prev.get(k, s), s)
    out["order_backsteps_gt_0.5s"] = back
    out["first_start"], out["last_start"] = rows[0][1], max(r[1] for r in rows)
    out["within_mic_started_and_t_fix"] = bool(starts) and min(r[1] for r in rows) >= starts[0] - 1 and \
        max(r[2] for r in rows) <= t_fix
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--t-fix", type=float, required=True, help="epoch s of the first mic_started on fixed code")
    ap.add_argument("--latency", type=float, default=None, help="Whisper latency L (s); default: estimate from post-fix rows")
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--apply", action="store_true", help="write (one transaction); default is a dry run")
    ap.add_argument("--busy-ms", type=int, default=3000)
    ap.add_argument("--show", default="", help="comma-separated row ids to print before/after")
    a = ap.parse_args(argv)

    db = sqlite3.connect(a.db, timeout=a.busy_ms / 1000, isolation_level=None)
    db.execute(f"PRAGMA busy_timeout = {int(a.busy_ms)}")
    lat, n_lat = (a.latency, 0) if a.latency is not None else estimate_latency(db, a.t_fix)
    if lat is None:
        sys.exit("no post-fix rows to estimate latency from; pass --latency")
    rows, starts = load(db, a.t_fix)
    updates, chunks = plan(rows, starts, lat)
    before = {r["id"]: (r["ts_start"], r["ts_end"]) for r in rows}
    rs = sorted({round(c["r"], 3) for c in chunks})
    hows = {}
    for c in chunks:
        hows[c["how"]] = hows.get(c["how"], 0) + 1
    report = {"mode": "apply" if a.apply else "dry-run", "start_method_counts": hows, "latency_s": round(lat, 3), "latency_samples": n_lat,
              "rows_in_scope": len(rows), "chunks": len(chunks), "updates": len(updates),
              "sessions": sorted({c["session"] for c in chunks}),
              "rate_min_med_max": [rs[0], statistics.median(sorted(c["r"] for c in chunks)), rs[-1]] if rs else None}

    db.execute("BEGIN IMMEDIATE")
    try:
        cols = columns(db)
        for col, typ in (("ts_start_orig", "REAL"), ("ts_end_orig", "REAL"), ("ts_fix", "TEXT")):
            if col not in cols:
                db.execute(f"ALTER TABLE voice_transcripts ADD COLUMN {col} {typ}")
        total_before = db.execute("SELECT COUNT(*) FROM voice_transcripts").fetchone()[0]
        db.executemany("UPDATE voice_transcripts SET ts_start_orig = ts_start, ts_end_orig = ts_end, "
                       "ts_start = ?, ts_end = ?, ts_fix = ? WHERE id = ? AND ts_fix IS NULL",
                       [(s, e, a.tag, i) for i, s, e in updates])
        report["total_rows_before_after"] = [total_before, db.execute("SELECT COUNT(*) FROM voice_transcripts").fetchone()[0]]
        report["checks"] = checks(db, a.t_fix, a.tag, starts)
        for i in [int(x) for x in a.show.split(",") if x.strip()]:
            r = db.execute("SELECT ts_start, ts_end FROM voice_transcripts WHERE id = ?", (i,)).fetchone()
            report.setdefault("show", {})[i] = {"before": before.get(i), "after": r}
        db.execute("COMMIT" if a.apply else "ROLLBACK")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    print(json.dumps(report, indent=1, default=str))
    return report


if __name__ == "__main__":
    main()
