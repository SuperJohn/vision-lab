"""HOU-84: the synthetic fixture matches the production schema and is reproducible.

    .venv/bin/python tests/event_model.py
"""
import os, sqlite3, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
import make_fixture

FIX = ROOT / "tests" / "fixtures" / "vlab_fixture.db"
REAL = Path(os.environ.get("VLAB_DB") or ROOT / "data" / "vlab.db")


def columns(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tabs = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        cols = {t: [(c[1], c[2], c[5]) for c in con.execute(f'PRAGMA table_info("{t}")')] for t in tabs}
        idx = sorted(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"))
        return cols, idx
    finally:
        con.close()


def dump(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return list(con.iterdump())
    finally:
        con.close()


def test_fixture_matches_schema():
    cols, idx = columns(FIX)
    assert set(cols) == {"events", "presence", "keyframes", "config", "utterances", "plan_blocks", "plan_changes", "batch_annotations"}
    assert [c[0] for c in cols["events"]][-4:] == ["modality", "written_at", "data", "source_key"]
    if REAL.exists():   # local dev: compare against the live DB (read-only), never commit its contents
        rcols, ridx = columns(REAL)
        assert rcols == cols, "fixture schema drifted from data/vlab.db; update make_fixture.SCHEMA and docs/event-model.md"
        assert ridx == idx


def test_fixture_reproducible():
    out = Path(tempfile.mkdtemp()) / "fx.db"
    make_fixture.build(out)
    assert dump(out) == dump(FIX), "committed fixture differs from make_fixture.py output; regenerate it"


def test_fixture_links_and_cases():
    con = sqlite3.connect(f"file:{FIX}?mode=ro", uri=True)
    q = lambda s: con.execute(s).fetchone()[0]
    assert q("SELECT COUNT(*) FROM utterances u LEFT JOIN keyframes k ON k.id=u.keyframe_id WHERE k.id IS NULL") == 0
    assert q("SELECT COUNT(*) FROM events e LEFT JOIN keyframes k ON k.id=e.keyframe_id WHERE e.keyframe_id IS NOT NULL AND k.id IS NULL") == 0
    assert q("SELECT COUNT(DISTINCT length(embedding)) FROM keyframes") == 1 and q("SELECT length(embedding) FROM keyframes LIMIT 1") == 3072
    assert q("SELECT COUNT(*) FROM batch_annotations WHERE source_type='screen' AND detections LIKE '%couch%'") == 2
    assert q("SELECT COUNT(DISTINCT modality) FROM events") == 5


if __name__ == "__main__":
    test_fixture_matches_schema(); test_fixture_reproducible(); test_fixture_links_and_cases()
    print("event_model: OK")
