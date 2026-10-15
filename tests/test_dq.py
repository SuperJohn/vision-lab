"""HOU-85: DQ checks on the synthetic fixture (pytest-style; also runnable directly).

    .venv/bin/python tests/test_dq.py
"""
import json, sqlite3, subprocess, sys, tempfile, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision_lab import dq

FIX = ROOT / "tests" / "fixtures" / "vlab_fixture.db"


def test_iou():
    assert dq.iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert dq.iou([0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]) == 0.0


def test_couch_fp():
    c = dq.run(FIX)["couch_fp"]
    b, a = c["before"], c["after"]
    # fixture: 6 keyframe couches (3 on a bed, 2 low-conf) + 2 couches on screens (conf 0.45)
    assert b["couch_detections"] == 8 and b["on_screen"] == 2 and b["on_bed"] == 3 and b["low_conf"] == 4
    assert b["flagged"] == 7 and b["fp_rate"] == round(7 / 8, 4)
    assert a["couch_detections"] == 4 and a["on_screen"] == 0 and a["low_conf"] == 0 and a["flagged"] == 3


def test_dup_person_frame():
    f = dq.run(FIX)["dup_person"]["frame"]
    assert f["before"]["frames_with_person"] == 30
    assert f["before"]["frames_with_dup"] == 2, "kf 7/8 are dups; kf 9 is two real people (low IoU)"
    assert f["after"]["frames_with_dup"] == 2


def test_dup_person_track_embeddings():
    t = dq.run(FIX)["dup_person"]["track"]
    # 101/102 near-same start, near-identical embeddings -> confirmed; 103 overlaps but different embedding -> rejected
    assert t["confirmed"] == 1 and t["rejected"] == 2 and t["unverified"] == 0, t


def test_degrades_without_embeddings():
    tmp = Path(tempfile.mkdtemp()) / "noemb.db"
    shutil.copy(FIX, tmp)
    con = sqlite3.connect(tmp); con.execute("UPDATE keyframes SET embedding = NULL"); con.commit(); con.close()
    t = dq.run(tmp)["dup_person"]["track"]
    assert t["confirmed"] == 0 and t["unverified"] == t["candidate_pairs"] == 3 and t["embedding_coverage"] == 0


def test_read_only():
    try:
        dq.connect_ro(FIX).execute("DELETE FROM events")
        raise AssertionError("dq connection must be read-only")
    except sqlite3.OperationalError:
        pass


def test_cli():
    out = subprocess.run([sys.executable, "-m", "vision_lab.dq", "--db", str(FIX)], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert "| couch_fp | FP rate | 0.875 |" in out
    j = json.loads(subprocess.run([sys.executable, "-m", "vision_lab.dq", "--db", str(FIX), "--json"], cwd=ROOT,
                                  capture_output=True, text=True, check=True).stdout)
    assert j["couch_fp"]["before"]["couch_detections"] == 8


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
    print("test_dq: OK")
