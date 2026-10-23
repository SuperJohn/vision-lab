"""PYTHONPATH=. .venv/bin/python tests/time_format.py
UI clock times are 12-hour with AM/PM from one shared helper (vision_lab/static/timefmt.js).
Runs the helper in node (TZ=America/Los_Angeles) and checks the UI uses it everywhere."""
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

STATIC = Path(__file__).resolve().parent.parent / "vision_lab" / "static"
TZ = ZoneInfo("America/Los_Angeles")
NODE = shutil.which("node") or next(iter(sorted(Path.home().glob(".nvm/versions/node/*/bin/node"))), None)


def ts(h, m, s=0, day=20):
    return datetime(2026, 10, day, h, m, s, tzinfo=TZ).timestamp()


def run_node(calls):
    js = f"const T = require({json.dumps(str(STATIC / 'timefmt.js'))});\n" \
         f"console.log(JSON.stringify({json.dumps(calls)}.map(([f, a]) => T[f](...a))));"
    out = subprocess.run([str(NODE), "-e", js], capture_output=True, text=True, timeout=20,
                         env={**os.environ, "TZ": "America/Los_Angeles"})
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_formatter():
    cases = [
        (("clock", [ts(10, 31, 5)]), "10:31:05 AM"),
        (("clock", [ts(0, 0, 9)]), "12:00:09 AM"),
        (("clock", [ts(12, 5, 0)]), "12:05:00 PM"),
        (("clock", [ts(23, 59, 59)]), "11:59:59 PM"),
        (("clock", [ts(9, 2, 10)]), "9:02:10 AM"),
        (("clock", [ts(15, 45, 30), False]), "3:45 PM"),
        (("tick", [ts(10, 0)]), "10 AM"),
        (("tick", [ts(10, 30)]), "10:30 AM"),
        (("tick", [ts(0, 0)]), "12 AM"),
        (("tick", [ts(12, 0)]), "12 PM"),
        (("tick", [ts(18, 15)]), "6:15 PM"),
        (("tick", [ts(0, 0, 0, 21)]), "12 AM"),
        (("dayTime", [ts(13, 20, 25)]), "Oct 20, 1:20:25 PM"),
        (("range", [ts(10, 31, 5), ts(10, 45, 10)]), "10:31:05 AM–10:45:10 AM"),
        (("range", [ts(9, 2), ts(10, 31), False]), "9:02 AM–10:31 AM"),
    ]
    got = run_node([c for c, _ in cases])
    for (c, want), g in zip(cases, got):
        assert g == want, f"{c}: {g!r} != {want!r}"


def test_ui_uses_shared_helper():
    html = (STATIC / "index.html").read_text()
    assert html.index("/static/timefmt.js") < html.index("/static/app.js") < html.index("/static/voicelog.js")
    for name in ("app.js", "voicelog.js", "mic.js"):
        js = (STATIC / name).read_text()
        assert "hour12" not in js and "toLocaleTimeString" not in js, f"{name}: use VLTime, not 24-hour locale formats"
        assert not re.search(r"getHours\(\)\)?\.padStart", js), f"{name}: hand-rolled 24-hour clock"
        assert not re.search(r"toLocaleString\([^)]*hour", js), f"{name}: date+time must use VLTime.dayTime"
    app, vl = (STATIC / "app.js").read_text(), (STATIC / "voicelog.js").read_text()
    assert "VLTime.clock(ts)" in app and "VLTime.tick(t)" in app and "VLTime.dayTime(it.capture_ts)" in app
    assert "VLTime.clock(ts)" in vl and "VLTime.clock(ts, false)" in vl


if __name__ == "__main__":
    assert NODE, "node not found"
    test_formatter(); test_ui_uses_shared_helper()
    print("time_format: OK")
