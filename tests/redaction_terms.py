"""HOU-88: personal title/place terms load from a local untracked file, not from source.

    .venv/bin/python tests/redaction_terms.py
"""
import os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
code = """
import sys; sys.path.insert(0, %r)
from vision_lab import redaction_pack as r
assert r.scan_text('meet at Fakeville Plaza')['pii'] == ['local_term'], r.scan_text('meet at Fakeville Plaza')
assert r.mask_text('near fakeville plaza') == 'near [LOCAL_TERM]'
res = r.check_event({'modality': 'desktop', 'data': {'app': 'Safari', 'title': 'Fakeville Plaza hours'}}) if hasattr(r, 'check_event') else None
print('ok')
""" % str(ROOT)


def test_local_title_terms():
    f = Path(tempfile.mkdtemp()) / "t.txt"
    f.write_text("# synthetic\nFakeville Plaza\n")
    env = {**os.environ, "VLAB_REDACT_TITLE_TERMS": str(f)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_source_has_no_personal_terms():
    src = (ROOT / "vision_lab" / "redaction_pack.py").read_text().lower()
    assert "9212" not in src and "village" not in src


if __name__ == "__main__":
    test_local_title_terms(); test_source_has_no_personal_terms()
    print("redaction_terms: OK")
