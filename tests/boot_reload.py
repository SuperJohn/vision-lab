"""HOU-146: boot ID, no-cache UI headers, version-busted assets, and frontend auto-reload/error-banner wiring.

    .venv/bin/python tests/boot_reload.py
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from vision_lab import bootinfo

STATIC = ROOT / "vision_lab" / "static"


def test_boot_id_stable():
    import importlib
    a = bootinfo.BOOT_ID
    assert a and isinstance(a, str)
    assert importlib.import_module("vision_lab.bootinfo").BOOT_ID == a, "boot id changed within one process"
    m = json.loads(bootinfo.with_boot_id(json.dumps({"paused": True})))
    assert m == {"paused": True, "boot_id": a}
    assert json.loads(bootinfo.with_boot_id("{}")) == {"boot_id": a}


def test_headers_and_busting():
    app = FastAPI()
    bootinfo.install_no_cache(app)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.get("/")(lambda: bootinfo.render_index(STATIC))
    app.get("/api/x")(lambda: {"ok": 1})
    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache, must-revalidate"
    assert f'/static/app.js?v={bootinfo.BOOT_ID}"' in r.text and f'/static/style.css?v={bootinfo.BOOT_ID}"' in r.text
    r = c.get(f"/static/app.js?v={bootinfo.BOOT_ID}")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache, must-revalidate"
    assert "cache-control" not in c.get("/api/x").headers


def test_server_and_frontend_wiring():
    srv = (STATIC.parent / "server.py").read_text(); js = (STATIC / "app.js").read_text()
    assert '"boot_id": BOOT_ID' in srv and "with_boot_id(meta)" in srv and "install_no_cache(app)" in srv
    assert "checkBoot(m.boot_id)" in js and "location.reload()" in js and "checkBoot(s.boot_id)" in js
    assert "PRIVACY_TIMEOUT_MS = 8000" in js and "ctl.abort()" in js and "await syncPrivacy()" in js
    assert "!!r.paused !== turnOn" in js and "has not been released yet" in js


def test_real_state_route_has_boot_id():
    """Run the real server.state() source with stub globals (importing server.py would load models/DB)."""
    import ast, types
    src = (STATIC.parent / "server.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "state")
    deco = [ast.unparse(d) for d in fn.decorator_list]
    assert deco == ["app.get('/api/state')"], deco
    fn.decorator_list = []
    eng = types.SimpleNamespace(settings={"capture_paused": True}, cameras=[], zones=[], rules=[],
                                class_names=lambda: [], embedder=types.SimpleNamespace(error=None))
    llm = types.SimpleNamespace(list_models=lambda: [], capabilities=lambda n: [])
    g = {"engine": eng, "llm": llm, "BOOT_ID": bootinfo.BOOT_ID}
    exec(compile(ast.Module([fn], []), "server.py", "exec"), g)
    out = g["state"]()
    assert out["boot_id"] == bootinfo.BOOT_ID and out["paused"] is True, out


def test_no_optimistic_toggle():
    js = (STATIC / "app.js").read_text()
    h = js[js.index("$('#power').addEventListener"):]
    h = h[:h.index("\n});") + 4]
    pre = h[:h.index("await api('/api/privacy'")]
    assert "renderPrivacy('pending')" in pre
    assert "renderPrivacy(true" not in pre and "renderPrivacy(false" not in pre and "renderPrivacy(turnOn" not in pre and "setPrivacy(" not in pre, "button flips before the server answers"
    assert "renderPrivacy(!!r.paused)" in h and "await syncPrivacy()" in h
    assert "if (!privacyPending) renderPrivacy" in js, "ws meta must not override the pending look"


if __name__ == "__main__":
    test_boot_id_stable(); test_headers_and_busting(); test_server_and_frontend_wiring()
    test_real_state_route_has_boot_id(); test_no_optimistic_toggle()
    print("boot_reload: OK")
