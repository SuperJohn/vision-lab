"""Model lab catalog: plain-language fields, in-use detection, benchmark provenance, and the houghtonlabs restyle.

    PYTHONPATH=. .venv/bin/python tests/model_catalog.py
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision_lab import catalog

STATIC = ROOT / "vision_lab" / "static"
OLD_KEYS = ["yolo11n", "yolo11s", "yolov8s-worldv2", "yolo11s-seg", "sam2.1_b", "bytetrack", "yolo11s-pose", "depth", "siglip"]
OLD_IDS = ("mic-toggle mic-dot mic-label power privacy-label quit banner view-live stages camera detector world-wrap "
           "world-classes describe-model summary-model describe-every conf-val conf counts layers live hud-res hud-pose "
           "budget-total budget-bar budget-legend scene-meta scene-text scene-tags events ask-form ask ask-answer view-zones "
           "paused-at refresh-frame zcanvas tool-hint name-form zone-name name-cancel zone-list rules-title new-rule "
           "rule-editor rules view-timeline search-form q range search-meta shots tl-prev tl-day tl-next tl-today tl-zoom "
           "tl-refresh cap-meta cap-lanes tl-day-label lanes summarize summary-text summary-meta day-form ask-day view-models "
           "mem-total mem-bar mem-legend ab-form ab-a ab-b ab-classes ab catalog toast view-voice mic-device vl-expand "
           "vl-collapse vl-replies vl-summary vl-refresh vl-days vl-empty vl-more view-screens scr-date scr-refresh "
           "scr-grid scr-empty").split()


def live_ctx(**over):
    """Mirrors the live config on 2026-10-22: yolo11s + segment/pose/describe on, depth off, gemma4:26b everywhere."""
    ctx = {
        "settings": {"detector": "yolo11s", "stages": {"detect": True, "segment": True, "pose": True, "describe": True,
                                                        "depth": False},
                     "describe_model": "gemma4:26b", "summary_model": "gemma4:26b"},
        "voice": {"stt_model": "mlx-community/whisper-large-v3-turbo", "tts_engine": "kokoro", "ambient": False,
                  "llm_model": "gemma4:26b"},
        "planloop": {"camera_probe": True, "probe_model": "weights/yolo11n.pt", "caption_model": "gemma4:26b",
                     "screen_caption": True},
        "batch": {"detect": True, "detect_model": "weights/yolo11s-seg.pt", "model": "gemma4:26b"},
        "processes": {"planloop": True, "batch": True},
        "engine_cache": ["yolo11s-seg", "yolo11s"], "embedder_loaded": True, "voice_loaded": [],
        "whisper_loaded": False, "voice_log_recording": True, "ollama_loaded": ["gemma4:26b"],
        "describe_model": "gemma4:26b", "summary_model": "gemma4:26b", "voice_llm": "gemma4:26b",
        "batch_model": "gemma4:26b", "planloop_model": "gemma4:26b",
    }
    ctx.update(over)
    return ctx


def status(key, ctx):
    return catalog.usage(key, ctx)["status"]


def test_entries_are_plain_and_sourced():
    keys = [e["key"] for e in catalog.ENTRIES]
    assert len(keys) == len(set(keys)), "duplicate catalog keys"
    for k in OLD_KEYS:
        assert k in keys, f"{k} dropped from the catalog"
    for k in ("whisper-large-v3-turbo", "silero-vad", "kokoro", "ast-audioset", "minilm"):
        assert k in keys, f"{k} is used by Vision Lab and must be listed"
    for e in catalog.ENTRIES:
        assert e["group"] in ("Camera", "Search", "Voice", "Language"), e
        assert e["hw"] in ("CPU", "GPU", "ANE"), e
        assert len(e["task"]) >= 25 and len(e["gives"]) >= 40, f"{e['key']}: task/gives should be a real sentence"
        assert e["task"] not in ("Detect", "Segment", "Track", "Pose", "Depth", "Search", "Describe"), e["key"]
        assert e["published"], f"{e['key']}: needs at least one published fact with a source"
        for p in e["published"]:
            assert p["kind"] == "published" and p["source"] and p["value"], (e["key"], p)
            assert p["url"].startswith("https://"), (e["key"], p)


def test_usage_matches_live_config():
    ctx = live_ctx()
    want = {"yolo11s-seg": "active", "yolo11n": "active", "yolo11s": "on_demand", "yolov8s-worldv2": "idle",
            "sam2.1_b": "on_demand", "bytetrack": "active", "yolo11s-pose": "active", "depth": "idle",
            "siglip": "active", "minilm": "on_demand", "whisper-large-v3-turbo": "active", "silero-vad": "on_demand",
            "kokoro": "on_demand", "ast-audioset": "idle", "gemma4:26b": "active", "qwen3-vl:8b": "idle"}
    got = {k: status(k, ctx) for k in want}
    assert got == want, {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    u = catalog.usage("yolo11s-seg", ctx)
    assert u["label"] == "In use" and u["loaded"] and "Vision Lab server" in u["loaded_where"]
    assert any("Live camera" in w for w in u["where"]) and any("batch" in w.lower() for w in u["where"])
    g = catalog.usage("gemma4:26b", ctx)
    assert len(g["where"]) == 5 and g["loaded_where"] == ["Ollama (in memory)"], g
    assert catalog.usage("qwen3-vl:8b", ctx) == {"status": "idle", "label": "Not in use", "where": [],
                                                 "loaded": False, "loaded_where": []}


def test_usage_follows_settings():
    ctx = live_ctx(settings={"detector": "yolo11n", "stages": {"detect": True, "segment": False, "pose": False, "describe": True},
                             "describe_model": "qwen3-vl:8b"},
                   processes={"planloop": False, "batch": False}, voice_log_recording=False,
                   describe_model="qwen3-vl:8b", batch_model="qwen3-vl:8b")
    assert status("yolo11n", ctx) == "active" and status("yolo11s-seg", ctx) == "on_demand"
    assert status("yolo11s-pose", ctx) == "idle" and status("whisper-large-v3-turbo", ctx) == "on_demand"
    assert status("qwen3-vl:8b", ctx) == "active" and status("gemma4:26b", ctx) == "on_demand"
    ctx = live_ctx(settings={"detector": "yolov8s-worldv2", "stages": {"detect": True, "segment": True, "depth": True}})
    assert status("yolov8s-worldv2", ctx) == "active" and status("depth", ctx) == "active"
    assert not any("Live" in w for w in catalog.usage("yolo11s-seg", ctx)["where"]), "world disables segment"
    ctx = live_ctx(settings={"detector": "yolo11s", "stages": {"detect": False}})
    assert status("bytetrack", ctx) == "idle" and status("yolo11s-seg", ctx) == "active"  # batch still uses it
    assert status("ast-audioset", live_ctx(voice={"ambient": True})) == "on_demand"
    assert catalog.usage("kokoro", live_ctx(voice_loaded=["tts", "vad"]))["loaded"] is True
    assert catalog.usage("silero-vad", live_ctx(voice_loaded=["vad"]))["loaded_where"] == ["Vision Lab server"]
    assert catalog.usage("depth", live_ctx())["loaded"] is False
    assert catalog.usage("yolo11n", {})["status"] in ("idle", "on_demand"), "empty context must not crash"


def test_benchmarks_keep_provenance(tmp=Path("/tmp/vlab_catalog_test.json")):
    tmp.write_text(json.dumps({"date": "2026-10-22", "machine": "Test Mac", "models": {
        "yolo11n": {"metrics": [{"label": "Time per frame", "value": "5.0 ms", "note": "640 px", "date": "2026-10-22"}]},
        "depth": {"metrics": [], "error": "not installed"}}}))
    m = catalog.load_measured(tmp)
    rows = catalog.benchmarks("yolo11n", m)
    assert rows[0]["kind"] == "measured" and rows[0]["date"] == "2026-10-22" and rows[0]["machine"] == "Test Mac"
    assert all(r["kind"] == "published" and r["url"] for r in rows[1:])
    assert catalog.benchmarks("depth", m)[0]["kind"] == "published", "no measured row may be invented"
    assert catalog.measure_note("depth", m) == "not installed" and catalog.measure_note("yolo11n", m) is None
    live = catalog.benchmarks("yolo11n", m, {"fps": 120.0, "label": "120 fps"})
    assert live[0]["value"] == "120 fps" and live[0]["kind"] == "measured"
    assert catalog.benchmarks("some-unknown:1b", {"models": {}}) == []
    assert catalog.load_measured(Path("/nonexistent.json")) == {"models": {}}
    tmp.unlink()


def test_committed_measurements_are_well_formed():
    m = catalog.load_measured()
    assert m.get("date") and m.get("machine") and m.get("models"), "run tools/bench_catalog.py"
    keys = {e["key"] for e in catalog.ENTRIES} | set(catalog.LLM_PUBLISHED)
    for k, v in m["models"].items():
        assert k in keys, f"measured an unknown model {k}"
        assert v.get("metrics") or v.get("error"), k
        for r in v.get("metrics", []):
            assert r["label"] and r["value"] and re.match(r"\d{4}-\d\d-\d\d$", r["date"]), (k, r)


def test_benchable_kinds():
    assert all(catalog.benchable({"kind": k}) for k in ("yolo", "sam", "depth", "siglip", "builtin", "ollama"))
    assert not catalog.benchable({"kind": "info"})
    srv = (ROOT / "vision_lab" / "server.py").read_text()
    assert "if not catalog.benchable(entry):" in srv and "catalog.usage(" in srv and '"usage":' in srv
    assert "json.dump" not in srv[srv.index("def _read_cfg"):srv.index("def _running_processes")], "must stay read-only"


def test_ui_catalog_and_restyle():
    html, css, js = ((STATIC / n).read_text() for n in ("index.html", "style.css", "app.js"))
    ids = set(re.findall(r'id="([^"]+)"', html))
    missing = [i for i in OLD_IDS if i not in ids]
    assert not missing, f"element IDs removed: {missing}"
    head = html[html.index('<table class="catalog" id="catalog">'):html.index("</thead>")]
    for col in ("Model", "In use", "What it does for you", "Benchmarks", "Runs on"):
        assert f">{col}</th>" in head, col
    assert "data-bench=" in js and "data-download=" in js
    assert "measured here" in js and "published" in js and "not measured" in js
    # benchmarks: value only, provenance in the hover tooltip - no caption/pill/note below the number
    bl = js[js.index("function benchList(r)"):js.index("function usageCell(u)")]
    assert 'title="${esc(benchTip(b))}"' in bl and "bm-val" in bl
    for gone in ("src pub", "src here", "bm-note", "measure_note)})"):
        assert gone not in bl, gone
    assert "bm-note" not in css and 'class="src' not in html
    # houghtonlabs brand
    assert 'src="/static/brand/hl-flask-outline.svg"' in html
    assert '<span class="brand-name">houghton<span class="brand-labs">labs</span></span>' in html
    assert '<h1 class="app-name">Vision Lab</h1>' in html and 'class="skip"' in html
    for f in ("brand/hl-flask-outline.svg", "fonts/InterVariable.woff2", "fonts/Inter-LICENSE.txt", "brand/favicon.ico"):
        assert (STATIC / f).exists(), f
    assert 'url("/static/fonts/InterVariable.woff2")' in css and "googleapis" not in css + html
    for tok in ("--accent: #0d9488", "--teal-bright: #2dd4bf", "--dark: #0b1220", "--bg: #fafaf7", "letter-spacing: -.035em"):
        assert tok in css, tok
    assert "#F5A524" not in css and "IBM Plex" not in css, "old dark/amber theme left in style.css"
    assert ":focus-visible" in css and "@media (max-width: 720px)" in css and "position: sticky" in css


if __name__ == "__main__":
    test_entries_are_plain_and_sourced(); test_usage_matches_live_config(); test_usage_follows_settings()
    test_benchmarks_keep_provenance(); test_committed_measurements_are_well_formed(); test_benchable_kinds()
    test_ui_catalog_and_restyle()
    print("model_catalog: OK")
