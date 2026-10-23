"""Measure catalog models on this Mac with SYNTHETIC inputs -> vision_lab/catalog_measured.json.

    PYTHONPATH=. .venv/bin/python tools/bench_catalog.py [key ...]

Each model runs in its own subprocess (fresh memory numbers, nothing loaded into the running app). Offline only
(HF_HUB_OFFLINE=1): a model that isn't already on disk is reported "not installed", never downloaded. Ollama
models are only timed if they are already loaded (no cold loads). Inputs: a drawn 960x540 test scene, a `say`
sentence for audio, fixed sample sentences for text. Numbers are taken while the live app is running, so they
include contention from the live pipeline. Nothing here reads camera, screen, microphone or the database.
"""
import json, os, platform, resource, statistics, subprocess, sys, tempfile, time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "vision_lab" / "catalog_measured.json"
WEIGHTS = Path(os.environ.get("VLAB_WEIGHTS", ROOT / "weights"))
HF = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
YOLO = ["yolo11n", "yolo11s", "yolo11s-seg", "yolo11s-pose", "yolov8s-worldv2"]
ALL = YOLO + ["sam2.1_b", "siglip", "depth", "whisper-large-v3-turbo", "silero-vad", "kokoro", "ast-audioset",
              "minilm", "gemma4:26b"]
SENT = "The kettle is on the counter and the dog is asleep on the couch next to a blue blanket."


def peak_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


def mb(n):
    return f"{n / 1e6:,.0f} MB" if n < 1e9 else f"{n / 1e9:.2f} GB"


def scene():
    import numpy as np, cv2
    img = np.full((540, 960, 3), 200, np.uint8)
    img[:] = np.linspace(150, 230, 960, dtype=np.uint8)[None, :, None]
    cv2.rectangle(img, (80, 300), (420, 500), (60, 90, 140), -1)      # "couch"
    cv2.circle(img, (650, 260), 60, (40, 40, 180), -1)
    cv2.rectangle(img, (600, 330), (720, 520), (90, 60, 40), -1)
    cv2.ellipse(img, (250, 200), (50, 70), 0, 0, 360, (120, 150, 190), -1)
    cv2.rectangle(img, (210, 270), (290, 450), (30, 30, 30), -1)
    return img


def timed(fn, n, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def dir_bytes(p):
    p = Path(p)
    if p.is_file():
        return p.stat().st_size
    snaps = p / "snapshots"
    root = max(snaps.iterdir(), key=lambda q: q.stat().st_mtime) if snaps.exists() else p
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())   # follows HF blob symlinks


def say_audio(seconds_text=SENT * 2):
    import numpy as np
    d = Path(tempfile.mkdtemp())
    subprocess.run(["say", "-o", str(d / "s.aiff"), seconds_text], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(d / "s.aiff"), "-ac", "1", "-ar", "16000",
                    "-f", "s16le", str(d / "s.pcm")], check=True)
    return np.frombuffer((d / "s.pcm").read_bytes(), np.int16).astype(np.float32) / 32768


def m(label, value, note=None):
    return {"label": label, "value": value, "note": note}


def bench(key):
    import numpy as np
    out = []
    if key in YOLO or key == "sam2.1_b":
        import torch
        from ultralytics import SAM, YOLO as Y
        w = WEIGHTS / f"{key}.pt"
        if not w.exists():
            return None, "not installed"
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model = SAM(str(w)) if key.startswith("sam") else Y(str(w))
        if "world" in key:
            model.set_classes(["person", "cup", "laptop", "dog"])
        img = scene()
        if key.startswith("sam"):
            kw = {"device": dev, "verbose": False, "points": [[650, 260]], "labels": [1]}
            ms = timed(lambda: model(img, **kw), 5, warm=1)
            out.append(m("Outline one click", f"{ms:.0f} ms", f"960×540 test image, {dev.upper()}"))
        else:
            kw = {"device": dev, "verbose": False, "imgsz": 640}
            ms = timed(lambda: model(img, **kw), 40, warm=5)
            out.append(m("Time per frame", f"{ms:.1f} ms (≈{1000 / ms:.0f} fps)", f"640 px, {dev.upper()}, model only"))
        if dev == "mps":
            out.append(m("GPU memory", mb(torch.mps.driver_allocated_memory()), "Metal driver, after the runs"))
        out.append(m("Peak RAM", f"{peak_mb():,.0f} MB", "whole Python process incl. PyTorch"))
        out.append(m("Size on disk", mb(w.stat().st_size), None))
    elif key == "siglip":
        if not (HF / "models--google--siglip-base-patch16-224").exists():
            return None, "not installed"
        from vision_lab.llm import Embedder
        import torch
        e = Embedder(); img = scene()
        e.image(img)
        if e.error:
            return None, e.error
        ms = timed(lambda: e.image(img), 10)
        tms = timed(lambda: e.text("red mug on the desk"), 10)
        out += [m("Index one keyframe", f"{ms:.0f} ms", "224 px, " + e.device.upper()),
                m("Search query", f"{tms:.0f} ms", "embed one phrase")]
        if e.device == "mps":
            out.append(m("GPU memory", mb(torch.mps.driver_allocated_memory()), "Metal driver"))
        out += [m("Peak RAM", f"{peak_mb():,.0f} MB", "whole Python process incl. PyTorch"),
                m("Size on disk", mb(dir_bytes(HF / "models--google--siglip-base-patch16-224")), None)]
    elif key == "depth":
        if not (HF / "models--depth-anything--Depth-Anything-V2-Small-hf").exists():
            return None, "not installed"
        from transformers import pipeline
        from PIL import Image
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        p = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=dev)
        im = Image.fromarray(scene()[:, :, ::-1].copy()).resize((518, 292))
        ms = timed(lambda: p(im), 10)
        out += [m("Time per frame", f"{ms:.0f} ms", f"518×292, {dev.upper()}"),
                m("Peak RAM", f"{peak_mb():,.0f} MB", "whole Python process"),
                m("Size on disk", mb(dir_bytes(HF / "models--depth-anything--Depth-Anything-V2-Small-hf")), None)]
    elif key == "whisper-large-v3-turbo":
        repo = HF / "models--mlx-community--whisper-large-v3-turbo"
        if not repo.exists():
            return None, "not installed"
        import mlx.core as mx, mlx_whisper
        audio = say_audio(); dur = len(audio) / 16000
        kw = {"path_or_hf_repo": "mlx-community/whisper-large-v3-turbo", "language": "en"}
        t0 = time.perf_counter(); mlx_whisper.transcribe(audio, **kw); first = time.perf_counter() - t0
        t0 = time.perf_counter(); mlx_whisper.transcribe(audio, **kw); warm = time.perf_counter() - t0
        out += [m("Speed (real-time factor)", f"{warm / dur:.3f} (≈{dur / warm:.0f}× faster than real time)",
                  f"{dur:.0f} s synthetic speech, GPU (MLX), warm"),
                m("First run incl. load", f"{first:.1f} s", None)]
        peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
        out += [m("GPU memory peak", mb(peak()), "MLX"),
                m("Peak RAM", f"{peak_mb():,.0f} MB", "whole Python process"),
                m("Size on disk", mb(dir_bytes(repo)), None)]
    elif key == "silero-vad":
        from silero_vad import load_silero_vad
        import torch
        v = load_silero_vad(onnx=True)
        audio = say_audio(SENT)
        chunks = [torch.from_numpy(audio[i:i + 512].copy()) for i in range(0, len(audio) - 512, 512)][:200]
        for c in chunks[:5]:
            v(c, 16000)
        t0 = time.perf_counter()
        for c in chunks:
            v(c, 16000)
        ms = (time.perf_counter() - t0) * 1000 / len(chunks)
        import silero_vad
        f = next(Path(silero_vad.__file__).parent.rglob("silero_vad.onnx"), None)
        out += [m("Time per 32 ms chunk", f"{ms:.2f} ms", "CPU, ONNX")]
        if f:
            out.append(m("Size on disk", mb(f.stat().st_size), None))
        out.append(m("Peak RAM", f"{peak_mb():,.0f} MB", "whole Python process"))
    elif key == "kokoro":
        f = WEIGHTS / "kokoro" / "kokoro-v1.0.onnx"
        if not f.exists():
            return None, "not installed"
        from kokoro_onnx import Kokoro
        k = Kokoro(str(f), str(WEIGHTS / "kokoro" / "voices-v1.0.bin"))
        k.create("Warm up.", voice="af_heart", speed=1.0, lang="en-us")
        t0 = time.perf_counter()
        a, sr = k.create(SENT, voice="af_heart", speed=1.0, lang="en-us")
        secs = time.perf_counter() - t0; dur = len(a) / sr
        out += [m("Speed (real-time factor)", f"{secs / dur:.2f} (≈{dur / secs:.1f}× faster than real time)",
                  f"one {dur:.0f} s sentence, CPU (ONNX)"),
                m("Peak RAM", f"{peak_mb():,.0f} MB", "whole Python process"),
                m("Size on disk", mb(f.stat().st_size + (WEIGHTS / "kokoro" / "voices-v1.0.bin").stat().st_size),
                  "model + voices")]
    elif key == "ast-audioset":
        repo = HF / "models--MIT--ast-finetuned-audioset-10-10-0.4593"
        if not repo.exists() or dir_bytes(repo) < 10e6:
            return None, "not installed (weights not downloaded)"
        from transformers import pipeline
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        clf = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593", device=dev)
        clip = say_audio(SENT)[:16000 * 2]
        ms = timed(lambda: clf({"raw": clip, "sampling_rate": 16000}, top_k=10), 5)
        out += [m("Classify a 2 s clip", f"{ms:.0f} ms", dev.upper()),
                m("Peak RAM", f"{peak_mb():,.0f} MB", "whole Python process"),
                m("Size on disk", mb(dir_bytes(repo)), None)]
    elif key == "minilm":
        from vision_lab.find import Embedder
        try:
            e = Embedder()
        except Exception:
            return None, "not installed (ONNX export not cached)"
        texts = [SENT] * 64
        ms = timed(lambda: e(texts), 5)
        repo = HF / "models--sentence-transformers--all-MiniLM-L6-v2"
        out += [m("Embed 64 sentences", f"{ms:.0f} ms ({ms / 64:.2f} ms each)", "CPU, ONNX"),
                m("Peak RAM", f"{peak_mb():,.0f} MB", "whole Python process"),
                m("Size on disk", mb(next(repo.glob("snapshots/*/onnx/model.onnx")).stat().st_size), "ONNX model file")]
    else:  # Ollama model: only if already loaded
        import base64, urllib.request, cv2
        url = os.environ.get("OLLAMA", "http://127.0.0.1:11434")
        ps = json.load(urllib.request.urlopen(url + "/api/ps", timeout=5)).get("models", [])
        if key not in [p["name"] for p in ps]:
            return None, "not loaded (skipped to avoid a cold load)"
        jpg = base64.b64encode(cv2.imencode(".jpg", scene())[1].tobytes()).decode()
        body = {"model": key, "stream": False, "think": False, "keep_alive": "30m",
                "options": {"num_predict": 60, "temperature": 0},
                "prompt": "Describe this image in one sentence.", "images": [jpg]}
        req = urllib.request.Request(url + "/api/generate", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=300))
        tps = r["eval_count"] / (r["eval_duration"] / 1e9)
        out += [m("Caption one image", f"{r['total_duration'] / 1e9:.1f} s", "960×540 test image, ≤60 tokens, warm"),
                m("Writing speed", f"{tps:.0f} tokens/s", "Ollama eval rate")]
    return out, None


def child(key):
    try:
        metrics, err = bench(key)
    except Exception as e:
        metrics, err = None, f"{type(e).__name__}: {e}"[:200]
    print("RESULT " + json.dumps({"metrics": metrics, "error": err}))


def machine():
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
        return f"{chip}, {mem // 2**30} GB"
    except Exception:
        return platform.machine()


def main(keys):
    if len(keys) == 2 and keys[0] == "--child":
        return child(keys[1])
    prev = json.loads(OUT.read_text()) if OUT.exists() else {"models": {}}
    today = date.today().isoformat()
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": str(ROOT)}
    for k in keys or ALL:
        p = subprocess.run([sys.executable, __file__, "--child", k], capture_output=True, text=True, env=env,
                           cwd=ROOT)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        res = json.loads(line[7:]) if line else {"metrics": None, "error": (p.stderr or "no output")[-200:]}
        if res["metrics"]:
            prev["models"][k] = {"metrics": [{**x, "date": today} for x in res["metrics"]]}
        else:
            prev["models"][k] = {"metrics": [], "error": res["error"], "date": today}
        print(k, "->", res["error"] or "; ".join(f"{x['label']}: {x['value']}" for x in res["metrics"]), flush=True)
    prev.update(date=today, machine=machine(),
                note="Measured by tools/bench_catalog.py on synthetic inputs while the live app was running.")
    OUT.write_text(json.dumps(prev, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
