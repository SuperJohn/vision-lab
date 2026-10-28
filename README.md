# Vision Lab

A local, on-device camera understanding lab for Apple Silicon Macs. It runs object detection, segmentation, tracking, pose, depth and a vision-language model on a live webcam stream, all on your own machine. No video leaves the computer.

## What it does

| Screen | What you can do |
|---|---|
| **Live** | Camera feed with toggleable overlays (boxes, masks, pose skeletons, track trails, zones, depth). Turn pipeline stages on and off and watch the per-frame time budget. A local vision model describes the scene every few seconds, and you can ask questions about the current frame. |
| **Zones & rules** | Draw polygon zones and crossing lines on a paused frame, or click any object to outline it with SAM 2.1. Build rules such as *"a person stays in Desk zone for more than 45 min while sitting → notify"* or *"someone crosses Doorway line → ask the vision model to describe them"*. |
| **Timeline** | Search saved keyframes with plain text (SigLIP embeddings), see per-class activity across the day, and have a local LLM summarize or answer questions about the event log. |
| **Model lab** | Memory use across the app and Ollama, an A/B comparison of two detectors on the same frame (e.g. YOLO11n vs. open-vocabulary YOLO-World), and one-click download/benchmark of each model. |

## How it works

```
Camera ─► fast loop (every frame): YOLO11 detect/segment + ByteTrack + YOLO11-pose ─► zones & rules ─► SQLite event log
                                                     │ rule fires / every N seconds
                                                     ▼
                         slow loop: Ollama vision model (gemma3, qwen3-vl, …) describes or answers ─► event log
                                                     │
                       keyframes ─► SigLIP embeddings (text search)   ·   LLM summarizes the day's log
```

The fast loop runs at about 30 fps on an M3 Ultra (detect ≈ 5 ms, segment ≈ 5 ms, pose ≈ 7 ms at 960 px input). It drops for a moment while the vision-language model uses the GPU. Models get a 960 px copy of each frame, and all outputs are normalised coordinates.

## Metrics

Architecture diagram: [docs/architecture.md](docs/architecture.md). Event model: [docs/event-model.md](docs/event-model.md).

**Fast loop** (M3 Ultra, 960 px input; source: Model lab benchmark in this repo): about 30 fps; detect ≈ 5 ms, segment ≈ 5 ms, pose ≈ 7 ms per frame.

**Local LLM bake-off** (source: `~/bakeoff` results, 2026-10-14; 20 home keyframes for vision, 10 voice prompts, 9 tool-calling tasks; medians):

| Model | Vision caption | Valid JSON | Voice TTFT | Voice tok/s | Voice reply total | Agent tasks |
|---|---|---|---|---|---|---|
| **gemma4:26b** (slow loop) | **0.95 s** | 20/20 | 0.09 s | 91 | 0.42 s | 9/9 (0.83 s) |
| qwen3.6:35b | 1.15 s | 20/20 | 0.07 s | 94 | 0.57 s | 9/9 (1.45 s) |
| qwen3.8:27b | 3.30 s | 20/20 | 0.24 s | 41 | 1.13 s | 9/9 (low 6.4 s / high 5.2 s) |
| gemma3:27b | 4.11 s | 20/20 | 0.42 s | 25 | 2.49 s | 0/9 (no tool support) |
| gpt-oss:120b | n/a (no vision) | – | 0.60 s | 67 | 1.14 s | 9/9 (low 1.9 s / high 3.8 s) |

gemma4:26b is about 4× faster than gemma3:27b on captions with the same JSON validity, which is why it runs the slow loop.

**Data quality** (source: `python -m vision_lab.dq --db data/vlab.db`, 2026-10-14, aggregates only; details in [docs/dq.md](docs/dq.md)):

| Check | Before (all detections) | After (conf ≥ 0.5) |
|---|---|---|
| Couch false-positive rate (on screen / on bed / low conf) | 37.9% of 2,012 | 1.4% of 1,300 |
| Frames with duplicate person boxes (IoU ≥ 0.7) | 0.09% of 1,075 | 0% of 901 |
| Person tracks with an overlapping duplicate candidate (≤ 2 s) | 13.4% of 3,608 | – |

**Data volume** (2026-10-14): 95.6k events across 5 modalities, 33k presence spans, 6.2k embedded keyframes, 2,972 slow-loop annotations, 128 voice turns (all linked to a keyframe).

## Requirements

- macOS on Apple Silicon (uses PyTorch's MPS backend; falls back to CPU elsewhere)
- [uv](https://github.com/astral-sh/uv), `ffmpeg` (used to list camera names)
- [Ollama](https://ollama.com) with a vision model, e.g. `ollama pull gemma3:27b` (or a smaller one like `gemma3:12b` / `qwen3-vl:8b`)

## Run

```bash
./run.sh
```

Or double-click **`Start Vision Lab.command`** in Finder: it starts the server and opens the page (if it's already running, it just opens the page).

Then open <http://localhost:8765>. In the header, the **Privacy mode** toggle (green RECORDING / red PRIVATE) stops the camera, screen captures and mic together while the server keeps running. **Quit server** shuts it down; start it again with the launcher. The first run creates `.venv`, installs dependencies, and downloads YOLO weights (about 20 MB each). SAM, Depth Anything and SigLIP download the first time you use them.

macOS asks for camera permission the first time, for the app that launched the script (Terminal, iTerm, etc.). Restart after approving.

To test without a camera, point it at a video file (it loops):

```bash
VLAB_SOURCE=/path/to/video.mp4 ./run.sh
```

## Layout

```
vision_lab/
  engine.py   camera thread, model pipeline, tracker state, zones/lines/rules, VLM + embedding workers
  server.py   FastAPI: WebSocket stream (JSON metadata + JPEG per frame) and REST API
  llm.py      Ollama client and SigLIP embedder
  store.py    SQLite: events, presence spans, keyframes, settings/zones/rules
  static/     single-page UI (vanilla JS + canvas)
data/         created at runtime: vlab.db, keyframes/ (git-ignored)
weights/      downloaded model weights (git-ignored)
```

## Notes and limits

- Pose and posture (standing / sitting / lying) come from YOLO11-pose keypoints with a simple angle heuristic. When legs are hidden (e.g. behind a desk) posture stays "unknown".
- Track IDs come from ByteTrack and can change when an object is occluded for a while. Rule cooldowns are per rule to avoid repeat alerts.
- SAM click-to-segment outlines an object on the paused frame so you can save it as a zone. It does not yet follow that object through the video.
- Recordings are saved as H.264 MP4 to `~/Downloads` (set `VLAB_RECORDINGS=/some/folder` to change it). Events and keyframes stay in `data/`. Everything stays on your machine. Be mindful of other people when recording.

## Voice (step 1 of Sense Lab)
Voice tab: talk to gemma3:27b through the browser mic, all on this Mac.
Silero VAD (turns + barge-in) → mlx-whisper large-v3-turbo → Ollama gemma3:27b (streamed) → Kokoro TTS (streamed per sentence).
While a voice turn is live the vision caption worker pauses so voice gets Ollama first.
Turn metrics are shown in the tab, served at `/api/voice/state`, and appended to `data/voice_events.jsonl`
(`kind: voice_turn | voice_interrupt`) for loading into vlab.db in step 2. "Grok Voice (cloud)" is a disabled stub.
Tests (against a running server): `.venv/bin/python tests/voice_e2e.py 8765` and `tests/voice_bargein.py 8765`.

## Multimodal record (step 2)
vlab.db now holds vision, audio and voice together (migrated in place on first start; older rows count as vision).
- `events.modality` = vision | audio | voice | system, plus `written_at` (write lag = written_at − ts), `data` (JSON), `source_key` (dedupe).
- `utterances`: one row per user/assistant utterance — turn_id, role, text, ts_start/ts_end, voice, model,
  latencies (JSON), interrupted, error, nearest keyframe_id (±10 min) and a snapshot of tracked objects/zones/caption.
- Ambient sounds (dog, doorbell, knock, speech, alarm, music, water, loud noise) are classified with
  AST/AudioSet while the Voice mic is open (`ambient` setting) and stored as `modality=audio, kind=sound`.
- The voice brain gets a compact camera summary (now in view, time in view, last seen, recent notable events,
  half-hourly captions, or the log around a time you ask about). Preview it: `/api/context?q=...`.
- APIs: `/api/timeline/events` (merged; since/until/range/modality/q), `/api/search/all?q=`,
  `/api/export?format=jsonl|parquet`. `data/voice_events.jsonl` is imported idempotently at startup.
- Test: `.venv/bin/python tests/multimodal_e2e.py 8765`. Test servers: `VLAB_NO_CAMERA=1 VLAB_DB=/tmp/x.db VLAB_VOICE_LOG=/tmp/x.jsonl`.

## License

MIT. See [LICENSE](LICENSE).

## Cloudflare Pages (portfolio demo)

Static demo site root: `site/`. Sample JSON: `demo/data/data.json` (mirrored at `site/assets/data.json` for the browser).

Suggested Pages project: **vision-lab** · Production branch `main` · Framework none · Root directory `site` · no build command.
