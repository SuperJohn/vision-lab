# Architecture

All processing is on-device (Mac Studio, M3 Ultra). Nothing leaves the machine.

```mermaid
flowchart LR
    subgraph Capture
        CAM[Camera<br/>webcam]
        MIC[Mic<br/>browser getUserMedia]
        SCR[Screen<br/>screencapture every 10 s]
    end
    PRIV{{Privacy mode<br/>config.settings.capture_paused}}
    PRIV -. stops .-> CAM
    PRIV -. stops .-> MIC
    PRIV -. stops .-> SCR

    subgraph Fast["Fast loop (every frame, server :8765)"]
        DET[YOLO11 detect / seg / pose<br/>+ ByteTrack]
        ZR[Zones & rules]
        VOX[Voice: ASR + local LLM + TTS<br/>audio events]
    end
    subgraph Slow["Slow loop (vlabbatch / planloop)"]
        GEM[gemma4 VLM<br/>captions + detections]
        EMB[SigLIP embeddings]
        PL[planloop<br/>desktop samples, activity slices]
    end

    CAM --> DET --> ZR
    MIC --> VOX
    SCR --> GEM
    DET -- keyframes --> GEM
    DET -- keyframes --> EMB
    SCR --> PL

    DB[(data/vlab.db<br/>events · presence · keyframes<br/>utterances · batch_annotations · plan_*)]
    ZR --> DB
    VOX --> DB
    GEM --> DB
    EMB --> DB
    PL --> DB

    DQ[vision_lab.dq<br/>couch FP · duplicate person IDs]
    DIG[Digest<br/>timeline summary · nudges]
    DB --> DQ --> DIG
    DB --> DIG
```

- **Fast loop**: per-frame detection/tracking in the server process; writes `events` and `presence`, saves keyframes.
- **Slow loop**: `vision_lab.batch` (gemma4) captions keyframes and screen captures into `batch_annotations`; SigLIP embeds keyframes for text search; `planloop` samples the desktop and produces activity slices and nudges.
- **Storage**: one SQLite file; schema in [event-model.md](event-model.md).
- **DQ**: `python -m vision_lab.dq` (read-only) measures detector false positives and tracker ID churn; see [dq.md](dq.md).
- **Digest**: timeline LLM summary and planloop nudges read from the same DB.
