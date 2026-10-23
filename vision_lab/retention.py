"""Storage retention policy.

  screen  Screen captures (planner screenshots and screenrec segments): kept forever. No pruning.
  voice   Mic audio: KEPT as AAC m4a, mono 16 kHz, 32 kbps, in ~/Datasets/vision-lab-audio
          (vision_lab/audiostore.py). Kept forever unless VLAB_AUDIO_KEEP_DAYS is a positive number;
          VLAB_AUDIO_RETAIN=0 = the 10/20 rule (transcribed, then deleted; only text stays).
  video   Camera video segments (recording mode, vision_lab/recording.py): kept forever, like screens. Deletion happens only if VLAB_VIDEO_KEEP_DAYS is explicitly set to a positive
          number of days; unset, empty, 0, "none", "forever" or "keep" all mean keep everything.
          Keyframe JPEGs (data/keyframes) are not video and are never pruned.
"""
import os
import time
from pathlib import Path

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".ts")
FOREVER = ("", "0", "none", "forever", "keep")


def video_keep_days():
    """None = keep forever (the default). A positive number only when VLAB_VIDEO_KEEP_DAYS says so."""
    v = (os.environ.get("VLAB_VIDEO_KEEP_DAYS") or "").strip().lower()
    if v in FOREVER:
        return None
    try:
        d = float(v)
    except ValueError:
        return None          # unparseable: fail safe = keep
    return d if d > 0 else None


def policy():
    from . import audiostore
    a = audiostore.settings()
    return {"screen": "forever", "voice_audio": (a["keep_days"] if a["retain"] else "delete after transcription"),
            "audio_dir": a["dir"] if a["retain"] else None, "video_days": video_keep_days() or "forever"}


POLICY = policy()


def prune_video(dirpath, now=None, days=None, dry_run=False):
    """Delete camera video segments older than the video retention. Only touches video files in `dirpath`.
    With the default (keep forever) this never deletes anything."""
    days = video_keep_days() if days is None else days
    d = Path(dirpath)
    if not days or days <= 0 or not d.exists():
        return []
    cutoff = (now or time.time()) - days * 86400
    old = [f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXT and f.stat().st_mtime < cutoff]
    if not dry_run:
        for f in old:
            f.unlink(missing_ok=True)
    return old
