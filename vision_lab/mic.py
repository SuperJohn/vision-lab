"""Microphone on/off toggle and input-device selection (persisted in the config table).

Audio is captured in the browser (getUserMedia) and streamed to /ws/voice; the server also
enumerates macOS input devices with ffmpeg/avfoundation so the picker and a level test work
without a browser. When the mic is off the server drops all incoming audio, so no voice turns
and no ambient sound events (barks, knocks) are produced.

Privacy mode locks the mic: while it is on, the mic cannot be turned on or tested
(409) and enabled() reports False even if the stored flag says otherwise."""
import os
import re
import subprocess

import numpy as np
from fastapi import APIRouter, Body, HTTPException

router = APIRouter()
STORE = {"store": None}
# Set by the server: privacy() -> True while Privacy mode is on.
PRIVACY = {"on": None}
PRIVACY_MSG = "Privacy mode is on, so the mic stays off. Turn off Privacy mode (the PRIVATE button) first."


def privacy_on():
    f = PRIVACY["on"]
    if f is None:
        return False
    try:
        return bool(f())
    except Exception:
        return True   # fail closed: unknown privacy state locks the mic


def _refuse_if_private():
    if privacy_on():
        raise HTTPException(409, PRIVACY_MSG)


def list_devices():
    """Input devices as [{index, name, manufacturer}] from avfoundation + system_profiler."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                         capture_output=True, text=True, timeout=15).stderr
    devs, audio = [], False
    for line in out.splitlines():
        if "audio devices" in line:
            audio = True
            continue
        m = re.search(r"\] \[(\d+)\] (.+)$", line)
        if audio and m:
            devs.append({"index": int(m.group(1)), "name": m.group(2).strip(), "manufacturer": None})
    try:
        sp = subprocess.run(["system_profiler", "SPAudioDataType"], capture_output=True, text=True, timeout=15).stdout
        cur = None
        for line in sp.splitlines():
            h = re.match(r"^ {8}(\S.*):$", line)
            if h:
                cur = h.group(1)
            m = re.match(r"^\s+Manufacturer: (.+)$", line)
            if cur and m:
                for d in devs:
                    if d["name"] == cur:
                        d["manufacturer"] = m.group(1).strip()
    except Exception:
        pass
    return devs


def _is_lark(d):
    s = f"{d['name']} {d.get('manufacturer') or ''}".lower()
    return "lark" in s or "hollyland" in s


def pick_default(devs):
    for d in devs:
        if _is_lark(d):
            return d["name"]
    return None


def state():
    st = STORE["store"]
    private = privacy_on()
    return {"enabled": bool(st.get_config("mic_enabled", True)) and not private,
            "device": st.get_config("mic_device", None), "privacy": private}


def enabled():
    return state()["enabled"]


def ensure_default():
    """Startup log only. No device is auto-selected: unset (None) means the system default input."""
    devs = list_devices()
    lark = pick_default(devs)
    print(f"[mic] inputs: {[d['name'] for d in devs]}; Lark selectable: {lark or 'not detected'}")


@router.get("/api/mic")
def get_mic():
    devs = list_devices()
    s = state()
    sel = next((d for d in devs if d["name"] == s["device"]), None)
    lark = pick_default(devs)
    return {**s, "device_present": sel is not None, "device_is_lark": bool(sel and _is_lark(sel)),
            "manufacturer": sel and sel["manufacturer"], "lark_detected": lark,
            "warning": None if lark else f"Lark M2 not detected; visible inputs: {[d['name'] for d in devs]}",
            "devices": devs}


@router.get("/api/mic/state")
def get_mic_state():
    """Cheap state read (no device scan) for UI refreshes, e.g. after Privacy mode changes."""
    return state()


@router.post("/api/mic")
def set_mic(patch: dict = Body(...)):
    st = STORE["store"]
    if patch.get("enabled"):
        _refuse_if_private()
    if "enabled" in patch:
        st.set_config("mic_enabled", bool(patch["enabled"]))
    if "device" in patch:
        names = [d["name"] for d in list_devices()]
        if patch["device"] is not None and patch["device"] not in names:
            raise HTTPException(400, f"unknown input device; visible: {names}")
        st.set_config("mic_device", patch["device"])
    return state()


@router.post("/api/mic/toggle")
def toggle_mic():
    if not enabled():
        _refuse_if_private()
    STORE["store"].set_config("mic_enabled", not enabled())
    return state()


@router.post("/api/mic/test")
def test_mic(seconds: float = 2.0):
    """Record a short clip from the selected device server-side and report its level."""
    if os.environ.get("VLAB_NO_MIC"):   # preview/test servers never touch the mic
        raise HTTPException(409, "This server runs with VLAB_NO_MIC=1, so it never records from the mic.")
    _refuse_if_private()
    dev = state()["device"]
    idx = "default" if dev is None else next((d["index"] for d in list_devices() if d["name"] == dev), None)
    if idx is None:
        raise HTTPException(404, f"selected device {dev!r} not visible")
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "avfoundation", "-i", f":{idx}",
                        "-t", str(min(max(seconds, 0.5), 10)), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                       capture_output=True, timeout=30)
    a = np.frombuffer(p.stdout, np.float32)
    if not len(a):
        raise HTTPException(500, f"no audio captured: {p.stderr.decode()[-300:]}")
    rms, peak = float(np.sqrt(np.mean(a ** 2))), float(np.abs(a).max())
    db = lambda x: round(20 * np.log10(max(x, 1e-9)), 1)
    return {"device": dev or "system default", "samples": len(a), "rms": rms, "peak": peak, "rms_dbfs": db(rms), "peak_dbfs": db(peak),
            "nonzero": peak > 0}
