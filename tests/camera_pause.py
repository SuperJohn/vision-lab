"""Pause must close every capture handle (macOS camera light off); resume must reopen it.

Runs without a real camera: cv2.VideoCapture is replaced by a fake that counts open handles.
    .venv/bin/python tests/camera_pause.py
"""
import os, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["VLAB_NO_CAMERA"] = "1"
import numpy as np
from vision_lab import engine

OPEN = set()


class FakeCap:
    def __init__(self, *a, **k):
        self.alive = True
        OPEN.add(self)

    def isOpened(self): return self.alive
    def set(self, *a): return True
    def get(self, *a): return 30

    def read(self):
        time.sleep(0.03)  # blocking read, like AVFoundation
        return (True, np.zeros((4, 4, 3), np.uint8)) if self.alive else (False, None)

    def release(self):
        self.alive = False
        OPEN.discard(self)


engine.cv2.VideoCapture = FakeCap


def wait(cond, t=3.0):
    end = time.time() + t
    while time.time() < end:
        if cond(): return True
        time.sleep(0.02)
    return False


def main():
    cam = engine.Camera()
    cam.open(0)
    assert wait(lambda: cam.latest()[0] is not None), "camera never produced a frame"
    assert len(OPEN) == 1

    assert cam.pause() is True, "pause did not confirm release"
    assert not OPEN, f"{len(OPEN)} capture handle(s) still open after pause"
    assert cam.latest()[0] is None
    seq = cam.latest()[1]
    cam.open(0)                      # selecting a camera while paused must not reopen it
    time.sleep(0.5)
    assert not OPEN and cam.latest()[1] == seq, "device reopened while paused"

    cam.resume()
    assert wait(lambda: cam.latest()[0] is not None), "resume did not reopen the camera"
    assert len(OPEN) == 1

    for _ in range(5):               # rapid toggles never leak a handle
        cam.pause(); cam.resume()
    cam.pause()
    assert not OPEN, "handle leaked after rapid pause/resume"

    started_paused = engine.Camera(paused=True)
    started_paused.open(0)
    time.sleep(0.5)
    assert started_paused.paused and not OPEN, "a camera that starts paused must not open the device"
    print("camera_pause: OK")


if __name__ == "__main__":
    main()
