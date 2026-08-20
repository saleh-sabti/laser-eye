"""Thin wrapper around cv2.VideoCapture that degrades gracefully when no
camera is attached yet, so the rest of the app (and the UI) can be built
and exercised before real hardware is plugged in.

Flask serves each request on its own thread (threaded=True), and several
routes can hit the same shared Camera object at once - the live stream
polls it continuously while a calibration/design/training snapshot request
can land at the same moment. OpenCV's Windows DSHOW backend isn't safe to
call from multiple threads concurrently; without a lock this can silently
crash the whole process (no Python traceback - it's a native-level fault),
which is what was happening here before this lock was added.
"""
import threading

import cv2
import numpy as np

_NO_CAMERA_FRAME_TEXT = "No camera detected"

# A live DSHOW connection can go stale after running for a long time (the
# device still reports isOpened()==True, but every frame comes back flat
# black - observed after several hours of continuous use). Distinguish that
# from a legitimately dark scene (e.g. the recommended dark bed mat) by
# requiring near-zero *variance* too: a stuck/dead connection returns a
# genuinely uniform buffer, whereas a dark-but-real scene still has some
# texture/noise.
DEAD_FRAME_MEAN_THRESHOLD = 5.0
DEAD_FRAME_STD_THRESHOLD = 2.0
DEAD_FRAME_STREAK_LIMIT = 5


def probe_devices(max_index: int = 5) -> list[int]:
    """Return indices of camera devices that actually open."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found


class Camera:
    def __init__(self, index: int):
        self.index = index
        self._lock = threading.RLock()
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self._dead_frame_streak = 0

    def _ensure_open(self) -> None:
        # A device plugged in (or replugged) after this object was created
        # won't be picked up unless we retry opening it here - isOpened()
        # alone would stay stuck on whatever the first attempt saw.
        if not self._cap.isOpened():
            self._reconnect()

    def _reconnect(self) -> None:
        self._cap.release()
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        self._dead_frame_streak = 0

    @property
    def is_open(self) -> bool:
        with self._lock:
            self._ensure_open()
            return self._cap.isOpened()

    def read(self) -> np.ndarray:
        """Return a BGR frame. Returns a placeholder frame with a status
        message if no camera is available, instead of raising - callers
        (the live view, detection) can keep running and the UI can show
        the placeholder rather than crashing.
        """
        with self._lock:
            self._ensure_open()
            if not self._cap.isOpened():
                return self._placeholder_frame()
            ok, frame = self._cap.read()
            if not ok:
                self._reconnect()
                return self._placeholder_frame()

            if frame.mean() < DEAD_FRAME_MEAN_THRESHOLD and frame.std() < DEAD_FRAME_STD_THRESHOLD:
                self._dead_frame_streak += 1
                if self._dead_frame_streak >= DEAD_FRAME_STREAK_LIMIT:
                    self._reconnect()
                    ok, frame = self._cap.read()
                    if not ok:
                        return self._placeholder_frame()
            else:
                self._dead_frame_streak = 0
            return frame

    def _placeholder_frame(self) -> np.ndarray:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame, _NO_CAMERA_FRAME_TEXT, (60, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 220), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"(index {self.index})", (60, 280),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1, cv2.LINE_AA,
        )
        return frame

    def release(self) -> None:
        with self._lock:
            self._cap.release()
