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
    def __init__(self, index: int, width: int = 640, height: int = 480, rotate_180: bool = False):
        self.index = index
        self.width = width
        self.height = height
        # Only 180deg is supported for now - it's a plain flip, so frame
        # dimensions stay the same and nothing downstream (calibration,
        # detection, ROI) needs to know rotation happened at all, since
        # it's applied here before any of them ever see the frame. 90/270
        # would swap width<->height and need touching every place that
        # currently assumes camera_width x camera_height - not done yet.
        self.rotate_180 = rotate_180
        self._lock = threading.RLock()
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self._apply_resolution()
        self._dead_frame_streak = 0

    def _apply_resolution(self) -> None:
        # OpenCV/DSHOW default to 640x480 unless a higher resolution is
        # explicitly requested - most webcams support far more (this
        # project's actually supports up to 1920x1080). Setting these on an
        # unopened/failed capture is harmless; a genuinely unsupported
        # value just gets ignored by the driver and read() falls back to
        # whatever the device actually delivers.
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def _ensure_open(self) -> None:
        # A device plugged in (or replugged) after this object was created
        # won't be picked up unless we retry opening it here - isOpened()
        # alone would stay stuck on whatever the first attempt saw.
        if not self._cap.isOpened():
            self._reconnect()

    def _reconnect(self) -> None:
        self._cap.release()
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        self._apply_resolution()
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
            if self.rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            return frame

    def _placeholder_frame(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cx, cy = self.width // 2 - 180, self.height // 2
        cv2.putText(
            frame, _NO_CAMERA_FRAME_TEXT, (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 220), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"(index {self.index})", (cx, cy + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1, cv2.LINE_AA,
        )
        return frame

    def release(self) -> None:
        with self._lock:
            self._cap.release()
