import logging
import subprocess
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Pause between retries when the camera opens but stops delivering frames, so a
# stalled device can't spin the reader thread into a busy loop that pegs a core
# and floods the log.
_READ_RETRY_DELAY_SEC = 0.1

# Pause between attempts to (re)open a camera that is absent or has been
# unplugged. Long enough that a permanently missing device costs nothing, short
# enough that replugging one recovers within a couple of seconds.
_REOPEN_DELAY_SEC = 2.0

# Consecutive failed reads tolerated before the device is treated as lost and
# reopened from scratch. At _READ_RETRY_DELAY_SEC apart this rides out ~3s of
# driver hiccups, which is well beyond a transient stall but far short of the
# indefinite stream of warnings an unplugged camera used to produce.
_FAILED_READS_BEFORE_REOPEN = 30


class Camera:
    """Capture webcam frames on a background thread, exposing the latest one.

    A reader thread continuously grabs frames so the main loop never blocks on
    camera I/O; ``read_frame`` returns the freshest captured BGR frame.

    A missing or failed camera is not fatal: construction always succeeds, and
    the reader thread keeps trying to (re)open the device in the background. While
    no camera is available ``read_frame`` hands back a black placeholder and
    ``is_available`` reports ``False``, so the caller can skip pose inference and
    fall back to the camera-less modes rather than bringing the app down. Hot
    unplug and replug are handled by the same path, so the installation recovers
    on its own once the device returns.
    """

    def __init__(
        self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30
    ) -> None:
        self.camera_index = camera_index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None
        # Black stand-in handed out until a real frame arrives. Keeping the
        # return type a plain array (never None) means crop/pose/RenderContext
        # need no camera-specific special cases downstream.
        self._frame: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        self._available = False
        self._lock = threading.Lock()

        if not self._open():
            logger.warning(
                "Camera unavailable at startup (index=%s); continuing without it and "
                "retrying in the background. Pose-driven modes stay disabled until it opens.",
                camera_index,
            )

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _open(self) -> bool:
        """Open the capture device and grab one frame; return whether it worked."""
        try:
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        except Exception:
            logger.exception("Failed to construct capture device (index=%s)", self.camera_index)
            return False

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False

        self._apply_contrast()
        self._cap = cap
        with self._lock:
            self._frame = frame
            self._available = True
        logger.info(
            "Camera initialized index=%s width=%s height=%s fps=%s",
            self.camera_index,
            self._width,
            self._height,
            self._fps,
        )
        return True

    def _apply_contrast(self) -> None:
        """Fix washed-out webcam contrast via v4l2-ctl, logging (not raising) on failure."""
        # Pass the (integer) device index as a discrete argv element rather than
        # interpolating it into a shell string.
        try:
            subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    f"/dev/video{int(self.camera_index)}",
                    "--set-ctrl=contrast=128",
                ],
                check=False,
            )
        except (OSError, ValueError) as exc:
            logger.warning("Failed to set camera contrast (index=%s): %s", self.camera_index, exc)

    def _release(self) -> None:
        """Drop the capture device and fall back to the placeholder frame.

        The last captured frame is discarded rather than held: a frozen image
        would render as a stuck mirror in the camera-fed modes, whereas a black
        frame makes the outage obvious and matches ``is_available`` being False.
        """
        cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                logger.exception("Failed to release capture device (index=%s)", self.camera_index)
        with self._lock:
            self._frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            self._available = False

    def _reader(self) -> None:
        """Continuously grab frames, keeping only the most recent.

        Reopens the device whenever it stops delivering frames, so an unplugged
        camera is picked back up automatically once it returns.
        """
        failed_reads = 0
        while True:
            cap = self._cap
            if cap is None:
                time.sleep(_REOPEN_DELAY_SEC)
                if self._open():
                    failed_reads = 0
                continue

            try:
                ret, frame = cap.read()
            except Exception:
                logger.exception("Camera reader thread error (index=%s)", self.camera_index)
                ret, frame = False, None

            if ret:
                failed_reads = 0
                with self._lock:
                    self._frame = frame
                continue

            failed_reads += 1
            if failed_reads >= _FAILED_READS_BEFORE_REOPEN:
                logger.warning(
                    "Camera stopped delivering frames (index=%s); reopening", self.camera_index
                )
                self._release()
                failed_reads = 0
            else:
                # Avoid a busy loop when the device opens but stops yielding
                # frames (unplugged, driver hiccup).
                time.sleep(_READ_RETRY_DELAY_SEC)

    def is_available(self) -> bool:
        """Return whether a camera is currently open and delivering frames."""
        with self._lock:
            return self._available

    def read_frame(self) -> np.ndarray:
        """Return the most recent BGR frame, or a black placeholder if none is available."""
        with self._lock:
            return self._frame
