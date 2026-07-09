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


class Camera:
    """Capture webcam frames on a background thread, exposing the latest one.

    A reader thread continuously grabs frames so the main loop never blocks on
    camera I/O; ``read_frame`` returns the freshest captured BGR frame.
    """

    def __init__(
        self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30
    ) -> None:
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

        # Fix webcam contrast
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read initial frame from camera index {camera_index}")
        # Pass the (integer) device index as a discrete argv element rather than
        # interpolating it into a shell string.
        try:
            subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    f"/dev/video{int(camera_index)}",
                    "--set-ctrl=contrast=128",
                ],
                check=False,
            )
        except (OSError, ValueError) as exc:
            logger.warning("Failed to set camera contrast (index=%s): %s", camera_index, exc)
        logger.info(
            "Camera initialized index=%s width=%s height=%s fps=%s",
            camera_index,
            width,
            height,
            fps,
        )

        # Background reader: always keep the freshest frame ready so the main
        # loop never blocks on camera I/O.
        self._frame = frame
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        """Continuously grab frames, keeping only the most recent."""
        while True:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self._lock:
                        self._frame = frame
                else:
                    logger.warning("Camera read returned no frame (index=%s)", self.camera_index)
                    # Avoid a busy loop when the device opens but stops yielding
                    # frames (unplugged, driver hiccup).
                    time.sleep(_READ_RETRY_DELAY_SEC)
            except Exception:
                logger.exception("Camera reader thread error (index=%s)", self.camera_index)
                # Avoid hot-looping if the camera backend is temporarily unavailable.
                time.sleep(_READ_RETRY_DELAY_SEC)

    def read_frame(self) -> np.ndarray:
        """Return the most recently captured BGR frame."""
        with self._lock:
            return self._frame
