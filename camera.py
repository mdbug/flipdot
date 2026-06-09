import cv2
import os
import threading


class Camera:
    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        # Fix webcam contrast
        ret, frame = self.cap.read()
        os.system(f"v4l2-ctl -d /dev/video{camera_index} --set-ctrl=contrast=128")

        # Background reader: always keep the freshest frame ready so the main
        # loop never blocks on camera I/O.
        self._frame = frame
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read_frame(self):
        with self._lock:
            return self._frame