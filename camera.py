import cv2
import os

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

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame