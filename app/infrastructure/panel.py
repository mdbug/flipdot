from flippydot import flippydot
import numpy as np
import serial
import threading
import queue


class Panel:
    def __init__(self, preview=False):
        self.preview = preview
        self.serial = None
        self._write_queue = None
        self._writer_thread = None

        if not preview:
            self.serial = serial.Serial(
                port='/dev/ttyUSB0',
                baudrate=57600,
                timeout=1,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            # Background thread drains the write queue so the main loop
            # never blocks on serial I/O.
            self._write_queue = queue.SimpleQueue()
            self._writer_thread = threading.Thread(
                target=self._serial_writer, daemon=True)
            self._writer_thread.start()

        self.panel = flippydot.Panel([
            [1],
            [2],
            [3],
            [4],
        ], 28, 7, module_rotation=0, screen_preview=preview, screen_preview_scaling_factor=20)

        self.WIDTH = self.panel.get_total_width()
        self.HEIGHT = self.panel.get_total_height()
        self._prev_frame = np.full((self.HEIGHT, self.WIDTH), 255, dtype=np.uint8)

    def _serial_writer(self):
        """Drain write queue in a background thread."""
        while True:
            data = self._write_queue.get()
            self.serial.write(data)

    def update(self, frame):
        if self.preview:
            # Preview path: use the library's full apply_frame (drives cv2 window)
            self.panel.apply_frame(frame)
            return

        # Diff-based: only send serial commands for modules whose content changed.
        # Layout is 4 rows × 1 column of 28×7 modules stacked vertically.
        module_h = self.panel.module_height  # 7
        module_w = self.panel.module_width   # 28
        serial_bytes = bytearray()

        for row_idx, module_row in enumerate(self.panel.modules):
            for col_idx, module in enumerate(module_row):
                r0 = row_idx * module_h
                c0 = col_idx * module_w
                new_slice = frame[r0:r0 + module_h, c0:c0 + module_w]
                if not np.array_equal(new_slice, self._prev_frame[r0:r0 + module_h, c0:c0 + module_w]):
                    module.set_content(new_slice)
                    serial_bytes.extend(module.fetch_serial_command().tobytes())

        np.copyto(self._prev_frame, frame)
        if serial_bytes:
            self._write_queue.put(bytes(serial_bytes))