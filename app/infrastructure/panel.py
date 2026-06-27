import logging
import queue
import threading

import numpy as np
import serial
from flippydot import flippydot

from app.modes.contracts import Frame

logger = logging.getLogger(__name__)


class Panel:
    """Drive the 28×28 flip-dot display (or a pygame preview).

    Wraps the ``flippydot`` library: in hardware mode it diffs each frame
    against the last and writes only changed modules over serial from a
    background thread; in preview mode it renders to an on-screen window.
    """

    def __init__(self, preview: bool = False) -> None:
        self.preview = preview
        self.serial: serial.Serial | None = None
        self._write_queue: queue.SimpleQueue[bytes] | None = None
        self._writer_thread: threading.Thread | None = None

        if not preview:
            self.serial = serial.Serial(
                port="/dev/ttyUSB0",
                baudrate=57600,
                timeout=1,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
            )
            # Background thread drains the write queue so the main loop
            # never blocks on serial I/O.
            self._write_queue = queue.SimpleQueue()
            self._writer_thread = threading.Thread(target=self._serial_writer, daemon=True)
            self._writer_thread.start()
            logger.info("Panel serial writer started on /dev/ttyUSB0")

        self.panel = flippydot.Panel(
            [
                [1],
                [2],
                [3],
                [4],
            ],
            28,
            7,
            module_rotation=0,
            screen_preview=preview,
            screen_preview_scaling_factor=20,
        )

        self.WIDTH = self.panel.get_total_width()
        self.HEIGHT = self.panel.get_total_height()
        self._prev_frame = np.full((self.HEIGHT, self.WIDTH), 255, dtype=np.uint8)

    def _serial_writer(self) -> None:
        """Drain the write queue in a background thread, writing to serial."""
        assert self._write_queue is not None and self.serial is not None
        while True:
            try:
                data = self._write_queue.get()
                self.serial.write(data)
            except Exception:
                logger.exception("Panel serial writer thread error")

    def update(self, frame: Frame) -> None:
        """Render ``frame`` to the panel, sending serial only for changed modules."""
        if self.preview:
            # Preview path: use the library's full apply_frame (drives cv2 window)
            self.panel.apply_frame(frame)
            return

        # Diff-based: only send serial commands for modules whose content changed.
        # Layout is 4 rows × 1 column of 28×7 modules stacked vertically.
        module_h = self.panel.module_height  # 7
        module_w = self.panel.module_width  # 28
        serial_bytes = bytearray()

        for row_idx, module_row in enumerate(self.panel.modules):
            for col_idx, module in enumerate(module_row):
                r0 = row_idx * module_h
                c0 = col_idx * module_w
                new_slice = frame[r0 : r0 + module_h, c0 : c0 + module_w]
                if not np.array_equal(
                    new_slice, self._prev_frame[r0 : r0 + module_h, c0 : c0 + module_w]
                ):
                    module.set_content(new_slice)
                    serial_bytes.extend(module.fetch_serial_command().tobytes())

        np.copyto(self._prev_frame, frame)
        if serial_bytes and self._write_queue is not None:
            self._write_queue.put(bytes(serial_bytes))
