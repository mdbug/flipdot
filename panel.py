from flippydot import flippydot
import serial

class Panel:
    def __init__(self, preview=False):
        self.preview = preview
        self.serial = None
        if not preview:
            self.serial = serial.Serial(
                port='/dev/ttyUSB0',
                baudrate=57600,
                timeout=1,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )

        self.panel = flippydot.Panel([
            [1],
            [2],
            [3],
            [4],
        ], 28, 7, module_rotation=0, screen_preview=preview, screen_preview_scaling_factor=20)

        self.WIDTH = self.panel.get_total_width()
        self.HEIGHT = self.panel.get_total_height()

    def update(self, frame):
        serial_data = self.panel.apply_frame(frame)
        if not self.preview:
            self.serial.write(serial_data)