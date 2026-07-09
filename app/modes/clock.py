import logging
import math
import threading
import time
from datetime import datetime
from typing import Any

import numpy as np

from app.modes.contracts import Frame
from app.services.draw import draw_line, fill_circle, thick_line
from app.services.text import write, write_centered
from app.services.weather import get_weather_forecast

logger = logging.getLogger(__name__)


class Clock:
    """Clock mode: renders the time as a digital or analog face (web-configurable).

    The digital face shows date, time, an hour-progress bar, and a weather strip.
    The analog face fills the panel with a clock dial: a rim, 12 hour ticks, and
    hour/minute hands.
    """

    WEATHER_INTERVAL = 60 * 60
    CLOCK_INTERVAL = 1

    STYLE_DIGITAL = "digital"
    STYLE_ANALOG = "analog"
    STYLES = (STYLE_DIGITAL, STYLE_ANALOG)

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.last_weather_update = time.time() - Clock.WEATHER_INTERVAL
        self.last_frame_update = time.time() - Clock.CLOCK_INTERVAL
        self.weather: dict[str, Any] | None = None
        # A refresh is in flight on a background thread; guards against stacking
        # up concurrent fetches while one is still running.
        self._weather_refresh_in_flight = False
        self.style = Clock.STYLE_DIGITAL
        self.seconds = False
        self.frame = np.zeros((height, width), dtype=np.uint8)

    def get_settings(self) -> dict[str, Any]:
        """Return the current clock settings for the web UI."""
        return {"style": self.style, "seconds": self.seconds}

    def update_settings(self, *, style: str, seconds: bool | None = None) -> dict[str, Any]:
        """Update the clock face style and second-hand toggle; return the result.

        Unknown ``style`` values are ignored. ``seconds`` only affects the analog
        face but is stored regardless so the toggle persists across face changes.
        """
        changed = False
        if style in Clock.STYLES and style != self.style:
            self.style = style
            changed = True
        if seconds is not None and bool(seconds) != self.seconds:
            self.seconds = bool(seconds)
            changed = True
        if changed:
            # Force a re-render on the next get_frame so the change is immediate.
            self.last_frame_update = time.time() - Clock.CLOCK_INTERVAL
        return self.get_settings()

    def get_weather(self) -> dict[str, Any] | None:
        """Return the cached forecast, kicking off a throttled background refresh.

        The fetch runs on a daemon thread so a slow (or hung) API call can never
        stall the single-threaded render loop that calls this. The interval
        timestamp is advanced when the refresh is *scheduled* (not when it
        finishes) so a slow fetch does not spawn a new thread every frame.
        """
        now = time.time()
        if (
            now - self.last_weather_update > Clock.WEATHER_INTERVAL
            and not self._weather_refresh_in_flight
        ):
            self.last_weather_update = now
            self._weather_refresh_in_flight = True
            threading.Thread(
                target=self._refresh_weather, name="clock-weather", daemon=True
            ).start()
        return self.weather

    def _refresh_weather(self) -> None:
        """Fetch the forecast off the render thread, keeping only a valid payload.

        ``get_weather_forecast`` returns an ``{"error": ...}`` dict on failure;
        storing that would make the digital renderer raise on a missing
        ``current_temperature`` key, so an error (or exception) leaves the last
        good forecast in place instead.
        """
        try:
            result = get_weather_forecast()
        except Exception:
            logger.exception("Weather refresh failed")
            result = None
        if isinstance(result, dict) and "error" not in result:
            self.weather = result
        self._weather_refresh_in_flight = False

    def get_frame(self) -> Frame:
        """Return the clock frame, re-rendering at most once per ``CLOCK_INTERVAL``."""
        if time.time() - self.last_frame_update > Clock.CLOCK_INTERVAL:
            self.update_frame()

        return self.frame

    def update_frame(self) -> None:
        """Redraw the clock frame using the configured face style."""
        self.frame = np.zeros((self.height, self.width), dtype=np.uint8)
        if self.style == Clock.STYLE_ANALOG:
            self._render_analog()
        else:
            self._render_digital()

    def _render_digital(self) -> None:
        """Render the digital face: date, time, hour-progress bar, and weather/rain strip."""
        now = datetime.now()
        date_now = now.strftime("%d.%m.%y")
        time_now = now.strftime("%H:%M")
        write(self.frame, date_now, y=0, size=5, style="regular")
        write_centered(
            self.frame,
            time_now,
            y=6,
            font="scoreline",
            size=6,
            style="regular",
            spacing=1,
        )

        self.frame[25, 1:27] = 1
        self.frame[26, 1] = 1
        self.frame[26, 26] = 1
        self.frame[27, 1:27] = 1
        self.frame[26, 2 : now.hour + 2] = 1

        weather = self.get_weather()
        if weather is not None:
            write(
                self.frame,
                f"{weather['current_temperature']}°/{weather['max_temperature_today']}°",
                x=1,
                y=14,
                size=5,
                style="regular",
            )
            self.frame[20:24, 1:27] = 0
            for rain_forecasts in weather["hourly_rain_forecast"]:
                hour = int(rain_forecasts["time"].split(":")[0])
                rain_prob = round(rain_forecasts["rain_probability"] * 4)
                self.frame[20:24, hour + 1 : 26] = 0
                self.frame[24 - rain_prob : 24, hour + 1 : 26] = 1

    def _render_analog(self) -> None:
        """Render an analog face: a white disc with black hour/minute hands.

        The dial is drawn on an odd-diameter circle centered on a single pixel.
        It touches the left and top edges and leaves a 1px margin on the right
        and bottom, while staying exactly symmetric about that center. The face
        is a solid white disc; the hour and minute hands are cut out in black as
        1px, gap-free strokes (the hour hand shorter than the minute hand).
        """
        now = datetime.now()
        # Integer center so mirror pairs round identically; the radius reaches
        # the left/top edges and leaves a 1px margin on the right/bottom.
        cx = (self.width - 1) // 2
        cy = (self.height - 1) // 2
        radius = min(cx, cy)

        # White clock face. Drawing it half a pixel larger rounds off the four
        # cardinal tips so the rim has no protruding single pixels, while the
        # disc still reaches the left/top edges.
        fill_circle(self.frame, (cx, cy), radius + 0.5)

        # Hands measured clockwise from 12-o'clock-up: x = cx + L*sin, y = cy - L*cos.
        minute_angle = math.radians(now.minute * 6)
        hour_angle = math.radians((now.hour % 12) * 30 + now.minute * 0.5)
        minute_len = radius - 1
        hour_len = radius * 0.7
        # The minute hand is a single pixel wide; the hour hand is shorter and
        # 3px wide so it reads as the bolder of the two.
        draw_line(
            self.frame,
            (cx, cy),
            (cx + minute_len * math.sin(minute_angle), cy - minute_len * math.cos(minute_angle)),
            color=0,
        )
        thick_line(
            self.frame,
            (cx, cy),
            (cx + hour_len * math.sin(hour_angle), cy - hour_len * math.cos(hour_angle)),
            width=3,
            color=0,
        )

        # Optional second hand: a 1px black stroke reaching the rim, drawn last so
        # it sweeps over the hour and minute hands.
        if self.seconds:
            second_angle = math.radians(now.second * 6)
            second_len = radius
            draw_line(
                self.frame,
                (cx, cy),
                (
                    cx + second_len * math.sin(second_angle),
                    cy - second_len * math.cos(second_angle),
                ),
                color=0,
            )
