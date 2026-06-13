from datetime import datetime
from app.services.weather import get_weather_forecast
from app.services.text import write, write_centered
import time
import numpy as np

class Clock:
    WEATHER_INTERVAL = 60*60
    CLOCK_INTERVAL = 1

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.last_weather_update = time.time()-Clock.WEATHER_INTERVAL
        self.last_frame_update = time.time()-Clock.CLOCK_INTERVAL
        self.weather = None
        self.frame = np.zeros((height, width), dtype=np.uint8)

    def get_weather(self):
        if time.time() - self.last_weather_update > Clock.WEATHER_INTERVAL:
            self.weather = get_weather_forecast()
            self.last_weather_update = time.time()
        return self.weather

    def get_frame(self):
        if time.time() - self.last_frame_update > Clock.CLOCK_INTERVAL:
            self.update_frame()
        
        return self.frame

    def update_frame(self):
        self.frame = np.zeros((self.height, self.width), dtype=np.uint8)
        now = datetime.now()
        date_now = now.strftime('%d.%m.%y')
        time_now = now.strftime('%H:%M')
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
        self.frame[26, 2:now.hour+2] = 1

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
            for rain_forecasts in weather['hourly_rain_forecast']:
                hour = int(rain_forecasts['time'].split(':')[0])
                rain_prob = round(rain_forecasts['rain_probability'] * 4)
                self.frame[20:24, hour+1:26] = 0
                self.frame[24-rain_prob:24, hour+1:26] = 1