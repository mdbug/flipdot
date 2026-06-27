import os
from datetime import datetime
from typing import Any

import requests
from dotenv import load_dotenv

CITY = "Aachen"
COUNTRY_CODE = "DE"

load_dotenv()


def get_weather_forecast(
    city: str = CITY, country_code: str = COUNTRY_CODE, api_key: str | None = None
) -> dict[str, Any]:
    """Return current temp, today's max, and hourly rain probability from OpenWeatherMap.

    Falls back to ``OPENWEATHER_API_KEY`` when ``api_key`` is omitted, and returns a
    ``{"error": ...}`` payload (rather than raising) on missing key or API failure.
    """
    if api_key is None:
        api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return {"error": "Missing OPENWEATHER_API_KEY environment variable"}

    # Base URLs for OpenWeatherMap API
    current_weather_url = "http://api.openweathermap.org/data/2.5/weather"
    forecast_url = "http://api.openweathermap.org/data/2.5/forecast"

    # Parameters for API calls
    params = {
        "q": f"{city},{country_code}",
        "appid": api_key,
        "units": "metric",  # Use metric for Celsius
    }

    try:
        # Get current weather
        current_response = requests.get(current_weather_url, params=params)
        current_response.raise_for_status()
        current_data = current_response.json()

        # Get forecast data
        forecast_response = requests.get(forecast_url, params=params)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()

        # Extract current temperature
        current_temp = current_data["main"]["temp"]

        # Find today's data from forecast
        today = datetime.now().date()
        today_forecasts = []
        max_temp_today = current_temp  # Initialize with current temp

        for item in forecast_data["list"]:
            forecast_time = datetime.fromtimestamp(item["dt"])
            if forecast_time.date() == today:
                today_forecasts.append(item)
                # Update max temperature
                temp = item["main"]["temp"]
                if temp > max_temp_today:
                    max_temp_today = temp

        # Create hourly rain probability data
        hourly_rain_data = []
        for forecast in today_forecasts:
            time_str = datetime.fromtimestamp(forecast["dt"]).strftime("%H:%M")
            rain_prob = forecast.get("pop", 0)
            rain_amount = forecast.get("rain", {}).get("3h", 0)

            hourly_rain_data.append(
                {"time": time_str, "rain_probability": rain_prob, "rain_amount_mm": rain_amount}
            )

        # Compile results
        weather_report = {
            "location": f"{city}, {country_code}",
            "current_temperature": round(current_temp),
            "max_temperature_today": round(max_temp_today),
            "hourly_rain_forecast": hourly_rain_data,
        }

        return weather_report

    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except KeyError as e:
        return {"error": f"Unexpected API response format: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
