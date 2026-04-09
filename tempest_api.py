"""
Tempest Weather Station REST API Client
Fetches current/historical weather conditions from WeatherFlow Tempest API.
Complements the UDP-based tempest_monitor.py with on-demand data retrieval.
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict

import requests

from config_manager import get_config


class TempestAPI:
    """Fetch weather observations from the Tempest REST API"""

    BASE_URL = "https://swd.weatherflow.com/swd/rest"

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        self.station_id = self.config.get('tempest.station_id', '')
        self.api_token = (
            os.getenv('TEMPEST_API_TOKEN')
            or self.config.get('tempest.api_token', '')
        )
        self.timeout = self.config.get('advanced.connection_timeout_seconds', 30)

    def is_configured(self) -> bool:
        """Check if API credentials are present"""
        return bool(self.station_id and self.api_token)

    def get_current_observation(self) -> Optional[Dict]:
        """
        Fetch the latest observation from the Tempest station.

        Returns:
            Dict with parsed weather fields, or None on failure.
        """
        if not self.is_configured():
            self.logger.warning("Tempest API not configured (missing station_id or TEMPEST_API_TOKEN)")
            return None

        url = f"{self.BASE_URL}/observations/station/{self.station_id}"
        params = {"token": self.api_token}

        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return self._parse_observation(data)

        except requests.RequestException as e:
            self.logger.error(f"Tempest API request failed: {e}")
            return None
        except (KeyError, IndexError, TypeError) as e:
            self.logger.error(f"Failed to parse Tempest API response: {e}")
            return None

    def get_weather_block(self) -> Optional[Dict]:
        """
        Build the weather metadata block for the sunset JSON.

        Returns:
            Dict ready to be inserted as the 'weather' key in metadata,
            or None if data is unavailable.
        """
        obs = self.get_current_observation()
        if obs is None:
            return None

        return {
            "source": "tempest",
            "station_id": self.station_id,
            "observed_at": obs["timestamp"],
            "temperature_f": obs["temperature_f"],
            "feels_like_f": obs["feels_like_f"],
            "humidity_pct": obs["humidity"],
            "wind_speed_mph": obs["wind_speed_mph"],
            "wind_direction_deg": obs["wind_direction"],
            "cloud_cover_pct": obs.get("cloud_cover_pct"),
            "precipitation_today_in": obs["precip_today_in"],
            "conditions": obs["conditions"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_observation(self, data: Dict) -> Dict:
        """Parse the JSON response from /observations/station/{id}"""
        obs = data["obs"][0]  # most recent observation

        temp_c = obs.get("air_temperature")
        feels_c = obs.get("feels_like")
        humidity = obs.get("relative_humidity")
        wind_avg = obs.get("wind_avg")           # m/s
        wind_dir = obs.get("wind_direction")
        precip_day = obs.get("precip_accum_local_day")  # mm
        solar_rad = obs.get("solar_radiation")
        conditions = obs.get("conditions", "")     # sometimes absent

        # Unit conversions
        temp_f = round(temp_c * 9 / 5 + 32, 1) if temp_c is not None else None
        feels_f = round(feels_c * 9 / 5 + 32, 1) if feels_c is not None else None
        wind_mph = round(wind_avg * 2.237, 1) if wind_avg is not None else None
        precip_in = round(precip_day / 25.4, 2) if precip_day is not None else None

        # Derive cloud cover from solar radiation when not provided directly.
        # Tempest doesn't expose a cloud_cover field via the station API;
        # this is a best-effort estimate (None when solar is unavailable or
        # after sunset when radiation is naturally 0).
        cloud_cover_pct = self._estimate_cloud_cover(solar_rad)

        # Derive a human-readable conditions string when the API doesn't
        # supply one (the station-level endpoint often omits it).
        if not conditions:
            conditions = self._derive_conditions(
                cloud_cover_pct, precip_day, wind_avg
            )

        timestamp_epoch = obs.get("timestamp")
        timestamp_iso = (
            datetime.fromtimestamp(timestamp_epoch).isoformat()
            if timestamp_epoch else datetime.now().isoformat()
        )

        return {
            "timestamp": timestamp_iso,
            "temperature_f": temp_f,
            "feels_like_f": feels_f,
            "humidity": humidity,
            "wind_speed_mph": wind_mph,
            "wind_direction": wind_dir,
            "cloud_cover_pct": cloud_cover_pct,
            "precip_today_in": precip_in,
            "solar_radiation": solar_rad,
            "conditions": conditions,
        }

    @staticmethod
    def _estimate_cloud_cover(solar_radiation: Optional[float]) -> Optional[int]:
        """
        Rough cloud-cover estimate from solar radiation.
        Returns None when solar data is missing or zero (nighttime).
        """
        if solar_radiation is None or solar_radiation <= 0:
            return None
        # Clear-sky baseline ~1000 W/m²; scale linearly.
        clear_sky = 1000.0
        ratio = min(solar_radiation / clear_sky, 1.0)
        cloud_pct = int(round((1.0 - ratio) * 100))
        return max(0, min(cloud_pct, 100))

    @staticmethod
    def _derive_conditions(
        cloud_pct: Optional[int],
        precip_mm: Optional[float],
        wind_ms: Optional[float],
    ) -> str:
        """Derive a simple conditions label from available metrics."""
        if precip_mm is not None and precip_mm > 1.0:
            return "rainy"
        if cloud_pct is None:
            return "unknown"
        if cloud_pct < 15:
            return "clear"
        if cloud_pct < 50:
            return "partly cloudy"
        if cloud_pct < 85:
            return "mostly cloudy"
        return "overcast"
