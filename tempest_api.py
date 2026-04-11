"""
Tempest Weather Station REST API Client
Fetches current/historical weather conditions from WeatherFlow Tempest API.
Complements the UDP-based tempest_monitor.py with on-demand data retrieval.
"""

import logging
import os
from datetime import datetime, date
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

    def get_historical_observation(self, target_date: date, sunset_hour: int = 19) -> Optional[Dict]:
        """
        Fetch the observation closest to sunset time on a past date.

        Uses the Tempest device-level API which supports time_start/time_end
        for historical minute-by-minute observations.

        Args:
            target_date: The date to fetch weather for
            sunset_hour: Approximate local sunset hour (24h format, default 19 = 7PM)

        Returns:
            Dict with parsed weather fields, or None on failure.
        """
        if not self.is_configured():
            self.logger.warning("Tempest API not configured")
            return None

        device_id = self.config.get('tempest.device_id', '')
        if not device_id:
            self.logger.warning("Tempest device_id not configured, falling back to current observation")
            return self.get_current_observation()

        # Build window from 3h before sunset to 1h after (need early data for cloud cover)
        sunset_time = datetime(target_date.year, target_date.month, target_date.day, sunset_hour, 0, 0)
        time_start = int(sunset_time.timestamp()) - 10800  # 3 hours before
        time_end = int(sunset_time.timestamp()) + 3600     # 1 hour after

        url = f"{self.BASE_URL}/observations/device/{device_id}"
        params = {
            "token": self.api_token,
            "time_start": time_start,
            "time_end": time_end,
        }

        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            observations = data.get("obs", [])
            if not observations:
                self.logger.warning(f"No historical observations found for {target_date}")
                return None

            # Device obs are fixed-position arrays:
            # [0]  timestamp       [7]  air_temperature (C)
            # [1]  wind_lull (m/s) [8]  relative_humidity (%)
            # [2]  wind_avg (m/s)  [9]  illuminance (lux)
            # [3]  wind_gust (m/s) [10] uv index
            # [4]  wind_direction  [11] solar_radiation (W/m2)
            # [5]  wind_interval   [12] precip_accumulated (mm)
            # [6]  pressure (mb)   [13-17] precip_type, lightning, battery, etc.

            # Pick observation closest to sunset for temp/humidity/wind
            target_ts = sunset_time.timestamp()
            closest = min(observations, key=lambda o: abs(o[0] - target_ts))

            # For cloud cover, use observations 2-3h before sunset when sun is
            # high enough for the 1000 W/m² clear-sky baseline to be meaningful.
            # Near sunset, solar radiation drops regardless of cloud cover.
            early_start = target_ts - 10800  # 3h before sunset
            early_end = target_ts - 7200     # 2h before sunset
            early_obs = [o for o in observations
                         if early_start <= o[0] <= early_end
                         and o[11] is not None and o[11] > 0]
            if early_obs:
                # Average solar radiation over the window for stability
                avg_solar = sum(o[11] for o in early_obs) / len(early_obs)
                cloud_cover_pct = self._estimate_cloud_cover(avg_solar)
            else:
                cloud_cover_pct = None

            temp_c = closest[7]
            humidity = closest[8]
            wind_avg = closest[2]
            wind_dir = closest[4]
            solar_rad = closest[11]
            precip_mm = closest[12]

            temp_f = round(temp_c * 9 / 5 + 32, 1) if temp_c is not None else None
            wind_mph = round(wind_avg * 2.237, 1) if wind_avg is not None else None
            conditions = self._derive_conditions(cloud_cover_pct, precip_mm, wind_avg)

            timestamp_iso = datetime.fromtimestamp(closest[0]).isoformat()

            return {
                "timestamp": timestamp_iso,
                "temperature_f": temp_f,
                "feels_like_f": temp_f,  # device obs don't include feels_like
                "humidity": humidity,
                "wind_speed_mph": wind_mph,
                "wind_direction": wind_dir,
                "cloud_cover_pct": cloud_cover_pct,
                "precip_today_in": round(precip_mm / 25.4, 2) if precip_mm is not None else None,
                "solar_radiation": solar_rad,
                "conditions": conditions,
            }

        except requests.RequestException as e:
            self.logger.error(f"Tempest historical API request failed: {e}")
            return None
        except (KeyError, IndexError, TypeError) as e:
            self.logger.error(f"Failed to parse historical Tempest response: {e}")
            return None

    def get_weather_block(self, target_date: date = None, sunset_hour: int = 19) -> Optional[Dict]:
        """
        Build the weather metadata block for the sunset JSON.

        Args:
            target_date: If provided and not today, fetch historical weather
                         for that date. Otherwise fetch current conditions.
            sunset_hour: Approximate local sunset hour for historical lookups.

        Returns:
            Dict ready to be inserted as the 'weather' key in metadata,
            or None if data is unavailable.
        """
        today = date.today()
        if target_date and target_date < today:
            obs = self.get_historical_observation(target_date, sunset_hour)
        else:
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
