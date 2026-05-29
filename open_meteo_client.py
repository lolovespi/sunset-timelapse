"""
Open-Meteo Forecast Client

Polls Open-Meteo forecast API and identifies storm watch windows
based on CAPE, lifted index, wind direction, and precipitation triggers.

See docs/superpowers/specs/2026-05-22-storm-capture-design.md for design.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

from config_manager import get_config
from geography_calculator import GeographyCalculator


API_URL = 'https://api.open-meteo.com/v1/forecast'


@dataclass
class StormWindow:
    """A contiguous range of hours that meet storm-watch criteria."""
    start: datetime
    end: datetime
    confidence: float                          # 0-1; mean across hours
    reasons: List[str] = field(default_factory=list)

    @property
    def duration_hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0


class OpenMeteoClient:
    """Polls Open-Meteo, produces storm watch windows for the camera's FOV."""

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.geography = GeographyCalculator()

        # Trigger thresholds (configurable)
        self.cape_min = self.config.get('open_meteo.triggers.cape_min', 1500)
        self.lifted_index_max = self.config.get('open_meteo.triggers.lifted_index_max', -3.0)
        self.wind_fov_margin = self.config.get('open_meteo.triggers.wind_fov_margin_degrees', 30)
        self.precip_prob_min = self.config.get(
            'open_meteo.triggers.require_any_of.precipitation_probability_min', 40)
        self.weather_code_min = self.config.get(
            'open_meteo.triggers.require_any_of.weather_code_min', 80)
        self.cloud_cover_min = self.config.get(
            'open_meteo.triggers.require_any_of.cloud_cover_min', 80)

        # Location
        location = self.config.get('location', {})
        self.latitude = location.get('latitude')
        self.longitude = location.get('longitude')
        self.timezone = location.get('timezone', 'America/Chicago')

        if self.latitude is None or self.longitude is None:
            raise ValueError(
                "OpenMeteoClient requires 'location.latitude' and 'location.longitude' "
                "in config"
            )

    def _qualifies(self, hour_data: Dict) -> tuple[bool, List[str]]:
        """
        Evaluate whether one hour's forecast meets storm-watch criteria.

        Returns (qualifies, reasons). qualifies=False on any None input.
        """
        cape = hour_data.get('cape')
        lifted_index = hour_data.get('lifted_index')
        wind_dir = hour_data.get('wind_direction_10m')
        precip_prob = hour_data.get('precipitation_probability') or 0
        weather_code = hour_data.get('weather_code') or 0
        cloud_cover = hour_data.get('cloud_cover') or 0

        # Bail on None for required fields
        if cape is None or lifted_index is None or wind_dir is None:
            return False, []

        reasons = []

        # CAPE
        if cape < self.cape_min:
            return False, []
        reasons.append(f"CAPE {cape:.0f}")

        # Lifted index (negative = unstable; smaller = more unstable)
        if lifted_index > self.lifted_index_max:
            return False, []
        reasons.append(f"LI {lifted_index:.1f}")

        # Wind direction in camera arc + margin
        cone = self.geography.get_viewing_cone()
        margin = self.wind_fov_margin
        lo = (cone.azimuth_min - margin) % 360
        hi = (cone.azimuth_max + margin) % 360
        if lo > hi:
            in_arc = wind_dir >= lo or wind_dir <= hi
        else:
            in_arc = lo <= wind_dir <= hi
        if not in_arc:
            return False, []
        reasons.append(f"wind {wind_dir}°")

        # At least one of: precip prob, weather code, cloud cover
        any_triggered = False
        if precip_prob >= self.precip_prob_min:
            reasons.append(f"precip {precip_prob}%")
            any_triggered = True
        if weather_code >= self.weather_code_min:
            reasons.append(f"WMO {weather_code}")
            any_triggered = True
        if cloud_cover >= self.cloud_cover_min:
            reasons.append(f"cloud {cloud_cover}%")
            any_triggered = True

        if not any_triggered:
            return False, []

        return True, reasons

    def fetch_forecast(
        self,
        past_days: int = 0,
        forecast_days: int = 2,
        timeout: float = 10.0,
    ) -> Dict:
        """
        Fetch forecast from Open-Meteo. Returns the full JSON response.

        Raises: requests.exceptions.* on network errors.
        """
        params = {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'hourly': ','.join([
                'temperature_2m', 'precipitation_probability', 'precipitation',
                'weather_code', 'cloud_cover', 'pressure_msl',
                'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
                'cape', 'lifted_index',
            ]),
            'past_days': past_days,
            'forecast_days': forecast_days,
            'timezone': self.timezone,
            'temperature_unit': 'fahrenheit',
            'wind_speed_unit': 'mph',
            'precipitation_unit': 'inch',
        }
        response = requests.get(API_URL, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def get_storm_watch_windows(
        self,
        past_days: int = 0,
        forecast_days: int = 2,
    ) -> List[StormWindow]:
        """
        Fetch the forecast, evaluate each hour, return merged storm watch windows.

        Returns empty list on fetch failure (logged warning, not raised).
        """
        try:
            data = self.fetch_forecast(past_days=past_days, forecast_days=forecast_days)
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Open-Meteo fetch failed: {e}")
            return []
        except (ValueError, KeyError) as e:
            self.logger.warning(f"Open-Meteo response malformed: {e}")
            return []

        hourly = data.get('hourly', {})
        times = hourly.get('time', [])
        if not times:
            self.logger.warning("Open-Meteo response missing hourly.time")
            return []

        keys = [
            'cape', 'lifted_index', 'wind_direction_10m',
            'precipitation_probability', 'weather_code', 'cloud_cover',
        ]
        # Sanity: every key must be present (None values are OK)
        for k in keys:
            if k not in hourly:
                self.logger.warning(f"Open-Meteo response missing hourly.{k}")
                return []

        qualifying = []
        for i, time_str in enumerate(times):
            hour_data = {k: hourly[k][i] for k in keys}
            qualifies, reasons = self._qualifies(hour_data)
            if qualifies:
                # Use a simple confidence proxy: 0.4 base + 0.1 per "any_of" trigger met
                conf = 0.4 + 0.1 * sum(1 for r in reasons if any(
                    tag in r for tag in ('precip', 'WMO', 'cloud')))
                hour_dt = datetime.fromisoformat(time_str)
                qualifying.append((hour_dt, min(conf, 1.0), reasons))

        return self._merge_into_windows(qualifying)

    def _merge_into_windows(
        self,
        qualifying: List[tuple],  # list of (datetime, confidence, reasons)
    ) -> List[StormWindow]:
        """
        Merge strictly contiguous qualifying hours into windows.
        A gap of any size between qualifying hours creates separate windows.
        """
        if not qualifying:
            return []

        windows: List[StormWindow] = []
        cur_start = qualifying[0][0]
        cur_end = cur_start + timedelta(hours=1)
        cur_confidences = [qualifying[0][1]]
        cur_reasons = list(qualifying[0][2])

        for hour_time, conf, reasons in qualifying[1:]:
            if hour_time == cur_end:
                # Contiguous: extend
                cur_end = hour_time + timedelta(hours=1)
                cur_confidences.append(conf)
                # Dedupe reasons across hours; first occurrence wins
                for r in reasons:
                    if r not in cur_reasons:
                        cur_reasons.append(r)
            else:
                # Gap: emit current window, start new one
                windows.append(StormWindow(
                    start=cur_start, end=cur_end,
                    confidence=sum(cur_confidences) / len(cur_confidences),
                    reasons=cur_reasons,
                ))
                cur_start = hour_time
                cur_end = hour_time + timedelta(hours=1)
                cur_confidences = [conf]
                cur_reasons = list(reasons)

        windows.append(StormWindow(
            start=cur_start, end=cur_end,
            confidence=sum(cur_confidences) / len(cur_confidences),
            reasons=cur_reasons,
        ))
        return windows
