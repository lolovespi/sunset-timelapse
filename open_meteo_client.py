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
