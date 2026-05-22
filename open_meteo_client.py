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
