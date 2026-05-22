"""Tests for Open-Meteo client."""

from datetime import datetime

import pytest

from open_meteo_client import OpenMeteoClient, StormWindow


def test_storm_window_dataclass():
    w = StormWindow(
        start=datetime(2026, 5, 21, 17, 0),
        end=datetime(2026, 5, 21, 20, 0),
        confidence=0.75,
        reasons=['CAPE 1820', 'LI -4.7', 'wind 252°'],
    )
    assert w.start.hour == 17
    assert w.end.hour == 20
    assert 'CAPE' in w.reasons[0]
    assert w.duration_hours == 3.0


def test_client_init_reads_config():
    client = OpenMeteoClient()
    assert client.cape_min == 1500
    assert client.lifted_index_max == -3.0
    assert client.wind_fov_margin == 30


def test_qualifies_returns_true_for_storm_hour(openmeteo_response):
    """May 21 19:00 had CAPE 1960, LI -5.1, wind 249° — should qualify."""
    client = OpenMeteoClient()
    hour_data = {
        'cape': 1960.0,
        'lifted_index': -5.1,
        'wind_direction_10m': 249,
        'precipitation_probability': 25,
        'weather_code': 3,
        'cloud_cover': 100,
    }
    qualifies, reasons = client._qualifies(hour_data)
    assert qualifies is True
    assert any('CAPE' in r for r in reasons)
    assert any('100' in r for r in reasons)  # cloud cover satisfies require_any_of


def test_qualifies_returns_false_for_low_cape():
    client = OpenMeteoClient()
    hour_data = {
        'cape': 800.0,
        'lifted_index': -5.0,
        'wind_direction_10m': 250,
        'precipitation_probability': 60,
        'weather_code': 95,
        'cloud_cover': 100,
    }
    qualifies, _ = client._qualifies(hour_data)
    assert qualifies is False


def test_qualifies_returns_false_for_wind_outside_arc():
    """Wind from 90° (East) — opposite the camera (WNW)."""
    client = OpenMeteoClient()
    hour_data = {
        'cape': 2000.0,
        'lifted_index': -5.0,
        'wind_direction_10m': 90,
        'precipitation_probability': 60,
        'weather_code': 95,
        'cloud_cover': 100,
    }
    qualifies, _ = client._qualifies(hour_data)
    assert qualifies is False


def test_qualifies_handles_none_values():
    """Open-Meteo can return None for fields when data is unavailable."""
    client = OpenMeteoClient()
    hour_data = {
        'cape': None,
        'lifted_index': -5.0,
        'wind_direction_10m': 250,
        'precipitation_probability': 60,
        'weather_code': 95,
        'cloud_cover': 100,
    }
    qualifies, _ = client._qualifies(hour_data)
    assert qualifies is False
