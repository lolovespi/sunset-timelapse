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
