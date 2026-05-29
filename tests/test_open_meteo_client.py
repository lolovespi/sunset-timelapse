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


def test_merge_contiguous_hours():
    """Three consecutive qualifying hours merge into one window."""
    client = OpenMeteoClient()
    qualifying = [
        (datetime(2026, 5, 21, 17, 0), 0.6, ['CAPE 1570', 'LI -4.6']),
        (datetime(2026, 5, 21, 18, 0), 0.7, ['CAPE 1820', 'LI -4.7']),
        (datetime(2026, 5, 21, 19, 0), 0.8, ['CAPE 1960', 'LI -5.1']),
    ]
    windows = client._merge_into_windows(qualifying)
    assert len(windows) == 1
    assert windows[0].start == datetime(2026, 5, 21, 17, 0)
    assert windows[0].end == datetime(2026, 5, 21, 20, 0)  # last hour + 1
    assert windows[0].confidence == pytest.approx(0.7, abs=0.01)


def test_merge_with_gap_splits():
    """Hours with a gap produce separate windows."""
    client = OpenMeteoClient()
    qualifying = [
        (datetime(2026, 5, 21, 17, 0), 0.6, ['a']),
        (datetime(2026, 5, 21, 18, 0), 0.6, ['b']),
        (datetime(2026, 5, 21, 22, 0), 0.6, ['c']),  # gap
        (datetime(2026, 5, 21, 23, 0), 0.6, ['d']),
    ]
    windows = client._merge_into_windows(qualifying)
    assert len(windows) == 2
    assert windows[0].end == datetime(2026, 5, 21, 19, 0)
    assert windows[1].start == datetime(2026, 5, 21, 22, 0)


def test_merge_empty_returns_empty():
    client = OpenMeteoClient()
    assert client._merge_into_windows([]) == []


def test_get_storm_watch_windows_from_fixture(monkeypatch, openmeteo_response):
    """May 21 should produce a watch window for evening hours."""
    client = OpenMeteoClient()
    # Force the fetch to return the recorded fixture
    monkeypatch.setattr(client, 'fetch_forecast', lambda **kwargs: openmeteo_response)

    windows = client.get_storm_watch_windows()

    # We expect at least one window covering May 21 evening
    may_21_windows = [w for w in windows if w.start.date() == datetime(2026, 5, 21).date()]
    assert len(may_21_windows) >= 1
    w = may_21_windows[0]
    assert w.start.hour <= 17    # qualifying starts no later than 17:00
    assert w.end.hour >= 20      # extends through at least 19:00 inclusive
    assert any('CAPE' in r for r in w.reasons)


@pytest.mark.integration
def test_fetch_forecast_live():
    """Hit the real Open-Meteo API. Marked integration to allow opt-out."""
    client = OpenMeteoClient()
    data = client.fetch_forecast()
    assert 'hourly' in data
    assert 'cape' in data['hourly']
    assert len(data['hourly']['time']) > 24
