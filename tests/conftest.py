"""Shared pytest fixtures for storm capture tests."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


FIXTURES_DIR = Path(__file__).parent / 'fixtures'
REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True, scope='session')
def use_example_config():
    """
    Point config_manager's global singleton at example-config.yaml so tests
    don't require a real config.yaml (which is not checked in to the worktree).
    """
    import config_manager
    config_manager.config = config_manager.ConfigManager(
        str(REPO_ROOT / 'example-config.yaml')
    )
    yield
    config_manager.config = None


@pytest.fixture(autouse=True, scope='session')
def patch_heavy_components(tmp_path_factory):
    """
    Patch heavy component constructors so StormWorkflow (and any other class
    that instantiates CameraInterface, VideoProcessor, etc.) can be unit-tested
    without live credentials or network access.

    Also patches get_storage_paths so tests don't create real filesystem dirs
    at paths referencing the example-config placeholder user.
    """
    base = tmp_path_factory.mktemp('data')
    fake_paths = {
        'base': base,
        'images': base / 'images',
        'videos': base / 'videos',
        'logs': base / 'logs',
        'temp': base / 'temp',
    }

    with patch('camera_interface.CameraInterface.__init__', return_value=None), \
         patch('video_processor.VideoProcessor.__init__', return_value=None), \
         patch('drive_uploader.DriveUploader.__init__', return_value=None), \
         patch('facebook_uploader.FacebookUploader.__init__', return_value=None), \
         patch('youtube_uploader.YouTubeUploader.__init__', return_value=None), \
         patch('email_notifier.EmailNotifier.__init__', return_value=None), \
         patch('config_manager.ConfigManager.get_storage_paths', return_value=fake_paths):
        yield


@pytest.fixture
def openmeteo_response():
    """Recorded Open-Meteo response covering May 20-23 2026 for Pelham, AL."""
    with open(FIXTURES_DIR / 'openmeteo_sample.json') as f:
        return json.load(f)


@pytest.fixture
def camera_arc():
    """Camera FOV arc for Pelham deployment: 236°-324° (azimuth 280° ± 43.5°)."""
    return {'azimuth_min': 236.5, 'azimuth_max': 323.5, 'azimuth_center': 280.0}


@pytest.fixture
def storm_observations():
    """Synthetic Tempest observations during a thunderstorm."""
    from tempest_monitor import WeatherObservation
    now = datetime(2026, 5, 21, 21, 0, 0)
    return [
        WeatherObservation(
            timestamp=now + timedelta(minutes=i),
            temperature=72.0 - i * 0.3,
            humidity=85.0 + i * 0.5,
            pressure=1015.0 - i * 0.2,
            wind_speed=8.0 + i * 0.4,
            wind_gust=15.0 + i * 0.5,
            wind_direction=240,
            rain_rate=12.0 if i >= 3 else 0.0,
            rain_accumulation=i * 0.5,
            solar_radiation=0.0,
            uv_index=0.0,
            battery_voltage=2.8,
        )
        for i in range(20)
    ]


@pytest.fixture
def storm_strikes():
    """Synthetic Tempest lightning strikes during a thunderstorm."""
    from tempest_monitor import LightningStrike
    base = datetime(2026, 5, 21, 21, 5, 0)
    return [
        LightningStrike(timestamp=base + timedelta(minutes=i), distance_km=5.0 + i, energy=50000)
        for i in range(8)
    ]
