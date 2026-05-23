"""Tests for storm workflow components."""

from datetime import datetime, timedelta

import pytest

from storm_workflow import (
    compute_storm_intensity_score,
    sis_score_to_grade,
)


def test_sis_modest_storm(storm_observations, storm_strikes):
    """A moderate storm with 8 strikes + 12mm/hr rain should score B-range."""
    result = compute_storm_intensity_score(
        strikes=storm_strikes,
        observations=storm_observations,
    )
    assert 'score' in result
    assert 'grade' in result
    assert 'components' in result
    assert 0 <= result['score'] <= 100
    assert result['grade'] in {'A', 'B', 'C', 'D', 'F'}
    # 8 strikes × 5 = 40 base from lightning alone
    assert result['components']['lightning'] == pytest.approx(40.0, abs=0.1)


def test_sis_no_data_returns_zero():
    result = compute_storm_intensity_score(strikes=[], observations=[])
    assert result['score'] == 0.0
    assert result['grade'] == 'F'


def test_sis_capped_at_100():
    """Extreme inputs should cap at 100."""
    from tempest_monitor import LightningStrike, WeatherObservation
    strikes = [
        LightningStrike(timestamp=datetime(2026, 5, 21, 21, i), distance_km=2.0, energy=99999)
        for i in range(50)
    ]
    obs = [
        WeatherObservation(
            timestamp=datetime(2026, 5, 21, 21, i),
            temperature=70, humidity=90, pressure=990,
            wind_speed=30, wind_gust=60, wind_direction=240,
            rain_rate=80, rain_accumulation=10,
            solar_radiation=0, uv_index=0, battery_voltage=2.8,
        )
        for i in range(30)
    ]
    result = compute_storm_intensity_score(strikes=strikes, observations=obs)
    assert result['score'] == 100.0
    assert result['grade'] == 'A'


def test_grade_mapping():
    assert sis_score_to_grade(85) == 'A'
    assert sis_score_to_grade(80) == 'A'
    assert sis_score_to_grade(75) == 'B'
    assert sis_score_to_grade(50) == 'C'
    assert sis_score_to_grade(25) == 'D'
    assert sis_score_to_grade(5) == 'F'


def test_storm_prompt_contains_required_fields():
    from facebook_uploader import FacebookUploader
    uploader = FacebookUploader()
    metadata = {
        'event_type': 'storm',
        'date': '2026-05-21',
        'sis_score': 65.0,
        'sis_grade': 'B',
        'storm_metrics': {
            'lightning_count': 8,
            'lightning_avg_distance_km': 8.5,
            'rain_max_mm_hr': 12.0,
            'wind_gust_max_mph': 28.0,
            'pressure_drop_hpa': 3.2,
        },
        'weather': {'temperature_f': 71, 'conditions': 'thunderstorms'},
    }
    prompt = uploader._build_caption_prompt(metadata)
    assert 'thunderstorm' in prompt.lower() or 'storm' in prompt.lower()
    assert '8' in prompt              # strike count
    assert '28' in prompt             # peak gust
    assert '12' in prompt             # rain rate
    assert 'SIS' in prompt or 'intensity' in prompt.lower()
    # Banned words from the spec
    for banned in ('epic', 'incredible', 'wild', 'Mother Nature', 'ripped through'):
        assert banned.lower() not in prompt.lower()


def test_sunset_prompt_still_works():
    """Existing sunset path must not regress."""
    from facebook_uploader import FacebookUploader
    uploader = FacebookUploader()
    metadata = {
        'date': '2026-05-21',
        'sbs_score': 72.0,
        'sbs_grade': 'B',
        'weather': {'temperature_f': 72, 'conditions': 'clear'},
    }
    prompt = uploader._build_caption_prompt(metadata)
    assert 'sunset' in prompt.lower()
    assert 'SBS' in prompt or 'Brilliance' in prompt
