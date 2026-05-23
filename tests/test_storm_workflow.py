"""Tests for storm workflow components."""

from datetime import datetime, timedelta
from pathlib import Path

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


def test_storm_workflow_init():
    """Workflow class instantiates and exposes required methods."""
    from storm_workflow import StormWorkflow
    wf = StormWorkflow()
    assert hasattr(wf, 'complete_storm_workflow')
    assert hasattr(wf, '_compute_storm_duration_seconds')


def test_compute_storm_duration_uses_config():
    """Duration = min(max_duration_hours, post_storm_minutes) by default."""
    from storm_workflow import StormWorkflow
    wf = StormWorkflow()
    # Default post_storm_minutes=60, max_duration_hours=4
    seconds = wf._compute_storm_duration_seconds()
    assert seconds == 60 * 60   # 60 minutes


def test_storm_metadata_path(tmp_path, monkeypatch):
    """Storm metadata file path is generated per-date."""
    from storm_workflow import StormWorkflow
    from datetime import date

    wf = StormWorkflow()
    # Override storms_dir for test isolation
    wf.storms_dir = tmp_path

    target = date(2026, 5, 21)
    path = wf._storm_metadata_path(target)
    assert path.parent.name == '2026-05-21'
    assert path.name == 'storm_metadata.json'


def test_capture_storm_sequence_calls_camera(monkeypatch):
    """_capture_storm_sequence delegates to camera.capture_video_sequence
    with the right time bounds and a cancel_event."""
    from storm_workflow import StormWorkflow
    from datetime import datetime
    import threading

    wf = StormWorkflow()

    calls = []
    def fake_capture(start_time, end_time, interval_seconds, cancel_event=None):
        calls.append({
            'start': start_time, 'end': end_time,
            'interval': interval_seconds,
            'cancel_event': cancel_event,
        })
        return [Path('/tmp/fake_frame.jpg')]

    monkeypatch.setattr(wf.camera, 'capture_video_sequence', fake_capture)

    start = datetime(2026, 5, 21, 21, 0)
    cancel_event = threading.Event()
    images = wf._capture_storm_sequence(start, cancel_event=cancel_event)

    assert len(images) == 1
    assert len(calls) == 1
    assert calls[0]['cancel_event'] is cancel_event
    duration = (calls[0]['end'] - calls[0]['start']).total_seconds()
    assert duration == wf._compute_storm_duration_seconds()


def test_recovery_triggered_when_zero_frames():
    """Zero captured frames → recovery should be attempted."""
    from storm_workflow import StormWorkflow
    wf = StormWorkflow()
    expected = 720   # 60 min × 60s ÷ 5s
    assert wf._should_attempt_recovery(captured=[], expected_frame_count=expected) is True


def test_recovery_triggered_when_partial(tmp_path):
    """< 50% frames → recovery."""
    from storm_workflow import StormWorkflow
    wf = StormWorkflow()
    expected = 720
    partial = [tmp_path / f"frame_{i:04d}.jpg" for i in range(100)]
    assert wf._should_attempt_recovery(captured=partial, expected_frame_count=expected) is True


def test_recovery_not_triggered_when_complete(tmp_path):
    """>= 50% frames → no recovery needed."""
    from storm_workflow import StormWorkflow
    wf = StormWorkflow()
    expected = 720
    full = [tmp_path / f"frame_{i:04d}.jpg" for i in range(500)]
    assert wf._should_attempt_recovery(captured=full, expected_frame_count=expected) is False


def test_complete_storm_workflow_happy_path(monkeypatch, tmp_path, storm_strikes, storm_observations):
    """End-to-end through complete_storm_workflow with mocked I/O."""
    from storm_workflow import StormWorkflow
    from tempest_monitor import StormConditions
    from datetime import datetime
    from pathlib import Path
    import threading

    wf = StormWorkflow()
    wf.storms_dir = tmp_path

    # Mock all I/O components
    fake_video = tmp_path / 'storm_video.mp4'
    fake_video.touch()

    monkeypatch.setattr(wf, '_capture_storm_sequence',
                        lambda start, cancel_event=None: [tmp_path / f'f{i}.jpg' for i in range(800)])
    monkeypatch.setattr(wf.video_processor, 'create_timelapse',
                        lambda images, output_path, **kwargs: fake_video)
    monkeypatch.setattr(wf.drive_uploader, 'upload_video',
                        lambda video, target_date, metadata: {'filename': 'storm.mp4'})
    monkeypatch.setattr(wf.facebook_uploader, 'generate_caption',
                        lambda metadata, video_path=None: "Caption text")
    monkeypatch.setattr(wf.facebook_uploader, 'post_sunset',
                        lambda video_path, metadata, caption_override=None: True)
    monkeypatch.setattr(wf.youtube_uploader, 'upload_video_with_sbs_enhancements',
                        lambda video_path, video_date, start_time, end_time, **kwargs: 'yt_id_123')
    monkeypatch.setattr(wf.email_notifier, 'send_notification', lambda subject, body: None)

    conditions = StormConditions(
        storm_detected=True, confidence=0.7,
        trigger_reasons=['Lightning: 8 strikes', 'High winds'],
        lightning_active=True,
    )

    # Inject Tempest observations + strikes
    monkeypatch.setattr(wf, '_collect_tempest_data',
                        lambda monitor, window_minutes: (storm_strikes, storm_observations))

    # Build a tempest_monitor stub
    class StubMonitor:
        pass

    result = wf.complete_storm_workflow(
        conditions=conditions,
        start_time=datetime(2026, 5, 21, 21, 0),
        tempest_monitor=StubMonitor(),
    )

    assert result is True
    # Metadata file should exist
    assert (tmp_path / '2026-05-21' / 'storm_metadata.json').exists()
