# Storm Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automated storm timelapse capture to the sunset-timelapse system using Open-Meteo for prediction and Tempest for real-time confirmation, with social parity to the existing sunset workflow.

**Architecture:** Two new modules (`open_meteo_client.py`, `storm_workflow.py`) plus targeted edits to `sunset_scheduler.py`, `facebook_uploader.py`, `tempest_monitor.py`, and `main.py`. Layered: Open-Meteo polls every 15-30 min and schedules storm watch windows; Tempest UDP listener fires the actual capture trigger when ground conditions confirm within an active window. Storm wins over sunset (active sunset capture aborts cleanly). Captured footage processes through the existing video → caption → social pipeline with a storm-flavored prompt.

**Tech Stack:** Python 3.9+, pytest, requests (HTTP), existing libs (Astral, FFmpeg, OpenCV, Anthropic SDK, Google APIs).

**Spec:** See `docs/superpowers/specs/2026-05-22-storm-capture-design.md`.

---

## File Structure

| File | Role | Action |
|---|---|---|
| `open_meteo_client.py` | Forecast layer: poll Open-Meteo, identify storm watch windows | **CREATE** |
| `storm_workflow.py` | Storm-specific capture/upload pipeline + SIS computation | **CREATE** |
| `tempest_monitor.py` | Add `armed` flag gating storm callbacks; commit existing visibility-heuristic edits | **MODIFY** |
| `sunset_scheduler.py` | Wire new modules, add storm callback, capture-state machine, cancellation | **MODIFY** |
| `facebook_uploader.py` | Storm-prompt branch in `_build_caption_prompt`; add `#alabamawx` to HASHTAGS | **MODIFY** |
| `camera_interface.py` | Add `cancel_event` parameter to `capture_video_sequence` | **MODIFY** |
| `main.py` | Add `test --weather`, `test --tempest`, `test --storm-prompt`, `weather --backtest` | **MODIFY** |
| `config.yaml` | New `open_meteo:` block; activate `tempest.enabled` | **MODIFY** |
| `config-pi.yaml` | Add missing `tempest:` block + `open_meteo:` block | **MODIFY** |
| `CLAUDE.md` | Note storm workflow path | **MODIFY** |
| `tests/conftest.py` | pytest shared fixtures (config stub, sample observations) | **CREATE** |
| `tests/fixtures/openmeteo_sample.json` | Recorded Open-Meteo response for stable tests | **CREATE** |
| `tests/test_open_meteo_client.py` | Unit tests for parsing, qualification, merging | **CREATE** |
| `tests/test_storm_workflow.py` | Unit tests for SIS, prompt builder | **CREATE** |
| `tests/test_tempest_monitor.py` | Unit tests for armed-flag gating, visibility heuristic | **CREATE** |

---

## Task 0: Foundation — commit pending edits and bootstrap pytest

**Files:**
- Modify: `tempest_monitor.py` (uncommitted edits from brainstorming — already on disk)
- Modify: `CLAUDE.md` (uncommitted rewrite — already on disk)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`

- [ ] **Step 1: Verify uncommitted state**

Run: `git status --short`

Expected:
```
 M CLAUDE.md
 M tempest_monitor.py
```

- [ ] **Step 2: Commit existing tempest_monitor.py + CLAUDE.md edits as foundation**

```bash
git add tempest_monitor.py CLAUDE.md
git commit -m "$(cat <<'EOF'
Foundation for storm capture: visibility heuristic + CLAUDE.md refresh

- Add is_storm_likely_visible() to TempestMonitor (log-only diagnostic)
- Rewrite CLAUDE.md to reflect current shared-caption architecture and
  social posting pipeline

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Create pytest.ini**

Create `pytest.ini` with:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -ra --strict-markers
markers =
    integration: marks tests requiring network or hardware (deselect with -m 'not integration')
```

- [ ] **Step 4: Create tests/__init__.py (empty file)**

```bash
touch tests/__init__.py
```

- [ ] **Step 5: Create tests/conftest.py**

```python
"""Shared pytest fixtures for storm capture tests."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / 'fixtures'


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
```

- [ ] **Step 6: Verify pytest runs (no tests yet)**

Run: `source venv/bin/activate && pytest`

Expected: `no tests ran in 0.XXs` (pytest finds the empty tests/ dir, exits cleanly).

- [ ] **Step 7: Commit pytest infrastructure**

```bash
git add pytest.ini tests/__init__.py tests/conftest.py
git commit -m "Bootstrap pytest infrastructure for storm capture tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 1: Save Open-Meteo response fixture

**Files:**
- Create: `tests/fixtures/openmeteo_sample.json`

- [ ] **Step 1: Create fixtures directory**

```bash
mkdir -p tests/fixtures
```

- [ ] **Step 2: Fetch a stable Open-Meteo sample covering May 20-23 2026**

Run:
```bash
curl -sS 'https://api.open-meteo.com/v1/forecast?latitude=33.2856&longitude=-86.8097&current=temperature_2m,precipitation,weather_code,cloud_cover,pressure_msl,wind_speed_10m,wind_direction_10m,wind_gusts_10m,cape&hourly=temperature_2m,precipitation_probability,precipitation,weather_code,cloud_cover,pressure_msl,wind_speed_10m,wind_direction_10m,wind_gusts_10m,cape,lifted_index&past_days=2&forecast_days=2&timezone=America%2FChicago&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch' -o tests/fixtures/openmeteo_sample.json
```

- [ ] **Step 3: Verify the fixture has hourly data**

Run:
```bash
python3 -c "import json; d=json.load(open('tests/fixtures/openmeteo_sample.json')); print('hourly times:', len(d['hourly']['time']), 'cape values:', len([v for v in d['hourly']['cape'] if v is not None]))"
```

Expected: `hourly times: 96  cape values: 96` (or similar — both > 0).

- [ ] **Step 4: Commit the fixture**

```bash
git add tests/fixtures/openmeteo_sample.json
git commit -m "Add Open-Meteo response fixture (May 20-23 2026, Pelham AL)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: StormWindow dataclass + OpenMeteoClient initialization

**Files:**
- Create: `open_meteo_client.py`
- Create: `tests/test_open_meteo_client.py`

- [ ] **Step 1: Write failing test for StormWindow dataclass + client init**

Add to `tests/test_open_meteo_client.py`:
```python
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
```

- [ ] **Step 2: Run tests — should fail**

Run: `source venv/bin/activate && pytest tests/test_open_meteo_client.py -v`

Expected: ImportError on `open_meteo_client`.

- [ ] **Step 3: Create open_meteo_client.py**

```python
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
```

- [ ] **Step 4: Run tests — should pass**

Run: `pytest tests/test_open_meteo_client.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add open_meteo_client.py tests/test_open_meteo_client.py
git commit -m "Add OpenMeteoClient skeleton with StormWindow dataclass

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: OpenMeteoClient — qualifying hours logic

**Files:**
- Modify: `open_meteo_client.py` (add `_qualifies` method)
- Modify: `tests/test_open_meteo_client.py`

- [ ] **Step 1: Write failing tests for `_qualifies`**

Add to `tests/test_open_meteo_client.py`:
```python
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
```

- [ ] **Step 2: Run tests — should fail**

Run: `pytest tests/test_open_meteo_client.py -v`

Expected: AttributeError on `_qualifies`.

- [ ] **Step 3: Add `_qualifies` to OpenMeteoClient**

Add inside `OpenMeteoClient` class in `open_meteo_client.py`:
```python
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
```

Also add `from typing import Tuple` to imports if not already covered by Python 3.9+ `tuple[...]` syntax.

- [ ] **Step 4: Run tests — should pass**

Run: `pytest tests/test_open_meteo_client.py -v`

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add open_meteo_client.py tests/test_open_meteo_client.py
git commit -m "OpenMeteoClient: add hour-qualification logic

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: OpenMeteoClient — window merging

**Files:**
- Modify: `open_meteo_client.py` (add `_merge_into_windows`)
- Modify: `tests/test_open_meteo_client.py`

- [ ] **Step 1: Write failing tests for `_merge_into_windows`**

Add to `tests/test_open_meteo_client.py`:
```python
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
```

- [ ] **Step 2: Run tests — should fail**

Run: `pytest tests/test_open_meteo_client.py::test_merge_contiguous_hours -v`

Expected: AttributeError on `_merge_into_windows`.

- [ ] **Step 3: Add `_merge_into_windows`**

Add inside `OpenMeteoClient`:
```python
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
```

- [ ] **Step 4: Run tests — should pass**

Run: `pytest tests/test_open_meteo_client.py -v`

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add open_meteo_client.py tests/test_open_meteo_client.py
git commit -m "OpenMeteoClient: add window merging logic

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: OpenMeteoClient — full pipeline + HTTP fetch

**Files:**
- Modify: `open_meteo_client.py` (add `fetch_forecast`, `get_storm_watch_windows`)
- Modify: `tests/test_open_meteo_client.py`

- [ ] **Step 1: Write failing tests using the recorded fixture**

Add to `tests/test_open_meteo_client.py`:
```python
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
```

- [ ] **Step 2: Run unit tests — should fail**

Run: `pytest tests/test_open_meteo_client.py::test_get_storm_watch_windows_from_fixture -v`

Expected: AttributeError on `get_storm_watch_windows`.

- [ ] **Step 3: Add `fetch_forecast` and `get_storm_watch_windows`**

Add inside `OpenMeteoClient`:
```python
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
```

- [ ] **Step 4: Run unit tests — should pass**

Run: `pytest tests/test_open_meteo_client.py -m 'not integration' -v`

Expected: 10 passed (or however many — all non-integration).

- [ ] **Step 5: Optionally run integration test against live API**

Run: `pytest tests/test_open_meteo_client.py::test_fetch_forecast_live -v`

Expected: PASS (requires internet).

- [ ] **Step 6: Commit**

```bash
git add open_meteo_client.py tests/test_open_meteo_client.py
git commit -m "OpenMeteoClient: complete pipeline (fetch + parse + merge)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Storm Intensity Score (pure function)

**Files:**
- Create: `storm_workflow.py`
- Create: `tests/test_storm_workflow.py`

- [ ] **Step 1: Write failing tests for SIS computation**

Create `tests/test_storm_workflow.py`:
```python
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
```

- [ ] **Step 2: Run tests — should fail**

Run: `pytest tests/test_storm_workflow.py -v`

Expected: ImportError on `storm_workflow`.

- [ ] **Step 3: Create storm_workflow.py with SIS computation**

```python
"""
Storm Workflow

Storm-specific capture and upload pipeline.
Mirrors the sunset workflow inside sunset_scheduler.complete_daily_workflow,
but with storm-specific scoring, captions, and orchestration.

See docs/superpowers/specs/2026-05-22-storm-capture-design.md for design.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config_manager import get_config


def sis_score_to_grade(score: float) -> str:
    """Map numeric Storm Intensity Score (0-100) to letter grade."""
    if score >= 80:
        return 'A'
    if score >= 60:
        return 'B'
    if score >= 40:
        return 'C'
    if score >= 20:
        return 'D'
    return 'F'


def compute_storm_intensity_score(
    strikes: list,
    observations: list,
    weights: Optional[Dict] = None,
) -> Dict:
    """
    Compute Storm Intensity Score from Tempest sensor data only.

    No video analysis. Pure function over inputs.

    Args:
        strikes: list of LightningStrike from tempest_monitor
        observations: list of WeatherObservation from tempest_monitor
        weights: optional dict overriding default weights (from config)

    Returns:
        {
            'score': float (0-100),
            'grade': str ('A'-'F'),
            'components': {
                'lightning': float,
                'distance': float,
                'rain': float,
                'wind': float,
                'pressure': float,
            },
            'metrics': {
                'lightning_count': int,
                'lightning_avg_distance_km': float | None,
                'rain_max_mm_hr': float,
                'wind_gust_max_mph': float,
                'pressure_drop_hpa': float,
            },
        }
    """
    if weights is None:
        config = get_config()
        weights = {
            'lightning': config.get('storm_analysis.scoring.lightning_weight', 5.0),
            'distance': config.get('storm_analysis.scoring.distance_weight', 20.0),
            'rain': config.get('storm_analysis.scoring.rain_weight', 2.0),
            'wind': config.get('storm_analysis.scoring.wind_weight', 0.5),
            'pressure': config.get('storm_analysis.scoring.pressure_weight', 10.0),
            'max_score': config.get('storm_analysis.scoring.max_score', 100),
        }

    # Lightning metrics
    lightning_count = len(strikes)
    avg_distance = (
        sum(s.distance_km for s in strikes) / lightning_count
        if lightning_count > 0 else None
    )

    # Observation-derived metrics
    rain_max = max((o.rain_rate for o in observations), default=0.0)
    wind_gust_max = max((o.wind_gust for o in observations), default=0.0)

    pressure_drop = 0.0
    if len(observations) >= 2:
        # Drop = max - min over the observation window
        pressures = [o.pressure for o in observations]
        pressure_drop = max(pressures) - min(pressures)

    # Component scores
    lightning_pts = lightning_count * weights['lightning']
    if avg_distance is not None:
        distance_pts = (40.0 / max(avg_distance, 5.0)) * weights['distance'] / 20.0
    else:
        distance_pts = 0.0
    rain_pts = rain_max * weights['rain']
    wind_pts = wind_gust_max * weights['wind']
    pressure_pts = abs(pressure_drop) * weights['pressure']

    total = lightning_pts + distance_pts + rain_pts + wind_pts + pressure_pts
    score = min(total, float(weights['max_score']))

    return {
        'score': round(score, 1),
        'grade': sis_score_to_grade(score),
        'components': {
            'lightning': round(lightning_pts, 2),
            'distance': round(distance_pts, 2),
            'rain': round(rain_pts, 2),
            'wind': round(wind_pts, 2),
            'pressure': round(pressure_pts, 2),
        },
        'metrics': {
            'lightning_count': lightning_count,
            'lightning_avg_distance_km': avg_distance,
            'rain_max_mm_hr': round(rain_max, 2),
            'wind_gust_max_mph': round(wind_gust_max, 2),
            'pressure_drop_hpa': round(pressure_drop, 2),
        },
    }
```

- [ ] **Step 4: Run tests — should pass**

Run: `pytest tests/test_storm_workflow.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add storm_workflow.py tests/test_storm_workflow.py
git commit -m "Add storm_workflow with SIS computation (Tempest-data only)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Storm caption prompt + facebook_uploader integration

**Files:**
- Modify: `facebook_uploader.py` (extend `_build_caption_prompt`, update `HASHTAGS`)
- Modify: `tests/test_storm_workflow.py` (test the storm-branch behavior via a tiny adapter)

- [ ] **Step 1: Write failing test for storm-branch prompt**

Add to `tests/test_storm_workflow.py`:
```python
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
```

- [ ] **Step 2: Run tests — should fail**

Run: `pytest tests/test_storm_workflow.py::test_storm_prompt_contains_required_fields -v`

Expected: Some failure — the existing `_build_caption_prompt` doesn't branch on event_type.

- [ ] **Step 3: Modify `facebook_uploader.py:HASHTAGS` to add #alabamawx**

Edit line 33 of `facebook_uploader.py`:
```python
    HASHTAGS = "#Pelham #Alabama #AlabamaWx #SunsetTimelapse #Sunset #BirminghamAL #AlabamaSky #SunsetLovers #Timelapse #GoldenHour #NaturePhotography"
```

(Inserts `#AlabamaWx` after `#Alabama`.)

- [ ] **Step 4: Add storm-prompt branch at the top of `_build_caption_prompt`**

In `facebook_uploader.py`, replace the start of `_build_caption_prompt` (around line 371) with:

```python
    def _build_caption_prompt(self, metadata: Dict[str, Any]) -> str:
        """Build the prompt for Anthropic API.

        Branches on metadata['event_type']: 'sunset' (default) or 'storm'.
        """
        event_type = metadata.get('event_type', 'sunset')
        if event_type == 'storm':
            return self._build_storm_caption_prompt(metadata)

        # ---- Sunset prompt (existing logic) ----
        # [keep the existing implementation below this point unchanged]
```

Then add a new method right after `_build_caption_prompt` (before `_sbs_score_to_grade`):

```python
    def _build_storm_caption_prompt(self, metadata: Dict[str, Any]) -> str:
        """Build the prompt for storm-flavored captions."""
        date_str = metadata.get('date', 'today')
        try:
            from datetime import datetime as dt
            d = dt.fromisoformat(date_str)
            date_display = d.strftime('%B %d, %Y')
        except (ValueError, TypeError):
            date_display = date_str

        sm = metadata.get('storm_metrics', {})
        lightning_count = sm.get('lightning_count', 0)
        avg_dist = sm.get('lightning_avg_distance_km')
        rain_max = sm.get('rain_max_mm_hr', 0.0)
        wind_gust = sm.get('wind_gust_max_mph', 0.0)
        pressure_drop = sm.get('pressure_drop_hpa', 0.0)
        sis_score = metadata.get('sis_score')
        sis_grade = metadata.get('sis_grade')

        weather = metadata.get('weather', {})
        temp = weather.get('temperature_f')

        sensor_lines = []
        if lightning_count > 0:
            line = f"- Lightning: {lightning_count} strike(s)"
            if avg_dist is not None:
                line += f", avg distance {avg_dist:.1f} km"
            sensor_lines.append(line)
        if wind_gust > 0:
            sensor_lines.append(f"- Peak wind gust: {wind_gust:.0f} mph")
        if rain_max > 0:
            sensor_lines.append(f"- Peak rain rate: {rain_max:.1f} mm/hr")
        if pressure_drop > 0:
            sensor_lines.append(f"- Pressure drop: {pressure_drop:.1f} hPa over the capture window")
        if sis_score is not None:
            sensor_lines.append(f"- Storm Intensity Score (SIS): {sis_score:.0f}/100 (grade {sis_grade or '?'})")
        if temp is not None:
            sensor_lines.append(f"- Temperature: {temp}°F")

        sensor_block = '\n'.join(sensor_lines) if sensor_lines else '- (no sensor data available)'

        return f"""Write a 2-3 sentence caption for a thunderstorm timelapse from Mont Vaughn Purvis (MVP), a home in Pelham, Alabama (Birmingham area), captured on {date_display}.

Sensor data:
{sensor_block}

VOICE — direct, dry, observational. No hype, no slang, no weather-channel drama.

RULES:
- 2-3 sentences max
- Include the date and "Pelham, AL" naturally
- Lead with the most concrete sensor reading available (lightning count, peak gust, peak rain, or pressure drop, whichever is most striking)
- Mention the storm's apparent direction of travel only if confident from wind data
- Reference visual character only from the frames provided — actual cloud structure, lightning visible, sky color, rain visibility
- End with "Camera: Reolink RLC810-WA" on its own line
- No emojis, no hashtags
- Avoid: "epic", "incredible", "wild", "intense", "Mother Nature", "ripped through", "unleashed", "fury"

Caption:"""
```

- [ ] **Step 5: Run tests — both new and the sunset-doesn't-regress test should pass**

Run: `pytest tests/test_storm_workflow.py -v`

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add facebook_uploader.py tests/test_storm_workflow.py
git commit -m "FacebookUploader: storm caption prompt branch + #AlabamaWx hashtag

- _build_caption_prompt now branches on metadata['event_type']
- New _build_storm_caption_prompt for thunderstorm captions
- HASHTAGS updated to include #AlabamaWx (applies to sunset + storm)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: TempestMonitor — add `armed` flag gating callbacks

**Files:**
- Modify: `tempest_monitor.py`
- Modify: `tests/test_tempest_monitor.py` (CREATE)

- [ ] **Step 1: Write failing tests for armed-flag behavior**

Create `tests/test_tempest_monitor.py`:
```python
"""Tests for Tempest monitor armed-flag gating."""

from datetime import datetime

import pytest

from tempest_monitor import TempestMonitor, StormConditions


def test_armed_flag_defaults_to_false():
    monitor = TempestMonitor()
    assert monitor.armed is False


def test_arm_and_disarm():
    monitor = TempestMonitor()
    monitor.arm()
    assert monitor.armed is True
    monitor.disarm()
    assert monitor.armed is False


def test_callback_not_fired_when_disarmed(monkeypatch):
    monitor = TempestMonitor()
    fired = []
    monitor.register_storm_callback(lambda c: fired.append(c))
    monitor.disarm()

    # Synthesize storm conditions and try to fire
    conditions = StormConditions(
        storm_detected=True, confidence=0.5,
        trigger_reasons=['test'], lightning_active=True,
    )
    monitor._fire_storm_callbacks(conditions)
    assert fired == []


def test_callback_fired_when_armed():
    monitor = TempestMonitor()
    fired = []
    monitor.register_storm_callback(lambda c: fired.append(c))
    monitor.arm()

    conditions = StormConditions(
        storm_detected=True, confidence=0.5,
        trigger_reasons=['test'], lightning_active=True,
    )
    monitor._fire_storm_callbacks(conditions)
    assert len(fired) == 1
```

- [ ] **Step 2: Run tests — should fail**

Run: `pytest tests/test_tempest_monitor.py -v`

Expected: AttributeError on `armed` / `arm` / `disarm` / `_fire_storm_callbacks`.

- [ ] **Step 3: Modify `tempest_monitor.py` to add the armed flag and extract callback firing**

In `tempest_monitor.py`, find the `__init__` and add after `self.storm_callbacks`:
```python
        # Armed flag — gates whether storm callbacks fire.
        # Set true when inside a storm watch window (arm_lead/trailing buffer applied).
        # Defaults to False; SunsetScheduler manages transitions via arm()/disarm().
        self.armed = False
```

Add two methods:
```python
    def arm(self):
        """Allow storm callbacks to fire. Called by scheduler when entering a watch window."""
        if not self.armed:
            self.logger.info("TempestMonitor armed — storm callbacks now active")
        self.armed = True

    def disarm(self):
        """Suppress storm callbacks. Called by scheduler when outside watch windows."""
        if self.armed:
            self.logger.info("TempestMonitor disarmed — storm callbacks suppressed")
        self.armed = False

    def _fire_storm_callbacks(self, conditions):
        """Fire all registered storm callbacks IF armed. Centralized for testability."""
        if not self.armed:
            self.logger.debug(
                f"Storm conditions met (confidence {conditions.confidence:.1%}) "
                "but TempestMonitor is disarmed — callbacks suppressed"
            )
            return
        for callback in self.storm_callbacks:
            try:
                callback(conditions)
            except Exception as e:
                self.logger.error(f"Error in storm callback {callback.__name__}: {e}")
```

Find the existing block inside `_evaluate_storm_conditions` that loops over `self.storm_callbacks` and replace it with `self._fire_storm_callbacks(conditions)`. The block currently looks like:

```python
                for callback in self.storm_callbacks:
                    try:
                        callback(conditions)
                    except Exception as e:
                        self.logger.error(f"Error in storm callback {callback.__name__}: {e}")
```

Replace with:
```python
                self._fire_storm_callbacks(conditions)
```

- [ ] **Step 4: Run tests — should pass**

Run: `pytest tests/test_tempest_monitor.py -v`

Expected: 4 passed.

- [ ] **Step 5: Verify standalone test mode still works**

Run: `python tempest_monitor.py` then Ctrl+C within 5 seconds (don't need an actual Tempest, just verify no startup errors).

Expected: Logs "UDP listener started on port 50222" (or "disabled in configuration" — both are fine; we just want no crashes).

- [ ] **Step 6: Commit**

```bash
git add tempest_monitor.py tests/test_tempest_monitor.py
git commit -m "TempestMonitor: add armed flag gating storm callbacks

Callbacks now suppressed when monitor is disarmed (outside watch windows).
SunsetScheduler will arm/disarm based on Open-Meteo storm watch windows.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: StormWorkflow — capture sequence and recovery hooks

**Files:**
- Modify: `storm_workflow.py` (add `StormWorkflow` class)
- Modify: `tests/test_storm_workflow.py`

- [ ] **Step 1: Write failing tests for the workflow class structure**

Add to `tests/test_storm_workflow.py`:
```python
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
```

- [ ] **Step 2: Run tests — should fail**

Run: `pytest tests/test_storm_workflow.py::test_storm_workflow_init -v`

Expected: ImportError on `StormWorkflow`.

- [ ] **Step 3: Add the `StormWorkflow` class to `storm_workflow.py`**

Append to `storm_workflow.py`:
```python
import json
import threading
from datetime import date
from pathlib import Path


class StormWorkflow:
    """Orchestrates storm capture: capture → process → SIS → caption → social posts.

    Mirrors SunsetScheduler.complete_daily_workflow but storm-flavored.
    Reuses (does not duplicate) camera, video processor, drive uploader, social uploaders.
    """

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # Reuse the same shared components as SunsetScheduler.
        # Lazy import avoids circular dependency at module load time.
        from camera_interface import CameraInterface
        from video_processor import VideoProcessor
        from drive_uploader import DriveUploader
        from facebook_uploader import FacebookUploader
        from youtube_uploader import YouTubeUploader
        from email_notifier import EmailNotifier

        self.camera = CameraInterface()
        self.video_processor = VideoProcessor()
        self.drive_uploader = DriveUploader()
        self.facebook_uploader = FacebookUploader()
        self.youtube_uploader = YouTubeUploader()
        self.email_notifier = EmailNotifier()

        # Storms directory under data/storms/
        paths = self.config.get_storage_paths()
        self.storms_dir = paths['base'] / 'storms'
        self.storms_dir.mkdir(parents=True, exist_ok=True)

    def _compute_storm_duration_seconds(self) -> int:
        """Capture duration: min(max_duration_hours, post_storm_minutes)."""
        post_minutes = self.config.get('tempest.capture.post_storm_minutes', 60)
        max_hours = self.config.get('tempest.capture.max_duration_hours', 4)
        return int(min(max_hours * 3600, post_minutes * 60))

    def _storm_metadata_path(self, target_date: date) -> Path:
        """Where to persist storm metadata for a given date."""
        d = self.storms_dir / target_date.isoformat()
        d.mkdir(parents=True, exist_ok=True)
        return d / 'storm_metadata.json'

    def _persist_storm_metadata(self, target_date: date, metadata: Dict):
        """Write storm metadata JSON (survives across restarts for deferred recovery)."""
        path = self._storm_metadata_path(target_date)
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        self.logger.info(f"Storm metadata persisted to {path}")
```

- [ ] **Step 4: Run tests — should pass**

Run: `pytest tests/test_storm_workflow.py -v`

Expected: 9 passed total.

- [ ] **Step 5: Commit**

```bash
git add storm_workflow.py tests/test_storm_workflow.py
git commit -m "StormWorkflow: class skeleton with duration + metadata path

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: StormWorkflow — capture method with cancellation support

**Files:**
- Modify: `camera_interface.py` (add `cancel_event` param to `capture_video_sequence`)
- Modify: `storm_workflow.py` (add `_capture_storm_sequence`)
- Modify: `tests/test_storm_workflow.py`

- [ ] **Step 1: Add `cancel_event` parameter to `camera_interface.capture_video_sequence`**

In `camera_interface.py`, modify the signature of `capture_video_sequence` (line 190):
```python
    def capture_video_sequence(self, start_time: datetime, end_time: datetime,
                              interval_seconds: int = 5,
                              cancel_event: Optional['threading.Event'] = None) -> List[Path]:
```

Add `import threading` near the top imports (around line 7).

Inside the chunk-recording for-loop (around line 230, the `for chunk_idx in range(chunks_needed):` line), add at the start of each iteration:
```python
            for chunk_idx in range(chunks_needed):
                # Cancellation check between chunks (granularity = chunk_duration_minutes)
                if cancel_event is not None and cancel_event.is_set():
                    self.logger.info(
                        f"Capture cancelled after chunk {chunk_idx}/{chunks_needed} — "
                        f"returning {len(captured_images)} images captured so far"
                    )
                    return captured_images

                chunk_end_time = min(current_time + timedelta(seconds=chunk_duration), end_time)
                # [existing chunk logic continues unchanged]
```

- [ ] **Step 2: Write failing test for `_capture_storm_sequence`**

Add to `tests/test_storm_workflow.py`:
```python
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
```

- [ ] **Step 3: Run test — should fail**

Run: `pytest tests/test_storm_workflow.py::test_capture_storm_sequence_calls_camera -v`

Expected: AttributeError on `_capture_storm_sequence`.

- [ ] **Step 4: Add `_capture_storm_sequence` to `StormWorkflow`**

Add inside `StormWorkflow`:
```python
    def _capture_storm_sequence(
        self,
        start_time: datetime,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[Path]:
        """
        Capture storm frames using the existing camera capture path.

        Duration is fixed (no continuous Tempest polling during capture) — see
        spec non-goals. Cancellation supported between camera chunks (~15 min).

        Args:
            start_time: when capture starts (naive datetime, local time)
            cancel_event: optional threading.Event; capture exits early when set

        Returns:
            List of captured frame paths (possibly empty if camera failed)
        """
        duration_seconds = self._compute_storm_duration_seconds()
        end_time = start_time + timedelta(seconds=duration_seconds)
        interval = self.config.get('capture.interval_seconds', 5)

        self.logger.info(
            f"Storm capture window: {start_time} → {end_time} "
            f"({duration_seconds // 60} min, interval {interval}s)"
        )

        try:
            return self.camera.capture_video_sequence(
                start_time, end_time, interval, cancel_event=cancel_event,
            )
        except Exception as e:
            self.logger.error(f"Storm capture failed: {e}")
            return []
```

- [ ] **Step 5: Run tests — should pass**

Run: `pytest tests/test_storm_workflow.py -v`

Expected: 10 passed.

- [ ] **Step 6: Commit**

```bash
git add camera_interface.py storm_workflow.py tests/test_storm_workflow.py
git commit -m "Storm capture: add cancellation support to camera + workflow

camera.capture_video_sequence now accepts an optional cancel_event that's
checked between recording chunks. StormWorkflow._capture_storm_sequence
delegates with a fixed-duration window.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: StormWorkflow — recovery via historical_retrieval

**Files:**
- Modify: `storm_workflow.py`
- Modify: `tests/test_storm_workflow.py`

- [ ] **Step 1: Write failing test for recovery decision logic**

Add to `tests/test_storm_workflow.py`:
```python
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
```

- [ ] **Step 2: Run tests — should fail**

Run: `pytest tests/test_storm_workflow.py::test_recovery_triggered_when_zero_frames -v`

Expected: AttributeError on `_should_attempt_recovery`.

- [ ] **Step 3: Add recovery decision + recovery attempt to `StormWorkflow`**

Add inside `StormWorkflow`:
```python
    def _expected_frame_count(self) -> int:
        """How many frames a complete storm capture should yield."""
        duration = self._compute_storm_duration_seconds()
        interval = self.config.get('capture.interval_seconds', 5)
        return duration // interval

    def _should_attempt_recovery(
        self, captured: List[Path], expected_frame_count: int,
    ) -> bool:
        """Recovery triggered when captured < 50% of expected."""
        if expected_frame_count <= 0:
            return False
        return len(captured) < (expected_frame_count // 2)

    def _attempt_immediate_recovery(
        self, target_date: date, start_time: datetime, end_time: datetime,
    ) -> List[Path]:
        """
        Pull camera SD-card recordings for the storm window via historical_retrieval.
        Returns recovered frames (possibly empty if recovery also fails).
        """
        from historical_retrieval import HistoricalRetrieval
        self.logger.info(
            f"Attempting immediate recovery from camera recordings: "
            f"{start_time} → {end_time}"
        )
        try:
            hr = HistoricalRetrieval()
            # HistoricalRetrieval's interface returns processed video paths per-day.
            # For storm windows we need the underlying frame extraction.
            # If the existing API doesn't expose frame-extraction for arbitrary
            # time windows, the simplest path is to use create_historical_timelapse
            # for the day and then point storm_workflow at the resulting video.
            videos = hr.create_historical_timelapse(
                target_date, target_date, upload_to_youtube=False,
            )
            # Recovery succeeded if any video was created for the date.
            if videos:
                self.logger.info(f"Recovery produced {len(videos)} video(s) for {target_date}")
                return [Path(v) for v in videos]
            return []
        except Exception as e:
            self.logger.error(f"Immediate recovery failed: {e}")
            return []

    def _queue_deferred_recovery(
        self, target_date: date, start_time: datetime, end_time: datetime,
        sis: Dict, storm_conditions_summary: Dict,
    ):
        """Queue this storm window for retry during morning maintenance."""
        queue_path = self.storms_dir / 'pending_recovery.json'
        queue: List[Dict] = []
        if queue_path.exists():
            try:
                with open(queue_path) as f:
                    queue = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Could not read pending_recovery.json: {e}")
                queue = []

        queue.append({
            'date': target_date.isoformat(),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'sis': sis,
            'conditions': storm_conditions_summary,
            'queued_at': datetime.now().isoformat(),
        })

        with open(queue_path, 'w') as f:
            json.dump(queue, f, indent=2, default=str)
        self.logger.info(f"Storm queued for deferred recovery in {queue_path}")
```

- [ ] **Step 4: Run tests — should pass**

Run: `pytest tests/test_storm_workflow.py -v`

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add storm_workflow.py tests/test_storm_workflow.py
git commit -m "StormWorkflow: recovery decision + immediate/deferred recovery hooks

- _should_attempt_recovery: < 50% expected frames triggers recovery
- _attempt_immediate_recovery: pulls camera SD via HistoricalRetrieval
- _queue_deferred_recovery: persists to pending_recovery.json for morning retry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: StormWorkflow — complete_storm_workflow orchestrator

**Files:**
- Modify: `storm_workflow.py`
- Modify: `tests/test_storm_workflow.py`

- [ ] **Step 1: Write failing test using monkeypatched components**

Add to `tests/test_storm_workflow.py`:
```python
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
    monkeypatch.setattr(wf.youtube_uploader, 'upload_video',
                        lambda video_path, title, description, **kwargs: 'yt_id_123')
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
```

- [ ] **Step 2: Run test — should fail**

Run: `pytest tests/test_storm_workflow.py::test_complete_storm_workflow_happy_path -v`

Expected: AttributeError on `complete_storm_workflow`.

- [ ] **Step 3: Add `complete_storm_workflow` and `_collect_tempest_data` to `StormWorkflow`**

Add inside `StormWorkflow`:
```python
    def _collect_tempest_data(
        self, tempest_monitor, window_minutes: int = 90,
    ) -> tuple:
        """Snapshot strikes + observations from the monitor's ring buffers."""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        strikes = [s for s in tempest_monitor.lightning_strikes if s.timestamp >= cutoff]
        observations = [o for o in tempest_monitor.observations if o.timestamp >= cutoff]
        return strikes, observations

    def complete_storm_workflow(
        self,
        conditions,                 # StormConditions
        start_time: datetime,
        tempest_monitor,            # TempestMonitor instance
        cancel_event: Optional[threading.Event] = None,
    ) -> bool:
        """
        Full storm capture pipeline: capture → process → SIS → caption → post.

        Returns True on overall success (some upload failures are non-fatal and
        emit emails; only capture+process failure with no recovery returns False).
        """
        target_date = start_time.date()
        self.logger.info(f"[STORM] Starting workflow for {target_date} at {start_time.time()}")

        try:
            # Step 1: Capture
            captured = self._capture_storm_sequence(start_time, cancel_event=cancel_event)

            # Step 2: Recovery if needed
            expected = self._expected_frame_count()
            recovered_videos: List[Path] = []
            recovered = False
            if self._should_attempt_recovery(captured, expected):
                self.logger.warning(
                    f"[STORM] Only {len(captured)}/{expected} frames captured; "
                    "attempting immediate recovery from camera SD"
                )
                duration = self._compute_storm_duration_seconds()
                end_time = start_time + timedelta(seconds=duration)
                recovered_videos = self._attempt_immediate_recovery(
                    target_date, start_time, end_time,
                )
                recovered = bool(recovered_videos)

            # Step 3: Determine working video path
            if recovered and recovered_videos:
                video_path = recovered_videos[0]
                source_tag = '[RECOVERED]'
            elif captured:
                paths = self.config.get_storage_paths()
                video_output = paths['videos'] / f"storm_{target_date.isoformat()}.mp4"
                video_path = self.video_processor.create_timelapse(
                    captured, video_output,
                )
                if not video_path:
                    self.logger.error("[STORM] Video processing failed")
                    self._handle_capture_failure(
                        target_date, start_time, conditions, tempest_monitor,
                        reason='video_processing_failed',
                    )
                    return False
                source_tag = ''
            else:
                # No frames and no recovery succeeded — queue deferred recovery
                self.logger.error("[STORM] Capture and immediate recovery both failed")
                self._handle_capture_failure(
                    target_date, start_time, conditions, tempest_monitor,
                    reason='capture_and_recovery_failed',
                )
                return False

            # Step 4: Compute SIS from Tempest data
            strikes, observations = self._collect_tempest_data(tempest_monitor)
            sis = compute_storm_intensity_score(strikes, observations)
            self.logger.info(
                f"[STORM] SIS {sis['score']} (Grade {sis['grade']}), "
                f"lightning={sis['metrics']['lightning_count']}, "
                f"gust={sis['metrics']['wind_gust_max_mph']}"
            )

            # Step 5: Build metadata for downstream consumers
            metadata = {
                'event_type': 'storm',
                'date': target_date.isoformat(),
                'capture_date': target_date.isoformat(),
                'start_time': start_time.isoformat(),
                'sis_score': sis['score'],
                'sis_grade': sis['grade'],
                'storm_metrics': sis['metrics'],
                'storm_components': sis['components'],
                'trigger_reasons': list(conditions.trigger_reasons),
                'recovered': recovered,
            }
            self._persist_storm_metadata(target_date, metadata)

            # Step 6: Drive backup
            try:
                drive_result = self.drive_uploader.upload_video(
                    video_path, target_date, metadata,
                )
                if drive_result:
                    self.logger.info(f"[STORM] Drive upload OK: {drive_result.get('filename')}")
            except Exception as e:
                self.logger.warning(f"[STORM] Drive upload failed (non-fatal): {e}")

            # Step 7: Generate one caption for all socials
            try:
                shared_caption = self.facebook_uploader.generate_caption(
                    metadata, video_path=str(video_path),
                )
                self.logger.info(f"[STORM] Caption: {shared_caption[:100]}...")
            except Exception as e:
                self.logger.warning(f"[STORM] Caption generation failed; using fallback: {e}")
                shared_caption = (
                    f"Thunderstorm timelapse from Pelham, AL on "
                    f"{target_date.strftime('%B %d, %Y')}. "
                    f"SIS {sis['score']:.0f}/100 (grade {sis['grade']}). "
                    f"Camera: Reolink RLC810-WA"
                )

            # Step 8: Post to FB+IG+Reels
            try:
                self.facebook_uploader.post_sunset(
                    video_path, metadata, caption_override=shared_caption,
                )
            except Exception as e:
                self.logger.warning(f"[STORM] Facebook/Instagram post failed: {e}")
                self.email_notifier.send_notification(
                    f"{source_tag} Storm FB/IG Post Failed - {target_date}",
                    f"Storm video {video_path} created but FB/IG post failed:\n{e}",
                )

            # Step 9: YouTube
            try:
                youtube_title = f"Storm {target_date.strftime('%m/%d/%y')}"
                description = self._build_youtube_description(
                    shared_caption, sis, metadata,
                )
                self.youtube_uploader.upload_video(
                    video_path, youtube_title, description,
                )
            except Exception as e:
                self.logger.warning(f"[STORM] YouTube upload failed: {e}")
                self.email_notifier.send_notification(
                    f"{source_tag} Storm YouTube Upload Failed - {target_date}",
                    f"Storm video {video_path} created but YouTube upload failed:\n{e}",
                )

            # Step 10: Success email
            subject_prefix = source_tag if source_tag else ''
            self.email_notifier.send_notification(
                f"{subject_prefix} Storm captured - {target_date} (SIS {sis['score']:.0f}/{sis['grade']})".strip(),
                f"Storm timelapse captured and posted.\n\n"
                f"SIS: {sis['score']:.0f}/100 (grade {sis['grade']})\n"
                f"Lightning: {sis['metrics']['lightning_count']} strikes\n"
                f"Peak gust: {sis['metrics']['wind_gust_max_mph']:.0f} mph\n"
                f"Peak rain: {sis['metrics']['rain_max_mm_hr']:.1f} mm/hr\n"
                f"Pressure drop: {sis['metrics']['pressure_drop_hpa']:.1f} hPa\n"
                f"\nVideo: {video_path}",
            )

            self.logger.info(f"[STORM] Workflow completed for {target_date}")
            return True

        except Exception as e:
            self.logger.error(f"[STORM] Workflow exception: {e}", exc_info=True)
            self.email_notifier.send_notification(
                f"Storm Workflow Error - {start_time.date()}",
                f"Unhandled exception in storm workflow:\n\n{str(e)}",
            )
            return False

    def _handle_capture_failure(
        self, target_date: date, start_time: datetime,
        conditions, tempest_monitor, reason: str,
    ):
        """Queue deferred recovery + email notification."""
        try:
            strikes, observations = self._collect_tempest_data(tempest_monitor)
            sis = compute_storm_intensity_score(strikes, observations)
        except Exception:
            sis = {'score': 0.0, 'grade': 'F', 'metrics': {}, 'components': {}}

        duration = self._compute_storm_duration_seconds()
        end_time = start_time + timedelta(seconds=duration)
        self._queue_deferred_recovery(
            target_date, start_time, end_time, sis,
            {'reasons': list(conditions.trigger_reasons), 'confidence': conditions.confidence},
        )
        self.email_notifier.send_notification(
            f"Storm Capture Failed - {target_date}",
            f"Storm capture failed (reason: {reason}). "
            f"Queued for deferred recovery in morning maintenance.\n\n"
            f"SIS estimate from Tempest data: {sis['score']:.0f}/100",
        )

    def _build_youtube_description(
        self, caption: str, sis: Dict, metadata: Dict,
    ) -> str:
        """Compose YouTube description: caption + structured sensor block + hashtags."""
        m = sis['metrics']
        avg_dist = m.get('lightning_avg_distance_km')
        avg_dist_str = f"{avg_dist:.1f}km" if avg_dist is not None else "n/a"
        block = (
            f"\n\n---\n"
            f"Lightning: {m['lightning_count']} strikes, avg distance {avg_dist_str}\n"
            f"Peak wind gust: {m['wind_gust_max_mph']:.0f} mph\n"
            f"Peak rain rate: {m['rain_max_mm_hr']:.1f} mm/hr\n"
            f"Pressure drop: {m['pressure_drop_hpa']:.1f} hPa\n"
            f"Storm Intensity Score: {sis['score']:.0f} (Grade {sis['grade']})\n\n"
            f"#thunderstorm #lightning #timelapse #alabama #alabamawx #weather #pelham"
        )
        return caption + block
```

- [ ] **Step 4: Run tests — should pass**

Run: `pytest tests/test_storm_workflow.py -v`

Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add storm_workflow.py tests/test_storm_workflow.py
git commit -m "StormWorkflow: complete_storm_workflow orchestrator

End-to-end: capture → recovery-if-needed → process → SIS → metadata →
Drive → caption → FB/IG/Reels → YouTube → success email. Mirrors
SunsetScheduler.complete_daily_workflow but with storm-specific scoring
and prompt.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: SunsetScheduler — capture-state machine + cancel_event

**Files:**
- Modify: `sunset_scheduler.py`

- [ ] **Step 1: Add capture-state and cancel_event attributes**

In `sunset_scheduler.py`, modify `__init__` (around line 34) — after the existing `# State tracking` block, add:

```python
        # Capture-state machine for storm/sunset coordination
        self.capture_state = 'IDLE'  # IDLE | SUNSET_ACTIVE | STORM_ACTIVE
        self.current_capture_cancel_event = None  # threading.Event when capture active
```

Add `import threading` at the top if not already present.

- [ ] **Step 2: Modify `capture_sunset_sequence` to use a cancel_event**

Replace the body of `capture_sunset_sequence` (around line 171). The signature stays the same; just thread a cancel_event through:

```python
    def capture_sunset_sequence(self, target_date: Optional[date] = None) -> Optional[List[Path]]:
        """
        Capture a complete sunset sequence for the specified date.

        Honors self.current_capture_cancel_event for storm-interrupts-sunset.
        """
        if target_date is None:
            target_date = date.today()

        self.logger.info(f"Starting sunset capture sequence for {target_date}")

        # Per-capture cancellation event (cleared each run)
        self.current_capture_cancel_event = threading.Event()
        self.capture_state = 'SUNSET_ACTIVE'

        try:
            start_time, end_time = self.sunset_calc.get_capture_window(target_date)
            interval = self.config.get('capture.interval_seconds', 5)

            if hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
                start_time = start_time.replace(tzinfo=None)
            if hasattr(end_time, 'tzinfo') and end_time.tzinfo is not None:
                end_time = end_time.replace(tzinfo=None)

            self.logger.info(f"Capture window: {start_time} to {end_time}")

            captured_images = self.camera.capture_video_sequence(
                start_time, end_time, interval,
                cancel_event=self.current_capture_cancel_event,
            )

            if self.current_capture_cancel_event.is_set():
                self.logger.warning(
                    f"Sunset capture for {target_date} cancelled by storm"
                )
                # Delete the partial frames so they don't get processed into a video
                for img in (captured_images or []):
                    try:
                        Path(img).unlink(missing_ok=True)
                    except Exception:
                        pass
                return None

            if captured_images:
                self.logger.info(f"Capture sequence completed: {len(captured_images)} images")
                return captured_images

            self.logger.error("No images captured")
            self.email_notifier.send_capture_failure(
                "Camera failed to capture any images during the sunset window",
                datetime.combine(target_date, datetime.min.time())
            )
            return None

        except Exception as e:
            self.logger.error(f"Failed to capture sunset sequence: {e}")
            self.camera.disconnect()
            self.email_notifier.send_capture_failure(
                f"Exception during sunset capture: {str(e)}",
                datetime.combine(target_date, datetime.min.time())
            )
            return None
        finally:
            self.current_capture_cancel_event = None
            if self.capture_state == 'SUNSET_ACTIVE':
                self.capture_state = 'IDLE'
```

- [ ] **Step 3: Smoke-test by running sunset validation**

Run: `python main.py test --sunset`

Expected: PASS (no regressions from cancel_event addition).

- [ ] **Step 4: Commit**

```bash
git add sunset_scheduler.py
git commit -m "SunsetScheduler: add capture-state machine + cancel_event hook

capture_sunset_sequence now creates a threading.Event each run and threads
it through to camera.capture_video_sequence. Partial frames are deleted
on cancellation so cancelled sunsets don't produce half-baked videos.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: SunsetScheduler — wire OpenMeteo, Tempest, StormWorkflow

**Files:**
- Modify: `sunset_scheduler.py`

- [ ] **Step 1: Add new imports + component instances**

In `sunset_scheduler.py`, add imports near the top:
```python
from tempest_monitor import TempestMonitor
from open_meteo_client import OpenMeteoClient, StormWindow
from storm_workflow import StormWorkflow
```

In `__init__`, after `self.visual_analyzer = VisualAnalyzer()` (around line 50), add:
```python
        # Storm capture pipeline
        self.tempest_monitor = TempestMonitor()
        self.open_meteo_client = OpenMeteoClient()
        self.storm_workflow = StormWorkflow()
        self.storm_watch_windows: List[StormWindow] = []
        self.tempest_monitor.register_storm_callback(self.on_storm_detected)
```

Add `from typing import List` if not already imported.

- [ ] **Step 2: Add `update_storm_watch_windows` and `_reconcile_tempest_arming`**

Add to `SunsetScheduler` (e.g., near `daily_maintenance`):
```python
    def update_storm_watch_windows(self):
        """Poll Open-Meteo and reconcile storm watch window state.

        Cadence-gating: when outside daylight hours AND there's no currently-active
        watch window, skip the poll. This implements the spec's 'daylight only
        baseline / 15 min during active' policy without needing two scheduled jobs.
        """
        if not self.config.get('open_meteo.enabled', False):
            return

        # Cadence gate
        daylight = self.config.get('open_meteo.daylight_hours', [6, 22])
        now = datetime.now()
        in_daylight = daylight[0] <= now.hour < daylight[1]
        in_active_window = any(w.start <= now <= w.end for w in self.storm_watch_windows)
        if not in_daylight and not in_active_window:
            self.logger.debug("[STORM] Skipping Open-Meteo poll (outside daylight, no active window)")
            return

        try:
            new_windows = self.open_meteo_client.get_storm_watch_windows()
        except Exception as e:
            self.logger.warning(f"[STORM] Open-Meteo poll failed: {e}")
            return

        # Diff: find newly-appearing windows for notifications
        existing = {(w.start, w.end) for w in self.storm_watch_windows}
        new_set = {(w.start, w.end) for w in new_windows}
        appearing = new_set - existing
        if appearing:
            for window in new_windows:
                if (window.start, window.end) in appearing:
                    msg = (
                        f"Storm watch window forecast: "
                        f"{window.start.strftime('%a %H:%M')} to "
                        f"{window.end.strftime('%H:%M')}, "
                        f"reasons: {', '.join(window.reasons[:4])}"
                    )
                    self.logger.info(f"[STORM] {msg}")
                    try:
                        self.email_notifier.send_notification(
                            f"Storm watch window forecast — {window.start.date()}", msg,
                        )
                    except Exception:
                        pass

        self.storm_watch_windows = new_windows
        self._reconcile_tempest_arming()

    def _reconcile_tempest_arming(self):
        """Arm/disarm Tempest based on whether we're inside (or near) a watch window."""
        lead = self.config.get('open_meteo.tempest_arm_lead_minutes', 30)
        trail = self.config.get('open_meteo.tempest_arm_trailing_minutes', 30)
        now = datetime.now()

        should_arm = any(
            (window.start - timedelta(minutes=lead)) <= now <= (window.end + timedelta(minutes=trail))
            for window in self.storm_watch_windows
        )

        if should_arm and not self.tempest_monitor.armed:
            self.tempest_monitor.arm()
        elif not should_arm and self.tempest_monitor.armed:
            # Only disarm if no capture is in progress
            if self.capture_state != 'STORM_ACTIVE':
                self.tempest_monitor.disarm()
```

- [ ] **Step 3: Add `on_storm_detected` callback**

Add to `SunsetScheduler`:
```python
    def on_storm_detected(self, conditions):
        """
        Storm callback fired by TempestMonitor. Aborts sunset if active,
        runs the storm workflow.
        """
        if self.capture_state == 'STORM_ACTIVE':
            self.logger.info("[STORM] Storm callback ignored — capture already STORM_ACTIVE")
            return

        # Note: tempest_monitor.armed check already happened upstream in
        # _fire_storm_callbacks — if we got here, we're inside a watch window
        # OR caller explicitly armed it.

        if self.capture_state == 'SUNSET_ACTIVE':
            self.logger.info("[STORM] Aborting active sunset capture")
            if self.current_capture_cancel_event is not None:
                self.current_capture_cancel_event.set()
            # Wait briefly for sunset capture loop to notice the cancel
            wait_until = datetime.now() + timedelta(seconds=10)
            while (self.capture_state == 'SUNSET_ACTIVE'
                   and datetime.now() < wait_until):
                time.sleep(0.5)

        self.capture_state = 'STORM_ACTIVE'
        self.current_capture_cancel_event = threading.Event()

        try:
            success = self.storm_workflow.complete_storm_workflow(
                conditions=conditions,
                start_time=datetime.now(),
                tempest_monitor=self.tempest_monitor,
                cancel_event=self.current_capture_cancel_event,
            )
            if success:
                self.logger.info("[STORM] Storm workflow completed successfully")
            else:
                self.logger.warning("[STORM] Storm workflow returned failure")
        finally:
            self.capture_state = 'IDLE'
            self.current_capture_cancel_event = None
            self._reconcile_tempest_arming()
```

(`time` must be imported — `import time` at top.)

- [ ] **Step 4: Smoke-test imports**

Run: `python -c "from sunset_scheduler import SunsetScheduler; print('OK')"`

Expected: `OK` (no import errors).

- [ ] **Step 5: Commit**

```bash
git add sunset_scheduler.py
git commit -m "SunsetScheduler: wire OpenMeteo, Tempest, StormWorkflow

- New instance attributes: tempest_monitor, open_meteo_client, storm_workflow
- update_storm_watch_windows polls + diffs + emails on new windows
- _reconcile_tempest_arming uses arm_lead/trailing config
- on_storm_detected callback aborts active sunset and runs storm workflow

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: SunsetScheduler — schedule the polls + start/stop Tempest

**Files:**
- Modify: `sunset_scheduler.py`

- [ ] **Step 1: Schedule Open-Meteo polls and start Tempest in `schedule_daily_capture`**

Locate `schedule_daily_capture` (around line 971). After the existing meteor scan scheduling block, add:

```python
        # Storm capture: poll Open-Meteo forecasts.
        # Scheduled at the *tighter* cadence (15 min); function early-exits when
        # outside daylight hours AND no active window (matches spec "daylight only
        # baseline" + "15 min active window" without needing two scheduled jobs).
        if self.config.get('open_meteo.enabled', False):
            poll_minutes = self.config.get('open_meteo.active_window_poll_minutes', 15)
            schedule.every(poll_minutes).minutes.do(self.update_storm_watch_windows)
            self.logger.info(f"Open-Meteo storm watch polling scheduled every {poll_minutes} min (gated by daylight/active-window logic)")
            # Run once at startup so first iteration has data
            self.update_storm_watch_windows()
```

- [ ] **Step 2: Start the Tempest UDP listener in `run_scheduler`**

Locate `run_scheduler` (around line 1003). Right after `self.running = True`, add:

```python
        # Start Tempest UDP listener (callbacks gated by armed flag)
        if self.config.get('tempest.enabled', False) and self.tempest_monitor.is_enabled():
            started = self.tempest_monitor.start_udp_listener()
            if started:
                self.logger.info("Tempest UDP listener started")
            else:
                self.logger.warning("Tempest UDP listener failed to start — storm reactive layer disabled")
```

- [ ] **Step 3: Stop the Tempest listener cleanly in `stop`**

Locate `stop` (around line 1027). Before the existing body or at the start:

```python
    def stop(self):
        """Stop the scheduler gracefully"""
        self.logger.info("Stopping scheduler...")
        self.running = False
        # Stop Tempest listener if running
        try:
            self.tempest_monitor.stop_udp_listener()
        except Exception as e:
            self.logger.warning(f"Error stopping Tempest listener: {e}")
        # ... existing cleanup ...
```

- [ ] **Step 4: Smoke-test scheduler startup**

Run: `python main.py status` (the existing status command instantiates SunsetScheduler).

Expected: No import errors, no exceptions during init.

- [ ] **Step 5: Commit**

```bash
git add sunset_scheduler.py
git commit -m "SunsetScheduler: schedule Open-Meteo polls + start/stop Tempest

Open-Meteo polled every open_meteo.poll_interval_minutes (default 30) when
open_meteo.enabled. Tempest UDP listener started in run_scheduler and
stopped in stop. Listener runs 24/7 but callbacks gated by armed flag.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Morning maintenance — pending recovery processing

**Files:**
- Modify: `sunset_scheduler.py`

- [ ] **Step 1: Add a helper `_process_pending_storm_recovery`**

Add to `SunsetScheduler`:
```python
    def _process_pending_storm_recovery(self):
        """Retry deferred storm recoveries queued in storms/pending_recovery.json."""
        paths = self.config.get_storage_paths()
        queue_path = paths['base'] / 'storms' / 'pending_recovery.json'
        if not queue_path.exists():
            return

        try:
            with open(queue_path) as f:
                queue = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning(f"[STORM] Could not read {queue_path}: {e}")
            return

        if not queue:
            return

        self.logger.info(f"[STORM] Processing {len(queue)} pending recovery item(s)")
        for item in queue:
            try:
                target_date = date.fromisoformat(item['date'])
                start_time = datetime.fromisoformat(item['start_time'])
                end_time = datetime.fromisoformat(item['end_time'])
            except (KeyError, ValueError) as e:
                self.logger.warning(f"[STORM] Skipping malformed recovery item: {e}")
                continue

            self.logger.info(f"[STORM] Deferred recovery attempt for {target_date}")
            videos = self.storm_workflow._attempt_immediate_recovery(
                target_date, start_time, end_time,
            )
            if videos:
                self.logger.info(
                    f"[STORM] Deferred recovery succeeded for {target_date}; "
                    "downstream upload not auto-triggered (manual review recommended)"
                )
            else:
                self.logger.warning(
                    f"[STORM] Deferred recovery failed for {target_date}; "
                    "dropping (spec: one retry per storm)"
                )

        # Per spec: one retry per storm. All items processed once → queue cleared.
        with open(queue_path, 'w') as f:
            json.dump([], f)

    def daily_maintenance(self, skip_historical_recovery=False):
        # [existing body unchanged — at the start, after the initial log line, add:]
        # (actually we want to add this near the end so historical recovery runs first)
```

Then locate `daily_maintenance` (around line 402) and add at the end of its happy path (before the final logger / return):
```python
        # Process any storms queued for deferred recovery
        try:
            self._process_pending_storm_recovery()
        except Exception as e:
            self.logger.warning(f"[STORM] Pending recovery processing failed: {e}")
```

Also add `import json` at the top of the file if not already imported.

- [ ] **Step 2: Smoke-test**

Run: `python main.py status`

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add sunset_scheduler.py
git commit -m "SunsetScheduler: process pending storm recoveries in morning maintenance

Reads data/storms/pending_recovery.json and retries each entry exactly
once (per spec: 'one retry per storm'). Successful retries are logged
but downstream upload is left for manual review.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Configuration additions

**Files:**
- Modify: `config.yaml`
- Modify: `config-pi.yaml`

- [ ] **Step 1: Append `open_meteo:` block to `config.yaml`**

At the end of `config.yaml`, append:
```yaml

# =============================================================================
# OPEN-METEO FORECAST CLIENT
# =============================================================================

open_meteo:
  enabled: true
  poll_interval_minutes: 30           # baseline cadence (06:00-22:00)
  active_window_poll_minutes: 15      # tighter when inside active watch window
  daylight_hours: [6, 22]

  triggers:
    cape_min: 1500                    # J/kg — minimum CAPE to qualify
    lifted_index_max: -3.0            # lifted index must be at or below this
    wind_fov_margin_degrees: 30       # margin beyond camera arc (236°-324°)
    require_any_of:
      precipitation_probability_min: 40
      weather_code_min: 80
      cloud_cover_min: 80

  tempest_arm_lead_minutes: 30        # arm Tempest this far before window start
  tempest_arm_trailing_minutes: 30    # keep armed this long after window end
```

- [ ] **Step 2: Set `tempest.enabled: true`**

In `config.yaml`, find the existing `tempest:` block and change:
```yaml
tempest:
  enabled: false  # Set to true when Tempest is configured
```
to:
```yaml
tempest:
  enabled: true
```

(Station ID is already set via `${TEMPEST_STATION_ID}` env var per existing config.)

- [ ] **Step 3: Add full `tempest:` + `open_meteo:` blocks to `config-pi.yaml`**

Read the current `config-pi.yaml`. Find the position after the existing `sbs:` block, append the entire `tempest:` block from `config.yaml` (lines 164-241 of `config.yaml` based on initial read) AND the new `open_meteo:` block. The Pi config currently has no `tempest:` block at all.

- [ ] **Step 4: Smoke-test config validation**

Run: `python main.py config --validate --show`

Expected: PASS with the new keys visible in `--show` output.

- [ ] **Step 5: Commit**

```bash
git add config.yaml config-pi.yaml
git commit -m "Config: add open_meteo block, activate tempest, sync Pi config

config.yaml and config-pi.yaml now both contain tempest + open_meteo
blocks. Defaults match design spec thresholds (CAPE 1500, LI -3, wind
margin 30°, arm lead/trailing 30 min, poll 30 min baseline).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: CLI extensions for testing + backtesting

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add `--weather`, `--tempest`, `--storm-prompt` to `test` command**

In `main.py`, locate the `test_parser` setup (around line 867). Add three new arguments:
```python
    test_parser.add_argument('--weather', action='store_true',
                           help='Test Open-Meteo connectivity and parse response')
    test_parser.add_argument('--tempest', action='store_true',
                           help='Test Tempest UDP reception (5s listen)')
    test_parser.add_argument('--storm-prompt', action='store_true',
                           help='Generate sample storm caption prompt')
```

In `cmd_test` (around line 104), add three new branches inside the function:
```python
    if args.weather:
        logger.info("Testing Open-Meteo connectivity...")
        from open_meteo_client import OpenMeteoClient
        client = OpenMeteoClient()
        try:
            windows = client.get_storm_watch_windows()
            logger.info(f"✓ Open-Meteo returned {len(windows)} storm watch window(s)")
            for w in windows[:3]:
                logger.info(
                    f"  {w.start} → {w.end} (confidence {w.confidence:.0%}): {', '.join(w.reasons[:3])}"
                )
        except Exception as e:
            logger.error(f"✗ Open-Meteo test failed: {e}")
            sys.exit(1)

    if args.tempest:
        logger.info("Testing Tempest UDP reception (5 seconds)...")
        from tempest_monitor import TempestMonitor
        import time as _time
        monitor = TempestMonitor()
        if not monitor.is_enabled():
            logger.warning("✗ Tempest is disabled in configuration")
            sys.exit(1)
        if not monitor.start_udp_listener():
            logger.error("✗ Could not start UDP listener (port already in use?)")
            sys.exit(1)
        _time.sleep(5)
        status = monitor.get_status()
        monitor.stop_udp_listener()
        if status.get('observations_cached', 0) > 0 or status.get('last_message'):
            logger.info(f"✓ Tempest UDP test passed: {status}")
        else:
            logger.error("✗ No messages received from Tempest in 5 seconds")
            sys.exit(1)

    if args.storm_prompt:
        logger.info("Generating sample storm caption prompt...")
        from facebook_uploader import FacebookUploader
        from datetime import date as _date
        uploader = FacebookUploader()
        sample_metadata = {
            'event_type': 'storm',
            'date': _date.today().isoformat(),
            'sis_score': 65.0, 'sis_grade': 'B',
            'storm_metrics': {
                'lightning_count': 8,
                'lightning_avg_distance_km': 8.5,
                'rain_max_mm_hr': 12.0,
                'wind_gust_max_mph': 28.0,
                'pressure_drop_hpa': 3.2,
            },
            'weather': {'temperature_f': 71, 'conditions': 'thunderstorms'},
        }
        prompt = uploader._build_caption_prompt(sample_metadata)
        print("\n" + "=" * 70)
        print(prompt)
        print("=" * 70 + "\n")
        logger.info("✓ Storm prompt generated")
```

- [ ] **Step 2: Add `weather --backtest` subcommand**

After the `meteor_parser` block in `main.py`, add:
```python
    # Weather command (backtest forecasts)
    weather_parser = subparsers.add_parser('weather', help='Weather forecast utilities')
    weather_parser.add_argument('--backtest', action='store_true',
                              help='Run storm-window logic against historical Open-Meteo data')
    weather_parser.add_argument('--start', dest='start_date',
                              help='Backtest start date (YYYY-MM-DD); up to 92 days back')
    weather_parser.add_argument('--end', dest='end_date',
                              help='Backtest end date (YYYY-MM-DD)')
    weather_parser.set_defaults(func=cmd_weather)
```

Add the `cmd_weather` function above `def main()`:
```python
def cmd_weather(args):
    """Weather forecast utilities (currently: --backtest)."""
    setup_logging()
    logger = logging.getLogger(__name__)

    if not args.backtest:
        logger.error("Use --backtest --start YYYY-MM-DD --end YYYY-MM-DD")
        sys.exit(1)

    if not args.start_date or not args.end_date:
        logger.error("--backtest requires --start and --end")
        sys.exit(1)

    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)

    days = (date.today() - start_date).days
    if days > 92:
        logger.warning(
            f"Start date is {days} days back; Open-Meteo free tier supports ~92 days. "
            "Results may be truncated."
        )
        days = 92

    from open_meteo_client import OpenMeteoClient
    client = OpenMeteoClient()

    # Pull forecast with past_days large enough to cover the range
    data = client.fetch_forecast(past_days=days, forecast_days=1)

    hourly = data.get('hourly', {})
    times = hourly.get('time', [])
    keys = ['cape', 'lifted_index', 'wind_direction_10m',
            'precipitation_probability', 'weather_code', 'cloud_cover']

    by_date: Dict[date, List[tuple]] = {}
    for i, t in enumerate(times):
        try:
            dt = datetime.fromisoformat(t)
        except ValueError:
            continue
        d = dt.date()
        if d < start_date or d > end_date:
            continue
        hour_data = {k: hourly[k][i] for k in keys}
        qualifies, reasons = client._qualifies(hour_data)
        if qualifies:
            by_date.setdefault(d, []).append((dt, 0.5, reasons))

    print(f"\nBacktest: {start_date} → {end_date}")
    print("=" * 78)
    cur = start_date
    while cur <= end_date:
        hours = by_date.get(cur, [])
        if hours:
            windows = client._merge_into_windows(hours)
            for w in windows:
                print(
                    f"{cur}  WATCH {w.start.strftime('%H:%M')}-{w.end.strftime('%H:%M')}  "
                    f"{', '.join(w.reasons[:4])}"
                )
        else:
            # Find the max CAPE for context
            max_cape = max(
                (hourly['cape'][i] for i, t in enumerate(times)
                 if datetime.fromisoformat(t).date() == cur
                 and hourly['cape'][i] is not None),
                default=None,
            )
            cape_str = f"CAPE max {max_cape:.0f}" if max_cape else "no data"
            print(f"{cur}  no watch — {cape_str}")
        cur += timedelta(days=1)
    print("=" * 78)
```

Add `from typing import Dict` to imports if not already present.

- [ ] **Step 3: Smoke-test the new CLI**

Run: `python main.py test --weather`

Expected: prints storm watch windows (or "0 storm watch window(s)" if none).

Run: `python main.py test --storm-prompt`

Expected: prints the generated storm prompt.

Run: `python main.py weather --backtest --start 2026-05-20 --end 2026-05-22`

Expected: prints a table with at least one WATCH entry for May 21.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "CLI: add test --weather/--tempest/--storm-prompt + weather --backtest

- test --weather: hit Open-Meteo, summarize watch windows
- test --tempest: 5s UDP listen, assert ≥1 message received
- test --storm-prompt: print sample storm caption prompt
- weather --backtest --start/--end: run trigger logic against historical
  Open-Meteo data, prints daily summary

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: Update CLAUDE.md to document storm workflow

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a Storm Workflow section to CLAUDE.md**

In `CLAUDE.md`, after the existing "Daily workflow" subsection (under "Architecture"), add:

```markdown
### Storm capture workflow (parallel to sunset)

`StormWorkflow.complete_storm_workflow()` in `storm_workflow.py` mirrors the sunset workflow but for thunderstorms. Triggered by a layered detection chain:

1. `OpenMeteoClient.get_storm_watch_windows()` polls Open-Meteo every 30 min (15 min when inside a window). Identifies hours that pass CAPE/LI/wind-direction/precip triggers; merges contiguous qualifying hours into windows.
2. `SunsetScheduler.update_storm_watch_windows()` reconciles state; calls `_reconcile_tempest_arming()` to arm/disarm `TempestMonitor` based on window proximity (default: arm 30 min before, disarm 30 min after).
3. When armed AND Tempest's `_evaluate_storm_conditions` fires, `SunsetScheduler.on_storm_detected` aborts any active sunset capture (via `cancel_event`) and runs `complete_storm_workflow`.
4. Storm workflow: capture → recovery-if-needed (via `historical_retrieval` from camera SD) → SIS computation (Tempest data only, no video analysis) → metadata persistence → Drive backup → shared AI caption (`event_type='storm'` branch of `_build_caption_prompt`) → FB/IG/Reels/YouTube.
5. Failed captures queue in `data/storms/pending_recovery.json` and get one retry during morning maintenance.

**Key config keys:**
- `open_meteo.enabled` / `open_meteo.poll_interval_minutes` / `open_meteo.triggers.*`
- `tempest.enabled` / `tempest.capture.*` (cooldown, max duration, post-storm minutes)
- `storm_analysis.scoring.*` (SIS weights)

**Storm vs sunset precedence:** storm wins. Active sunset capture is cancelled cleanly (partial frames deleted, no half-baked video). Decided in design — see `docs/superpowers/specs/2026-05-22-storm-capture-design.md`.

**Backtest mode** for threshold tuning: `python main.py weather --backtest --start YYYY-MM-DD --end YYYY-MM-DD` (works up to 92 days back per Open-Meteo free tier).
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "CLAUDE.md: document storm capture workflow + entry points

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 20: Final validation + end-to-end smoke test

**Files:**
- (no edits — verification only)

- [ ] **Step 1: Run full unit test suite**

Run: `pytest -m 'not integration' -v`

Expected: all tests pass (~27 tests).

- [ ] **Step 2: Run Open-Meteo integration test**

Run: `pytest tests/test_open_meteo_client.py -v`

Expected: all pass including `test_fetch_forecast_live`.

- [ ] **Step 3: Run CLI smoke tests**

Run each in sequence:
```bash
python main.py config --validate --show
python main.py test --camera --youtube --sunset
python main.py test --weather
python main.py test --storm-prompt
python main.py weather --backtest --start 2026-05-20 --end 2026-05-22
```

Expected:
- `config --validate`: PASS, shows new `open_meteo:` keys
- `test --camera --youtube --sunset`: PASS (no regression)
- `test --weather`: prints ≥ 0 watch windows
- `test --storm-prompt`: prints prompt without banned words
- `weather --backtest`: prints at least one WATCH line for May 21 2026

- [ ] **Step 4: Run the Tempest UDP test only if running on Pi (skip on Mac)**

Run (on Pi): `python main.py test --tempest`

Expected: PASS with ≥1 message received in 5s.

- [ ] **Step 5: Run scheduler in foreground briefly to confirm clean startup/shutdown**

Run: `python main.py schedule --validate` then Ctrl+C within 10s.

Expected: logs include:
- `Sunset scheduler initialized`
- `Tempest UDP listener started` (or "disabled")
- `Open-Meteo storm watch polling scheduled every 30 min` (if `open_meteo.enabled`)
- `Storm watch window forecast` line if a window is currently active
- Clean shutdown on Ctrl+C without traceback

- [ ] **Step 6: Tag the completed work**

```bash
git tag storm-capture-mvp-v1
git log --oneline storm-capture-mvp-v1~20..storm-capture-mvp-v1
```

Expected: ~20 commits on the storm-capture work, in clear forward order.

- [ ] **Step 7: Final cleanup commit if needed**

If any uncommitted edits remain from the smoke tests, commit them with a "Final cleanup" message. Otherwise skip.

---

## Self-review notes

**Spec coverage (verified):**
- Open-Meteo polling cadence (30/15) → Task 15 + config in Task 17 ✓
- State reconciliation per poll → Task 14 (`update_storm_watch_windows`) ✓
- Storm watch trigger logic → Tasks 3-4 ✓
- Tempest arming policy → Task 14 (`_reconcile_tempest_arming`) ✓
- SIS computation → Task 6 ✓
- Storm caption prompt + #alabamawx → Task 7 ✓
- Sunset cancellation → Tasks 10 (camera), 13 (scheduler) ✓
- Recovery: immediate (Task 11) + deferred (Tasks 11, 16) ✓
- Storage layout (`data/storms/`) → Task 9 (`storms_dir`) ✓
- All error-handling rows → Tasks 5 (Open-Meteo failure), 12 (caption fallback, upload failures), 11+16 (recovery), 14 (cooldown via existing TempestMonitor logic) ✓
- Backtest mode → Task 18 ✓
- New validation commands → Task 18 ✓
- Pi config has tempest + open_meteo → Task 17 ✓
- Rollback path (enabled flags) → Task 17 ✓

**Type/signature consistency check:** verified that `cancel_event` parameter signature matches across `camera_interface.capture_video_sequence`, `storm_workflow._capture_storm_sequence`, `storm_workflow.complete_storm_workflow`, and `sunset_scheduler.capture_sunset_sequence`. Verified `StormWindow` dataclass fields used consistently. Verified `StormConditions` and `LightningStrike` / `WeatherObservation` imports match the dataclasses defined in `tempest_monitor.py`.

**Non-goals not implemented (intentional, per spec):**
- Video-based SBS-for-storms
- NWS API integration
- Storm cell motion vectors
- Stormy-sunset merge
- Dynamic cooldown override
