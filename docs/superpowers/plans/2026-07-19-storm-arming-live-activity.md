# Storm Watch Live-Activity Disarm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `TempestMonitor` from disarming mid-storm when Open-Meteo's forecast window disappears on a later poll — disarm should be driven by live Tempest activity going quiet, not by forecast churn.

**Architecture:** `TempestMonitor` gains a live-activity timestamp (`last_storm_activity_time`) updated by real UDP observations/strikes, plus an `armed_since` timestamp, and two query methods (`recent_activity`, `armed_duration_exceeds`). `SunsetScheduler._reconcile_tempest_arming` keeps forecast-driven arming unchanged but replaces forecast-driven disarming with "stay armed while forecast says so OR live activity is recent, capped by a safety limit."

**Tech Stack:** Python 3.13, no pytest (repo has no test runner — standalone verify scripts following the existing `verify_storm_lightning_arm.py` pattern), YAML config.

## Global Constraints

- No pytest suite exists in this repo — `main.py test` is the validation harness, and one-off logic gets a standalone `verify_*.py` script with plain `assert`, not a `tests/` file. Follow the existing `verify_storm_lightning_arm.py` pattern exactly.
- The Raspberry Pi's systemd service loads `config.yaml`, **not** `config-pi.yaml` — any new config key must be added to `config.yaml`.
- Never read, display, or edit `.env`, `.env.pi`, `.env.mac`, or `facebook_config.json`.
- Deploying to the Pi (`update_pi.sh`) restarts the live production service — do not run it without explicit user confirmation immediately beforehand.
- Reference design doc: `docs/superpowers/specs/2026-07-19-storm-arming-live-activity-design.md`. Every task below implements a specific section of it.

---

### Task 1: `TempestMonitor` — live-activity timestamp + query helpers

**Files:**
- Modify: `tempest_monitor.py:110-116` (state fields), `tempest_monitor.py:146-156` (`arm`/`disarm`)
- Create: `verify_storm_disarm_quiet_period.py`

**Interfaces:**
- Produces: `TempestMonitor.last_storm_activity_time: Optional[datetime]`, `TempestMonitor.armed_since: Optional[datetime]`, `TempestMonitor.recent_activity(within_minutes: float) -> bool`, `TempestMonitor.armed_duration_exceeds(hours: float) -> bool`. Task 2 and Task 3 depend on all four.

- [ ] **Step 1: Write the failing verify script**

Create `verify_storm_disarm_quiet_period.py`:

```python
"""
Standalone verification for the storm-watch live-activity disarm change.

Run from repo root with the venv active:
    python verify_storm_disarm_quiet_period.py

Exits 0 and prints "ALL CHECKS PASSED" on success; raises AssertionError otherwise.
No pytest dependency — this repo has no test runner; main.py test is the harness.
See docs/superpowers/specs/2026-07-19-storm-arming-live-activity-design.md.
"""

from datetime import datetime, timedelta

from tempest_monitor import TempestMonitor


def fresh_monitor():
    """A monitor with clean arm/activity state; UDP listener never started."""
    m = TempestMonitor()
    m.armed = False
    m.armed_since = None
    m.last_storm_activity_time = None
    m.storm_active = False
    m.last_storm_capture_time = None
    return m


def check_recent_activity_helper():
    m = fresh_monitor()
    assert m.recent_activity(45) is False, "no activity yet -> False"

    m.last_storm_activity_time = datetime.now() - timedelta(minutes=10)
    assert m.recent_activity(45) is True, "10 min ago within 45 min window -> True"

    m.last_storm_activity_time = datetime.now() - timedelta(minutes=50)
    assert m.recent_activity(45) is False, "50 min ago outside 45 min window -> False"
    print("  ok: recent_activity()")


def check_armed_duration_exceeds_helper():
    m = fresh_monitor()
    assert m.armed_duration_exceeds(6) is False, "never armed -> False"

    m.armed_since = datetime.now() - timedelta(hours=2)
    assert m.armed_duration_exceeds(6) is False, "2h armed, 6h cap -> False"

    m.armed_since = datetime.now() - timedelta(hours=7)
    assert m.armed_duration_exceeds(6) is True, "7h armed, 6h cap -> True"
    print("  ok: armed_duration_exceeds()")


def check_arm_disarm_track_timestamps():
    m = fresh_monitor()
    m.arm()
    assert m.armed_since is not None, "arm() must set armed_since"
    first = m.armed_since

    m.arm()  # already armed — must not reset the clock
    assert m.armed_since == first, "re-arming while already armed must not reset armed_since"

    m.disarm()
    assert m.armed_since is None, "disarm() must clear armed_since"
    print("  ok: arm()/disarm() track armed_since")


if __name__ == "__main__":
    check_recent_activity_helper()
    check_armed_duration_exceeds_helper()
    check_arm_disarm_track_timestamps()
    print("ALL CHECKS PASSED")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python verify_storm_disarm_quiet_period.py`
Expected: `AttributeError: 'TempestMonitor' object has no attribute 'armed_since'` (or `recent_activity`) — the fields/methods don't exist yet.

- [ ] **Step 3: Add the state fields**

In `tempest_monitor.py`, in `__init__`, immediately after the existing block:

```python
        # Storm detection state
        self.storm_active = False
        self.last_storm_capture_time = None
```

add:

```python
        self.last_storm_activity_time: Optional[datetime] = None
        self.armed_since: Optional[datetime] = None
```

- [ ] **Step 4: Update `arm()` / `disarm()` and add the two query helpers**

Replace the existing `arm`/`disarm` methods (`tempest_monitor.py:146-156`):

```python
    def arm(self):
        """Allow storm callbacks to fire. Called by scheduler when entering a watch window."""
        if not self.armed:
            self.logger.info("TempestMonitor armed — storm callbacks now active")
            self.armed_since = datetime.now()
        self.armed = True

    def disarm(self):
        """Suppress storm callbacks. Called by scheduler when outside watch windows."""
        if self.armed:
            self.logger.info("TempestMonitor disarmed — storm callbacks suppressed")
        self.armed = False
        self.armed_since = None

    def recent_activity(self, within_minutes: float) -> bool:
        """True if a live storm-relevant signal was observed within the last
        `within_minutes` minutes. See
        docs/superpowers/specs/2026-07-19-storm-arming-live-activity-design.md.
        """
        if self.last_storm_activity_time is None:
            return False
        elapsed_minutes = (datetime.now() - self.last_storm_activity_time).total_seconds() / 60.0
        return elapsed_minutes <= within_minutes

    def armed_duration_exceeds(self, hours: float) -> bool:
        """True if continuously armed for longer than `hours` (safety cap)."""
        if self.armed_since is None:
            return False
        elapsed_hours = (datetime.now() - self.armed_since).total_seconds() / 3600.0
        return elapsed_hours > hours
```

- [ ] **Step 5: Run the verify script to confirm it passes**

Run: `python verify_storm_disarm_quiet_period.py`
Expected:
```
  ok: recent_activity()
  ok: armed_duration_exceeds()
  ok: arm()/disarm() track armed_since
ALL CHECKS PASSED
```

- [ ] **Step 6: Commit**

```bash
git add tempest_monitor.py verify_storm_disarm_quiet_period.py
git commit -m "TempestMonitor: add live-activity timestamp + armed-duration helpers"
```

---

### Task 2: Wire live-activity detection into observation/lightning processing

**Files:**
- Modify: `tempest_monitor.py:284-360` (`_process_observation`), `tempest_monitor.py:362-392` (`_process_lightning_strike`)
- Modify: `verify_storm_disarm_quiet_period.py` (append checks)

**Interfaces:**
- Consumes: `TempestMonitor.last_storm_activity_time`, `self.wind_gust_threshold`, `self.lightning_max_distance` (all exist already or from Task 1).
- Produces: `_process_observation` and `_process_lightning_strike` now set `last_storm_activity_time` on a loose "is weather happening" bar. Task 3's scheduler checks depend on this being wired correctly (they set `last_storm_activity_time` manually, so Task 3 doesn't strictly require this task's code — but Task 2 must land first per the design doc's live-detection requirement).

- [ ] **Step 1: Write the failing checks**

Append to `verify_storm_disarm_quiet_period.py`, above the `if __name__ == "__main__":` block:

```python
def check_process_observation_rain_sets_activity():
    m = fresh_monitor()
    epoch = datetime.now().timestamp()
    # fields: [ts, wind_lull, wind_avg, wind_gust, wind_dir, wind_interval, pressure,
    #          temp_c, humidity, lux, uv, solar_rad, rain_accum, precip_type,
    #          lightning_avg_dist, lightning_count, battery, report_interval]
    msg = {'type': 'obs_st', 'obs': [[epoch, 0.0, 1.0, 2.0, 200, 3, 1010.0, 20.0, 60, 0, 0, 0, 0.5, 1, 0, 0, 2.6, 1]]}
    m._process_observation(msg)
    assert m.last_storm_activity_time is not None, "precip_type=1 (rain) must set activity"
    print("  ok: rain observation sets last_storm_activity_time")


def check_process_observation_calm_does_not_set_activity():
    m = fresh_monitor()
    epoch = datetime.now().timestamp()
    msg = {'type': 'obs_st', 'obs': [[epoch, 0.0, 1.0, 2.0, 200, 3, 1010.0, 20.0, 60, 0, 0, 0, 0.0, 0, 0, 0, 2.6, 1]]}
    m._process_observation(msg)
    assert m.last_storm_activity_time is None, "calm observation (no rain/wind/pressure drop) must not set activity"
    print("  ok: calm observation leaves last_storm_activity_time unset")


def check_process_observation_high_gust_sets_activity():
    m = fresh_monitor()
    epoch = datetime.now().timestamp()
    gust_mph = m.wind_gust_threshold / 2 + 1  # just above the "elevated" bar
    gust_ms = gust_mph / 2.237
    msg = {'type': 'obs_st', 'obs': [[epoch, 0.0, gust_ms, gust_ms, 200, 3, 1010.0, 20.0, 60, 0, 0, 0, 0.0, 0, 0, 0, 2.6, 1]]}
    m._process_observation(msg)
    assert m.last_storm_activity_time is not None, "gust >= half of wind_gust_threshold must set activity"
    print("  ok: elevated wind gust sets last_storm_activity_time")


def check_process_observation_pressure_drop_sets_activity():
    m = fresh_monitor()
    now = datetime.now()
    old_epoch = (now - timedelta(minutes=35)).timestamp()
    old_msg = {'type': 'obs_st', 'obs': [[old_epoch, 0.0, 1.0, 1.0, 200, 3, 1015.0, 20.0, 60, 0, 0, 0, 0.0, 0, 0, 0, 2.6, 1]]}
    m._process_observation(old_msg)
    for i in range(4):
        epoch = (now - timedelta(minutes=25 - i * 5)).timestamp()
        msg = {'type': 'obs_st', 'obs': [[epoch, 0.0, 1.0, 1.0, 200, 3, 1013.0, 20.0, 60, 0, 0, 0, 0.0, 0, 0, 0, 2.6, 1]]}
        m._process_observation(msg)
    assert m.last_storm_activity_time is None, "priming observations (calm, <6 total) must not set activity yet"

    epoch = now.timestamp()
    msg = {'type': 'obs_st', 'obs': [[epoch, 0.0, 1.0, 1.0, 200, 3, 1010.0, 20.0, 60, 0, 0, 0, 0.0, 0, 0, 0, 2.6, 1]]}
    m._process_observation(msg)  # 6th observation: pressure has fallen since the 35-min-old reading
    assert m.last_storm_activity_time is not None, "falling pressure over 30 min must set activity"
    print("  ok: pressure drop sets last_storm_activity_time")


def check_process_lightning_strike_within_range_sets_activity():
    m = fresh_monitor()
    epoch = datetime.now().timestamp()
    msg = {'type': 'evt_strike', 'evt': [epoch, m.lightning_max_distance - 1, 1000]}
    m._process_lightning_strike(msg)
    assert m.last_storm_activity_time is not None, "nearby strike must set activity"
    print("  ok: nearby lightning strike sets last_storm_activity_time")


def check_process_lightning_strike_far_away_does_not_set_activity():
    m = fresh_monitor()
    epoch = datetime.now().timestamp()
    msg = {'type': 'evt_strike', 'evt': [epoch, m.lightning_max_distance + 10, 1000]}
    m._process_lightning_strike(msg)
    assert m.last_storm_activity_time is None, "distant strike must not set activity"
    print("  ok: distant lightning strike does not set last_storm_activity_time")
```

And update the `__main__` block to call all six new checks:

```python
if __name__ == "__main__":
    check_recent_activity_helper()
    check_armed_duration_exceeds_helper()
    check_arm_disarm_track_timestamps()
    check_process_observation_rain_sets_activity()
    check_process_observation_calm_does_not_set_activity()
    check_process_observation_high_gust_sets_activity()
    check_process_observation_pressure_drop_sets_activity()
    check_process_lightning_strike_within_range_sets_activity()
    check_process_lightning_strike_far_away_does_not_set_activity()
    print("ALL CHECKS PASSED")
```

- [ ] **Step 2: Run it to verify the new checks fail**

Run: `python verify_storm_disarm_quiet_period.py`
Expected: first new check fails —
`AssertionError: precip_type=1 (rain) must set activity` (nothing sets `last_storm_activity_time` yet).

- [ ] **Step 3: Wire activity detection into `_process_observation`**

In `tempest_monitor.py`, inside `_process_observation`, immediately after:

```python
            # Store observation
            self.observations.append(observation)
```

insert:

```python
            # Live-activity signal for the disarm quiet-period timer — a looser
            # bar than the capture-trigger thresholds below: "is weather still
            # happening here," not "is it severe enough to record." See
            # docs/superpowers/specs/2026-07-19-storm-arming-live-activity-design.md.
            activity = precip_type != 0 or wind_gust_mph >= self.wind_gust_threshold / 2
            if not activity and len(self.observations) >= 6:
                now = datetime.now()
                pressure_window = timedelta(minutes=30)
                old_obs = [o for o in self.observations
                           if now - o.timestamp >= pressure_window]
                if old_obs and (pressure - old_obs[-1].pressure) < 0:
                    activity = True
            if activity:
                self.last_storm_activity_time = datetime.now()
```

- [ ] **Step 4: Wire activity detection into `_process_lightning_strike`**

In `tempest_monitor.py`, inside `_process_lightning_strike`, immediately after:

```python
            # Store strike
            self.lightning_strikes.append(strike)
```

insert:

```python
            # Live-activity signal — any nearby strike counts, not gated by
            # lightning_min_strikes (that's the capture-trigger bar, not the
            # "is a storm happening" bar). See
            # docs/superpowers/specs/2026-07-19-storm-arming-live-activity-design.md.
            if distance_km <= self.lightning_max_distance:
                self.last_storm_activity_time = datetime.now()
```

- [ ] **Step 5: Run the verify script to confirm all checks pass**

Run: `python verify_storm_disarm_quiet_period.py`
Expected:
```
  ok: recent_activity()
  ok: armed_duration_exceeds()
  ok: arm()/disarm() track armed_since
  ok: rain observation sets last_storm_activity_time
  ok: calm observation leaves last_storm_activity_time unset
  ok: elevated wind gust sets last_storm_activity_time
  ok: pressure drop sets last_storm_activity_time
  ok: nearby lightning strike sets last_storm_activity_time
  ok: distant lightning strike does not set last_storm_activity_time
ALL CHECKS PASSED
```

- [ ] **Step 6: Regression-check the existing lightning-override verify script**

Run: `python verify_storm_lightning_arm.py`
Expected: `ALL CHECKS PASSED` (unchanged — confirms the new activity-tracking side effects don't break the existing lightning-override behavior).

- [ ] **Step 7: Commit**

```bash
git add tempest_monitor.py verify_storm_disarm_quiet_period.py
git commit -m "TempestMonitor: detect live storm activity from observations + strikes"
```

---

### Task 3: `SunsetScheduler` — disarm on quiet period instead of forecast churn

**Files:**
- Modify: `sunset_scheduler.py:484-500` (`_reconcile_tempest_arming`)
- Modify: `config.yaml:329-330` (add two keys under `open_meteo:`)
- Modify: `verify_storm_disarm_quiet_period.py` (append checks)

**Interfaces:**
- Consumes: `TempestMonitor.recent_activity`, `TempestMonitor.armed_duration_exceeds`, `TempestMonitor.armed`, `TempestMonitor.armed_since` (Task 1), `TempestMonitor.last_storm_activity_time` set live (Task 2).
- Produces: `SunsetScheduler._reconcile_tempest_arming` now reads `open_meteo.tempest_disarm_quiet_minutes` (default 45) and `open_meteo.tempest_arm_safety_cap_hours` (default 6).

- [ ] **Step 1: Write the failing checks**

Append to `verify_storm_disarm_quiet_period.py`, above the `if __name__ == "__main__":` block. Add these imports to the top of the file (alongside the existing `from tempest_monitor import TempestMonitor`):

```python
from types import SimpleNamespace

from sunset_scheduler import SunsetScheduler
from open_meteo_client import StormWindow
from config_manager import get_config
```

```python
def fake_scheduler(storm_watch_windows, tempest_monitor, capture_state='IDLE'):
    """A minimal stand-in for SunsetScheduler exposing only the attributes
    _reconcile_tempest_arming reads. Avoids constructing the real
    SunsetScheduler, whose __init__ wires up camera/YouTube/Drive/etc."""
    return SimpleNamespace(
        config=get_config(),
        storm_watch_windows=storm_watch_windows,
        tempest_monitor=tempest_monitor,
        capture_state=capture_state,
    )


def check_forecast_window_still_arms_and_disarms_as_before():
    m = fresh_monitor()
    now = datetime.now()
    window = StormWindow(start=now - timedelta(minutes=5), end=now + timedelta(minutes=30), confidence=0.6)
    fake = fake_scheduler([window], m)
    SunsetScheduler._reconcile_tempest_arming(fake)
    assert m.armed is True, "inside a forecast window must arm (baseline unchanged)"

    fake.storm_watch_windows = []
    m.last_storm_activity_time = None  # no live activity at all
    SunsetScheduler._reconcile_tempest_arming(fake)
    assert m.armed is False, "forecast window gone + no live activity must still disarm"
    print("  ok: forecast-window baseline arm/disarm unchanged")


def check_live_activity_keeps_armed_when_forecast_window_vanishes():
    m = fresh_monitor()
    m.arm()
    m.last_storm_activity_time = datetime.now() - timedelta(minutes=10)  # rain 10 min ago
    fake = fake_scheduler([], m)  # forecast window has vanished
    SunsetScheduler._reconcile_tempest_arming(fake)
    assert m.armed is True, "recent live activity must keep it armed despite no forecast window"
    print("  ok: live activity overrides a vanished forecast window (2026-07-19 bug, fixed)")


def check_disarms_after_quiet_period_elapses():
    m = fresh_monitor()
    m.arm()
    m.last_storm_activity_time = datetime.now() - timedelta(minutes=50)  # older than default 45 min quiet window
    fake = fake_scheduler([], m)
    SunsetScheduler._reconcile_tempest_arming(fake)
    assert m.armed is False, "must disarm once the quiet period has elapsed"
    print("  ok: disarms after the quiet period elapses")


def check_safety_cap_overrides_continuous_activity():
    m = fresh_monitor()
    m.arm()
    m.armed_since = datetime.now() - timedelta(hours=7)  # past the default 6h safety cap
    m.last_storm_activity_time = datetime.now()  # activity right now
    fake = fake_scheduler([], m)
    SunsetScheduler._reconcile_tempest_arming(fake)
    assert m.armed is False, "safety cap must win over continuous live activity"
    print("  ok: safety cap overrides continuous activity")


def check_storm_active_capture_still_blocks_disarm():
    m = fresh_monitor()
    m.arm()
    m.last_storm_activity_time = None  # no activity, would normally disarm
    fake = fake_scheduler([], m, capture_state='STORM_ACTIVE')
    SunsetScheduler._reconcile_tempest_arming(fake)
    assert m.armed is True, "must not disarm mid-capture regardless of activity/forecast"
    print("  ok: STORM_ACTIVE capture still blocks disarm (existing guard, unchanged)")
```

Update the `__main__` block to call the five new checks after the existing ones:

```python
if __name__ == "__main__":
    check_recent_activity_helper()
    check_armed_duration_exceeds_helper()
    check_arm_disarm_track_timestamps()
    check_process_observation_rain_sets_activity()
    check_process_observation_calm_does_not_set_activity()
    check_process_observation_high_gust_sets_activity()
    check_process_observation_pressure_drop_sets_activity()
    check_process_lightning_strike_within_range_sets_activity()
    check_process_lightning_strike_far_away_does_not_set_activity()
    check_forecast_window_still_arms_and_disarms_as_before()
    check_live_activity_keeps_armed_when_forecast_window_vanishes()
    check_disarms_after_quiet_period_elapses()
    check_safety_cap_overrides_continuous_activity()
    check_storm_active_capture_still_blocks_disarm()
    print("ALL CHECKS PASSED")
```

- [ ] **Step 2: Run it to verify the new checks fail**

Run: `python verify_storm_disarm_quiet_period.py`
Expected: `check_live_activity_keeps_armed_when_forecast_window_vanishes` fails —
`AssertionError: recent live activity must keep it armed despite no forecast window`
(current `_reconcile_tempest_arming` only looks at forecast windows).

- [ ] **Step 3: Update `_reconcile_tempest_arming`**

Replace `sunset_scheduler.py:484-500`:

```python
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

with:

```python
    def _reconcile_tempest_arming(self):
        """Arm/disarm Tempest based on forecast proximity and live activity.

        The forecast decides when to start watching. Once armed, live Tempest
        data decides when it's safe to stop — a forecast window disappearing
        must never, by itself, disarm mid-storm. See
        docs/superpowers/specs/2026-07-19-storm-arming-live-activity-design.md.
        """
        lead = self.config.get('open_meteo.tempest_arm_lead_minutes', 30)
        trail = self.config.get('open_meteo.tempest_arm_trailing_minutes', 30)
        quiet_minutes = self.config.get('open_meteo.tempest_disarm_quiet_minutes', 45)
        safety_cap_hours = self.config.get('open_meteo.tempest_arm_safety_cap_hours', 6)
        now = datetime.now()

        near_forecast_window = any(
            (window.start - timedelta(minutes=lead)) <= now <= (window.end + timedelta(minutes=trail))
            for window in self.storm_watch_windows
        )

        live_activity_keeping_armed = (
            self.tempest_monitor.armed
            and self.tempest_monitor.recent_activity(quiet_minutes)
            and not self.tempest_monitor.armed_duration_exceeds(safety_cap_hours)
        )

        should_arm = near_forecast_window or live_activity_keeping_armed

        if should_arm and not self.tempest_monitor.armed:
            self.tempest_monitor.arm()
        elif not should_arm and self.tempest_monitor.armed:
            # Only disarm if no capture is in progress
            if self.capture_state != 'STORM_ACTIVE':
                self.tempest_monitor.disarm()
```

- [ ] **Step 4: Add the new config keys**

In `config.yaml`, replace lines 329-330:

```yaml
  tempest_arm_lead_minutes: 30        # arm Tempest this far before window start
  tempest_arm_trailing_minutes: 30    # keep armed this long after window end
```

with:

```yaml
  tempest_arm_lead_minutes: 30        # arm Tempest this far before window start
  tempest_arm_trailing_minutes: 30    # keep armed this long after window end
  tempest_disarm_quiet_minutes: 45    # once armed, stay armed until this many quiet minutes pass (live activity, not forecast)
  tempest_arm_safety_cap_hours: 6     # hard ceiling on continuous armed duration, regardless of live activity
```

- [ ] **Step 5: Run the verify script to confirm all checks pass**

Run: `python verify_storm_disarm_quiet_period.py`
Expected: all 14 `ok:` lines print, followed by `ALL CHECKS PASSED`.

- [ ] **Step 6: Regression-check the existing lightning-override verify script**

Run: `python verify_storm_lightning_arm.py`
Expected: `ALL CHECKS PASSED`.

- [ ] **Step 7: Commit**

```bash
git add sunset_scheduler.py config.yaml verify_storm_disarm_quiet_period.py
git commit -m "Disarm storm watch on live-activity quiet period, not forecast churn"
```

---

### Task 4: Deploy to the Pi

**Files:** none (deployment only)

**Interfaces:** none — this task ships Tasks 1-3's already-committed changes.

- [ ] **Step 1: Confirm with the user before touching the live service**

`update_pi.sh` restarts the production `sunset-timelapse` systemd service on the Pi.
Ask the user to confirm it's OK to deploy now before running anything in this task.

- [ ] **Step 2: Push local commits**

Run: `git push origin main`
Expected: the three commits from Tasks 1-3 appear on `origin/main`.

- [ ] **Step 3: Deploy on the Pi**

Run (after user confirmation from Step 1):
```bash
ssh sunset@sunset.mvp 'cd sunset-timelapse && ./update_pi.sh'
```
Expected output ends with `🎉 Update complete!` and a systemd status block showing `Active: active (running)`.

- [ ] **Step 4: Confirm the new config keys loaded**

Run:
```bash
ssh sunset@sunset.mvp 'cd sunset-timelapse && source venv/bin/activate && python3 -c "
from config_manager import get_config
cfg = get_config()
print(\"quiet_minutes:\", cfg.get(\"open_meteo.tempest_disarm_quiet_minutes\", \"<missing>\"))
print(\"safety_cap_hours:\", cfg.get(\"open_meteo.tempest_arm_safety_cap_hours\", \"<missing>\"))
"'
```
Expected: `quiet_minutes: 45` and `safety_cap_hours: 6` (not `<missing>`).

- [ ] **Step 5: Run the verify script on the Pi as a smoke check**

Run:
```bash
ssh sunset@sunset.mvp 'cd sunset-timelapse && source venv/bin/activate && python3 verify_storm_disarm_quiet_period.py'
```
Expected: `ALL CHECKS PASSED`.

- [ ] **Step 6: Watch the next arm cycle in the journal**

Run:
```bash
ssh sunset@sunset.mvp 'sudo journalctl -u sunset-timelapse -f | grep -i --line-buffered "storm\|tempest"'
```
Watch (don't block on this — just confirm the service is emitting the expected log lines) for the next `TempestMonitor armed` event and confirm it does **not** immediately disarm on the following poll while `data/logs/sunset_timelapse.log` or the live Tempest observation shows ongoing rain. This is confirmation, not a blocking step — end the watch once satisfied or after a reasonable check-in, since the next real storm may not happen for days.
