# Storm Lightning Auto-Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fire a storm capture immediately on confirmed nearby lightning, independent of the Open-Meteo forecast arming window, and stop suppressed detections from consuming the capture cooldown.

**Architecture:** Two edits in `tempest_monitor.py`: (1) `_fire_storm_callbacks` fires when `armed` OR `conditions.lightning_active`, and returns whether it actually fired; (2) `_evaluate_storm_conditions` sets `last_storm_capture_time` only when a capture actually fired. One comment correction in `sunset_scheduler.py`. No new config keys.

**Tech Stack:** Python 3, existing `TempestMonitor` / `SunsetScheduler`. No pytest suite exists in this repo — verification is a standalone script run with `python`.

**Spec:** `docs/superpowers/specs/2026-06-01-storm-lightning-auto-arm-design.md`

---

## File Structure

- **Modify** `tempest_monitor.py`
  - `_fire_storm_callbacks` (currently lines 158-170): add lightning override, return `bool`.
  - `_evaluate_storm_conditions` fire block (currently lines 508-524): set cooldown only on fire.
- **Modify** `sunset_scheduler.py`
  - Comment at lines 511-513: note the lightning-override path.
- **Create** `verify_storm_lightning_arm.py` (repo root): standalone verification script, mirrors the existing `diag_*.py` convention. Asserts the new behavior. Not wired into any runner.

---

## Task 1: Verification script (failing first)

**Files:**
- Create: `verify_storm_lightning_arm.py`

- [ ] **Step 1: Write the verification script**

Create `verify_storm_lightning_arm.py` with exactly this content:

```python
"""
Standalone verification for the storm lightning auto-arm change.

Run from repo root with the venv active:
    python verify_storm_lightning_arm.py

Exits 0 and prints "ALL CHECKS PASSED" on success; raises AssertionError otherwise.
No pytest dependency — this repo has no test runner; main.py test is the harness.
"""

from datetime import datetime

from tempest_monitor import TempestMonitor, StormConditions, LightningStrike, WeatherObservation


def make_obs(rain_rate=0.0, wind_gust=0.0):
    """Minimal valid WeatherObservation timestamped now."""
    return WeatherObservation(
        timestamp=datetime.now(),
        temperature=70.0,
        humidity=50.0,
        pressure=1010.0,
        wind_speed=5.0,
        wind_gust=wind_gust,
        wind_direction=200,
        rain_rate=rain_rate,
        rain_accumulation=0.0,
        solar_radiation=0.0,
        uv_index=0.0,
        battery_voltage=2.6,
    )


def fresh_monitor():
    """A monitor with a recording callback; UDP listener never started."""
    m = TempestMonitor()
    fired = []
    m.register_storm_callback(lambda conditions: fired.append(conditions))
    m.armed = False
    m.storm_active = False
    m.last_storm_capture_time = None
    return m, fired


def check_fire_callbacks_direct():
    # disarmed + lightning -> fires, returns True
    m, fired = fresh_monitor()
    lightning = StormConditions(storm_detected=True, confidence=0.4, lightning_active=True)
    assert m._fire_storm_callbacks(lightning) is True, "lightning override should fire while disarmed"
    assert len(fired) == 1, "callback should have been invoked once"

    # disarmed + no lightning -> suppressed, returns False
    m, fired = fresh_monitor()
    no_lightning = StormConditions(storm_detected=True, confidence=0.4, heavy_rain=True, high_winds=True)
    assert m._fire_storm_callbacks(no_lightning) is False, "no-lightning disarmed should be suppressed"
    assert len(fired) == 0, "callback must not fire when suppressed"

    # armed + no lightning -> fires, returns True
    m, fired = fresh_monitor()
    m.armed = True
    assert m._fire_storm_callbacks(no_lightning) is True, "armed should fire regardless of lightning"
    assert len(fired) == 1, "callback should fire when armed"
    print("  ok: _fire_storm_callbacks override + return value")


def check_lightning_disarmed_captures_and_sets_cooldown():
    m, fired = fresh_monitor()  # disarmed
    m.observations.append(make_obs())
    now = datetime.now()
    for _ in range(m.lightning_min_strikes):
        m.lightning_strikes.append(LightningStrike(timestamp=now, distance_km=1.0, energy=1000))

    m._evaluate_storm_conditions()

    assert len(fired) == 1, "lightning while disarmed should fire a capture"
    assert m.last_storm_capture_time is not None, "a real capture must start the cooldown"
    print("  ok: disarmed lightning fires + sets cooldown")


def check_disarmed_no_lightning_does_not_burn_cooldown():
    m, fired = fresh_monitor()  # disarmed, no lightning strikes
    m.observations.append(make_obs(rain_rate=20.0, wind_gust=40.0))  # heavy_rain + high_winds = 0.4

    m._evaluate_storm_conditions()

    assert len(fired) == 0, "no-lightning while disarmed must not fire"
    assert m.last_storm_capture_time is None, "suppressed detection must NOT consume the cooldown"
    print("  ok: disarmed non-lightning does not fire or burn cooldown")


if __name__ == "__main__":
    check_fire_callbacks_direct()
    check_lightning_disarmed_captures_and_sets_cooldown()
    check_disarmed_no_lightning_does_not_burn_cooldown()
    print("ALL CHECKS PASSED")
```

- [ ] **Step 2: Run it to confirm it fails against current code**

Run: `python verify_storm_lightning_arm.py`

Expected: FAIL. Against current code `_fire_storm_callbacks` returns `None` (not `True`), so the first assertion `is True` fails with `AssertionError: lightning override should fire while disarmed`.

- [ ] **Step 3: Commit the failing verification**

```bash
git add verify_storm_lightning_arm.py
git commit -m "Add verification for storm lightning auto-arm (failing)"
```

---

## Task 2: Lightning override + cooldown-on-fire

**Files:**
- Modify: `tempest_monitor.py` (`_fire_storm_callbacks`, `_evaluate_storm_conditions`)

- [ ] **Step 1: Replace `_fire_storm_callbacks`**

Find (currently lines 158-170):

```python
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

Replace with:

```python
    def _fire_storm_callbacks(self, conditions) -> bool:
        """Fire all registered storm callbacks if armed OR if live lightning is
        confirmed.

        Confirmed nearby lightning overrides the forecast-driven arming gate so a
        real storm is captured even before (or without) a forecast watch window.
        See docs/superpowers/specs/2026-06-01-storm-lightning-auto-arm-design.md.

        Returns True if callbacks were actually invoked, False if suppressed.
        """
        if not self.armed and not conditions.lightning_active:
            self.logger.debug(
                f"Storm conditions met (confidence {conditions.confidence:.1%}) "
                "but TempestMonitor is disarmed and no lightning — callbacks suppressed"
            )
            return False
        if not self.armed and conditions.lightning_active:
            self.logger.info(
                "Lightning override — confirmed nearby lightning; firing storm "
                "callbacks despite disarmed forecast state"
            )
        for callback in self.storm_callbacks:
            try:
                callback(conditions)
            except Exception as e:
                self.logger.error(f"Error in storm callback {callback.__name__}: {e}")
        return True
```

- [ ] **Step 2: Set the cooldown only when a capture fires**

In `_evaluate_storm_conditions`, find (currently lines 521-524):

```python
                # Call registered callbacks
                self._fire_storm_callbacks(conditions)

                self.last_storm_capture_time = now
```

Replace with:

```python
                # Call registered callbacks. Only start the cooldown when a
                # capture actually fires — a suppressed detection must not
                # consume the cooldown (root cause of the 2026-06-01 miss).
                fired = self._fire_storm_callbacks(conditions)
                if fired:
                    self.last_storm_capture_time = now
```

- [ ] **Step 3: Run the verification**

Run: `python verify_storm_lightning_arm.py`

Expected: PASS — prints three `ok:` lines then `ALL CHECKS PASSED`.

- [ ] **Step 4: Commit**

```bash
git add tempest_monitor.py
git commit -m "Arm storm capture off live lightning; fix cooldown accounting

Lightning-confirmed detections now fire storm callbacks even when the
forecast arming window is closed, and the capture cooldown only starts
when a capture actually fires. Fixes the 2026-06-01 missed capture."
```

---

## Task 3: Correct the scheduler comment

**Files:**
- Modify: `sunset_scheduler.py:511-513`

- [ ] **Step 1: Update the comment**

Find (currently lines 511-513):

```python
        # Note: tempest_monitor.armed check already happened upstream in
        # _fire_storm_callbacks — if we got here, we're inside a watch window
        # OR caller explicitly armed it.
```

Replace with:

```python
        # Note: the arming check already happened upstream in
        # _fire_storm_callbacks — if we got here we're either inside a forecast
        # watch window, explicitly armed, OR confirmed nearby lightning fired
        # the lightning-override path (capture independent of the forecast window).
```

- [ ] **Step 2: Re-run the verification (still green)**

Run: `python verify_storm_lightning_arm.py`

Expected: PASS — `ALL CHECKS PASSED` (comment change does not affect behavior; confirms nothing regressed).

- [ ] **Step 3: Commit**

```bash
git add sunset_scheduler.py
git commit -m "Document lightning-override path in on_storm_detected"
```

---

## Self-Review

- **Spec coverage:**
  - Lightning overrides arming gate → Task 2 Step 1.
  - Cooldown only on actual capture → Task 2 Step 2.
  - Comment fix → Task 3.
  - Lightning-only bar (no rain/wind auto-arm) → enforced by `conditions.lightning_active` in Task 2 Step 1; verified by `check_disarmed_no_lightning_does_not_burn_cooldown`.
  - 24/7, no time gate → no time check added anywhere.
  - No new config → confirmed; only existing fields used.
  - Verification (3 spec scenarios) → Task 1 covers all three.
- **Placeholder scan:** none — every step has concrete code/commands.
- **Type consistency:** `_fire_storm_callbacks` returns `bool` in Task 2 and is consumed as `fired` in the same task; `WeatherObservation`/`LightningStrike`/`StormConditions` fields match the dataclasses in `tempest_monitor.py`.

## Deployment (after all tasks)

```bash
git checkout main && git merge --ff-only storm-lightning-auto-arm   # or open a PR
./update_pi.sh                                                       # deploy to the Pi
```

`update_pi.sh` restarts `sunset-timelapse.service`. Confirm with:
`ssh sunset@192.168.40.7 'systemctl is-active sunset-timelapse.service'`
