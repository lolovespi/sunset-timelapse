# Lightning Detection obs_st Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make lightning-based storm detection actually work by feeding `obs_st`'s per-minute `lightning_strike_count`/`lightning_avg_distance` fields — confirmed reliably delivered — into the same strike-tracking `evt_strike` already feeds, since `evt_strike` UDP messages are evidently not being delivered by this Tempest hub.

**Architecture:** Single-file change to `tempest_monitor.py`. `_process_observation` (the `obs_st` handler) already parses `lightning_avg_distance`/`lightning_strike_count` and discards them. Add a block that synthesizes `LightningStrike` entries into the existing `self.lightning_strikes` deque — the same structure `_process_lightning_strike` (the `evt_strike` handler) populates — so `_evaluate_storm_conditions`'s existing windowing/threshold logic needs zero changes.

**Tech Stack:** Python 3, no new dependencies. Testing via a new standalone `verify_*.py` script (this repo's convention — no pytest suite, see `verify_storm_lightning_arm.py` for the pattern).

## Global Constraints

- No config keys change. No changes to arm/disarm timing, thresholds, or any trigger other than lightning.
- `evt_strike` handling stays exactly as-is (not removed, not disabled) — it may start working on its own hub firmware/behavior, and the two channels are additive, not exclusive.
- Synthesis is unconditional (not gated on `self.lightning_enabled`), matching how `_process_lightning_strike` behaves today — the enabled flag only gates the trigger check inside `_evaluate_storm_conditions`, not data collection.
- Number of synthesized entries per observation is clamped to `self.lightning_strikes.maxlen` (currently 50) to guard against a corrupt/absurd count field.
- The new code path must log its own INFO line, distinct from `_process_lightning_strike`'s `⚡ Lightning strike detected` line — the complete absence of lightning logging is what made the original bug invisible, so the fallback path needs its own visible signal.
- `verify_*.py` files are gitignored in this repo (`.gitignore` line 22) — write and run them, but never `git add` them.
- `config.yaml` is also gitignored — do not create or modify it as part of this task; the implementer's local Mac config.yaml already has a valid `tempest:` block for `TempestMonitor()` construction (same as prior plans in this repo).

---

### Task 1: Synthesize obs_st lightning data into the strike-tracking deque

**Files:**
- Modify: `tempest_monitor.py:370-394` (inside `_process_observation`, between the existing live-activity block and the `# Log observation` comment)
- Test: `verify_storm_lightning_obs_fallback.py` (new, repo root, gitignored — follow the pattern in `verify_storm_lightning_arm.py`)

**Interfaces:**
- Consumes: `TempestMonitor` (from `tempest_monitor.py`), specifically:
  - `self.lightning_strikes: collections.deque[LightningStrike]` (existing, `maxlen=50`)
  - `self.lightning_max_distance: float` (existing, from `tempest.triggers.lightning.max_distance_km` config)
  - `self.last_storm_activity_time: Optional[datetime]` (existing)
  - `LightningStrike(timestamp: datetime, distance_km: float, energy: int, bearing: Optional[float] = None)` dataclass (existing)
  - `self._evaluate_storm_conditions()` (existing, already called unconditionally at the end of `_process_observation` — no new call needed)
- Produces: nothing new is exposed publicly. `_evaluate_storm_conditions()` (unchanged) will now see synthesized strikes in `self.lightning_strikes` and can set `StormConditions.lightning_active = True` through its existing logic.

Current code at `tempest_monitor.py:370-394` (for orientation — do not copy this into the diff, it's what's already there):

```python
            # Store observation
            self.observations.append(observation)

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

            # Log observation
            self.logger.debug(f"Observation: {temp_f:.1f}°F, {humidity:.0f}% RH, "
                            f"{pressure:.1f} hPa, Wind {wind_speed_mph:.1f}/{wind_gust_mph:.1f} mph, "
                            f"Rain {rain_rate:.1f} mm/hr")

            # Check for storm conditions
            self._evaluate_storm_conditions()
```

Note `lightning_avg_distance` and `lightning_strike_count` are already parsed a few lines earlier at `tempest_monitor.py:335-336` (`lightning_avg_distance = obs_data[14]`, `lightning_strike_count = obs_data[15]`) — they are in scope at this point in the function, just currently unused.

- [ ] **Step 1: Write the failing test file**

Create `verify_storm_lightning_obs_fallback.py` in the repo root:

```python
"""
Standalone verification for the obs_st lightning-detection fallback.

evt_strike UDP messages aren't reliably delivered by this Tempest hub (see
docs/superpowers/specs/2026-07-31-lightning-obs-fallback-design.md), but
obs_st's own lightning_strike_count/lightning_avg_distance fields are. This
verifies that data gets synthesized into the same strike-tracking deque
_process_lightning_strike (the evt_strike path) already populates.

Run from repo root with the venv active:
    python verify_storm_lightning_obs_fallback.py

Exits 0 and prints "ALL CHECKS PASSED" on success; raises AssertionError otherwise.
No pytest dependency — this repo has no test runner; main.py test is the harness.
"""

import time
from datetime import datetime

from tempest_monitor import TempestMonitor


def make_obs_st_message(lightning_count=0, lightning_distance=0.0, precip_type=0,
                         wind_gust_ms=0.0, pressure=1010.0, ts=None):
    """Raw obs_st UDP message dict, matching the Tempest UDP v171 field order."""
    ts = ts or time.time()
    obs = [
        ts,             # 0 time epoch
        0.0,            # 1 wind lull m/s
        1.0,            # 2 wind avg m/s
        wind_gust_ms,   # 3 wind gust m/s
        200,            # 4 wind direction
        3,              # 5 wind sample interval
        pressure,       # 6 station pressure hPa
        21.0,           # 7 air temp C
        50,             # 8 relative humidity %
        0,              # 9 illuminance lux
        0,              # 10 UV
        0,              # 11 solar radiation W/m2
        0.0,            # 12 rain accumulated mm
        precip_type,    # 13 precip type
        lightning_distance,  # 14 lightning avg distance km
        lightning_count,     # 15 lightning strike count
        2.6,            # 16 battery volts
        1,              # 17 reporting interval
    ]
    return {"type": "obs_st", "obs": [obs]}


def fresh_monitor():
    """A monitor with a recording callback; UDP listener never started."""
    m = TempestMonitor()
    fired = []
    m.register_storm_callback(lambda conditions: fired.append(conditions))
    m.armed = True  # armed so a real trigger isn't suppressed by arming state
    m.storm_active = False
    m.last_storm_capture_time = None
    return m, fired


def check_synthesizes_strikes_and_updates_activity():
    m, fired = fresh_monitor()
    assert len(m.lightning_strikes) == 0
    assert m.last_storm_activity_time is None

    msg = make_obs_st_message(lightning_count=5, lightning_distance=10.0)
    m._process_observation(msg)

    assert len(m.lightning_strikes) == 5, f"expected 5 synthesized strikes, got {len(m.lightning_strikes)}"
    assert all(s.distance_km == 10.0 for s in m.lightning_strikes)
    assert m.last_storm_activity_time is not None, "nearby lightning must extend the quiet-period timer"
    print("  ok: obs_st strike count synthesizes entries + updates activity timer")


def check_zero_or_none_count_synthesizes_nothing():
    m, fired = fresh_monitor()

    m._process_observation(make_obs_st_message(lightning_count=0, lightning_distance=10.0))
    assert len(m.lightning_strikes) == 0, "count=0 must not synthesize strikes"

    print("  ok: zero strike count synthesizes nothing")


def check_count_above_maxlen_is_clamped():
    m, fired = fresh_monitor()
    maxlen = m.lightning_strikes.maxlen

    m._process_observation(make_obs_st_message(lightning_count=maxlen + 70, lightning_distance=15.0))

    assert len(m.lightning_strikes) == maxlen, (
        f"expected clamp to maxlen={maxlen}, got {len(m.lightning_strikes)}"
    )
    print("  ok: strike count above maxlen is clamped, no error")


def check_distance_beyond_threshold_does_not_update_activity():
    m, fired = fresh_monitor()
    far_distance = m.lightning_max_distance + 50.0

    m._process_observation(make_obs_st_message(lightning_count=3, lightning_distance=far_distance))

    assert len(m.lightning_strikes) == 3, "strikes are still recorded regardless of distance"
    assert m.last_storm_activity_time is None, "distant lightning must not extend the quiet-period timer"
    print("  ok: distant lightning recorded but does not extend activity timer")


def check_end_to_end_triggers_storm_conditions():
    m, fired = fresh_monitor()

    for _ in range(m.lightning_min_strikes):
        msg = make_obs_st_message(lightning_count=1, lightning_distance=5.0)
        m._process_observation(msg)

    assert len(fired) >= 1, "qualifying obs_st lightning data must trigger a storm callback"
    conditions = fired[-1]
    assert conditions.lightning_active is True, "lightning_active must be set once threshold is met"
    print("  ok: qualifying obs_st lightning data triggers _evaluate_storm_conditions")


if __name__ == "__main__":
    check_synthesizes_strikes_and_updates_activity()
    check_zero_or_none_count_synthesizes_nothing()
    check_count_above_maxlen_is_clamped()
    check_distance_beyond_threshold_does_not_update_activity()
    check_end_to_end_triggers_storm_conditions()
    print("ALL CHECKS PASSED")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python verify_storm_lightning_obs_fallback.py`

Expected: `AssertionError` on `check_synthesizes_strikes_and_updates_activity` — `len(m.lightning_strikes) == 5` fails because nothing synthesizes strikes yet (obs_st lightning fields are parsed and discarded on `main`).

- [ ] **Step 3: Implement the obs_st lightning synthesis**

In `tempest_monitor.py`, inside `_process_observation`, insert this block immediately after the existing `if activity: self.last_storm_activity_time = datetime.now()` block and before the `# Log observation` comment (i.e., between current lines 386 and 388):

```python
            # Lightning fallback: evt_strike UDP messages aren't reliably
            # delivered by this hub, but obs_st's own strike count/distance
            # fields are. Synthesize entries into the same deque
            # _process_lightning_strike (the evt_strike path) already
            # populates, so _evaluate_storm_conditions needs no changes. See
            # docs/superpowers/specs/2026-07-31-lightning-obs-fallback-design.md.
            if lightning_strike_count is not None and lightning_strike_count > 0 \
                    and lightning_avg_distance is not None:
                synthesized_count = min(lightning_strike_count, self.lightning_strikes.maxlen)
                for _ in range(synthesized_count):
                    self.lightning_strikes.append(
                        LightningStrike(timestamp=timestamp, distance_km=lightning_avg_distance, energy=0)
                    )
                if lightning_avg_distance <= self.lightning_max_distance:
                    self.last_storm_activity_time = datetime.now()
                self.logger.info(
                    f"⚡ Lightning strikes from obs_st: {lightning_strike_count} strikes, "
                    f"avg distance {lightning_avg_distance:.1f} km"
                )
```

This uses `timestamp`, `lightning_avg_distance`, and `lightning_strike_count`, all already assigned earlier in the same function (`tempest_monitor.py:321,335-336`). `LightningStrike` is already imported/defined in this module (used by `_process_lightning_strike`).

- [ ] **Step 4: Run the test to verify it passes**

Run: `python verify_storm_lightning_obs_fallback.py`

Expected:
```
  ok: obs_st strike count synthesizes entries + updates activity timer
  ok: zero strike count synthesizes nothing
  ok: strike count above maxlen is clamped, no error
  ok: distant lightning recorded but does not extend activity timer
  ok: qualifying obs_st lightning data triggers _evaluate_storm_conditions
ALL CHECKS PASSED
```

- [ ] **Step 5: Run the existing lightning regression test**

Run: `python verify_storm_lightning_arm.py`

Expected: `ALL CHECKS PASSED` (unchanged — confirms the `evt_strike` path still works exactly as before; this task only adds a second, additive data source).

- [ ] **Step 6: Commit**

```bash
git add tempest_monitor.py
git commit -m "Feed obs_st lightning strike data into storm detection

evt_strike UDP messages aren't reliably delivered by this Tempest hub, so
the lightning trigger has likely never fired. obs_st already carries the
same strike count/distance data reliably; synthesize it into the existing
strike-tracking deque so detection and the disarm quiet-period timer both
pick it up."
```

(`verify_storm_lightning_obs_fallback.py` is gitignored — do not `git add` it.)

---

### Task 2: Deploy to the Pi (controller-run, post-merge)

This task is not run by a subagent. The controller runs it directly after Task 1 is reviewed and merged to `main`, mirroring the deployment step from `docs/superpowers/plans/2026-07-19-storm-arming-live-activity.md`.

- [ ] **Step 1: Push to origin**

```bash
git push origin main
```

Confirm the pushed SHA matches the local merge commit — a prior deploy in this repo was caught pulling a stale `origin/main` because the merge hadn't been pushed yet. Check with:

```bash
git log --oneline -1 origin/main
git log --oneline -1 main
```

Both must match before continuing.

- [ ] **Step 2: Run `update_pi.sh`**

```bash
ssh sunset@sunset.mvp "cd sunset-timelapse && ./update_pi.sh"
```

Confirm the script reports success and the service is active afterward.

- [ ] **Step 3: Confirm the Pi is running the new code**

```bash
ssh sunset@sunset.mvp "cd sunset-timelapse && git log --oneline -1"
```

Must show the Task 1 commit (or a merge commit containing it).

- [ ] **Step 4: Watch the journal through the next obs_st cycle**

```bash
ssh sunset@sunset.mvp "journalctl -u sunset-timelapse -f"
```

Watch for at least 90 seconds (obs_st arrives roughly once a minute). Expected: no new tracebacks or errors. If lightning is active at deploy time, look for the new `⚡ Lightning strikes from obs_st` line; if not, its absence is expected (no lightning to report) and not a failure signal on its own.

- [ ] **Step 5: Report back to the user**

Summarize: commit deployed, `update_pi.sh` succeeded, Pi confirmed on new commit, journal watched with no errors. Note that full confirmation of real-world triggering requires an actual nearby storm — this cannot be manufactured for a live smoke test, so the verification at this step is code-correctness (tests) plus deployment-correctness (journal clean), not an observed live trigger.
