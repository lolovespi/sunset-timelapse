# Storm capture: disarm on live-activity quiet period, not forecast churn

**Date:** 2026-07-19
**Status:** Approved, pending implementation

## Problem

Storm capture has been silently missing real storms since deployment. Confirmed
live on the Pi during an actual storm on 2026-07-19:

- 21:13:10 — Open-Meteo forecast produced a window "21:00–23:00"; `_reconcile_tempest_arming`
  armed `TempestMonitor`.
- 21:43:11 — `TempestMonitor` disarmed, 30 minutes later, while it was actively
  raining (live Tempest observation at 21:44 showed 99% humidity, conditions
  "rainy").
- A fresh `OpenMeteoClient.get_storm_watch_windows()` poll run manually right
  after returned **zero windows** — the forecast had simply stopped predicting
  a storm for the current hour, despite one actively happening.

Root cause: `_reconcile_tempest_arming` (`sunset_scheduler.py`) computes
`should_arm` entirely from `self.storm_watch_windows`, which
`update_storm_watch_windows` fully replaces with the latest Open-Meteo poll
every 15 minutes. Near-term hourly forecasts are volatile — a window that
justified arming can vanish on the very next poll. When it does, the system
disarms mid-storm, before enough live lightning strikes can accumulate to
cross the capture trigger threshold via the existing lightning-override path
(see `2026-06-01-storm-lightning-auto-arm-design.md`). Rain-only or
wind/pressure-only storm phases have no override at all today.

Net effect: arming is real, but it doesn't reliably *stay* armed through an
actual storm, because the system trusts the forecast more than its own
sensor.

## Goal

Forecast still decides when to start watching (avoids running the UDP
listener armed continuously). Live Tempest data decides when it's safe to
stop watching. A forecast window disappearing must never, by itself, disarm
mid-storm.

## Decisions (from brainstorming)

- **Arming trigger unchanged.** Open-Meteo forecast proximity (existing
  lead/trail window) is still what starts a watch. Live data does not arm on
  its own — the existing lightning-override path already covers "storm the
  forecast entirely missed."
- **Disarm gated by a live-activity quiet-period timer**, not a point-in-time
  check. Chosen over sticky/merged forecast windows (Approach 2, rejected —
  still fundamentally trusts the forecast) and a single point-in-time live
  guard at poll time (Approach 3, rejected — too coarse, only re-evaluated
  every 15 min).
- **Quiet period: 45 minutes.** Armed state persists until 45 minutes pass
  with no live storm-relevant activity, regardless of forecast state.
- **Activity bar is deliberately looser than the capture-trigger thresholds.**
  This timer answers "is weather still happening here," not "is it severe
  enough to record." Any of the following resets the quiet timer:
  - measurable precipitation occurring (not the full heavy-rain trigger)
  - a lightning strike within `lightning_max_distance`, any count (not gated
    by `lightning_min_strikes`)
  - wind gust at or above half of `wind_gust_threshold_mph`
  - pressure actively falling over the last 30 min (any negative trend, not
    the full drop threshold)
- **Safety cap: 6 hours.** Hard ceiling on continuous armed duration,
  independent of live activity, in case a stuck/misbehaving sensor reading
  keeps resetting the quiet timer indefinitely.
- **Existing guards unchanged:** still won't disarm mid-`STORM_ACTIVE`
  capture; the lightning-override-fires-despite-disarmed path is untouched.

## Changes

### 1. `tempest_monitor.py` — track live storm activity

- Add `self.last_storm_activity_time: Optional[datetime] = None` and
  `self.armed_since: Optional[datetime] = None` (set in `arm()`, cleared in
  `disarm()`) to `TempestMonitor.__init__`.
- In `_process_observation` (obs_st messages, ~once/min): if precipitation is
  occurring, wind gust ≥ half `wind_gust_threshold`, or the 30-min pressure
  trend is negative, set `self.last_storm_activity_time = now`.
- In `_process_lightning_strike` (evt_strike messages): if the strike is
  within `lightning_max_distance`, set `self.last_storm_activity_time = now`
  (regardless of `lightning_min_strikes` — one nearby strike counts as
  "activity" even if it doesn't yet meet the capture-trigger bar).
- Add `TempestMonitor.recent_activity(within_minutes: float) -> bool`:
  returns `True` if `last_storm_activity_time` is set and within the given
  window of now.
- Add `TempestMonitor.armed_duration_exceeds(hours: float) -> bool`: returns
  `True` if `armed_since` is set and armed longer than the given duration.

### 2. `sunset_scheduler.py` — `_reconcile_tempest_arming`

Replace the current `should_arm` computation:

```python
should_arm = any(
    (window.start - timedelta(minutes=lead)) <= now <= (window.end + timedelta(minutes=trail))
    for window in self.storm_watch_windows
)
```

with:

```python
near_forecast_window = any(
    (window.start - timedelta(minutes=lead)) <= now <= (window.end + timedelta(minutes=trail))
    for window in self.storm_watch_windows
)

quiet_minutes = self.config.get('open_meteo.tempest_disarm_quiet_minutes', 45)
safety_cap_hours = self.config.get('open_meteo.tempest_arm_safety_cap_hours', 6)

live_activity_keeping_armed = (
    self.tempest_monitor.armed
    and self.tempest_monitor.recent_activity(quiet_minutes)
    and not self.tempest_monitor.armed_duration_exceeds(safety_cap_hours)
)

should_arm = near_forecast_window or live_activity_keeping_armed
```

The rest of `_reconcile_tempest_arming` (arm/disarm calls, the
`capture_state != 'STORM_ACTIVE'` guard on disarm) is unchanged.

### 3. New config keys (`config.yaml`, under `open_meteo:`)

```yaml
open_meteo:
  tempest_disarm_quiet_minutes: 45     # keep armed until this many quiet minutes pass
  tempest_arm_safety_cap_hours: 6      # hard ceiling on continuous armed duration
```

Both read via `config.get(...)` with the defaults shown above already in the
code, so a missing key degrades gracefully rather than breaking — but they
should be added explicitly to `config.yaml` for visibility/tunability, matching
how `tempest_arm_lead_minutes` / `tempest_arm_trailing_minutes` are already
documented there.

## Behavior on the 2026-07-19 case

21:13 — forecast window arms Tempest. 21:44 — live observation shows active
rain → `last_storm_activity_time` updates. At the next reconcile (poll or
scheduler tick), even though the forecast window has vanished,
`recent_activity(45)` is `True` → stays armed. It only disarms once 45
minutes pass with no measurable rain/wind/pressure-drop/lightning — i.e. once
the storm has actually left the area, not once Open-Meteo's model revises its
mind.

## Verification

No pytest suite exists; `main.py test` is the validation harness. Add a
standalone verification script (following the pattern of
`verify_storm_lightning_arm.py`) that:

1. Arms a `TempestMonitor` with a stubbed config, feeds it a rain-bearing
   observation, then simulates the forecast window disappearing — asserts
   `_reconcile_tempest_arming` keeps it armed.
2. Advances the clock past `tempest_disarm_quiet_minutes` with no further
   activity — asserts it disarms.
3. Feeds continuous rain activity past `tempest_arm_safety_cap_hours` —
   asserts it disarms anyway (safety cap wins over live activity).
4. Confirms mid-`STORM_ACTIVE` capture still blocks disarm regardless of the
   above (existing guard, must not regress).

## Deployment

Edit on `main`, deploy to the Pi via `update_pi.sh`. Add the two new
`open_meteo:` config keys to `config.yaml` (the file the Pi actually loads —
see project note that the Pi loads `config.yaml`, not `config-pi.yaml`) as
part of the same change, with the defaults above.

## Out of scope

- The separate config/code key-path mismatch in `tempest.triggers.*`
  (`tempest_monitor.py` reads nested `tempest.triggers.lightning.*` /
  `tempest.triggers.rain.*` / etc. paths that don't exist in `config.yaml`'s
  flat `tempest.triggers:` block, so every trigger threshold silently uses
  hardcoded defaults). Confirmed live on the Pi as a real, separate bug —
  tracked for a follow-up fix, not part of this change.
- The duplicate top-level `tempest:` key in `config.yaml` that drops
  `device_id` (breaks historical Tempest weather lookups only, not live
  detection). Separate follow-up.
- `tempest.capture.cooldown_hours` vs. configured `cooldown_minutes: 120`
  naming mismatch (cooldown always defaults to 1h instead of the intended
  2h). Separate follow-up.
- Letting live Tempest data arm on its own without any forecast window
  (rejected in brainstorming — the existing lightning-override path already
  covers the "forecast missed it entirely" case for lightning; extending that
  to rain/wind/pressure is a bigger behavior change than this fix needs).
