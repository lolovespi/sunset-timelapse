# Storm capture: arm off live lightning

**Date:** 2026-06-01
**Status:** Approved, pending implementation

## Problem

On 2026-06-01 a thunderstorm passed around 15:23 CDT with confirmed lightning
strikes ~1 km from the station, but no storm capture ran. Two compounding
defects in `tempest_monitor.py`, confirmed from the Pi's journal:

1. **Forecast-only arming lags real storms.** `TempestMonitor` fires storm
   callbacks only while `armed`, and arming is driven entirely by the
   Open-Meteo forecast window via `SunsetScheduler._reconcile_tempest_arming`.
   The forecast window for the day was 16:00–17:00, so arming opened at 15:31
   (30 min lead). The real storm arrived at 15:23 — 8 minutes *before* arming —
   and the detection was suppressed.

2. **Cooldown is consumed by suppressed detections.** In
   `_evaluate_storm_conditions`, `self.last_storm_capture_time = now` is set
   whenever a storm is detected and the (possibly suppressed) callback path
   runs — not when a capture actually fires. The 15:23 detection therefore
   started the 1-hour cooldown even though nothing was captured. Every later
   detection (15:31 armed window, 16:09, 16:23) then logged "in cooldown" and
   was dropped. The field name `last_storm_capture_time` is a lie: it records
   the last *detection*, not the last *capture*.

Net effect: the storm fell into the 8-minute gap before forecast arming, the
cooldown was burned by a capture that never happened, and the brief armed
window (15:31–15:46, before the forecast window shifted away) was entirely
inside that bogus cooldown.

## Goal

When the Tempest confirms nearby lightning, trigger a storm capture
immediately, independent of the forecast-driven arming window.

## Decisions (from brainstorming)

- **Trigger bar:** confirmed lightning only (existing `lightning_active`, which
  requires `lightning_min_strikes` within `lightning_max_distance`). Rain/wind
  without lightning still requires forecast arming. Lightning is the most
  reliable and most visually compelling storm signal.
- **Time window:** any hour, 24/7. No daylight gate on the lightning override.
- **FOV visibility heuristic:** unchanged — remains informational (logged, not
  gating), exactly as today.
- **No new config keys.** Reuses existing lightning thresholds and cooldown.

## Changes

Both changes are in `tempest_monitor.py`; one comment update in
`sunset_scheduler.py`.

### 1. Lightning overrides the arming gate

`_fire_storm_callbacks(conditions)` currently returns early unless
`self.armed`. Add a force path so it also fires on confirmed lightning:

- Fire when `self.armed OR conditions.lightning_active`.
- Otherwise (disarmed, no lightning) keep the existing debug-log + suppress.

A lightning-only storm already scores 0.4, so it clears the existing
`storm_detected` confidence threshold and reaches this gate.

### 2. Cooldown starts only on an actual capture

- `_fire_storm_callbacks` returns a boolean: `True` if it actually invoked the
  callbacks, `False` if suppressed.
- In `_evaluate_storm_conditions`, set `self.last_storm_capture_time = now`
  **only when** `_fire_storm_callbacks` returned `True`.

This makes the cooldown reflect real captures, so a suppressed detection no
longer silences the rest of the storm.

### 3. Comment fix

`sunset_scheduler.py:511-513` claims "if we got here, we're inside a watch
window OR caller explicitly armed it." Update to note the lightning-override
path (a live-lightning detection can fire callbacks while disarmed).

## Coordination (verified in current code)

- `on_storm_detected` sets `capture_state = 'STORM_ACTIVE'` before running
  `complete_storm_workflow` synchronously, so `_reconcile_tempest_arming` will
  not disarm mid-capture (`sunset_scheduler.py:499` guards on `STORM_ACTIVE`).
- After the workflow, the `finally` block calls `_reconcile_tempest_arming`,
  returning arming to the forecast baseline. The now-correct cooldown prevents
  an immediate re-fire.
- The lightning override does not call `self.arm()` — it bypasses the gate for
  a single fire. Arming state stays owned by the scheduler's forecast
  reconciliation.

## Behavior on the 2026-06-01 case

15:23 — lightning ~1 km, `lightning_active` true, monitor disarmed →
`_fire_storm_callbacks` fires via the override → `on_storm_detected` runs the
storm workflow → `last_storm_capture_time` set to 15:23 from a real capture →
1-hour cooldown correctly spaces subsequent detections.

## Verification

No pytest suite exists; `main.py test` is the validation harness. Add a
standalone verification script (not wired into a runner) that constructs a
`TempestMonitor` with a stub config and a recording callback, then asserts:

1. Disarmed + lightning detection → callback fires, `last_storm_capture_time`
   is set.
2. Disarmed + non-lightning detection (e.g. heavy rain + high winds, no
   lightning) → callback does **not** fire, `last_storm_capture_time` stays
   `None`.
3. After a fire, a second qualifying detection within the cooldown window is
   suppressed.

## Deployment

Edit on `main` (Mac fast-forwarded to `origin/main` `ec08bae`). Deploy to the
Pi via `update_pi.sh`. No config migration.

## Out of scope

- Changing the forecast-arming logic or lead/trail timing.
- Daylight/civil-dusk gating of storm captures.
- Pressure/rain/wind auto-arm paths (lightning only for now).
- Renaming `last_storm_capture_time` (kept to minimize churn; semantics are now
  correct).
