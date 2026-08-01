# Lightning detection: use obs_st strike data, not just evt_strike

## Problem

Storm capture is still not triggering on real storms, even after the
arm/disarm timing fix ([[2026-07-19-storm-arming-live-activity]]) and the
Pi config-key fix were deployed and verified.

Root cause, confirmed live on the Pi on 2026-07-31:

- On 2026-07-27, WeatherFlow's own cloud data for this station (fetched via
  the `observations/device/{id}` REST endpoint) shows a real, intense,
  nearby lightning event: up to **120 strikes/minute at 20-34km**, well
  inside the configured trigger (`max_distance_km: 30`, `min_strike_count: 3`).
- The app's journal shows **zero** `⚡ Lightning strike detected` log lines
  across the entire window checked (5+ days, including that storm) — despite
  `tcpdump` confirming the Tempest hub *is* broadcasting UDP packets and the
  app's socket *is* bound and receiving `obs_st`/`rapid_wind` traffic live.
- `tempest_monitor.py`'s lightning trigger depends entirely on `evt_strike`
  UDP messages (`_process_lightning_strike`, fed only from the `evt_strike`
  branch of the UDP dispatch loop). That message type is evidently not
  being delivered/processed by this hub, for reasons outside this app's
  control (hardware/firmware behavior — not investigated further here, see
  Out of Scope).
- The regular `obs_st` message — confirmed reliably delivered — already
  carries the same information: `lightning_avg_distance` (index 14) and
  `lightning_strike_count` (index 15). `_process_observation` parses both
  fields today (`tempest_monitor.py:335-336`) and then **never uses them**.

Net effect: the lightning trigger, and the "lightning overrides disarmed
state" safety net from an earlier fix, have likely never fired on this
hub, independent of the arming-timing and config fixes.

## Goal

Make the lightning trigger and the disarm quiet-period's activity signal
work off `obs_st`'s `lightning_strike_count`/`lightning_avg_distance` —
the channel we've confirmed is actually delivered — while leaving
`evt_strike` wired up as-is, in case it ever starts working. Do not touch
arming/disarming logic, thresholds, or any other trigger (rain, wind,
pressure) — those are unaffected by this bug.

## Design

`_process_observation` already parses `lightning_avg_distance` and
`lightning_strike_count` from each `obs_st` message. When
`lightning_strike_count > 0`, synthesize that many `LightningStrike`
entries into the existing `self.lightning_strikes` deque — the same
structure `_process_lightning_strike` (the `evt_strike` path) already
populates — so `_evaluate_storm_conditions`'s existing windowing/counting
logic (`tempest_monitor.py:484-504`) needs no changes at all.

Per synthesized entry:
- `timestamp` = the observation's timestamp (obs_st only gives one
  per-minute count, not individual strike times)
- `distance_km` = `lightning_avg_distance`
- `energy` = `0` (obs_st has no per-strike energy; `energy` is display-only
  today, unused by any detection logic)

Guard rails:
- Skip entirely if `lightning_strike_count` is falsy/`None`/`<= 0`, or
  `lightning_avg_distance` is `None`.
- Clamp the number of synthesized entries to `self.lightning_strikes.maxlen`
  (50) per observation, so a corrupt/absurd count field can't cause a large
  loop. (A single real burst — e.g. the 120/min seen on 2026-07-27 — will
  fill the deque with fresh entries and evict old ones; since
  `_evaluate_storm_conditions` only looks at the last `strike_window_minutes`
  anyway, this is harmless.)

Live-activity signal (the quiet-period disarm timer,
[[2026-07-19-storm-arming-live-activity]]): mirror what
`_process_lightning_strike` already does today — if
`lightning_avg_distance <= self.lightning_max_distance`, update
`self.last_storm_activity_time`. This check is unconditional (not gated on
`self.lightning_enabled`), matching the existing `evt_strike` code path
exactly.

This synthesis is also unconditional (not gated on `self.lightning_enabled`)
for the deque append, matching how `_process_lightning_strike` behaves
today — the enabled flag only gates the trigger check inside
`_evaluate_storm_conditions`, not data collection.

Log an INFO line when synthesizing (e.g. `⚡ Lightning strikes from obs_st:
{count} strikes, avg distance {distance:.1f} km`), distinct from the
existing `_process_lightning_strike` log line. This is deliberate: the
complete absence of any lightning-strike logging is exactly what made this
bug invisible for as long as it was, so the new path needs its own visible
signal in the journal rather than silently feeding the same log line.

Double-counting: if `evt_strike` ever does start working, a real storm
could be counted once via `evt_strike` and roughly once more via `obs_st`
in the same minute. This is an accepted, harmless tradeoff — it can only
make the trigger *slightly more eager*, never cause a missed storm.

## Changes needed

**`tempest_monitor.py`** — `_process_observation`, right after the existing
`activity` block (around line 386):
- If `lightning_strike_count` and `lightning_avg_distance` are present and
  count > 0: append `min(count, self.lightning_strikes.maxlen)` synthesized
  `LightningStrike` entries to `self.lightning_strikes`, and log the new
  INFO line described above.
- If `lightning_avg_distance <= self.lightning_max_distance`: set
  `self.last_storm_activity_time = datetime.now()`.
- No extra call to `_evaluate_storm_conditions()` needed — it already runs
  unconditionally at the end of `_process_observation`.

No other files change. No config keys change.

## Testing

Extend the existing `verify_storm_lightning_arm.py` (or add a new
`verify_storm_lightning_obs_fallback.py`, gitignored like other
`verify_*.py` scripts) with checks covering:
- An `obs_st` observation with `lightning_strike_count = 5` at a distance
  under `lightning_max_distance` synthesizes 5 entries into
  `lightning_strikes` and updates `last_storm_activity_time`.
- `lightning_strike_count = 0` or `None` synthesizes nothing.
- A count larger than `maxlen` (e.g. 120) is clamped and doesn't error.
- Distance beyond `lightning_max_distance`: entries are still recorded
  (matching existing `evt_strike` behavior — the distance filter is applied
  at evaluation time via `nearby_strikes`, not at ingestion), but
  `last_storm_activity_time` is not updated by this signal.
- End-to-end: feeding a synthetic `obs_st` message with a qualifying strike
  count causes `_evaluate_storm_conditions()` to set
  `conditions.lightning_active = True`, matching the existing
  `min_strike_count`/`strike_window_minutes` thresholds.
- Regression: existing `evt_strike`-driven tests in
  `verify_storm_lightning_arm.py` still pass unchanged.

## Deployment

No config changes. Same deploy path as the arming fix: push to `main`,
`update_pi.sh` on the Pi, restart `sunset-timelapse`, watch the journal
through the next `obs_st` cycle (≤60s) for no new errors, and confirm the
next real lightning activity (or a manual test using a synthetic UDP
packet, if available) produces the new `⚡ Lightning strikes from obs_st`
log line and, above threshold, a trigger.

## Out of scope

- Investigating *why* the hub isn't delivering `evt_strike` locally
  (firmware behavior, WeatherFlow app settings, hub placement). Deferred;
  `evt_strike` stays wired as-is in case it starts working on its own.
- Rain/wind/pressure triggers — confirmed unaffected by this bug, not
  touched.
- Any change to arm/disarm timing, thresholds, or config keys.
