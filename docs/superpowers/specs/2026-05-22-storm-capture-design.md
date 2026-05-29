# Storm Capture — Design

**Status**: Draft, pending implementation
**Date**: 2026-05-22
**Author**: Lo + Claude Code (brainstorming session)

## Goal

Extend the sunset-timelapse system to automatically capture and post timelapses of thunderstorms visible from the Pelham, AL camera, with the same social distribution parity as the existing sunset workflow (YouTube + Facebook + Instagram + Reels + Google Drive backup).

The system must:
- Predict storm windows hours in advance (forecast-driven)
- Confirm in real-time using local Tempest weather sensor (ground truth)
- Filter out storms not visible from the camera's WNW-facing FOV
- Cancel an active sunset capture if a storm fires during sunset
- Recover from failed live captures by pulling 24x7 continuous recordings off the camera SD card

## Architecture overview

Layered detection: **Open-Meteo (predictive)** schedules storm watch windows; **Tempest (reactive)** fires the actual capture trigger when ground-level sensor conditions confirm. NWS deferred — only adds incremental value beyond what these two cover for the common cases.

```
open_meteo_client.py   ← NEW: daily/intra-day forecast → storm watch windows
storm_workflow.py      ← NEW: storm-specific capture/upload pipeline
sunset_scheduler.py    ← MODIFY: wire Tempest + Open-Meteo, add storm callback,
                                 add cancellation hook to sunset capture
facebook_uploader.py   ← MODIFY: storm-prompt branch in _build_caption_prompt;
                                 add #alabamawx to existing sunset hashtag list
config.yaml,           ← MODIFY: add open_meteo block; activate tempest.enabled;
config-pi.yaml                  add tempest block to Pi config
```

### Component boundaries

**`OpenMeteoClient` (`open_meteo_client.py`)**
- `get_storm_watch_windows(target_date) -> list[StormWindow]` — hits api.open-meteo.com, merges contiguous storm-trigger hours into windows
- `StormWindow` dataclass: `start`, `end`, `confidence`, `reasons[]`
- Backtest entry point for tuning thresholds against past_days (~92 days back, free tier)
- Pure data layer: no scheduling, no side effects beyond HTTP

**`StormWorkflow` (`storm_workflow.py`)**
- `complete_storm_workflow(conditions, start_time) -> bool` — orchestrator mirroring `SunsetScheduler.complete_daily_workflow`
- `compute_storm_intensity_score(conditions, observations) -> dict` — Tempest-only SIS from existing `storm_analysis.scoring:` weights
- `_build_storm_caption_prompt(metadata)` — storm-flavored Haiku 4.5 prompt
- Reuses existing modules: `camera_interface`, `video_processor`, `drive_uploader`, `facebook_uploader`, `youtube_uploader`, `historical_retrieval` (for recovery)

**`SunsetScheduler` modifications**
- New attrs: `tempest_monitor`, `open_meteo_client`, `storm_workflow`, `storm_watch_windows[]`, `current_capture_cancel_event`, `capture_state` (IDLE | SUNSET_ACTIVE | STORM_ACTIVE)
- New scheduled jobs: storm watch window polling every 30 min during 06:00–22:00 (every 15 min inside active windows)
- Sunset capture loop reads `cancel_event` between frame intervals so storm callbacks can interrupt it
- Storm callback handler — abort sunset if active, run `storm_workflow.complete_storm_workflow()`

## Data flow

```
30/15-min poll  ──► OpenMeteoClient.get_storm_watch_windows()
                              │
                              ▼
              reconcile against storm_watch_windows state
              (new / extended / shrunken / disappeared)
                              │
                              ▼
              compute Tempest arm/disarm transitions
              (arm 30 min before window start, disarm 30 min after window end)
                              │
                              ▼
              TempestMonitor listener active 24/7;
              callbacks gated by `armed` flag

           ┌──────────────────┴──────────────────┐
           ▼ (armed)                              ▼ (unarmed)
   _evaluate_storm_conditions             callback suppressed
   fires storm callback                   (observations still buffered)
           │
           ▼
   on_storm_detected(conditions)
           │
   ┌───────┴──────────┐
   ▼                  ▼
SUNSET_ACTIVE      IDLE
   │                  │
   ▼                  │
set cancel_event      │
wait ≤5s for          │
sunset to exit        │
delete partial        │
sunset images         │
   │                  │
   └──────┬───────────┘
          ▼
   capture_state = STORM_ACTIVE
          │
          ▼
   storm_workflow.complete_storm_workflow(conditions)
          │
          ▼
   capture → process → SIS → Drive → caption → FB/IG/Reels/YouTube
          │
          ▼
   capture_state = IDLE
```

## Polling cadence

- **Every 30 min during 06:00–22:00 local** — baseline (~32 polls/day)
- **Every 15 min inside an active watch window** — tighter cadence for arm/disarm timing
- **Outside daylight hours** — no polling unless already inside an active window
- Open-Meteo free-tier limit is 10,000/day; usage stays under 1%

### Storm watch window state reconciliation

Each poll diffs the new windows list against the current state:
- **New window appears** → log + email notification with forecast detail
- **Existing window extends** → update end time, keep Tempest armed
- **Existing window shrinks/disappears** → if not capturing, disarm Tempest; if capturing, let it complete naturally
- **CAPE/LI improves dramatically** (≥500 J/kg jump) → email alert "storm conditions strengthening"

## Storm watch trigger logic

For each forecast hour to qualify as a watch window, ALL must be true:
- `CAPE > cape_min` (default 1500 J/kg)
- `lifted_index < lifted_index_max` (default -3.0)
- `wind_direction` within camera arc (236°–324°) + `wind_fov_margin_degrees` (default 30°)
- AT LEAST ONE of:
  - `precipitation_probability > 40`
  - `weather_code >= 80`
  - `cloud_cover > 80`

Strictly contiguous qualifying hours (no gap hours between them) merge into one window. Any gap → separate windows.

## Tempest arming policy

| Time relative to watch window | Tempest UDP listener | Callbacks fire? |
|---|---|---|
| 30 min before window start | thread running | yes |
| Inside window | thread running | yes |
| 30 min after window end | thread running | yes |
| Otherwise | thread running | no (gated by `armed` flag) |

The UDP listener runs 24/7 (lightweight, no socket churn). The `armed` flag only gates whether storm callbacks fire — observations still buffer for context.

## Storm Intensity Score (SIS)

Tempest sensor data only. No video analysis. Computed after capture from data in Tempest ring buffers.

```
sis = min(100,
    lightning_count_in_window × 5.0       (lightning_weight)
  + (40 / max(avg_distance_km, 5)) × 20.0  (distance_weight, inverted)
  + max_rain_rate_mm_hr × 2.0              (rain_weight)
  + max_wind_gust_mph × 0.5                (wind_weight)
  + abs(pressure_drop_hpa) × 10.0          (pressure_weight)
)
```

Grade mapping: A (≥80), B (60-79), C (40-59), D (20-39), F (<20).

The `storm_analysis.scoring.visual_weight: 3.0` field in existing config remains dormant in MVP — documents the future-add hook for video-based lightning-flash counting without requiring implementation.

## Captions and titles

**Storm caption prompt** (storm-flavored branch in existing `FacebookUploader._build_caption_prompt`):

> Write a 2-3 sentence caption for a thunderstorm timelapse. Stay direct, dry, observational — no hype, no slang, no weather-channel drama. Lead with the most concrete sensor reading available (lightning strike count, peak wind gust, peak rain rate, or pressure drop, whichever is most striking). Mention the storm's apparent direction of travel only if confident from wind data. Reference visual character only from the frames provided — actual cloud structure, lightning visible, sky color, rain visibility. Avoid: "epic", "incredible", "wild", "intense", "Mother Nature", "ripped through".

Falls back to a templated caption if Anthropic unreachable (existing fallback pattern):

> "Thunderstorm timelapse from Pelham, AL on {date}. {lightning_count} lightning strikes, peak gust {gust} mph, {rain_total}" of rain. SIS {sis}/{grade}. Camera: Reolink RLC810-WA."

**YouTube title**: `"Storm {date}"` (deterministic, matches existing `"Sunset MM/DD/YY"` convention)

**YouTube description**: same shared caption as Facebook/Instagram/Reels, plus auto-appended structured block:

```
---
Lightning: {count} strikes within {window} min, avg distance {avg}km
Peak wind gust: {gust} mph
Rain total: {accumulation}"
Pressure drop: {drop} hPa over {duration} min
Storm Intensity Score: {sis} (Grade {grade})

#thunderstorm #lightning #timelapse #alabama #alabamawx #weather #pelham
```

## Hashtag update (cross-cutting)

Add `#alabamawx` to BOTH storm and sunset social posts:

- **Sunset** (`facebook_uploader.append_hashtags()` or equivalent): `#sunset #timelapse #alabama #alabamawx #pelham`
- **Storm**: `#thunderstorm #lightning #timelapse #alabama #alabamawx #weather #pelham`

## Error handling

| Failure | Behavior |
|---|---|
| Open-Meteo unreachable | Log warning, email once per 6h. Fall back to Tempest armed continuously (degrades to reactive-only). Last-known windows valid for 24h. |
| Open-Meteo malformed data | Ignore poll, log at WARN. Next poll attempt in 15-30 min. |
| Tempest UDP silent (existing behavior) | Existing `last_message_time` check warns after 5 min. No new code. |
| Sunset cancellation timeout (>5s) | Force-kill camera grab task. Storm capture proceeds. Log warning. Should never happen given 5s frame intervals. |
| Storm capture: zero frames, or fewer than 50% of `(duration_seconds / capture.interval_seconds)` expected frames | **Immediate recovery**: pull camera SD recordings for storm window via existing `historical_retrieval`. Process recovered footage and continue workflow. Email subject prefixed `[RECOVERED]`. |
| Immediate recovery also fails | **Deferred recovery**: add storm window to `pending_recovery.json`. Existing 06:00 morning maintenance run picks it up and retries. One retry per storm — no infinite loops. |
| Storm capture: complete failure, no recoverable footage | Email notification + log. SIS recorded for posterity. No social posts. Workflow returns False. |
| Anthropic caption failure | Existing templated-fallback path with sensor data block. |
| FB/IG/Reels/YouTube upload failure | Existing per-platform try/except. Log, email per failed platform, don't block others. |
| Cooldown edge case (second storm within `tempest.capture.cooldown_hours` of first storm's *start* time) | MVP: second storm suppressed, logged with `storm_suppressed_during_cooldown: true`. |
| Tempest fires outside any forecast window | Capture anyway. Email prefixed `[UNFORECAST]` to flag the forecast model needs tuning. |

24x7 continuous camera recording guarantees recovery footage always exists for any storm window — "no recording exists" is not a failure mode we need to handle.

## Testing

| Component | Test approach |
|---|---|
| `OpenMeteoClient.get_storm_watch_windows` | Unit test with recorded JSON response. Asserts window merging, trigger thresholds, wind-direction arc filtering. |
| `TempestMonitor._evaluate_storm_conditions` | Use existing standalone test mode. Add fixtures simulating the May 21 storm observation stream. |
| `compute_storm_intensity_score` | Pure function unit tests. Sample StormConditions + observations → assert SIS score and grade. |
| `_build_storm_caption_prompt` | Unit test: required fields present, banned words absent. |
| Sunset capture cancellation | Mock camera; verify capture loop exits within one frame interval after `cancel_event.set()`. |

### Backtest mode

```bash
python main.py weather --backtest --start 2024-05-01 --end 2024-05-31
```

For each day: pull historical Open-Meteo, run watch-window logic, print:

```
2024-05-21  WATCH 17:00–22:00  CAPE 1820  LI -4.7  wind 252°  ✓ (storm at 21:00)
2024-05-22  no watch — CAPE max 540
2024-05-23  WATCH 14:00–16:00  CAPE 1650  LI -3.8  wind 180°  ✗ (wind outside arc)
```

Allows tuning thresholds against real history before deployment.

### New validation commands

```bash
python main.py test --weather       # Open-Meteo connectivity + response parsing
python main.py test --tempest       # UDP reception (5s listen, assert ≥1 message)
python main.py test --storm-prompt  # generate sample storm caption against synthetic data
```

Wires into existing `cmd_test()` alongside `--camera --youtube --sunset`.

## Configuration

New `config.yaml` block (also added to `config-pi.yaml` — Pi config currently has no `tempest:` block):

```yaml
open_meteo:
  enabled: true
  poll_interval_minutes: 30
  active_window_poll_minutes: 15
  daylight_hours: [6, 22]

  triggers:
    cape_min: 1500
    lifted_index_max: -3.0
    wind_fov_margin_degrees: 30
    require_any_of:
      precipitation_probability_min: 40
      weather_code_min: 80
      cloud_cover_min: 80

  tempest_arm_lead_minutes: 30
  tempest_arm_trailing_minutes: 30
```

Existing `tempest:` block: change `enabled: false` → `true`, set `station_id`.

## Storage layout

```
data/
├── storms/
│   ├── 2026-05-21/
│   │   ├── images/             # captured frames
│   │   ├── video.mp4           # rendered timelapse
│   │   ├── storm_metadata.json # Tempest snapshot + SIS + caption
│   │   └── recovery.log        # if recovered from camera SD
│   └── pending_recovery.json   # deferred recovery queue
```

Retention follows existing `storage.keep_videos_locally_days` (currently 14). Metadata JSON kept indefinitely for backtest cross-referencing.

## Deployment

Standard path: develop on Mac, commit, push, `./update_pi.sh` on Pi. One extra step the first time:

1. On Pi, set `tempest.enabled: true` + `tempest.station_id` in `config-pi.yaml`
2. Verify UDP reception: `python tempest_monitor.py` (standalone test mode, 60s)
3. Verify forecast: `python main.py test --weather`
4. Backtest: `python main.py weather --backtest --start <recent_date> --end <recent_date>`
5. Restart service: `sudo systemctl restart sunset-timelapse`

### Rollback path

Set `tempest.enabled: false` and `open_meteo.enabled: false` in config. Scheduler reverts to sunset-only — no storm callbacks, no Open-Meteo polling. No code changes needed.

### Pi resource impact

- Open-Meteo polls: ~32-48 HTTP requests/day, ~200ms each → negligible
- Tempest UDP listener: existing thread, no new resources
- Storm capture: same camera/CPU load as sunset
- Worst case storm (4h max duration): ~3000 frames, ~2GB video pre-cleanup. Within existing storage envelope.

## Explicit non-goals (MVP)

These were considered and deferred — listing them so future-us doesn't re-litigate during implementation:

- **Video-based Storm Brilliance Score** (frame-by-frame lightning flash detection, cloud darkness analysis). Adds ~8-10 hours; tempest-data SIS is sufficient for MVP captions.
- **NWS API integration**. High-precision/low-recall alert layer. Open-Meteo + Tempest catches the common cases; revisit only if backtest shows gaps.
- **Storm cell motion vectors / radar polygon visibility**. Tempest doesn't provide bearing per strike; getting cell motion requires NWS or third-party lightning API.
- **Sunset/storm merge into one "stormy sunset" video**. Decided: storm wins, sunset aborts cleanly. Cleaner separation, simpler workflow.
- **Per-storm dynamic cooldown override**. MVP uses fixed `tempest.capture.cooldown_hours`. If pulsing storms cut off interesting events, revisit.
- **CLI commands beyond `--backtest` and `test --weather/--tempest/--storm-prompt`**. The plan's full `weather` subcommand suite is YAGNI for MVP.
- **`tempest.visibility.fov_margin_degrees` config + `is_storm_likely_visible()` heuristic** added in uncommitted edits to `tempest_monitor.py`. Stays as a log-only diagnostic in MVP; future tuning may promote it to a scoring signal.

## Open questions

None — all design questions resolved during brainstorming.
