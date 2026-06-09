# Sunset capture: extend post-sunset window on Pi

**Date:** 2026-06-09
**Status:** Approved, pending implementation

## Problem

The Pi's daily capture cuts off 30 minutes after sunset
(`config-pi.yaml: capture.time_after_sunset_minutes: 30`). Civil twilight in
Pelham ends ~25–30 minutes after sunset, but the deep red/magenta afterglow —
the "burn out" — frequently peaks 25–45 minutes after sunset and can persist
through nautical twilight. Recent timelapses have been clipping right when the
best color appears.

The Mac config (`config.yaml`) already uses 60 min on the back side; the Pi was
set to 30 separately and was never bumped.

## Goal

Extend the Pi's post-sunset capture window to 60 minutes so daily timelapses
include the full afterglow.

## Decisions

- Change `config-pi.yaml: capture.time_after_sunset_minutes` from `30` to `60`.
- Leave `time_before_sunset_minutes` at `30`. The pre-sunset side is rarely the
  problem and the existing 30-min lead is sufficient.
- No code changes. `sunset_calculator.get_capture_window()` reads both keys
  with sensible defaults; the scheduler and capture loop use that result
  directly with no separate assumption about window length (verified by
  grepping `capture.time_before_sunset_minutes` /
  `capture.time_after_sunset_minutes` across the codebase — both are read only
  in `sunset_calculator.py`).
- `config.yaml` (Mac) is already at 60/60 and needs no change.

## Impact

- **Frames:** at the configured 5-second interval, +360 frames per evening
  (30 min × 60 s ÷ 5 s).
- **Video duration:** at `video.fps: 60`, +6 seconds of rendered timelapse.
- **Storage/processing:** negligible. The Pi already handles the 60-min lead
  side without strain, and the assembled video stays well under upload limits.
- **Scheduler:** the daily job is keyed to the capture-window start, so the
  end shift has no scheduling side effects.
- **SBS, captions, social posts:** all consume the captured frames directly;
  they automatically benefit from the longer window with no change.

## Out of scope

- No change to the pre-sunset window.
- No change to the Mac config.
- No new config keys, defaults, or code paths.
- No retroactive change for historical reprocessing — `historical_retrieval`
  uses its own window logic.

## Validation

- After deploying, the next evening's log line
  `Capture window: <start> to <end>` should show `<end>` ≈ sunset + 60 min.
- The last captured frame timestamp should land ~60 min after the predicted
  sunset.
- Spot-check the assembled video for the additional ~6 seconds of dusk content
  at the tail.
