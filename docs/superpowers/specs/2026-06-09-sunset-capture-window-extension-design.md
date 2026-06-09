# Sunset capture: extend post-sunset window on Pi

**Date:** 2026-06-09
**Status:** Approved, pending implementation

## Problem

The Pi's daily capture cuts off 30 minutes after sunset. Civil twilight in
Pelham ends ~25–30 minutes after sunset, but the deep red/magenta afterglow —
the "burn out" — frequently peaks 25–45 minutes after sunset and can persist
through nautical twilight. Recent timelapses have been clipping right when the
best color appears.

## Goal

Extend the Pi's post-sunset capture window to 60 minutes so daily timelapses
include the full afterglow.

## Decisions

- **Active config file on the Pi is `config.yaml`, not `config-pi.yaml`.**
  `ConfigManager` defaults to `config.yaml`, and the `sunset-timelapse.service`
  unit runs `main.py schedule --validate` with no `--config` override. Both
  files exist on the Pi (both gitignored, separate per machine), but only
  `config.yaml` is read. Runtime logs confirmed the Pi's effective window was
  30/30, matching `config.yaml`'s values — `config-pi.yaml` was 60/60 but
  unused. The CLAUDE.md narrative refers to `config-pi.yaml` as "Pi config",
  but in practice it is dormant. Cleaning that up is out of scope here.
- Change `config.yaml: capture.time_after_sunset_minutes` from `30` to `60`
  **directly on the Pi via SSH**, since `config.yaml` is gitignored and
  doesn't propagate through `git pull` / `update_pi.sh`.
- Leave `time_before_sunset_minutes` at `30`. The pre-sunset side is rarely the
  problem and the existing 30-min lead is sufficient.
- No code changes. `sunset_calculator.get_capture_window()` reads both keys
  with sensible defaults; the scheduler and capture loop use that result
  directly with no separate assumption about window length (verified by
  grepping both keys across the codebase — both are read only in
  `sunset_calculator.py`).
- Mac-local `config.yaml` is already at 60/60 and needs no change.

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
