# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated sunset timelapse system. A Reolink RLC810-WA camera captures images around sunset; FFmpeg assembles them into a video; the video is then distributed to YouTube, Facebook (Page + Reels), Instagram, and Google Drive. The system also scans overnight Reolink recordings for meteors.

Two deployment targets share the same codebase:
- **Raspberry Pi** — daily capture + scheduling, runs as a systemd service (`config-pi.yaml`)
- **MacBook/desktop** — historical processing, meteor scans, manual operations (`config.yaml`)

## Key Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Validate everything before any deploy
python main.py test --camera --youtube --sunset

# Daily operations (Pi)
python main.py schedule --validate          # primary entry, runs forever
python main.py schedule --immediate --duration 30  # immediate capture for testing

# Historical (desktop)
python main.py historical --start 2024-01-01 --end 2024-01-07 --list
python main.py historical --start 2024-01-01 --end 2024-01-07 --upload

# Meteor scan over date range (auto-uses sunset/sunrise window)
python main.py meteor --start 2024-01-01 --end 2024-01-07
python main.py meteor --analyze-video /path/to/file.mp4 --date 2024-01-01
python main.py meteor --stats

# One-off operations
python main.py capture --duration 7200 --process --upload   # capture N seconds, then assemble + upload
python main.py upload --date 2024-09-15                      # re-upload an existing video
python main.py stream --duration 30 --platforms facebook,youtube --test
python main.py sbs --date 2024-09-15 --report
python main.py status
python main.py config --validate --show
python main.py cleanup --dry-run
```

There is **no pytest suite wired up** — `pytest` is in requirements but `tests/` does not exist. `main.py test` is the de facto validation harness; add new component checks there rather than under a separate test runner.

## Architecture

### Daily workflow (the most important code path to understand)

`SunsetScheduler.complete_daily_workflow()` in `sunset_scheduler.py` is the orchestrator. Reading top-to-bottom is the fastest way to understand how pieces fit together:

1. **Capture** — `camera_interface.py` pulls JPEGs at `capture.interval_seconds` between `time_before_sunset_minutes` and `time_after_sunset_minutes`. Sunset times come from `sunset_calculator.py` (Astral).
2. **Process** — `video_processor.py` runs FFmpeg (`libx264`, `video.fps` from config, currently 60).
3. **SBS report** — `sbs_reporter.py` aggregates per-frame metrics that `sunset_brilliance_score.py` computed *during capture*. SBS is real-time on the Pi, not a post-process.
4. **Weather + visual** — `tempest_api.py` pulls a weather block; `visual_analyzer.py` samples frames for sunset-type/intensity tags. Both feed downstream metadata; both are best-effort (caught and logged).
5. **Drive upload** — `drive_uploader.py` pushes the video + JSON metadata.
6. **One shared AI caption** — `FacebookUploader.generate_caption()` calls Anthropic (Haiku 4.5) with several sampled frames (vision-based) to write a 2-3 sentence caption. The *same* caption is then used for Facebook, Instagram, Facebook Reel, and the YouTube description. There is no separate AI-generated YouTube title — titles are deterministic `Sunset MM/DD/YY` (see commit `3a7213d`).
7. **Social posts** — `facebook_uploader.py` posts the standard video to FB Page, cross-posts to Instagram, and posts a portrait-rendered version as a Facebook Reel.
8. **YouTube** — `youtube_uploader.py` uploads with the shared caption as the description.
9. **Maintenance** — token refresh, file cleanup, SBS retention.

If a step needs to change, edit `complete_daily_workflow` first — it's the contract every other module is plugged into.

### Module map

| File | Responsibility |
|---|---|
| `main.py` | argparse CLI; one `cmd_*` function per subcommand |
| `sunset_scheduler.py` | Orchestrator. `complete_daily_workflow`, `run_scheduler` (the systemd entry), `run_overnight_meteor_scan` |
| `config_manager.py` | Loads `config.yaml` (or `config-pi.yaml` on the Pi) + `.env`. Centralizes secrets and paths |
| `sunset_calculator.py` | Astral-based sunset times + capture windows |
| `camera_interface.py` | ONVIF (port 8000) + Reolink API; capture, recording search, download |
| `video_processor.py` | FFmpeg wrapper for assembling JPEGs into MP4 |
| `youtube_uploader.py` | OAuth + upload; includes proactive token refresh and health alerts |
| `facebook_uploader.py` | FB Page + Instagram + FB Reels posting. Also owns AI-caption generation (Anthropic). `_render_portrait_for_reel` handles 9:16 reformat |
| `drive_uploader.py` | Google Drive backup for videos, meteor clips, and JSON metadata |
| `tempest_api.py` / `tempest_monitor.py` | Tempest weather station UDP + REST; storm-trigger detection |
| `visual_analyzer.py` | Per-video sunset classification (type, intensity, dominant colors) |
| `sunset_brilliance_score.py` / `sbs_reporter.py` | Real-time per-frame scoring during capture + daily aggregation |
| `historical_retrieval.py` | Pulls historical recordings from the camera and assembles per-day videos |
| `meteor_detector.py` | OpenCV multi-frame + single-frame meteor detection, dedup, clip extraction |
| `email_notifier.py` | SMTP notifications for uploads, failures, token expiry, meteor finds |
| `live_streamer.py` / `live_youtube.py` / `live_facebook.py` / `rtmp_streamer.py` | RTMP-based live broadcast support |

### Configuration system

- `config.yaml` — desktop/Mac config, the canonical reference. **Has duplicate `sbs:` blocks intentionally** (legacy v1 + v2 keys merged); both are read.
- `config-pi.yaml` — Pi-specific config (different `storage.base_path`, sometimes different camera IP). The Pi's `update_pi.sh` swaps this in. Don't assume `config.yaml` is what's running on the Pi.
- `example-config.yaml` — sanitized template for new deploys.
- `.env`, `.env.pi`, `.env.mac` — secrets. **Never read or display these.** All sensitive values (camera password, `ANTHROPIC_API_KEY`, Google credentials path, SMTP password, Tempest token, Facebook tokens in `facebook_config.json`) live here.

### AI captions

`facebook_uploader.py` uses `claude-haiku-4-5-20251001` with vision (4–5 sampled frames from the rendered video) to produce the social caption. If vision fails, it falls back to text-only with the same prompt; if Anthropic is unreachable, `_build_fallback_caption` synthesizes one from metadata.

**Caption voice (enforced by the prompt and by user preference):** direct, dry, observational. No hype, no slang ("vibes", "epic", "literal magic"), no repeating the word "muted". When editing the prompt in `_build_caption_prompt`, preserve this tone — it has been tuned through multiple iterations. Vision-based captions are known to be meaningfully better than text-only; do not regress to text-only as a default.

## Known issues and gotchas

**Meteor parallel-processing bottleneck.** `meteor_detector.py` uses `ProcessPoolExecutor` (3 workers) on desktop and sequential on Pi (detected via `platform.machine()` and `/proc/device-tree/model`). Workers re-initialize heavy CV objects per video → roughly 36× slowdown vs. an ideal pool. Full-night scans take ~12h instead of ~20min. Workaround: use targeted time windows. Real fix would refactor workers to reuse state.

**SBS config has two blocks.** `config.yaml` defines `sbs:` twice; both contribute (later block overrides earlier on key collision). When changing SBS keys, search for both blocks.

**FPS source-of-truth is `config.yaml`, not the CLAUDE/README narrative.** Current value is `video.fps: 60` (5s interval → 5× real-time playback). Older docs say 12 — trust the config.

**Token refresh is proactive, not lazy.** `youtube_uploader.refresh_token_proactively()` is called in daily maintenance; expiration emails fire only when refresh *fails*. Don't add lazy-refresh paths that would double-alert.

**Pi vs desktop branching is by platform detection, not by env var.** `meteor_detector.py` checks `platform.machine()` and the device-tree file. If you add platform-conditional logic elsewhere, follow that same pattern rather than introducing a new flag.

**Don't auto-fix `historical_process.log` or other large checked-in logs.** They're intentional artifacts of past runs.

## Meteor detection — quick reference

Detection combines:
- Multi-frame tracking for sustained meteors (`min_frames`–`max_frames` window)
- Single-frame streak detection for very fast events

False-positive filters (all in `meteor:` config):
- `max_velocity: 25.0` px/frame — rejects aircraft/satellites (confirmed meteors: 6.4–20.3 px/frame)
- `min_linearity: 0.80` — rejects erratic paths (insects, birds)
- `sky_region_top` / `sky_region_bottom` — restricts to upper half of the frame
- `min_brightness_threshold`, `min_area`/`max_area`

Output: `data/meteors/<clip>.mp4` + matching `.json` with timestamp, velocity, linearity, brightness, source recording. Clips with CST/CDT timestamps in filenames. Auto-uploaded to a `meteors` subfolder in Drive; email notifications fire on each detection.

## Deployment

- **Pi**: `update_pi.sh` (stash → pull → restore → restart `sunset-timelapse.service` → status check). Service runs `python main.py schedule --validate` under `/home/pi/sunset-timelapse`.
- **Desktop**: manual `git pull`; no service.
- `sync_credentials.sh` syncs OAuth tokens between machines when needed.
