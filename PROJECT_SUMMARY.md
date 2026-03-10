# Sunset Timelapse — Project Summary

## Platform Overview

An automated sunset timelapse system that captures daily sunset images via a **Reolink RLC810-WA camera**, processes them into timelapse videos, and uploads to **YouTube**. Designed for two deployment targets:

- **Raspberry Pi** — Continuous 24/7 scheduled capture via systemd
- **Mac/Desktop** — Manual historical processing and batch video creation

---

## Core Modules

| Module | Purpose |
|--------|---------|
| **main.py** | CLI entry point with subcommands: `schedule`, `historical`, `test`, `meteor`, `sbs`, `capture`, `config`, `upload`, `status`, `cleanup` |
| **config_manager.py** | YAML + `.env` config loading, secrets management, dot-notation access, validation |
| **sunset_calculator.py** | Astronomical sunset/sunrise/twilight calculations using the `astral` library with timezone support |
| **camera_interface.py** | RTSP stream capture from Reolink camera, frame extraction at 5-second intervals |
| **video_processor.py** | FFmpeg-based timelapse creation (H.264, 12 FPS, 1920x1080) from image sequences |
| **youtube_uploader.py** | OAuth 2.0 YouTube uploads with resumable transfers, auto-generated metadata/tags |
| **sunset_scheduler.py** | Central orchestrator — coordinates capture, processing, upload, maintenance, and meteor scanning |
| **historical_retrieval.py** | Bulk retrieval of camera SD card recordings by date range with fallback search strategies |
| **meteor_detector.py** | OpenCV-based meteor detection with multi-frame tracking, velocity/linearity filters, and clip extraction |
| **sbs_reporter.py** + **sunset_brilliance_score.py** | Sunset Brilliance Score (0-100) — colorfulness, saturation, gradient analysis per frame |

## Supporting Modules

| Module | Purpose |
|--------|---------|
| **email_notifier.py** | SMTP notifications for capture success/failure and token expiry alerts |
| **drive_uploader.py** | Google Drive backup of videos and meteor clips |
| **meteor_showers.py** | Known meteor shower calendar for context-aware detection |

---

## Key Workflows

1. **Daily Capture & Upload** — Scheduler calculates sunset window (configurable offset), captures RTSP frames, creates timelapse, scores sunset quality (SBS), uploads to YouTube, sends email notification
2. **Historical Recovery** — Downloads camera recordings for a date range, extracts frames, creates timelapses, optionally uploads
3. **Meteor Detection** — Analyzes overnight recordings (astronomical dusk to dawn) for bright moving objects using multi-frame tracking and single-frame streak detection with false-positive filtering
4. **System Validation** — Tests camera connectivity, YouTube auth, email, sunset calculations independently

---

## Technology Stack

- **Python 3** with `astral`, `opencv-python-headless`, `numpy/scipy`, `ffmpeg-python`
- **Camera**: RTSP streaming, Reolink API, ONVIF protocol
- **Cloud**: Google YouTube Data API v3, Google Drive API, OAuth 2.0
- **Scheduling**: `schedule` library (Pi), systemd service
- **Notifications**: SMTP email (Gmail compatible)

---

## Security Practices

- All secrets in environment variables (`.env`), never in config or logs
- RTSP URLs sanitized before logging (credentials redacted)
- Path traversal protection in file handling
- OAuth tokens stored with `0o600` permissions
- Configuration validation on startup

---

## Known Limitations

- **Meteor parallel processing**: ProcessPoolExecutor workers re-initialize per video causing ~36x slowdown; sequential processing recommended for large scans
- **Camera session limits**: Reolink supports 2-3 concurrent connections max
- **YouTube token expiry**: 30-day refresh tokens require proactive daily maintenance

---

## Storage & Retention

```
{base_path}/
├── images/{YYYY-MM-DD}/     # 7-day retention
├── videos/                   # 1-day retention (post-upload)
├── meteors/                  # 30-day retention (clips + JSON metadata)
├── logs/                     # Rotating log files
└── temp/                     # FFmpeg working files
```

---

## CLI Quick Reference

```bash
# Daily operations
python main.py schedule --validate          # Run scheduler (Raspberry Pi)
python main.py schedule --immediate         # Immediate 5-min capture
python main.py status                       # System health check

# Testing
python main.py test --camera --youtube --sunset --email --sbs

# Historical processing
python main.py historical --start 2024-01-01 --end 2024-01-07 --upload

# Meteor detection
python main.py meteor --start 2024-01-01 --end 2024-01-07

# Configuration
python main.py config --show --validate
```
