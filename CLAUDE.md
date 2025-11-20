# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an automated sunset timelapse system that captures daily sunset images using a Reolink camera, processes them into videos, and uploads to YouTube. The system is designed for two deployment scenarios:
- **Raspberry Pi**: Daily automated capture and scheduling
- **MacBook/Desktop**: Historical footage processing and video creation

## Key Commands

### Development and Testing
```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# System validation and testing
python main.py test --camera --youtube --sunset  # Test all components
python main.py test --camera      # Test camera connection only
python main.py test --youtube     # Test YouTube authentication only
python main.py test --sunset      # Show sunset calculation schedule

# Configuration management
python main.py config --validate --show
```

### Daily Operations
```bash
# Run scheduler with validation (primary command for Raspberry Pi)
python main.py schedule --validate

# Immediate capture for testing (default: 5 minutes)
python main.py schedule --immediate

# Immediate capture with custom duration (e.g., 30 minutes)
python main.py schedule --immediate --duration 30

# Check system status
python main.py status
```

### Historical Processing
```bash
# List available dates for processing
python main.py historical --start 2024-01-01 --end 2024-01-31 --list

# Process historical footage with YouTube upload
python main.py historical --start 2024-01-01 --end 2024-01-07 --upload

# Process without uploading
python main.py historical --start 2024-01-01 --end 2024-01-07
```

### Meteor Detection
```bash
# Search camera recordings for meteors in date range
# Automatically uses sunset/sunrise times for nighttime-only scanning
python main.py meteor --start 2024-01-01 --end 2024-01-07

# Search with specific time window (override automatic sunset/sunrise)
python main.py meteor --start 2024-01-01 --end 2024-01-07 --start-time 20:00 --end-time 05:00

# Analyze a specific local video file
python main.py meteor --analyze-video /path/to/video.mp4 --date 2024-01-01

# Show meteor detection statistics
python main.py meteor --stats
```

## Architecture

### Core Components

1. **main.py** - CLI entry point with subcommands for different operations
2. **config_manager.py** - Configuration loading from YAML and environment variables with secure secrets handling
3. **sunset_calculator.py** - Astronomical calculations for sunset times using the Astral library
4. **camera_interface.py** - ONVIF camera control for image capture
5. **video_processor.py** - FFmpeg-based video creation from image sequences
6. **youtube_uploader.py** - Google API integration for YouTube uploads
7. **sunset_scheduler.py** - Main orchestrator that coordinates daily operations
8. **historical_retrieval.py** - Bulk processing of historical camera footage
9. **meteor_detector.py** - OpenCV-based meteor detection and clip extraction

### Configuration System

The system uses a two-tier configuration approach:
- **config.yaml**: Main configuration for location, camera settings, capture parameters, video settings
- **.env**: Sensitive data like camera passwords and Google credentials paths

### Key Dependencies

- **astral**: Astronomical calculations for sunset timing
- **onvif-zeep**: Camera control via ONVIF protocol
- **ffmpeg-python**: Video processing wrapper
- **google-api-python-client**: YouTube API integration
- **schedule**: Task scheduling for Raspberry Pi operations
- **colorlog**: Enhanced logging with color output
- **opencv-python-headless**: Computer vision for meteor detection and SBS analysis
- **numpy/scipy**: Mathematical operations for image analysis

## Development Notes

### Camera Integration
The system connects to Reolink cameras via ONVIF protocol on port 8000. Camera credentials are stored securely in environment variables.

### Video Processing
Videos are created using FFmpeg with H.264 encoding at 12 FPS. The FPS is calculated based on 5-second capture intervals to create real-time playback.

### Storage Management
Images are organized by date and automatically cleaned up after 7 days. Videos are deleted locally after 1 day (post-upload).

### Deployment Differences
- **Raspberry Pi**: Uses systemd service for continuous operation, storage at `/home/pi/sunset_timelapse`
- **MacBook**: Manual operation for historical processing, storage at `/Users/username/sunset_timelapse`

### Security Practices
- All secrets in environment variables (never in code)
- Camera passwords and Google credentials are never logged
- Configuration validation prevents missing security settings

## Meteor Detection System

### Overview
The meteor detection system analyzes historical camera recordings to identify and extract meteor events using OpenCV-based computer vision algorithms.

### Detection Algorithm
- **Multi-frame tracking**: Tracks bright objects across consecutive frames for sustained meteors
- **Single-frame detection**: Identifies fast-moving streaks in individual frames
- **Validation filters**: Multiple criteria to distinguish meteors from false positives

### False Positive Filtering

**Max Velocity Filter** (config: `meteor.max_velocity`, default: 25.0 px/frame)
- Rejects fast-moving aircraft and satellites
- Based on confirmed meteor velocity analysis: 6.39 - 20.33 px/frame
- Threshold provides safety margin while filtering extreme velocities (26+ px/frame)

**Linearity Score** (config: `meteor.linearity_threshold`, default: 0.95)
- Validates meteor paths follow straight trajectories
- Rejects erratic movements (insects, birds, debris)

**Sky Region Filtering** (config: `meteor.sky_region_top/bottom`)
- Focuses detection on upper portion of frame where meteors occur
- Reduces ground-based false positives

### Platform-Specific Behavior

**Desktop (Mac/Windows/Linux x86)**
- Uses parallel processing with ProcessPoolExecutor (3 workers)
- Significantly faster for bulk historical scans
- Automatically detected via platform checks

**Raspberry Pi (ARM/aarch64)**
- Uses sequential processing to prevent resource exhaustion
- Detected via multiple methods:
  - Platform machine check (`'arm'` or `'aarch'` in platform.machine())
  - Raspberry Pi device tree file check (`/proc/device-tree/model` exists)
- Prevents worker process spawning that could overwhelm Pi resources

### Known Issues

**Performance Bottleneck in Parallel Processing**
- Each worker process re-initializes heavy objects for every video
- Causes ~36x slowdown compared to expected parallel performance
- Issue: ProcessPoolExecutor workers don't share state
- Impact: 173 videos takes ~12 hours instead of ~20 minutes
- Workaround: Use targeted time windows instead of full-night scans
- Fix needed: Refactor worker architecture to reuse initialized objects

### Configuration Example
```yaml
meteor:
  enabled: true
  linearity_threshold: 0.95
  max_velocity: 25.0              # Max 25 px/frame to reject aircraft
  sky_region_top: 0.0             # Top 0% of frame
  sky_region_bottom: 0.5          # Bottom 50% of frame
```

### Output Structure
Detected meteors are saved to `data/meteors/` with:
- **Video clip** (`.mp4`): Extracted frames showing the meteor
- **Metadata** (`.json`): Detection parameters, timestamps, velocity, linearity score

Example metadata:
```json
{
  "timestamp": "2024-11-15T18:53:24.123456",
  "duration_frames": 6,
  "duration_seconds": 0.45,
  "max_brightness": 255.0,
  "linearity_score": 0.958,
  "velocity": 9.05,
  "detection_type": "multi_frame",
  "source_recording": "RecM04_20241115_185320_185521.mp4"
}
```

## Testing Strategy

Always run system validation before deployment:
```bash
python main.py test --camera --youtube --sunset
```

This validates camera connectivity, YouTube authentication, and sunset calculation accuracy.