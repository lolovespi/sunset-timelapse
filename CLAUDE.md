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

## Testing Strategy

Always run system validation before deployment:
```bash
python main.py test --camera --youtube --sunset
```

This validates camera connectivity, YouTube authentication, and sunset calculation accuracy.