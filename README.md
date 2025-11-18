# Sunset Timelapse System

An automated system for capturing daily sunset timelapses using a Reolink camera, processing them into videos, and uploading to YouTube. Designed to run on Raspberry Pi for daily captures and MacBook for intensive historical processing.

## 🌅 Features

- **Automated Daily Capture**: Calculates sunset times based on location and captures images 1 hour before/after sunset
- **Smart Scheduling**: Uses astronomical calculations to determine optimal capture windows
- **🌟 Sunset Brilliance Score (SBS)**: AI-powered analysis to automatically enhance exceptional sunsets with special titles
- **☄️ Meteor Detection**: Automated overnight scanning for meteor events with intelligent deduplication
- **Video Processing**: Creates high-quality timelapse videos from image sequences using FFmpeg
- **YouTube Integration**: Automatically uploads videos with proper metadata and descriptions
- **📧 Email Notifications**: Real-time alerts for uploads, errors, meteor detections, and system status
- **☁️ Google Drive Backup**: Automatic cloud storage for videos, meteor clips, and metadata
- **Historical Retrieval**: Downloads and processes historical footage from camera storage
- **🔐 Enhanced Token Management**: Automatic OAuth token refresh with health monitoring
- **Multi-Platform**: Raspberry Pi for daily operations, MacBook for processing power
- **Secure Configuration**: Environment-based secrets management
- **ONVIF Support**: Camera control via standard ONVIF protocol
- **Comprehensive Logging**: Detailed logs with rotation and health monitoring
- **🚀 Easy Deployment**: Automated update scripts for Pi deployment

## 📋 Requirements

### Hardware
- **Reolink RLC810-WA Camera** (or compatible ONVIF camera)
- **Raspberry Pi 4** (for daily scheduling)
- **MacBook/Desktop** (for historical processing)
- **Stable Network Connection**

### Software
- **Python 3.9+**
- **FFmpeg** (for video processing)
- **Google Cloud Project** (for YouTube API access)
- **Email Account** (for notifications - Gmail recommended)
- **OpenCV** (for advanced sunset analysis)
- **NumPy & SciPy** (for mathematical computations)

### Supported Platforms
- Linux (Raspberry Pi OS, Ubuntu)
- macOS
- Windows (with some modifications)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/sunset-timelapse.git
cd sunset-timelapse
```

### 2. Install Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Install FFmpeg
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# macOS (with Homebrew)
brew install ffmpeg

# Verify installation
ffmpeg -version
```

### 4. Setup Configuration
```bash
# Copy example configuration
cp config.yaml.example config.yaml

# Edit configuration with your settings
nano config.yaml
```

### 5. Setup Environment Variables
```bash
# Create .env file
cp .env.example .env

# Edit with your secrets
nano .env
```

### 6. Test System
```bash
# Validate all components
python main.py test --camera --youtube --sunset

# Run immediate test capture
python main.py schedule --immediate
```

## ⚙️ Configuration

### Main Configuration (config.yaml)

The main configuration file controls all system behavior:

```yaml
# Location settings
location:
  city: "Pelham"
  state: "AL"
  latitude: 33.2856
  longitude: -86.8097
  timezone: "America/Chicago"

# Camera settings
camera:
  ip: "192.168.6.44"
  username: "apiuser"
  onvif_port: 8000

# Capture settings
capture:
  interval_seconds: 5
  time_before_sunset_minutes: 60
  time_after_sunset_minutes: 60

# Video settings
video:
  fps: 12
  resolution:
    width: 1920
    height: 1080

# Storage settings
storage:
  base_path: "/home/pi/sunset_timelapse"  # Change for Mac
  keep_images_days: 7
  keep_videos_locally_days: 1

# Sunset Brilliance Score (SBS) settings
sbs:
  enabled: true
  min_score_for_enhancement: 65
  retention_days: 30
  analysis_enabled: true

# Meteor detection settings
meteor:
  enabled: true                      # Enable automatic overnight meteor scanning
  sunset_offset_minutes: 30          # Start scanning after sunset + offset
  sunrise_offset_minutes: 30         # Stop scanning before sunrise - offset
  min_brightness_threshold: 200      # Meteor brightness threshold (0-255)
  min_frames: 3                      # Minimum frames to confirm meteor
  retention_days: 30                 # Keep meteor clips for 30 days

# Email notification settings
email:
  enabled: true
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  send_upload_notifications: true
  send_error_notifications: true
  send_daily_reports: false

# Google Drive backup (optional)
drive:
  enabled: false
  folder_name: "Sunset Timelapses"
  upload_videos: true
  upload_metadata: true
```

### Environment Variables (.env)

Store sensitive information in environment variables:

```bash
# Camera credentials
CAMERA_PASSWORD=your_camera_password

# Google Cloud credentials path
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Email notification settings
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
EMAIL_TO=recipient@example.com

# Google Drive integration (optional)
GOOGLE_DRIVE_FOLDER_ID=your_drive_folder_id
```

### Google Cloud Setup

1. **Create Google Cloud Project**
   ```bash
   # Install Google Cloud SDK
   curl https://sdk.cloud.google.com | bash
   gcloud init
   ```

2. **Enable YouTube API**
   ```bash
   gcloud services enable youtube.googleapis.com
   ```

3. **Create Service Account**
   ```bash
   gcloud iam service-accounts create sunset-timelapse \
     --display-name="Sunset Timelapse"
   
   gcloud iam service-accounts keys create credentials.json \
     --iam-account=sunset-timelapse@your-project.iam.gserviceaccount.com
   ```

4. **Setup OAuth Consent Screen** (in Google Cloud Console)
   - Add your YouTube channel as a test user
   - Configure OAuth consent screen for external users

## 🖥️ Usage

### Daily Operations (Raspberry Pi)

```bash
# Start daily scheduler with validation
python main.py schedule --validate

# Run as systemd service (recommended)
sudo systemctl enable sunset-timelapse
sudo systemctl start sunset-timelapse
```

### Historical Processing (MacBook)

```bash
# List available dates
python main.py historical --start 2024-01-01 --end 2024-01-31 --list

# Process specific date range
python main.py historical --start 2024-01-01 --end 2024-01-07 --upload

# Process without YouTube upload
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

Note: Meteor scanning is automatically scheduled to run every morning at 7 AM on the Raspberry Pi when `meteor.enabled: true` is set in config.yaml.

### System Management

```bash
# Check system status
python main.py status

# Validate configuration
python main.py config --validate --show

# Test individual components
python main.py test --camera      # Test camera connection
python main.py test --youtube     # Test YouTube authentication
python main.py test --sunset      # Show sunset schedule
python main.py test --email       # Test email notifications
python main.py test --drive       # Test Google Drive integration
python main.py test --sbs         # Test SBS analysis system
```

## 🌟 Sunset Brilliance Score (SBS) System

The SBS system uses advanced computer vision and machine learning to automatically analyze sunset quality and enhance exceptional captures with special titles.

### How SBS Works

1. **Image Analysis**: Analyzes sunset images for color intensity, cloud formations, and visual appeal
2. **Score Calculation**: Assigns a brilliance score from 0-100 based on multiple factors
3. **Automatic Enhancement**: Videos scoring above threshold get enhanced titles like "✨ SPECTACULAR"
4. **Historical Tracking**: Maintains database of sunset quality over time

### SBS Configuration

```yaml
sbs:
  enabled: true
  min_score_for_enhancement: 65  # Threshold for "spectacular" designation
  retention_days: 30             # How long to keep analysis data
  analysis_enabled: true         # Enable detailed analytics
```

### SBS Commands

```bash
# Test SBS analysis on recent sunset
python main.py test --sbs

# Analyze historical sunsets for patterns
python analyze_historical_sbs.py --start 2024-01-01 --end 2024-12-31

# Generate SBS analytics report
python sbs_reporter.py --report --days 30
```

## 📧 Email Notification System

Automated email notifications keep you informed about system status, successful uploads, and any issues.

### Notification Types

- **Upload Success**: Confirmation with video details and SBS score
- **Upload Failures**: Immediate alerts with error details
- **Token Expiration**: Warnings before YouTube token expires
- **System Errors**: Critical system issues requiring attention
- **Daily Reports**: Optional summary of daily operations

### Email Setup

1. **Enable App Password** (for Gmail):
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate password for "Mail"

2. **Configure Environment Variables**:
   ```bash
   EMAIL_SMTP_SERVER=smtp.gmail.com
   EMAIL_SMTP_PORT=587
   EMAIL_USERNAME=your-email@gmail.com
   EMAIL_PASSWORD=your-16-digit-app-password
   EMAIL_TO=recipient@example.com
   ```

3. **Test Email System**:
   ```bash
   python main.py test --email
   ```

## ☁️ Google Drive Integration

Automatic backup of videos and metadata to Google Drive for redundancy and easy access.

### Drive Setup

1. **Create Drive Folder**: Create a dedicated folder in Google Drive
2. **Get Folder ID**: Copy folder ID from URL
3. **Configure**: Set `GOOGLE_DRIVE_FOLDER_ID` in `.env`
4. **Enable**: Set `drive.enabled: true` in `config.yaml`

### Drive Commands

```bash
# Test Drive integration
python main.py test --drive

# Manual backup of specific video
python drive_uploader.py --video /path/to/video.mp4

# Backup all videos from date range
python drive_uploader.py --start 2024-01-01 --end 2024-01-07
```

## 🔐 Enhanced Token Management

Improved YouTube OAuth token handling prevents authentication failures and reduces false alerts.

### Token Features

- **Proactive Refresh**: Automatically refreshes tokens before expiration
- **Health Monitoring**: Tracks token status and remaining time
- **Smart Alerting**: Only sends expiration emails when refresh fails
- **Credential Sync**: Easy synchronization between development and Pi

### Token Commands

```bash
# Test token health and refresh
python main.py test --youtube-token

# Manually refresh token
python main.py refresh-token

# Sync credentials between machines
./sync_credentials.sh
```

## 🚀 Deployment and Updates

### Pi Update Script

Use the automated update script to safely deploy changes to your Raspberry Pi:

```bash
# On Raspberry Pi
./update_pi.sh
```

This script:
- Stashes local changes
- Pulls latest updates
- Restores local modifications
- Restarts the service
- Verifies system status

### Development Workflow

```bash
# On development machine
git add .
git commit -m "Your changes"
git push origin main

# On Raspberry Pi
./update_pi.sh
```

## 🔧 Installation Guide

### Raspberry Pi Setup

1. **Install Raspberry Pi OS**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Python and dependencies
   sudo apt install python3-pip python3-venv git ffmpeg
   ```

2. **Clone and Setup Project**
   ```bash
   cd /home/pi
   git clone https://github.com/yourusername/sunset-timelapse.git
   cd sunset-timelapse
   
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure for Raspberry Pi**
   ```bash
   # Edit config.yaml
   nano config.yaml
   # Set storage.base_path to "/home/pi/sunset_timelapse"
   
   # Setup environment
   nano .env
   # Add your camera password and credentials path
   ```

4. **Setup Systemd Service**
   ```bash
   # Create service file
   sudo nano /etc/systemd/system/sunset-timelapse.service
   ```
   
   ```ini
   [Unit]
   Description=Sunset Timelapse System
   After=network.target
   
   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/sunset-timelapse
   Environment=PATH=/home/pi/sunset-timelapse/venv/bin
   ExecStart=/home/pi/sunset-timelapse/venv/bin/python main.py schedule --validate
   Restart=always
   RestartSec=60
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   ```bash
   # Enable and start service
   sudo systemctl daemon-reload
   sudo systemctl enable sunset-timelapse
   sudo systemctl start sunset-timelapse
   
   # Check status
   sudo systemctl status sunset-timelapse
   ```

### MacBook Setup

1. **Install Dependencies**
   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install FFmpeg and Python
   brew install ffmpeg python3
   ```

2. **Setup Project**
   ```bash
   cd ~/Documents
   git clone https://github.com/yourusername/sunset-timelapse.git
   cd sunset-timelapse
   
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure for MacBook**
   ```bash
   # Edit config.yaml
   nano config.yaml
   # Set storage.base_path to "/Users/username/sunset_timelapse"
   ```

## 🔒 Security Best Practices

### Secrets Management
- **Never commit secrets** to version control
- Use **environment variables** for all sensitive data
- Store **Google credentials** outside the project directory
- Use **strong camera passwords** and change default credentials
- **Secure email credentials** with app passwords (never use main password)
- **Rotate tokens** regularly and monitor expiration dates

### Network Security
- **Change default camera ports** if possible
- Use **VPN** or **private networks** when accessing cameras remotely
- **Regularly update** camera firmware
- **Monitor network traffic** for unusual activity

### File System Security
```bash
# Set appropriate permissions
chmod 600 .env
chmod 600 /path/to/google-credentials.json

# Create dedicated user for the service (optional)
sudo useradd -r -s /bin/false sunset-timelapse
```

## 📁 Project Structure

```
sunset-timelapse/
├── config.yaml                    # Main configuration
├── .env                           # Environment variables (create from .env.example)
├── requirements.txt               # Python dependencies
├── main.py                        # CLI entry point
├── config_manager.py              # Configuration and secrets management
├── sunset_calculator.py           # Astronomical calculations
├── camera_interface.py            # Camera control via ONVIF
├── video_processor.py             # Video creation with FFmpeg
├── youtube_uploader.py            # YouTube API integration
├── sunset_scheduler.py            # Main orchestrator (includes meteor scheduling)
├── historical_retrieval.py        # Historical footage processing
├── meteor_detector.py             # Meteor detection and analysis engine
├── email_notifier.py              # Email notification system
├── drive_uploader.py              # Google Drive backup integration (videos + meteors)
├── sunset_brilliance_score.py     # SBS analysis engine
├── sbs_reporter.py                # SBS reporting and analytics
├── update_pi.sh                   # Pi deployment update script
├── sync_credentials.sh            # Credential synchronization script
├── logs/                          # Application logs
├── images/                        # Captured images (organized by date)
├── videos/                        # Created videos
├── data/                          # SBS analysis data and reports
│   └── meteors/                   # Detected meteor clips and metadata
└── temp/                          # Temporary files
```

## 🎥 Advanced Features

### ☄️ Meteor Detection System

The automated meteor detection system uses computer vision to scan overnight camera footage for meteor events.

#### How Meteor Detection Works

1. **Automated Scheduling**: Scans run every morning at 7 AM analyzing the previous night's footage
2. **Intelligent Time Windows**: Uses actual sunset/sunrise times (with configurable offsets) for nighttime-only scanning
3. **Multi-Method Detection**:
   - Multi-frame tracking for slower meteors
   - Single-frame streak detection for fast meteors
4. **Smart Deduplication**: Automatically eliminates duplicate detections of the same meteor event
5. **Timezone-Aware Timestamps**: All meteor clips have accurate CST/CDT timestamps in filenames
6. **Google Drive Backup**: Meteor clips automatically uploaded to separate "meteors" folder
7. **Email Notifications**: Get alerts when meteors are detected with clip details and Drive links

#### Meteor Configuration

```yaml
meteor:
  enabled: true                      # Enable automatic overnight meteor scanning
  sunset_offset_minutes: 30          # Start 30 min after sunset (ensures darkness)
  sunrise_offset_minutes: 30         # Stop 30 min before sunrise (avoids daylight)

  # Detection thresholds
  min_brightness_threshold: 200      # Minimum brightness (0-255)
  min_area: 10                       # Minimum pixel area
  max_area: 5000                     # Maximum pixel area (filter large objects)

  # Track validation
  min_frames: 3                      # Minimum frames to confirm meteor
  min_linearity: 0.85                # Path linearity score (0-1)
  min_velocity: 5.0                  # Minimum velocity in pixels/frame

  # Storage
  retention_days: 30                 # Keep meteor clips for 30 days
```

#### Meteor Detection Commands

```bash
# Manual scan for specific date range
python main.py meteor --start 2024-11-16 --end 2024-11-17

# Analyze local video file
python main.py meteor --analyze-video /path/to/video.mp4 --date 2024-11-16

# View detection statistics
python main.py meteor --stats
```

### Historical Analysis

```bash
# Analyze sunset patterns over time
python analyze_historical_sbs.py --start 2024-01-01 --end 2024-12-31

# Generate monthly SBS reports
python sbs_reporter.py --monthly-report --year 2024

# Export SBS data for external analysis
python sbs_reporter.py --export-csv --days 365
```

### Custom Analysis Tools

```bash
# Analyze sky regions and cloud patterns
python analyze_sky_regions.py --date 2024-09-15

# Debug SBS scoring algorithm
python debug_sbs_analysis.py --image /path/to/sunset.jpg

# Calibrate SBS thresholds
python calibrate_sbs_v2.py --samples 100
```

### Batch Operations

```bash
# Bulk process historical footage
python main.py historical --start 2024-01-01 --end 2024-01-31 --upload

# Batch upload to Google Drive
python drive_uploader.py --bulk-upload --start 2024-01-01

# Regenerate SBS scores for date range
python sbs_reporter.py --recalculate --start 2024-01-01 --end 2024-01-31
```

## 🐛 Troubleshooting

### Common Issues

**Camera Connection Failed**
```bash
# Check network connectivity
ping 192.168.6.44

# Test ONVIF port
telnet 192.168.6.44 8000

# Verify credentials
python main.py test --camera
```

**FFmpeg Not Found**
```bash
# Install FFmpeg
sudo apt install ffmpeg  # Linux
brew install ffmpeg      # macOS

# Verify installation
ffmpeg -version
```

**YouTube Upload Failed**
```bash
# Check authentication
python main.py test --youtube

# Test token refresh
python main.py test --youtube-token

# Verify credentials file exists
ls -la $GOOGLE_APPLICATION_CREDENTIALS

# Check API quotas in Google Cloud Console
```

**Email Notifications Not Working**
```bash
# Test email configuration
python main.py test --email

# Check SMTP settings
echo $EMAIL_SMTP_SERVER $EMAIL_SMTP_PORT

# Verify app password (for Gmail)
# Make sure 2FA is enabled and app password is generated
```

**SBS Analysis Failing**
```bash
# Test SBS system
python main.py test --sbs

# Check OpenCV installation
python -c "import cv2; print(cv2.__version__)"

# Verify image analysis dependencies
python -c "import numpy, scipy; print('Dependencies OK')"
```

**Google Drive Upload Issues**
```bash
# Test Drive integration
python main.py test --drive

# Check folder permissions
# Ensure service account has access to target folder

# Verify folder ID
echo $GOOGLE_DRIVE_FOLDER_ID
```

**Sunset Times Incorrect**
```bash
# Validate location settings
python main.py test --sunset

# Check timezone configuration
python main.py config --show
```

### Log Analysis
```bash
# View recent logs
tail -f logs/sunset_timelapse.log

# Search for errors
grep -i error logs/sunset_timelapse.log

# Check service logs (Raspberry Pi)
sudo journalctl -u sunset-timelapse -f
```

### Performance Optimization

**Raspberry Pi Performance**
```bash
# Increase GPU memory split
sudo raspi-config
# Advanced Options -> Memory Split -> 128

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable wifi-power-save
```

**Storage Management**
```bash
# Monitor disk usage
df -h

# Clean old files manually
python -c "
from sunset_scheduler import SunsetScheduler
scheduler = SunsetScheduler()
scheduler.cleanup_old_files()
"
```

## 📊 Monitoring and Maintenance

### Health Checks
```bash
# System status
python main.py status

# Validate all components
python main.py test

# Check recent captures
ls -la images/$(date +%Y-%m-%d)/
```

### Regular Maintenance
```bash
# Weekly cleanup (automated via scheduler)
# Manual cleanup if needed:
python main.py schedule --validate

# Update system safely on Pi
./update_pi.sh

# Update dependencies monthly
pip list --outdated
pip install -U package_name

# Review SBS analytics monthly
python sbs_reporter.py --report --days 30

# Check token health
python main.py test --youtube-token
```

### Backup Strategy
```bash
# Backup configuration
cp config.yaml config.yaml.backup
cp .env .env.backup

# Backup important videos
rsync -av videos/ backup_location/videos/

# Backup logs for analysis
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit changes** (`git commit -m 'Add amazing feature'`)
4. **Push to branch** (`git push origin feature/amazing-feature`)
5. **Open Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .

# Lint code
flake8 .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Astral** library for astronomical calculations
- **ONVIF** protocol for camera standardization
- **FFmpeg** for video processing capabilities
- **Google APIs** for YouTube integration
- **Reolink** for reliable camera hardware

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/sunset-timelapse/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/sunset-timelapse/discussions)
- **Email**: your.email@example.com

---

**Made with ❤️ for capturing beautiful sunsets automatically**