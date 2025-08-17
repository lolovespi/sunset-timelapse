# Sunset Timelapse System

An automated system for capturing daily sunset timelapses using a Reolink camera, processing them into videos, and uploading to YouTube. Designed to run on Raspberry Pi for daily captures and MacBook for intensive historical processing.

## 🌅 Features

- **Automated Daily Capture**: Calculates sunset times based on location and captures images 1 hour before/after sunset
- **Smart Scheduling**: Uses astronomical calculations to determine optimal capture windows
- **Video Processing**: Creates high-quality timelapse videos from image sequences using FFmpeg
- **YouTube Integration**: Automatically uploads videos with proper metadata and descriptions
- **Historical Retrieval**: Downloads and processes historical footage from camera storage
- **Multi-Platform**: Raspberry Pi for daily operations, MacBook for processing power
- **Secure Configuration**: Environment-based secrets management
- **ONVIF Support**: Camera control via standard ONVIF protocol
- **Comprehensive Logging**: Detailed logs with rotation and health monitoring

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
```

### Environment Variables (.env)

Store sensitive information in environment variables:

```bash
# Camera credentials
CAMERA_PASSWORD=your_camera_password

# Google Cloud credentials path
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
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
├── config.yaml              # Main configuration
├── .env                     # Environment variables (create from .env.example)
├── requirements.txt         # Python dependencies
├── main.py                  # CLI entry point
├── config_manager.py        # Configuration and secrets management
├── sunset_calculator.py     # Astronomical calculations
├── camera_interface.py      # Camera control via ONVIF
├── video_processor.py       # Video creation with FFmpeg
├── youtube_uploader.py      # YouTube API integration
├── sunset_scheduler.py      # Main orchestrator
├── historical_retrieval.py  # Historical footage processing
├── logs/                    # Application logs
├── images/                  # Captured images (organized by date)
├── videos/                  # Created videos
└── temp/                    # Temporary files
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

# Verify credentials file exists
ls -la $GOOGLE_APPLICATION_CREDENTIALS

# Check API quotas in Google Cloud Console
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

# Update dependencies monthly
pip list --outdated
pip install -U package_name
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