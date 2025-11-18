# Raspberry Pi Update Guide - Meteor Detection Feature

This guide provides step-by-step instructions for updating your Raspberry Pi to include the new automated meteor detection feature.

## 📋 Prerequisites

- Raspberry Pi running the sunset-timelapse system
- SSH access to your Raspberry Pi
- Git configured on your Raspberry Pi
- Existing sunset-timelapse service running

## 🚀 Update Steps

### Step 1: SSH into Your Raspberry Pi

```bash
ssh pi@your-raspberry-pi-ip
```

### Step 2: Navigate to Project Directory

```bash
cd /home/pi/sunset-timelapse
```

### Step 3: Stop the Running Service

```bash
sudo systemctl stop sunset-timelapse
```

### Step 4: Backup Current Configuration

```bash
# Backup config and environment
cp config.yaml config.yaml.backup.$(date +%Y%m%d)
cp .env .env.backup.$(date +%Y%m%d)

# Optional: Backup logs
mkdir -p ~/backups
tar -czf ~/backups/sunset-logs-$(date +%Y%m%d).tar.gz logs/
```

### Step 5: Update Code from Git

```bash
# Stash any local changes
git stash

# Pull latest changes
git pull origin main

# If you have local modifications, restore them
git stash pop  # Only if you stashed changes
```

### Step 6: Update Python Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Update dependencies (includes OpenCV for meteor detection)
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 7: Update Configuration File

Edit your `config.yaml` to add the meteor detection configuration:

```bash
nano config.yaml
```

Add the following section (usually after the `sbs:` section):

```yaml
# Meteor detection settings
meteor:
  enabled: true                      # Set to true to enable automatic overnight meteor scanning

  # Nighttime scanning window (automatically uses sunset/sunrise times)
  sunset_offset_minutes: 30          # Start scanning this many minutes AFTER sunset (ensures darkness)
  sunrise_offset_minutes: 30         # Stop scanning this many minutes BEFORE sunrise (avoids daylight)

  # Detection thresholds
  min_brightness_threshold: 200      # Minimum brightness (0-255) to consider as potential meteor
  min_area: 10                       # Minimum pixel area for detection
  max_area: 5000                     # Maximum pixel area (filter out large bright objects)

  # Track validation
  min_frames: 3                      # Minimum frames to confirm meteor (filters noise)
  max_frames: 30                     # Maximum frames (meteors are brief events)
  min_linearity: 0.85                # Path linearity score (0-1, higher = more linear)
  min_velocity: 5.0                  # Minimum velocity in pixels/frame
  max_gap_frames: 2                  # Max frames between detections to continue track

  # Clip extraction
  clip_padding_seconds: 3            # Seconds of padding before/after meteor event

  # Storage
  retention_days: 30                 # Keep meteor clips for this many days
```

**Important Notes:**
- If `meteor.enabled: false`, meteor scanning will be disabled
- If `meteor.enabled: true`, the system will automatically scan overnight footage every morning at 7 AM
- Meteor scanning runs AFTER the 6 AM daily maintenance task

Save the file (`Ctrl+O`, `Enter`, `Ctrl+X`).

### Step 8: Update Google Drive Configuration (Optional)

If you want meteor clips uploaded to Google Drive, ensure your `config.yaml` has:

```yaml
# Google Drive backup (optional)
drive:
  enabled: true                      # Set to true to enable Drive uploads
  folder_name: "Sunset Timelapses"   # Main folder for sunset videos
  meteor_folder_name: "meteors"      # Subfolder for meteor clips (will be created automatically)
  upload_videos: true
  upload_metadata: true
```

Edit if needed:

```bash
nano config.yaml
```

### Step 9: Verify Configuration

```bash
# Validate configuration
python main.py config --validate --show

# Test meteor detection system (optional - will analyze a test video if available)
python main.py meteor --stats
```

### Step 10: Restart the Service

```bash
# Restart the sunset-timelapse service
sudo systemctl start sunset-timelapse

# Check service status
sudo systemctl status sunset-timelapse

# View real-time logs to confirm everything is working
sudo journalctl -u sunset-timelapse -f
```

You should see log output indicating:
- Service started successfully
- Daily maintenance scheduled for 6 AM
- **Overnight meteor scan scheduled for 07:00** (new!)

Press `Ctrl+C` to exit the log viewer.

### Step 11: Verify Meteor Scheduling

Check that the meteor scan is properly scheduled:

```bash
# View service logs for meteor scheduling confirmation
sudo journalctl -u sunset-timelapse --since "5 minutes ago" | grep -i meteor
```

You should see a line like:
```
INFO - Overnight meteor scan scheduled for 07:00
```

## 📊 Monitoring Meteor Detection

### Check Meteor Scan Status

The meteor scan runs automatically every morning at 7 AM. To check if it ran:

```bash
# View logs from this morning
sudo journalctl -u sunset-timelapse --since "07:00 today" | grep -i meteor

# Or check the application logs
tail -f logs/sunset_timelapse.log | grep -i meteor
```

### View Detected Meteors

```bash
# List all detected meteor clips
ls -lh data/meteors/

# View meteor statistics
python main.py meteor --stats

# Check specific meteor metadata
cat data/meteors/meteor-11-17-2025-*.json
```

### Google Drive Uploads

If you have Google Drive enabled, meteor clips will be automatically uploaded to a `meteors` folder in your Google Drive. Check your email for upload notifications.

## 🧪 Testing Meteor Detection

### Test with Past Footage

You can manually run a meteor scan on a specific date range to test:

```bash
# Scan last night's footage manually
python main.py meteor --start 2024-11-16 --end 2024-11-17

# This will use actual sunset/sunrise times automatically
```

### Test with Specific Time Window

```bash
# Override automatic sunset/sunrise times with specific hours
python main.py meteor --start 2024-11-16 --end 2024-11-17 --start-time 20:00 --end-time 06:00
```

## 🔧 Troubleshooting

### Service Won't Start

```bash
# Check for errors in service
sudo systemctl status sunset-timelapse

# View detailed error logs
sudo journalctl -u sunset-timelapse -n 50

# Common issues:
# 1. Missing dependencies - reinstall: pip install -r requirements.txt
# 2. Configuration errors - validate: python main.py config --validate
# 3. Permissions - check: ls -la config.yaml .env
```

### OpenCV Installation Issues

If you get errors about OpenCV:

```bash
# Install OpenCV headless (optimized for Pi)
pip install opencv-python-headless

# If that fails, try system package
sudo apt update
sudo apt install python3-opencv
```

### Meteor Scan Not Running

```bash
# Check if meteor detection is enabled
grep -A 2 "^meteor:" config.yaml

# Should show: enabled: true
# If it shows: enabled: false, change to true and restart service

# Check service logs for scheduling confirmation
sudo journalctl -u sunset-timelapse | grep "meteor scan scheduled"
```

### Meteor Detection Too Sensitive / Not Sensitive Enough

Edit the detection thresholds in `config.yaml`:

```yaml
meteor:
  # Make MORE sensitive (detect more meteors, may include false positives):
  min_brightness_threshold: 180      # Lower threshold (was 200)
  min_frames: 2                      # Accept shorter tracks (was 3)
  min_linearity: 0.80                # More lenient path (was 0.85)

  # Make LESS sensitive (fewer false positives, may miss faint meteors):
  min_brightness_threshold: 220      # Higher threshold (was 200)
  min_frames: 4                      # Require longer tracks (was 3)
  min_linearity: 0.90                # Stricter path (was 0.85)
```

After changing, restart the service:
```bash
sudo systemctl restart sunset-timelapse
```

## 📧 Email Notifications

When meteors are detected, you'll receive an email notification with:
- Number of meteors detected
- Timestamp of each meteor (in CST/CDT)
- Duration and brightness information
- Google Drive links (if enabled)

Example email subject: `Meteor Detection Report - 2024-11-17`

## 🗄️ Storage Management

Meteor clips are stored in: `/home/pi/sunset-timelapse/data/meteors/`

Each meteor has:
- **Video clip**: `meteor-MM-DD-YYYY-HH-MM-SS-AM/PM-TZ.mp4`
- **Metadata JSON**: `meteor-MM-DD-YYYY-HH-MM-SS-AM/PM-TZ.json`

### Cleanup

Old meteor clips are automatically cleaned up based on `retention_days` setting (default: 30 days).

To manually clean up old meteors:

```bash
# Remove meteors older than 30 days
find data/meteors/ -name "meteor-*.mp4" -mtime +30 -delete
find data/meteors/ -name "meteor-*.json" -mtime +30 -delete
```

## 📅 Daily Schedule Summary

After this update, your Raspberry Pi will run:

- **6:00 AM**: Daily maintenance (token refresh, cleanup)
- **7:00 AM**: Overnight meteor scan (analyzes previous night's footage)
- **2 hours before sunset**: Start sunset timelapse capture
- **2:00 AM Sunday**: Weekly cleanup (backup)

## 🔄 Rolling Back (If Needed)

If you encounter issues and need to roll back:

```bash
# Stop service
sudo systemctl stop sunset-timelapse

# Restore backup configuration
cp config.yaml.backup.YYYYMMDD config.yaml
cp .env.backup.YYYYMMDD .env

# Revert code changes
git reset --hard HEAD~1  # Goes back one commit

# Restart service
sudo systemctl start sunset-timelapse
```

## ✅ Verification Checklist

After completing the update, verify:

- [ ] Service is running: `sudo systemctl status sunset-timelapse`
- [ ] Meteor detection enabled: `grep "enabled: true" config.yaml` (under meteor section)
- [ ] Meteor scan scheduled: Check logs for "meteor scan scheduled for 07:00"
- [ ] Configuration valid: `python main.py config --validate`
- [ ] No errors in logs: `sudo journalctl -u sunset-timelapse -n 20`
- [ ] Storage directory exists: `ls -la data/meteors/`

## 📞 Getting Help

If you encounter issues:

1. **Check Logs**: `sudo journalctl -u sunset-timelapse -f`
2. **Validate Config**: `python main.py config --validate`
3. **Test Components**: `python main.py test --camera --youtube`
4. **Open Issue**: [GitHub Issues](https://github.com/yourusername/sunset-timelapse/issues)

---

**Update Complete!** Your Raspberry Pi will now automatically scan for meteors every morning at 7 AM.
