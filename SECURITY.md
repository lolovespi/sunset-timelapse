# Security Policy

## Overview

This document outlines the security measures implemented in the Sunset Timelapse system and provides guidance for secure deployment.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |

## Security Features

### 1. **Credential Management** ✅

- All secrets stored in environment variables (`.env` file)
- OAuth2 tokens stored in `~/.config/sunset-timelapse/credentials/` with 600 permissions
- No credentials hardcoded in source code
- RTSP URLs sanitized before logging
- Configuration repr redacts passwords

### 2. **Injection Prevention** ✅

- **Command Injection**: All `subprocess` calls use list format (never `shell=True`)
- **Code Injection**: Replaced unsafe `eval()` with safe fraction parsing
- **Path Traversal**: Filenames sanitized to prevent directory traversal attacks
- **YAML Injection**: Uses `yaml.safe_load()` instead of `yaml.load()`

### 3. **Input Validation** ✅

- IP addresses validated against regex pattern
- Port numbers validated (1-65535 range)
- Latitude/longitude validated against geographic bounds
- Time values validated as positive integers

### 4. **Session Management** ✅

- Camera API sessions properly closed with try/finally blocks
- OAuth tokens automatically refreshed before expiration
- Concurrent session limits respected

## Security Setup

### Initial Setup

```bash
# 1. Run security setup script
./setup_security.sh

# 2. Create .env file from example
cp example.env .env
chmod 600 .env

# 3. Edit .env with your secrets
nano .env

# 4. Validate configuration
python main.py config --validate
```

### Required File Permissions

```bash
# Environment file (contains secrets)
chmod 600 .env

# Credentials directory
chmod 700 ~/.config/sunset-timelapse/
chmod 600 ~/.config/sunset-timelapse/credentials/*

# Data directories
chmod 700 data/
chmod 700 data/logs/
chmod 700 data/images/
chmod 700 data/videos/
```

### Systemd Service Security (Raspberry Pi)

The systemd service includes security hardening:

```ini
[Service]
NoNewPrivileges=true  # Prevents privilege escalation
PrivateTmp=true       # Isolated /tmp directory
```

Additional recommended hardening:

```ini
# Add to /etc/systemd/system/sunset-timelapse.service
ProtectHome=read-only
ProtectSystem=strict
ReadWritePaths=/home/mvp/sunset-timelapse/data
```

## Security Best Practices

### 1. **Secrets Management**

- ✅ Store all secrets in `.env` file (never in code)
- ✅ Use app-specific passwords for email (not main password)
- ✅ Rotate OAuth tokens every 6 months
- ✅ Never commit `.env`, `*token*`, or `*credentials*` files to git

### 2. **Network Security**

- ✅ Use strong camera passwords (minimum 12 characters)
- ✅ Isolate camera on separate VLAN if possible
- ✅ Use wired Ethernet for camera (avoid WiFi)
- ✅ Restrict camera access to only the Pi's IP

### 3. **System Hardening**

```bash
# Keep system updated
sudo apt update && sudo apt upgrade

# Enable automatic security updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure --priority=low unattended-upgrades

# Configure firewall (if needed)
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow from 192.168.6.0/24  # Allow camera subnet
```

### 4. **Dependency Security**

```bash
# Check for vulnerable dependencies
pip install safety
safety check -r requirements.txt

# Or use pip-audit
pip install pip-audit
pip-audit -r requirements.txt
```

### 5. **Monitoring**

- Review logs regularly for suspicious activity
- Monitor failed authentication attempts
- Set up email alerts for critical errors
- Check disk usage to prevent DoS via log/image flooding

## Reported Vulnerabilities

### 2024-01-XX: eval() Code Injection (FIXED)

**Severity**: CRITICAL
**Status**: ✅ FIXED
**Description**: `eval()` used to parse FFmpeg FPS output, could allow code injection if video file metadata was maliciously crafted.
**Fix**: Replaced with safe fraction parsing function.
**Commit**: `7c08f68`

### 2024-01-XX: Path Traversal in Downloads (FIXED)

**Severity**: HIGH
**Status**: ✅ FIXED
**Description**: Camera-provided filenames not sanitized, could allow path traversal attacks.
**Fix**: Added `sanitize_filename()` function to strip directory traversal attempts.
**Commit**: `7c08f68`

## Reporting a Vulnerability

If you discover a security vulnerability, please:

1. **DO NOT** open a public issue
2. Email security details to: [your-email]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

We will respond within 48 hours and work with you to resolve the issue.

## Security Checklist

Before deploying to production:

- [ ] Run `./setup_security.sh` to secure file permissions
- [ ] Verify `.env` file has mode 600
- [ ] Ensure no secrets are committed to git
- [ ] Review systemd service security settings
- [ ] Enable automatic security updates on Pi
- [ ] Configure firewall rules
- [ ] Set up email notifications for errors
- [ ] Document camera access credentials in password manager
- [ ] Test backup/restore procedures
- [ ] Review logs for initial 48 hours after deployment

## Compliance

This system handles:
- **PII**: None (location data is configuration, not user data)
- **Credentials**: Camera passwords, OAuth tokens (secured with encryption at rest)
- **Network Data**: RTSP streams (transmitted over local network, consider VPN for remote access)

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [Raspberry Pi Security](https://www.raspberrypi.com/documentation/computers/configuration.html#securing-your-raspberry-pi)

## Changelog

| Date | Change | Commit |
|------|--------|--------|
| 2024-11-09 | Fixed eval() vulnerability | 7c08f68 |
| 2024-11-09 | Added path traversal protection | 7c08f68 |
| 2024-11-09 | Added config validation | 7c08f68 |
| 2024-11-05 | Fixed camera session leaks | 9c6b1ea |
| 2024-11-05 | Added YouTube quota logging | 6d0f09f |
