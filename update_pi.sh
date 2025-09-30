#!/bin/bash

# Update script for Raspberry Pi sunset-timelapse service
# This script pulls the latest changes and restarts the service

set -e  # Exit on any error

echo "🔄 Updating sunset-timelapse on Raspberry Pi..."

# Stash any uncommitted changes (including untracked files)
echo "📦 Stashing local changes..."
git stash -u

# Pull latest changes from main branch
echo "⬇️ Pulling latest changes..."
git pull origin main

# Restore stashed changes
echo "📤 Restoring local changes..."
git stash pop || echo "ℹ️ No stashed changes to restore"

# Restart the systemd service
echo "🔄 Restarting sunset-timelapse service..."
sudo systemctl restart sunset-timelapse

# Check service status
echo "✅ Checking service status..."
sudo systemctl status sunset-timelapse --no-pager -l

echo "🎉 Update complete!"