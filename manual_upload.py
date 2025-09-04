#!/usr/bin/env python3
"""
Manual YouTube upload script for historical videos
"""
import sys
from pathlib import Path
from datetime import datetime, date
from youtube_uploader import YouTubeUploader

def manual_upload():
    video_path = Path("/Users/lolovespi/Documents/GitHub/sunset-timelapse/data/videos/Sunset_08_25_25.mp4")
    
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return
    
    # August 25, 2025 sunset window (6:20 PM - 8:20 PM CDT)
    video_date = date(2025, 8, 25)
    start_time = datetime(2025, 8, 25, 18, 20, 39)  # 6:20:39 PM CDT
    end_time = datetime(2025, 8, 25, 20, 20, 39)    # 8:20:39 PM CDT
    
    uploader = YouTubeUploader()
    print(f"Uploading {video_path.name}...")
    
    # Authenticate first
    if not uploader.authenticate():
        print("❌ YouTube authentication failed")
        return
    
    try:
        video_id = uploader.upload_video(
            video_path=video_path,
            video_date=video_date,
            start_time=start_time,
            end_time=end_time,
            is_test=False
        )
        
        if video_id:
            print(f"✅ Upload successful!")
            print(f"Video URL: https://www.youtube.com/watch?v={video_id}")
        else:
            print("❌ Upload failed")
            
    except Exception as e:
        print(f"❌ Upload error: {e}")

if __name__ == "__main__":
    manual_upload()