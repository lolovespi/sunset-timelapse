"""
Historical Retrieval
Retrieves recorded footage from Reolink camera storage for specified date ranges
Creates timelapses from historical data
"""

import logging
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import requests
from requests.auth import HTTPDigestAuth
import json
import cv2
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config_manager import get_config
from sunset_calculator import SunsetCalculator
from video_processor import VideoProcessor
from youtube_uploader import YouTubeUploader


class HistoricalRetrieval:
    """Handles retrieval of historical footage from camera"""
    
    def __init__(self):
        """Initialize historical retrieval system"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.sunset_calc = SunsetCalculator()
        self.video_processor = VideoProcessor()
        self.youtube_uploader = YouTubeUploader()
        
        # Camera settings
        self.ip = self.config.get('camera.ip')
        self.username, self.password = self.config.get_camera_credentials()
        self.auth = HTTPDigestAuth(self.username, self.password)
        
        # Retrieval settings
        self.timeout = self.config.get('advanced.connection_timeout_seconds', 30)
        self.max_retries = self.config.get('advanced.max_retries', 3)
        self.max_workers = 4  # For parallel downloads
        
        self.logger.info("Historical retrieval system initialized")
        
    def get_camera_recordings(self, start_date: date, end_date: date) -> List[Dict]:
        """
        Get list of recordings from camera for date range
        First checks for locally stored images, then attempts camera API if available
        
        Args:
            start_date: Start date for search
            end_date: End date for search
            
        Returns:
            List of recording metadata dictionaries
        """
        self.logger.info(f"Searching for recordings from {start_date} to {end_date}")
        
        recordings = []
        
        # First check for locally stored images
        #local_recordings = self._check_local_images(start_date, end_date)
        #if local_recordings:
        #    self.logger.info(f"Found {len(local_recordings)} local image sets")
        #    recordings.extend(local_recordings)
        #    return recordings
        
        # Try RTSP playback detection for SD card recordings
        rtsp_recordings = self._try_rtsp_playback_search(start_date, end_date)
        if rtsp_recordings:
            self.logger.info(f"Found {len(rtsp_recordings)} potential SD card recordings via RTSP")
            recordings.extend(rtsp_recordings)
            return recordings
        
        # Try Reolink official API for SD card recordings
        reolink_recordings = self._try_reolink_official_api(start_date, end_date)
        if reolink_recordings:
            self.logger.info(f"Found {len(reolink_recordings)} recordings via Reolink official API")
            recordings.extend(reolink_recordings)
            return recordings

        # If no local images, attempt camera API (may fail due to HTTP API limitations)
        try:
            self.logger.warning("No local images found, attempting camera HTTP API (may not work on this camera)")
            # Reolink API endpoint for getting recordings
            # Note: This may vary by camera model and firmware version
            search_url = f"http://{self.ip}/api.cgi"
            
            # Construct search parameters
            search_params = {
                'cmd': 'Search',
                'token': self._get_auth_token(),
                'param': {
                    'Search': {
                        'channel': 0,
                        'onlyStatus': 0,
                        'streamType': 'main',
                        'StartTime': {
                            'year': start_date.year,
                            'mon': start_date.month,
                            'day': start_date.day,
                            'hour': 0,
                            'min': 0,
                            'sec': 0
                        },
                        'EndTime': {
                            'year': end_date.year,
                            'mon': end_date.month,
                            'day': end_date.day,
                            'hour': 23,
                            'min': 59,
                            'sec': 59
                        }
                    }
                }
            }
            
            response = requests.post(
                search_url,
                json=search_params,
                auth=self.auth,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('cmd') == 'Search' and result.get('code') == 0:
                file_list = result.get('value', {}).get('SearchResult', {}).get('File', [])
                
                for file_info in file_list:
                    recordings.append({
                        'name': file_info.get('name'),
                        'size': file_info.get('size'),
                        'start_time': self._parse_camera_time(file_info.get('StartTime')),
                        'end_time': self._parse_camera_time(file_info.get('EndTime')),
                        'type': file_info.get('type')
                    })
                    
                self.logger.info(f"Found {len(recordings)} recordings via camera API")
            else:
                self.logger.error(f"Camera search failed: {result}")
                
        except Exception as e:
            self.logger.error(f"Failed to get camera recordings via HTTP API: {e}")
            self.logger.info("Camera HTTP API not available - use live capture mode instead")
            
        return recordings

    def _try_reolink_official_api(self, start_date: date, end_date: date) -> List[Dict]:
        """
        Try to get recordings using the official Reolink API library
        
        Args:
            start_date: Start date for search
            end_date: End date for search
            
        Returns:
            List of recording metadata dictionaries
        """
        recordings = []
        
        try:
            # Try to import the official Reolink API library
            from reolinkapi import Camera as ReoCamera
            
            self.logger.info("Using official Reolink API to search for recordings")
            
            # Create camera instance
            reo_camera = ReoCamera(self.ip, self.username, self.password)
            
            # Search for motion recordings in date range
            current_date = start_date
            while current_date <= end_date:
                try:
                    # Get motion files for this date
                    start_datetime = datetime.combine(current_date, datetime.min.time())
                    end_datetime = datetime.combine(current_date, datetime.max.time())
                    
                    motion_files = reo_camera.get_motion_files(
                        start=start_datetime,
                        end=end_datetime,
                        streamtype='main',
                        channel=0
                    )
                    
                    if motion_files:
                        self.logger.info(f"Found {len(motion_files)} motion files for {current_date}")
                        
                        # Debug: log the first file structure
                        if motion_files:
                            self.logger.debug(f"Sample motion file data: {motion_files[0]}")
                        
                        for motion_file in motion_files:
                            # Convert to our standard format
                            # The motion_file structure: {'start': datetime, 'end': datetime, 'filename': 'path'}
                            file_name = motion_file.get('filename') or motion_file.get('name') or motion_file.get('fileName', 'unknown')
                            file_size = motion_file.get('size') or motion_file.get('fileSize') or motion_file.get('file_size', 0)
                            start_time = motion_file.get('start') or motion_file.get('start_time') or motion_file.get('startTime')
                            end_time = motion_file.get('end') or motion_file.get('end_time') or motion_file.get('endTime')
                            
                            recordings.append({
                                'name': file_name,
                                'file_name': file_name,
                                'size': file_size,
                                'start_time': start_time,
                                'end_time': end_time,
                                'type': 'reolink_motion',
                                'api_method': 'official',
                                'reolink_data': motion_file  # Store original data
                            })
                    else:
                        self.logger.debug(f"No motion files found for {current_date}")
                        
                except Exception as date_error:
                    self.logger.error(f"Failed to get motion files for {current_date}: {date_error}")
                    
                current_date += timedelta(days=1)
                
        except ImportError:
            self.logger.warning("Reolink API library not available. Install with: pip install git+https://github.com/ReolinkCameraAPI/reolinkapipy.git")
        except Exception as e:
            self.logger.error(f"Failed to use Reolink official API: {e}")
            
        return recordings
        
    def _try_rtsp_playback_search(self, start_date: date, end_date: date) -> List[Dict]:
        """
        Try to detect recordings using RTSP playback URLs with time parameters
        
        Args:
            start_date: Start date for search
            end_date: End date for search
            
        Returns:
            List of detected recording periods
        """
        recordings = []
        
        try:
            import subprocess
            
            self.logger.info("Attempting to detect recordings via RTSP playback")
            
            current_date = start_date
            while current_date <= end_date:
                # Try to probe for recordings on this date
                # Format: rtsp://username:password@ip:port/path?starttime=...&endtime=...
                start_time = datetime.combine(current_date, datetime.min.time())
                end_time = datetime.combine(current_date, datetime.max.time())
                
                # Common Reolink RTSP playback URL formats
                rtsp_urls = [
                    f"rtsp://{self.username}:{self.password}@{self.ip}:554/h264Preview_01_sub",
                    f"rtsp://{self.username}:{self.password}@{self.ip}:554/h264Preview_01_main"
                ]
                
                for rtsp_url in rtsp_urls:
                    try:
                        # Use ffprobe to check if stream is accessible
                        cmd = [
                            'ffprobe',
                            '-v', 'quiet',
                            '-select_streams', 'v:0',
                            '-show_entries', 'stream=duration',
                            '-of', 'csv=p=0',
                            '-t', '1',  # Only probe for 1 second
                            rtsp_url
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                        
                        if result.returncode == 0:
                            # Stream accessible, assume recordings exist for this date
                            recordings.append({
                                'name': f'rtsp_recording_{current_date}',
                                'size': 0,  # Unknown size
                                'start_time': start_time,
                                'end_time': end_time,
                                'type': 'rtsp_playback',
                                'rtsp_url': rtsp_url
                            })
                            self.logger.info(f"Detected potential recordings for {current_date}")
                            break  # Found working stream for this date
                            
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
                        continue  # Try next URL format
                        
                current_date += timedelta(days=1)
                
        except Exception as e:
            self.logger.error(f"RTSP playback detection failed: {e}")
            
        return recordings
        
    def _check_local_images(self, start_date: date, end_date: date) -> List[Dict]:
        """
        Check for locally stored images in the specified date range
        
        Args:
            start_date: Start date for search
            end_date: End date for search
            
        Returns:
            List of local image set metadata dictionaries
        """
        local_recordings = []
        
        try:
            paths = self.config.get_storage_paths()
            images_dir = paths['images']
            
            if not images_dir.exists():
                return local_recordings
                
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                date_dir = images_dir / date_str
                
                if date_dir.exists():
                    # Look for image files in this date directory
                    image_files = list(date_dir.glob('*.jpg')) + list(date_dir.glob('*.jpeg')) + list(date_dir.glob('*.png'))
                    
                    # Also check historical subdirectory
                    historical_dir = date_dir / 'historical'
                    if historical_dir.exists():
                        historical_images = list(historical_dir.glob('*.jpg')) + list(historical_dir.glob('*.jpeg')) + list(historical_dir.glob('*.png'))
                        image_files.extend(historical_images)
                    
                    if image_files:
                        # Sort by filename (which should contain timestamp)
                        image_files.sort()
                        
                        # Create recording metadata for this date
                        first_image = image_files[0]
                        last_image = image_files[-1]
                        
                        # Extract timestamps from filenames (format: img_YYYYMMDD_HHMMSS.jpg)
                        try:
                            first_time = self._extract_timestamp_from_filename(first_image.name)
                            last_time = self._extract_timestamp_from_filename(last_image.name)
                        except:
                            # Fallback to file modification times
                            first_time = datetime.fromtimestamp(first_image.stat().st_mtime)
                            last_time = datetime.fromtimestamp(last_image.stat().st_mtime)
                            
                        # Check if we have historical images
                        has_historical = any(f.parent.name == 'historical' for f in image_files)
                        image_type = 'local_historical' if has_historical else 'local_images'
                        
                        local_recordings.append({
                            'name': f'local_images_{date_str}',
                            'size': sum(f.stat().st_size for f in image_files),
                            'start_time': first_time,
                            'end_time': last_time,
                            'type': image_type,
                            'image_count': len(image_files),
                            'image_files': image_files
                        })
                        
                current_date += timedelta(days=1)
                
        except Exception as e:
            self.logger.error(f"Failed to check local images: {e}")
            
        return local_recordings
        
    def _extract_timestamp_from_filename(self, filename: str) -> datetime:
        """
        Extract timestamp from image filename
        
        Args:
            filename: Image filename (e.g., img_20250701_180000.jpg)
            
        Returns:
            Extracted datetime object
        """
        # Remove extension
        name_without_ext = filename.rsplit('.', 1)[0]
        
        # Extract timestamp part (assumes format: img_YYYYMMDD_HHMMSS or frame_YYYYMMDD_HHMMSS)
        if '_' in name_without_ext:
            parts = name_without_ext.split('_')
            if len(parts) >= 3:
                date_part = parts[-2]  # YYYYMMDD
                time_part = parts[-1]  # HHMMSS
                
                timestamp_str = f"{date_part}_{time_part}"
                return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
        # Fallback: try to parse the whole filename as timestamp
        raise ValueError(f"Cannot extract timestamp from filename: {filename}")
        
    def _get_auth_token(self) -> Optional[str]:
        """
        Get authentication token from camera
        
        Returns:
            Authentication token or None if failed
        """
        try:
            login_url = f"http://{self.ip}/api.cgi"
            login_params = {
                'cmd': 'Login',
                'param': {
                    'User': {
                        'userName': self.username,
                        'password': self.password
                    }
                }
            }
            
            response = requests.post(
                login_url,
                json=login_params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('cmd') == 'Login' and result.get('code') == 0:
                token = result.get('value', {}).get('Token', {}).get('name')
                return token
            else:
                self.logger.error(f"Login failed: {result}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get auth token: {e}")
            return None
            
    def _parse_camera_time(self, time_dict: Dict) -> datetime:
        """
        Parse camera time dictionary to datetime
        
        Args:
            time_dict: Dictionary with year, mon, day, hour, min, sec
            
        Returns:
            Parsed datetime object
        """
        return datetime(
            year=time_dict.get('year', 2000),
            month=time_dict.get('mon', 1),
            day=time_dict.get('day', 1),
            hour=time_dict.get('hour', 0),
            minute=time_dict.get('min', 0),
            second=time_dict.get('sec', 0)
        )
        
    def download_recording(self, recording_info: Dict, output_path: Path) -> bool:
        """
        Download a specific recording from camera
        
        Args:
            recording_info: Recording metadata dictionary
            output_path: Path to save downloaded file
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            filename = recording_info['name']
            recording_type = recording_info.get('type', 'unknown')
            self.logger.info(f"Downloading recording: {filename} (type: {recording_type})")
            
            # Use official Reolink API if available
            if recording_type == 'reolink_motion' and recording_info.get('api_method') == 'official':
                return self._download_via_official_api(recording_info, output_path)
            
            # Fallback to HTTP API method
            return self._download_via_http_api(recording_info, output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download recording {recording_info.get('name')}: {e}")
            return False
            
    def _download_via_official_api(self, recording_info: Dict, output_path: Path) -> bool:
        """Download recording using official Reolink API"""
        try:
            from reolinkapi import Camera as ReoCamera
            
            reolink_data = recording_info.get('reolink_data', {})
            filename = reolink_data.get('name', recording_info['name'])
            
            self.logger.info(f"Downloading via official Reolink API: {filename}")
            
            # Create camera instance
            reo_camera = ReoCamera(self.ip, self.username, self.password)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download the file directly to output path
            success = reo_camera.get_file(filename, str(output_path))
            
            if success and output_path.exists():
                file_size = output_path.stat().st_size
                self.logger.info(f"Downloaded: {output_path} ({file_size:,} bytes)")
                return True
            else:
                self.logger.error(f"Failed to download file {filename}")
                return False
                
        except ImportError:
            self.logger.warning("Reolink API library not available, falling back to HTTP API")
            return self._download_via_http_api(recording_info, output_path)
        except Exception as e:
            self.logger.error(f"Failed to download via official API: {e}")
            return False
            
    def _download_via_http_api(self, recording_info: Dict, output_path: Path) -> bool:
        """Download recording using HTTP API (fallback method)"""
        try:
            filename = recording_info['name']
            
            # Reolink download URL format
            download_url = f"http://{self.ip}/flv"
            download_params = {
                'port': 1935,
                'app': 'bcs',
                'stream': filename,
                'user': self.username,
                'password': self.password
            }
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress tracking
            response = requests.get(
                download_url,
                params=download_params,
                auth=self.auth,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            
            self.logger.info(f"Downloaded: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download via HTTP API: {e}")
            return False
            
    def extract_frames_from_recording(self, video_path: Path, 
                                    start_time: datetime, end_time: datetime,
                                    interval_seconds: int = 5) -> List[Path]:
        """
        Extract frames from downloaded recording at specified intervals
        
        Args:
            video_path: Path to downloaded video file
            start_time: Start time for frame extraction
            end_time: End time for frame extraction
            interval_seconds: Interval between extracted frames
            
        Returns:
            List of extracted frame paths
        """
        self.logger.info(f"Extracting frames from {video_path.name}")
        self.logger.info(f"Time range: {start_time} to {end_time}")
        
        extracted_frames = []
        
        try:
            # Open video file
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                self.logger.error(f"Failed to open video file: {video_path}")
                return extracted_frames
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            self.logger.info(f"Video properties: {fps:.2f} FPS, {total_frames} frames, {duration:.1f}s")
            
            # Calculate frame extraction parameters based on actual video duration
            video_duration_seconds = duration  # Use actual video duration
            frames_to_extract = int(video_duration_seconds / interval_seconds)
            frame_interval = int(fps * interval_seconds)
            
            self.logger.info(f"Extracting {frames_to_extract} frames at {interval_seconds}s intervals")
            
            # Create output directory
            date_str = start_time.strftime('%Y-%m-%d')
            paths = self.config.get_storage_paths()
            output_dir = paths['images'] / date_str / 'historical'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract frames
            frame_count = 0
            extracted_count = 0
            
            with tqdm(total=frames_to_extract, desc="Extracting frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                        
                    # Check if this frame should be extracted
                    if frame_count % frame_interval == 0 and extracted_count < frames_to_extract:
                        # Calculate timestamp for this frame
                        frame_time = start_time + timedelta(seconds=extracted_count * interval_seconds)
                        timestamp_str = frame_time.strftime('%Y%m%d_%H%M%S')
                        
                        # Save frame
                        frame_path = output_dir / f"frame_{timestamp_str}.jpg"
                        quality = self.config.get('capture.image_quality', 90)
                        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                        
                        extracted_frames.append(frame_path)
                        extracted_count += 1
                        pbar.update(1)
                        
                    frame_count += 1
                    
            cap.release()
            self.logger.info(f"Extracted {len(extracted_frames)} frames")
            
        except Exception as e:
            self.logger.error(f"Failed to extract frames: {e}")
            
        return extracted_frames
        
    def retrieve_historical_sunset(self, target_date: date) -> Optional[List[Path]]:
        """
        Retrieve and process historical sunset footage for a specific date
        
        Args:
            target_date: Date to retrieve sunset footage for
            
        Returns:
            List of extracted frame paths or None if failed
        """
        self.logger.info(f"Retrieving historical sunset footage for {target_date}")
        
        try:
            # Calculate sunset capture window for the date
            start_time, end_time = self.sunset_calc.get_capture_window(target_date)
            
            # Convert timezone-aware to naive datetimes for comparison
            if hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
                start_time = start_time.replace(tzinfo=None)
            if hasattr(end_time, 'tzinfo') and end_time.tzinfo is not None:
                end_time = end_time.replace(tzinfo=None)
            
            # Search for recordings that overlap with sunset time
            recordings = self.get_camera_recordings(target_date, target_date)
            
            if not recordings:
                self.logger.error(f"No recordings found for {target_date}")
                return None
                
            # Find recordings that overlap with sunset window
            relevant_recordings = []
            for recording in recordings:
                rec_start = recording['start_time']
                rec_end = recording['end_time']
                
                # Check if recording overlaps with sunset window
                if (rec_start <= end_time and rec_end >= start_time):
                    relevant_recordings.append(recording)
                    
            if not relevant_recordings:
                self.logger.error(f"No recordings found that overlap with sunset window")
                return None
                
            self.logger.info(f"Found {len(relevant_recordings)} relevant recordings")
            
            # Process recordings based on type
            all_frames = []
            temp_dir = self.config.get_storage_paths()['temp']
            
            for recording in relevant_recordings:
                recording_type = recording.get('type', 'unknown')
                
                if recording_type in ['local_images', 'local_historical']:
                    # For local images, return existing image files
                    self.logger.info(f"Using existing local images for {target_date}")
                    image_files = recording.get('image_files', [])
                    
                    # Filter images by sunset time window if needed
                    filtered_images = []
                    for img_path in image_files:
                        try:
                            # Extract timestamp from filename
                            img_time = self._extract_timestamp_from_filename(img_path.name)
                            if start_time <= img_time <= end_time:
                                filtered_images.append(img_path)
                        except:
                            # If we can't parse timestamp, include all images
                            filtered_images.append(img_path)
                    
                    all_frames.extend(filtered_images)
                    self.logger.info(f"Found {len(filtered_images)} images in sunset window")
                    
                else:
                    # For remote recordings, download and extract frames
                    video_filename = f"historical_{target_date}_{recording['name']}"
                    video_path = temp_dir / video_filename
                    
                    if self.download_recording(recording, video_path):
                        # Extract frames using actual recording timestamps
                        rec_start = recording['start_time']
                        rec_end = recording['end_time']
                        frames = self.extract_frames_from_recording(
                            video_path, rec_start, rec_end,
                            self.config.get('capture.interval_seconds', 5)
                        )
                        all_frames.extend(frames)
                        
                        # Clean up downloaded video
                        video_path.unlink()
                    
            if all_frames:
                # Sort frames by timestamp
                all_frames.sort(key=lambda x: x.name)
                self.logger.info(f"Retrieved {len(all_frames)} frames for {target_date}")
                return all_frames
            else:
                self.logger.error("No frames extracted from recordings")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve historical sunset footage: {e}")
            return None
            
    def create_historical_timelapse(self, start_date: date, end_date: date,
                                  upload_to_youtube: bool = True) -> List[Path]:
        """
        Create timelapses for a range of historical dates
        
        Args:
            start_date: Start date for processing
            end_date: End date for processing
            upload_to_youtube: Whether to upload videos to YouTube
            
        Returns:
            List of created video paths
        """
        self.logger.info(f"Creating historical timelapses from {start_date} to {end_date}")
        
        created_videos = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                self.logger.info(f"Processing {current_date}")
                
                # Retrieve historical footage
                frames = self.retrieve_historical_sunset(current_date)
                
                if frames and len(frames) > 10:  # Need minimum frames for video
                    # Create timelapse
                    video_path = self.video_processor.create_date_timelapse(current_date)
                    
                    if video_path:
                        created_videos.append(video_path)
                        self.logger.info(f"Created timelapse: {video_path}")
                        
                        # Upload to YouTube if requested
                        if upload_to_youtube:
                            try:
                                # Calculate start and end times for this date
                                sunset_start, sunset_end = self.sunset_calc.get_capture_window(current_date)
                                
                                self.youtube_uploader.upload_video(
                                    video_path, current_date, sunset_start, sunset_end
                                )
                            except Exception as e:
                                self.logger.warning(f"YouTube upload failed for {current_date}: {e}")
                    else:
                        self.logger.error(f"Failed to create timelapse for {current_date}")
                else:
                    self.logger.warning(f"Insufficient frames for {current_date} (got {len(frames) if frames else 0})")
                    
            except Exception as e:
                self.logger.error(f"Error processing {current_date}: {e}")
                
            current_date += timedelta(days=1)
            
        self.logger.info(f"Historical processing complete. Created {len(created_videos)} videos.")
        return created_videos
        
    def get_available_dates(self, start_date: date, end_date: date) -> List[date]:
        """
        Get list of dates that have available recordings
        
        Args:
            start_date: Start date for search
            end_date: End date for search
            
        Returns:
            List of dates with available recordings
        """
        self.logger.info(f"Checking available dates from {start_date} to {end_date}")
        
        recordings = self.get_camera_recordings(start_date, end_date)
        available_dates = set()
        
        for recording in recordings:
            recording_date = recording['start_time'].date()
            available_dates.add(recording_date)
            
        sorted_dates = sorted(list(available_dates))
        self.logger.info(f"Found recordings for {len(sorted_dates)} dates")
        
        return sorted_dates