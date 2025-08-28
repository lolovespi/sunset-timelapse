"""
Camera Interface for Reolink RLC810-WA
Uses direct RTSP for camera control and image capture
"""

import logging
import time
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
import requests
from requests.auth import HTTPDigestAuth
from PIL import Image
import cv2
import urllib3

# Disable SSL warnings for camera's self-signed certificate
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from onvif import ONVIFCamera
    from onvif.exceptions import ONVIFError
except ImportError:
    logging.warning("ONVIF library not available. Camera control will be limited.")
    ONVIFCamera = None
    ONVIFError = Exception

from config_manager import get_config


class CameraInterface:
    """Interface for controlling Reolink camera via RTSP
    
    Security note: All RTSP URLs containing credentials are sanitized before logging.
    """
    
    def __init__(self):
        """Initialize camera interface"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.camera = None
        self.media_service = None
        self.imaging_service = None
        self._connected = False
        
        # Camera settings from config
        self.ip = self.config.get('camera.ip')
        self.onvif_port = self.config.get('camera.onvif_port', 8000)
        self.rtsp_port = self.config.get('camera.rtsp_port', 554)
        
        # Get credentials securely
        try:
            self.username, self.password = self.config.get_camera_credentials()
        except ValueError as e:
            self.logger.error(f"Failed to get camera credentials: {e}")
            raise
            
        self.logger.info(f"Camera interface initialized for {self.ip}")
        
    def _sanitize_rtsp_url_for_logging(self, rtsp_url: str) -> str:
        """Remove credentials from RTSP URL for safe logging"""
        if '@' in rtsp_url:
            # Replace credentials with ***
            protocol, rest = rtsp_url.split('://', 1)
            if '@' in rest:
                _, after_creds = rest.split('@', 1)
                return f"{protocol}://***:***@{after_creds}"
        return rtsp_url
        
    def connect(self) -> bool:
        """
        Test RTSP connection (no ONVIF needed)
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info(f"Testing RTSP connection to camera at {self.ip}")
            
            # Build direct RTSP URL
            rtsp_url = f"rtsp://{self.username}:{self.password}@{self.ip}:{self.rtsp_port}/h264Preview_01_main"
            
            import cv2
            cap = cv2.VideoCapture(rtsp_url)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                self.logger.info("RTSP connection successful")
                self._connected = True
                return True
            else:
                self.logger.error("Could not capture frame from RTSP")
                self._connected = False
                return False
                
        except Exception as e:
            self.logger.error(f"RTSP connection failed: {e}")
            self._connected = False
            return False
            
    def disconnect(self):
        """Disconnect from camera"""
        self._connected = False
        self.logger.info("Disconnected from camera")
        
    def is_connected(self) -> bool:
        """Check if camera is connected"""
        return self._connected
        
    def get_stream_uri(self, profile_index: int = 0) -> Optional[str]:
        """
        Get direct RTSP stream URI
        
        Returns:
            RTSP URI for Reolink camera
        """
        return f"rtsp://{self.username}:{self.password}@{self.ip}:{self.rtsp_port}/h264Preview_01_main"
            
            
    def capture_snapshot_rtsp(self, save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Capture snapshot from RTSP stream using OpenCV
        
        Args:
            save_path: Path to save image, if None will auto-generate
            
        Returns:
            Path to saved image or None if failed
        """
        try:
            # Build direct RTSP URL for Reolink
            rtsp_uri = f"rtsp://{self.username}:{self.password}@{self.ip}:{self.rtsp_port}/h264Preview_01_main"
                           
            self.logger.debug("Capturing frame from RTSP stream")
            
            # Open video capture
            cap = cv2.VideoCapture(rtsp_uri)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frame
            
            if not cap.isOpened():
                self.logger.error("Failed to open RTSP stream")
                return None
                
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                self.logger.error("Failed to capture frame from RTSP stream")
                return None
                
            # Generate save path if not provided
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                paths = self.config.get_storage_paths()
                save_path = paths['images'] / f"snapshot_{timestamp}.jpg"
                
            # Save image
            quality = self.config.get('capture.image_quality', 90)
            cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            self.logger.debug(f"RTSP snapshot saved to {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Failed to capture snapshot via RTSP: {e}")
            return None
            
    def capture_snapshot(self, save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Capture snapshot using RTSP method
        
        Args:
            save_path: Path to save image, if None will auto-generate
            
        Returns:
            Path to saved image or None if failed
        """
        # Use RTSP method directly (skip ONVIF)
        return self.capture_snapshot_rtsp(save_path)
        
    def capture_video_sequence(self, start_time: datetime, end_time: datetime, 
                              interval_seconds: int = 5) -> List[Path]:
        """
        Capture video via RTSP and extract frames at specified intervals using chunked recording
        
        Args:
            start_time: When to start capturing
            end_time: When to stop capturing  
            interval_seconds: Seconds between extracted frames
            
        Returns:
            List of paths to extracted frame images
        """
        import subprocess
        import tempfile
        from pathlib import Path
        from datetime import timedelta
        
        captured_images = []
        chunk_duration_minutes = 15  # Record in 15-minute chunks for reliability
        
        try:
            # Calculate total duration and chunk strategy
            total_duration = (end_time - start_time).total_seconds()
            chunk_duration = chunk_duration_minutes * 60  # Convert to seconds
            chunks_needed = int(total_duration / chunk_duration) + (1 if total_duration % chunk_duration > 0 else 0)
            
            self.logger.info(f"Recording {total_duration:.0f} seconds in {chunks_needed} chunks of {chunk_duration_minutes} minutes each")
            
            # Build direct RTSP URL for Reolink
            rtsp_url = f"rtsp://{self.username}:{self.password}@{self.ip}:{self.rtsp_port}/h264Preview_01_main"
            
            chunk_videos = []
            current_time = start_time
            
            # Record each chunk
            for chunk_idx in range(chunks_needed):
                chunk_end_time = min(current_time + timedelta(seconds=chunk_duration), end_time)
                actual_chunk_duration = (chunk_end_time - current_time).total_seconds()
                
                if actual_chunk_duration <= 0:
                    break
                    
                self.logger.info(f"Recording chunk {chunk_idx + 1}/{chunks_needed}: {actual_chunk_duration:.0f} seconds")
                
                # Create temp video file for this chunk
                with tempfile.NamedTemporaryFile(suffix=f'_chunk_{chunk_idx}.mp4', delete=False) as temp_video:
                    temp_video_path = Path(temp_video.name)
            
                # Record RTSP stream for this chunk using ffmpeg with stream copy
                cmd = [
                    'ffmpeg', '-y',
                    '-rtsp_transport', 'tcp',
                    '-i', rtsp_url,
                    '-t', str(int(actual_chunk_duration)),
                    '-c', 'copy',  # Stream copy - much faster than re-encoding
                    '-avoid_negative_ts', 'make_zero',
                    '-fflags', '+genpts',  # Generate presentation timestamps
                    '-use_wallclock_as_timestamps', '1',  # Use system time for timestamps
                    str(temp_video_path)
                ]
                
                # Create sanitized command for logging (hide credentials)
                cmd_safe = cmd.copy()
                cmd_safe[cmd_safe.index(rtsp_url)] = self._sanitize_rtsp_url_for_logging(rtsp_url)
                self.logger.info(f"Starting RTSP recording chunk {chunk_idx + 1}: {' '.join(cmd_safe)}")
                
                # Set timeout with buffer for this chunk only
                buffer_time = max(60, actual_chunk_duration * 0.1)  # 10% buffer, min 60s
                timeout_seconds = actual_chunk_duration + buffer_time
                self.logger.info(f"Chunk timeout: {timeout_seconds:.0f} seconds ({buffer_time:.0f}s buffer)")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
                
                if result.returncode != 0:
                    self.logger.error(f"FFmpeg recording failed for chunk {chunk_idx + 1}: {result.stderr}")
                    # Try to continue with other chunks
                    temp_video_path.unlink(missing_ok=True)
                else:
                    self.logger.info(f"RTSP recording chunk {chunk_idx + 1} completed: {temp_video_path}")
                    chunk_videos.append(temp_video_path)
                
                current_time = chunk_end_time
            
            # If we have no successful chunks, return empty
            if not chunk_videos:
                self.logger.error("No successful video chunks recorded")
                return captured_images
            
            # Concatenate chunks into single video if multiple chunks
            if len(chunk_videos) > 1:
                final_video_path = self._concatenate_video_chunks(chunk_videos)
            else:
                final_video_path = chunk_videos[0]
            
            if final_video_path and final_video_path.exists():
                # Extract frames from the final concatenated video
                captured_images = self.extract_frames_from_video(
                    final_video_path, start_time, interval_seconds
                )
                
                # Clean up all chunk videos and final video
                for chunk_path in chunk_videos:
                    chunk_path.unlink(missing_ok=True)
                if len(chunk_videos) > 1:  # Don't double-delete if single chunk
                    final_video_path.unlink(missing_ok=True)
            
            return captured_images
            
        except subprocess.TimeoutExpired as e:
            self.logger.error(f"FFmpeg chunk recording timed out")
            self.logger.error("This may indicate network connectivity issues or camera problems")
            # Preserve any successful chunks and the failed chunk
            if 'chunk_videos' in locals():
                import shutil
                for i, chunk_path in enumerate(chunk_videos):
                    if chunk_path.exists():
                        try:
                            paths = self.config.get_storage_paths()
                            failed_filename = f"FAILED_CHUNK_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                            failed_path = paths['videos'] / failed_filename
                            shutil.move(str(chunk_path), str(failed_path))
                            self.logger.info(f"Failed chunk preserved: {failed_path}")
                        except Exception as move_error:
                            self.logger.error(f"Failed to preserve chunk: {move_error}")
                            chunk_path.unlink(missing_ok=True)
            return captured_images
        except Exception as e:
            self.logger.error(f"Failed to capture video sequence: {e}")
            # Clean up any partial chunks
            if 'chunk_videos' in locals():
                for chunk_path in chunk_videos:
                    chunk_path.unlink(missing_ok=True)
            return captured_images
    
    def _concatenate_video_chunks(self, chunk_paths: List[Path]) -> Path:
        """
        Concatenate multiple video chunks into a single video file
        
        Args:
            chunk_paths: List of paths to video chunks
            
        Returns:
            Path to concatenated video file
        """
        import subprocess
        import tempfile
        
        if not chunk_paths:
            return None
            
        if len(chunk_paths) == 1:
            return chunk_paths[0]
        
        try:
            # Create temp file for concatenated video
            with tempfile.NamedTemporaryFile(suffix='_concatenated.mp4', delete=False) as temp_concat:
                concat_path = Path(temp_concat.name)
            
            # Create concat file list for FFmpeg
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as concat_file:
                for chunk_path in chunk_paths:
                    concat_file.write(f"file '{chunk_path}'\n")
                concat_file_path = Path(concat_file.name)
            
            self.logger.info(f"Concatenating {len(chunk_paths)} video chunks")
            
            # Use FFmpeg concat demuxer for lossless concatenation
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file_path),
                '-c', 'copy',  # Stream copy for lossless concatenation
                str(concat_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Clean up concat file list
            concat_file_path.unlink()
            
            if result.returncode != 0:
                self.logger.error(f"Video concatenation failed: {result.stderr}")
                concat_path.unlink(missing_ok=True)
                return None
                
            self.logger.info(f"Video chunks concatenated successfully: {concat_path}")
            return concat_path
            
        except Exception as e:
            self.logger.error(f"Failed to concatenate video chunks: {e}")
            return None
    
    def extract_frames_from_video(self, video_path: Path, start_time: datetime, 
                                 interval_seconds: int) -> List[Path]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to video file
            start_time: Base time for frame naming
            interval_seconds: Seconds between frames
            
        Returns:
            List of extracted frame paths
        """
        import subprocess
        
        extracted_frames = []
        
        try:
            # Get video duration
            probe_cmd = [
                'ffprobe', '-v', 'quiet',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                str(video_path)
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error("Failed to probe video duration")
                return extracted_frames
                
            duration = float(result.stdout.strip())
            frame_count = int(duration / interval_seconds)
            
            self.logger.info(f"Extracting {frame_count} frames at {interval_seconds}s intervals")
            
            # Create date folder
            paths = self.config.get_storage_paths()
            date_folder = paths['images'] / start_time.strftime('%Y-%m-%d')
            date_folder.mkdir(exist_ok=True)
            
            # Extract frames
            for i in range(frame_count):
                timestamp_offset = i * interval_seconds
                frame_time = start_time + timedelta(seconds=timestamp_offset)
                
                # Generate filename
                timestamp_str = frame_time.strftime('%Y%m%d_%H%M%S')
                frame_path = date_folder / f"img_{timestamp_str}.jpg"
                
                # Extract frame at specific time
                extract_cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(timestamp_offset),
                    '-i', str(video_path),
                    '-vframes', '1',
                    '-q:v', '2',  # High quality
                    str(frame_path)
                ]
                
                result = subprocess.run(extract_cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and frame_path.exists():
                    extracted_frames.append(frame_path)
                    self.logger.info(f"Extracted frame {i+1}/{frame_count}: {frame_path.name}")
                else:
                    self.logger.warning(f"Failed to extract frame at {timestamp_offset}s")
                    
            return extracted_frames
            
        except Exception as e:
            self.logger.error(f"Failed to extract frames: {e}")
            return extracted_frames
        
    def capture_sequence(self, start_time: datetime, end_time: datetime, 
                        interval_seconds: int) -> List[Path]:
        """
        Capture a sequence of images at specified intervals
        
        Args:
            start_time: When to start capturing
            end_time: When to stop capturing
            interval_seconds: Seconds between captures
            
        Returns:
            List of paths to captured images
        """
        captured_images = []
        current_time = start_time
        
        self.logger.info(f"Starting capture sequence from {start_time} to {end_time}")
        self.logger.info(f"Interval: {interval_seconds} seconds")
        
        # Ensure we're connected
        if not self.is_connected():
            if not self.connect():
                self.logger.error("Failed to connect to camera for sequence capture")
                return captured_images
                
        while current_time <= end_time:
            # Wait until it's time for the next capture
            now = datetime.now()
            if now < current_time:
                sleep_time = (current_time - now).total_seconds()
                if sleep_time > 0:
                    self.logger.debug(f"Waiting {sleep_time:.1f} seconds for next capture")
                    time.sleep(sleep_time)
                    
            # Generate filename with timestamp
            timestamp = current_time.strftime('%Y%m%d_%H%M%S')
            paths = self.config.get_storage_paths()
            date_folder = paths['images'] / current_time.strftime('%Y-%m-%d')
            date_folder.mkdir(exist_ok=True)
            
            image_path = date_folder / f"img_{timestamp}.jpg"
            
            # Capture image
            captured_path = self.capture_snapshot(image_path)
            if captured_path:
                captured_images.append(captured_path)
                self.logger.info(f"Captured image {len(captured_images)}: {captured_path.name}")
            else:
                self.logger.warning(f"Failed to capture image at {current_time}")
                
            # Move to next capture time
            current_time += timedelta(seconds=interval_seconds)
            
        self.logger.info(f"Capture sequence completed. {len(captured_images)} images captured.")
        return captured_images
        
    def test_connection(self) -> bool:
        """
        Test camera connection using direct RTSP
        
        Returns:
            True if all tests pass, False otherwise
        """
        self.logger.info("Testing camera connection...")
        
        # Build direct RTSP URL for Reolink  
        rtsp_url = f"rtsp://{self.username}:{self.password}@{self.ip}:{self.rtsp_port}/h264Preview_01_main"
        
        try:
            import cv2
            cap = cv2.VideoCapture(rtsp_url)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                self.logger.info("Camera connection test passed")
                return True
            else:
                self.logger.error("Could not capture frame from RTSP")
                return False
        except Exception as e:
            self.logger.error(f"RTSP test failed: {e}")
            return False