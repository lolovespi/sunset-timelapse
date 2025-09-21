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
from sunset_brilliance_score import SunsetBrillianceScore, FrameMetrics


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
            
        # Initialize SBS analyzer for real-time processing
        self.sbs_analyzer = SunsetBrillianceScore()
        self.current_chunk_metrics = []
        self.current_chunk_number = 0
            
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
                # Extract frames from the final concatenated video with SBS analysis
                captured_images = self.extract_frames_from_video_with_sbs(
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
            # Check available disk space before concatenation
            import shutil
            available_space = shutil.disk_usage(tempfile.gettempdir()).free
            total_chunk_size = sum(chunk.stat().st_size for chunk in chunk_paths if chunk.exists())

            if available_space < total_chunk_size * 1.5:  # Need 1.5x space for safety
                self.logger.error(f"Insufficient disk space: {available_space // (1024**3):.1f}GB available, "
                                f"{total_chunk_size * 1.5 // (1024**3):.1f}GB needed for concatenation")
                return None

            self.logger.info(f"Concatenation space check: {available_space // (1024**3):.1f}GB available, "
                           f"{total_chunk_size // (1024**3):.1f}GB input data")

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
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 minutes for large files
            
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
    
    def extract_frames_from_video_with_sbs(self, video_path: Path, start_time: datetime, 
                                          interval_seconds: int) -> List[Path]:
        """
        Extract frames from video with real-time SBS analysis
        Pi-optimized: processes each frame as it's extracted
        
        Args:
            video_path: Path to video file
            start_time: Base time for frame naming and SBS timing
            interval_seconds: Seconds between frames
            
        Returns:
            List of extracted frame paths
        """
        import subprocess
        from datetime import timedelta
        
        extracted_frames = []
        self.sbs_analyzer.reset_performance_stats()
        self.current_chunk_metrics = []
        
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
            chunk_duration_minutes = 15 * 60  # 15 minutes in seconds
            
            self.logger.info(f"Extracting {frame_count} frames with SBS analysis at {interval_seconds}s intervals")
            
            # Create date folder
            paths = self.config.get_storage_paths()
            date_folder = paths['images'] / start_time.strftime('%Y-%m-%d')
            date_folder.mkdir(exist_ok=True)
            
            # Extract frames with real-time SBS analysis
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
                    
                    # Perform real-time SBS analysis on extracted frame
                    self._analyze_frame_sbs(frame_path, i, timestamp_offset)
                    
                    # Check if we need to complete a chunk analysis
                    if timestamp_offset > 0 and timestamp_offset % chunk_duration_minutes == 0:
                        self._complete_chunk_analysis(start_time, timestamp_offset)
                    
                    self.logger.debug(f"Extracted and analyzed frame {i+1}/{frame_count}: {frame_path.name}")
                else:
                    self.logger.warning(f"Failed to extract frame at {timestamp_offset}s")
                    
            # Complete final chunk if there are remaining metrics
            if self.current_chunk_metrics:
                final_offset = frame_count * interval_seconds
                self._complete_chunk_analysis(start_time, final_offset)
            
            # Log SBS performance stats
            perf_stats = self.sbs_analyzer.get_performance_stats()
            if perf_stats.get('status') != 'no_data':
                self.logger.info(f"SBS Analysis Performance: {perf_stats['avg_processing_time_ms']:.1f}ms avg, "
                               f"{perf_stats['frames_processed']} frames, "
                               f"Status: {perf_stats['pi_performance_status']}")
                    
            return extracted_frames
            
        except Exception as e:
            self.logger.error(f"Failed to extract frames with SBS: {e}")
            return extracted_frames
    
    def _analyze_frame_sbs(self, frame_path: Path, frame_number: int, timestamp_offset: float):
        """Analyze single frame for SBS metrics"""
        try:
            # Load frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                self.logger.warning(f"Could not load frame for SBS analysis: {frame_path}")
                return
                
            # Perform SBS analysis
            frame_metrics = self.sbs_analyzer.analyze_frame(
                frame, frame_number, timestamp_offset
            )
            
            if frame_metrics:
                self.current_chunk_metrics.append(frame_metrics)
                
        except Exception as e:
            self.logger.error(f"SBS frame analysis failed: {e}")
    
    def _complete_chunk_analysis(self, start_time: datetime, current_offset: float):
        """Complete analysis for a 15-minute chunk"""
        if not self.current_chunk_metrics:
            return
            
        try:
            # Calculate sunset offset for golden hour bonus
            from sunset_calculator import SunsetCalculator
            sunset_calc = SunsetCalculator()
            sunset_time = sunset_calc.get_sunset_time(start_time.date())
            chunk_time = start_time + timedelta(seconds=current_offset - (15 * 60))  # Middle of chunk
            sunset_offset_minutes = (chunk_time - sunset_time).total_seconds() / 60
            
            # Analyze chunk
            chunk_metrics = self.sbs_analyzer.analyze_chunk(
                self.current_chunk_metrics, 
                self.current_chunk_number, 
                sunset_offset_minutes
            )
            
            # Save chunk analysis
            date_str = start_time.strftime('%Y-%m-%d')
            self.sbs_analyzer.save_chunk_analysis(chunk_metrics, date_str)
            
            self.logger.info(f"Completed SBS analysis for chunk {self.current_chunk_number}: "
                           f"Score {chunk_metrics.brilliance_score:.1f}, "
                           f"{len(self.current_chunk_metrics)} frames")
            
            # Reset for next chunk
            self.current_chunk_metrics = []
            self.current_chunk_number += 1
            
        except Exception as e:
            self.logger.error(f"Chunk SBS analysis failed: {e}")
        
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

    def connect_onvif(self) -> bool:
        """
        Connect to camera via ONVIF for advanced features like replay
        
        Returns:
            True if ONVIF connection successful, False otherwise
        """
        if not ONVIFCamera:
            self.logger.error("ONVIF library not available")
            return False
            
        try:
            self.logger.info(f"Connecting to camera via ONVIF at {self.ip}:{self.onvif_port}")
            
            # Create ONVIF camera connection
            self.camera = ONVIFCamera(
                self.ip, 
                self.onvif_port, 
                self.username, 
                self.password
            )
            
            # Test connection by getting device information
            device_service = self.camera.create_devicemgmt_service()
            device_info = device_service.GetDeviceInformation()
            
            self.logger.info(f"ONVIF connected - {device_info.Manufacturer} {device_info.Model}")
            
            # Get media service for stream info
            self.media_service = self.camera.create_media_service()
            
            self._connected = True
            return True
            
        except Exception as e:
            self.logger.error(f"ONVIF connection failed: {e}")
            self._connected = False
            return False

    def get_recordings(self, start_time: datetime, end_time: datetime) -> List[dict]:
        """
        Search for recorded videos in the specified time range via ONVIF
        
        Args:
            start_time: Start time for search
            end_time: End time for search
            
        Returns:
            List of recording information dictionaries
        """
        if not self.camera:
            self.logger.error("Camera not connected via ONVIF")
            return []
            
        try:
            # Create search service
            search_service = self.camera.create_search_service()
            
            # Create search parameters
            search_params = {
                'StreamType': 'RtspUnicast',
                'Scope': {
                    'IncludeRecordings': True,
                    'RecordingInformationFilter': {}
                },
                'MaxMatches': 100,
                'StartPoint': start_time.isoformat(),
                'EndPoint': end_time.isoformat()
            }
            
            self.logger.info(f"Searching for recordings from {start_time} to {end_time}")
            
            # Find recordings
            search_result = search_service.FindRecordings(search_params)
            
            recordings = []
            if hasattr(search_result, 'RecordingInformation'):
                for recording in search_result.RecordingInformation:
                    recordings.append({
                        'token': recording.RecordingToken,
                        'start_time': recording.EarliestRecording,
                        'end_time': recording.LatestRecording,
                        'source': recording.Source
                    })
                    
            self.logger.info(f"Found {len(recordings)} recordings")
            return recordings
            
        except Exception as e:
            self.logger.error(f"Failed to search recordings: {e}")
            return []

    def download_recording(self, recording_token: str, save_path: Path) -> bool:
        """
        Download a recorded video via ONVIF replay
        
        Args:
            recording_token: Token identifying the recording
            save_path: Path where to save the downloaded video
            
        Returns:
            True if download successful, False otherwise
        """
        if not self.camera:
            self.logger.error("Camera not connected via ONVIF")
            return False
            
        try:
            # Create replay service
            replay_service = self.camera.create_replay_service()
            
            # Set up stream configuration
            stream_setup = {
                'Stream': 'RTP-Unicast',
                'Transport': {
                    'Protocol': 'RTSP'
                }
            }
            
            # Get replay URI
            replay_params = {
                'RecordingToken': recording_token,
                'StreamSetup': stream_setup
            }
            
            replay_result = replay_service.GetReplayUri(replay_params)
            replay_uri = replay_result.Uri
            
            self.logger.info(f"Got replay URI: {self._sanitize_rtsp_url_for_logging(replay_uri)}")
            
            # Download video using ffmpeg
            import subprocess
            
            cmd = [
                'ffmpeg', '-y',
                '-i', replay_uri,
                '-c', 'copy',  # Stream copy to avoid re-encoding
                '-avoid_negative_ts', 'make_zero',
                str(save_path)
            ]
            
            # Create sanitized command for logging
            cmd_safe = cmd.copy()
            cmd_safe[cmd_safe.index(replay_uri)] = self._sanitize_rtsp_url_for_logging(replay_uri)
            self.logger.info(f"Downloading recording: {' '.join(cmd_safe)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                self.logger.info(f"Successfully downloaded recording to {save_path}")
                return True
            else:
                self.logger.error(f"Failed to download recording: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Recording download timed out")
            return False
        except Exception as e:
            self.logger.error(f"Failed to download recording: {e}")
            return False

    def download_recordings_by_time(self, start_time: datetime, end_time: datetime, 
                                   save_dir: Path) -> List[Path]:
        """
        Download all recordings in a time range
        
        Args:
            start_time: Start time for recordings
            end_time: End time for recordings  
            save_dir: Directory to save downloaded videos
            
        Returns:
            List of paths to downloaded video files
        """
        downloaded_files = []
        
        # Ensure ONVIF connection
        if not self.camera and not self.connect_onvif():
            self.logger.error("Failed to connect via ONVIF for recording download")
            return downloaded_files
            
        # Search for recordings
        recordings = self.get_recordings(start_time, end_time)
        
        if not recordings:
            self.logger.warning("No recordings found in specified time range")
            return downloaded_files
            
        # Download each recording
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, recording in enumerate(recordings):
            # Generate filename
            filename = f"recording_{start_time.strftime('%Y%m%d_%H%M')}_{i+1}.mp4"
            save_path = save_dir / filename
            
            if self.download_recording(recording['token'], save_path):
                downloaded_files.append(save_path)
            else:
                self.logger.warning(f"Failed to download recording {i+1}")
                
        self.logger.info(f"Downloaded {len(downloaded_files)}/{len(recordings)} recordings")
        return downloaded_files

    def download_reolink_recording(self, start_time: datetime, end_time: datetime, 
                                   save_path: Path) -> bool:
        """
        Download recorded video via Reolink HTTP API
        
        Args:
            start_time: Start time for recording
            end_time: End time for recording
            save_path: Path where to save the downloaded video
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Build Reolink API URL for playback
            base_url = f"http://{self.ip}/cgi-bin/api.cgi"
            
            # Format times for Reolink API
            start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Reolink playback parameters
            params = {
                'cmd': 'Playback',
                'user': self.username,
                'password': self.password,
                'channel': 0,
                'streamType': 'main',  # or 'sub' for lower quality
                'startTime': start_str,
                'endTime': end_str
            }
            
            self.logger.info(f"Downloading Reolink recording from {start_str} to {end_str}")
            self.logger.info(f"API URL: http://{self.ip}/cgi-bin/api.cgi?cmd=Playback&...")
            
            # Make request with streaming (disable SSL verification for self-signed certs)
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            response = requests.get(base_url, params=params, stream=True, timeout=1800, 
                                  verify=False, allow_redirects=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            self.logger.info(f"Response content-type: {content_type}")
            
            if 'video' not in content_type.lower() and 'application' not in content_type.lower():
                self.logger.error(f"Unexpected content type: {content_type}")
                # Try to read response as text to see error message
                try:
                    error_text = response.text[:500]
                    self.logger.error(f"Response: {error_text}")
                except:
                    pass
                return False
            
            # Download the video file
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            total_size = 0
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
                        
                        # Log progress every 10MB
                        if total_size % (10 * 1024 * 1024) == 0:
                            self.logger.info(f"Downloaded {total_size // (1024 * 1024)}MB...")
            
            self.logger.info(f"Successfully downloaded {total_size // (1024 * 1024)}MB to {save_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"HTTP request failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to download Reolink recording: {e}")
            return False

    def list_reolink_recordings(self, start_time: datetime, end_time: datetime) -> List[dict]:
        """
        List available recordings via Reolink HTTP API
        
        Args:
            start_time: Start time for search
            end_time: End time for search
            
        Returns:
            List of recording information dictionaries
        """
        try:
            # Build Reolink API URL for search
            base_url = f"http://{self.ip}/cgi-bin/api.cgi"
            
            # Format times for Reolink API
            start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # RLC-810WA API formats - try JSON POST method first
            import json
            
            # Modern Reolink API uses JSON POST requests
            json_payload = [{
                "cmd": "Search",
                "action": 0,
                "param": {
                    "Search": {
                        "channel": 0,
                        "onlyStatus": 1,
                        "streamType": "main",
                        "StartTime": {
                            "year": start_time.year,
                            "mon": start_time.month,
                            "day": start_time.day,
                            "hour": start_time.hour,
                            "min": start_time.minute,
                            "sec": start_time.second
                        },
                        "EndTime": {
                            "year": end_time.year,
                            "mon": end_time.month,
                            "day": end_time.day,
                            "hour": end_time.hour,
                            "min": end_time.minute,
                            "sec": end_time.second
                        }
                    }
                }
            }]
            
            # Fallback GET parameter formats - try different password encodings
            import urllib.parse
            
            params_list = [
                # Standard encoding
                {
                    'cmd': 'Search',
                    'user': self.username,
                    'password': self.password,
                    'channel': 0,
                    'onlyStatus': 1,
                    'streamType': 'main',
                    'startTime': start_str,
                    'endTime': end_str
                },
                # No encoding for special characters
                {
                    'cmd': 'Search',
                    'user': self.username,
                    'password': urllib.parse.unquote(self.password),
                    'channel': 0,
                    'onlyStatus': 1,
                    'streamType': 'main',
                    'startTime': start_str,
                    'endTime': end_str
                },
                # Alternative command
                {
                    'cmd': 'GetRecFiles', 
                    'user': self.username,
                    'password': self.password,
                    'channel': 0,
                    'startTime': start_str,
                    'endTime': end_str
                },
                # HTTP Basic Auth method
                {
                    'cmd': 'Search',
                    'channel': 0,
                    'onlyStatus': 1,
                    'streamType': 'main',
                    'startTime': start_str,
                    'endTime': end_str
                }
            ]
            
            self.logger.info(f"Searching Reolink recordings from {start_str} to {end_str}")
            
            # Disable SSL warnings for self-signed certificates
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # First try modern JSON POST API
            try:
                self.logger.info("Trying modern JSON POST API format")
                
                # Login first to get token (for newer firmware)
                login_payload = [{
                    "cmd": "Login",
                    "action": 0,
                    "param": {
                        "User": {
                            "userName": self.username,
                            "password": self.password
                        }
                    }
                }]
                
                # Try login
                response = requests.post(base_url, json=login_payload, timeout=30, 
                                       verify=False, allow_redirects=True)
                
                if response.status_code == 200:
                    try:
                        login_result = response.json()
                        if login_result and len(login_result) > 0 and login_result[0].get('value'):
                            token = login_result[0]['value']['Token']['name']
                            self.logger.info(f"Got authentication token: {token[:10]}...")
                            
                            # Now search with token
                            json_payload[0]['param']['Search']['token'] = token
                            response = requests.post(base_url, json=json_payload, timeout=60,
                                                   verify=False, allow_redirects=True)
                            
                            if response.status_code == 200:
                                try:
                                    result = response.json()
                                    self.logger.info("Got JSON response from modern API")
                                    
                                    recordings = []
                                    if result and len(result) > 0 and 'value' in result[0]:
                                        search_result = result[0]['value']['SearchResult']
                                        if 'File' in search_result:
                                            for file_info in search_result['File']:
                                                recordings.append({
                                                    'start_time': f"{file_info['StartTime']['year']:04d}-{file_info['StartTime']['mon']:02d}-{file_info['StartTime']['day']:02d} {file_info['StartTime']['hour']:02d}:{file_info['StartTime']['min']:02d}:{file_info['StartTime']['sec']:02d}",
                                                    'end_time': f"{file_info['EndTime']['year']:04d}-{file_info['EndTime']['mon']:02d}-{file_info['EndTime']['day']:02d} {file_info['EndTime']['hour']:02d}:{file_info['EndTime']['min']:02d}:{file_info['EndTime']['sec']:02d}",
                                                    'file_size': file_info.get('size', 0),
                                                    'file_name': file_info.get('name', ''),
                                                    'type': file_info.get('type', 'unknown')
                                                })
                                    
                                    if recordings:
                                        self.logger.info(f"Found {len(recordings)} recordings with modern API")
                                        return recordings
                                    else:
                                        self.logger.info("Modern API worked but returned no recordings")
                                        
                                except ValueError:
                                    self.logger.warning("Modern API: Failed to parse JSON response")
                        else:
                            self.logger.warning("Modern API: Login failed")
                    except ValueError:
                        self.logger.warning("Modern API: Login response not JSON")
                else:
                    self.logger.warning(f"Modern API: Login returned status {response.status_code}")
                    
            except Exception as e:
                self.logger.warning(f"Modern API failed: {e}")
            
            # Fallback to GET parameter formats
            for i, params in enumerate(params_list):
                try:
                    self.logger.info(f"Trying API format {i+1}: {params.get('cmd', params.get('action'))}")
                    
                    # Try both parameter-based auth and HTTP Basic Auth
                    if 'password' in params:
                        # Parameter-based auth (current approach)
                        response = requests.get(base_url, params=params, timeout=60, 
                                              verify=False, allow_redirects=True)
                    else:
                        # HTTP Basic Auth approach
                        from requests.auth import HTTPBasicAuth
                        auth_params = {k: v for k, v in params.items() if k not in ['user', 'password']}
                        response = requests.get(base_url, params=auth_params, 
                                              auth=HTTPBasicAuth(self.username, self.password),
                                              timeout=60, verify=False, allow_redirects=True)
                    response.raise_for_status()
                    
                    # Check if response looks like JSON
                    content_type = response.headers.get('content-type', '').lower()
                    if 'json' in content_type or response.text.strip().startswith(('{', '[')):
                        try:
                            result = response.json()
                            self.logger.info(f"Got JSON response with format {i+1}")
                            
                            recordings = []
                            
                            # Debug: log the actual response structure
                            self.logger.debug(f"Response structure: {result}")
                            
                            # Check for authentication errors first
                            if isinstance(result, list) and len(result) > 0:
                                if 'error' in result[0]:
                                    error_info = result[0]['error']
                                    if error_info.get('rspCode') == -502:
                                        self.logger.error(f"Authentication failed: {error_info.get('detail', 'Unknown error')}")
                                        # Try without URL encoding special characters
                                        continue
                            
                            # Try different response structures
                            if result and 'value' in result and 'SearchResult' in result['value']:
                                search_results = result['value']['SearchResult']
                                for item in search_results:
                                    recordings.append({
                                        'start_time': item.get('StartTime'),
                                        'end_time': item.get('EndTime'),
                                        'file_size': item.get('Size', 0),
                                        'file_name': item.get('FileName', ''),
                                        'type': item.get('Type', 'unknown')
                                    })
                            elif isinstance(result, list):
                                # Direct list format - also try to parse different structures
                                for item in result:
                                    # Check if this has a 'value' key with recording data
                                    if 'value' in item and 'SearchResult' in item['value']:
                                        search_result = item['value']['SearchResult']
                                        if 'File' in search_result:
                                            for file_info in search_result['File']:
                                                recordings.append({
                                                    'start_time': f"{file_info['StartTime']['year']:04d}-{file_info['StartTime']['mon']:02d}-{file_info['StartTime']['day']:02d} {file_info['StartTime']['hour']:02d}:{file_info['StartTime']['min']:02d}:{file_info['StartTime']['sec']:02d}",
                                                    'end_time': f"{file_info['EndTime']['year']:04d}-{file_info['EndTime']['mon']:02d}-{file_info['EndTime']['day']:02d} {file_info['EndTime']['hour']:02d}:{file_info['EndTime']['min']:02d}:{file_info['EndTime']['sec']:02d}",
                                                    'file_size': file_info.get('size', 0),
                                                    'file_name': file_info.get('name', ''),
                                                    'type': file_info.get('type', 'unknown')
                                                })
                                        elif 'Status' in search_result:
                                            # Status-only response, no actual files
                                            status = search_result['Status']
                                            if status.get('rspCode') == 200:
                                                self.logger.info("Search successful but no recordings in time range")
                                    else:
                                        # Simple format
                                        recordings.append({
                                            'start_time': item.get('start_time', item.get('StartTime')),
                                            'end_time': item.get('end_time', item.get('EndTime')),
                                            'file_size': item.get('size', item.get('Size', 0)),
                                            'file_name': item.get('name', item.get('FileName', '')),
                                            'type': item.get('type', item.get('Type', 'unknown'))
                                        })
                            elif result and 'SearchResult' in result:
                                # Direct SearchResult format
                                search_result = result['SearchResult']
                                if 'File' in search_result:
                                    for file_info in search_result['File']:
                                        recordings.append({
                                            'start_time': f"{file_info['StartTime']['year']:04d}-{file_info['StartTime']['mon']:02d}-{file_info['StartTime']['day']:02d} {file_info['StartTime']['hour']:02d}:{file_info['StartTime']['min']:02d}:{file_info['StartTime']['sec']:02d}",
                                            'end_time': f"{file_info['EndTime']['year']:04d}-{file_info['EndTime']['mon']:02d}-{file_info['EndTime']['day']:02d} {file_info['EndTime']['hour']:02d}:{file_info['EndTime']['min']:02d}:{file_info['EndTime']['sec']:02d}",
                                            'file_size': file_info.get('size', 0),
                                            'file_name': file_info.get('name', ''),
                                            'type': file_info.get('type', 'unknown')
                                        })
                            
                            if recordings:
                                self.logger.info(f"Found {len(recordings)} recordings with format {i+1}")
                                return recordings
                            else:
                                self.logger.info(f"Format {i+1} worked but returned no recordings")
                                
                        except ValueError:
                            self.logger.warning(f"Format {i+1}: Response looked like JSON but failed to parse")
                            continue
                    else:
                        self.logger.warning(f"Format {i+1}: Got non-JSON response")
                        if i == 0:  # Show response details for first attempt only
                            response_text = response.text[:200]
                            self.logger.debug(f"Response preview: {response_text}")
                        continue
                        
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"Format {i+1} failed: {e}")
                    continue
            
            self.logger.info("No recordings found with any API format")
            return []
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"HTTP request failed: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to list Reolink recordings: {e}")
            return []

    def download_reolink_recordings_by_time(self, start_time: datetime, end_time: datetime, 
                                           save_dir: Path) -> List[Path]:
        """
        Download Reolink recordings in a time range
        
        Args:
            start_time: Start time for recordings
            end_time: End time for recordings  
            save_dir: Directory to save downloaded videos
            
        Returns:
            List of paths to downloaded video files
        """
        downloaded_files = []
        
        # List available recordings first
        recordings = self.list_reolink_recordings(start_time, end_time)
        
        if not recordings:
            self.logger.warning("No Reolink recordings found in specified time range")
            # Try downloading the entire time range as one file
            filename = f"reolink_recording_{start_time.strftime('%Y%m%d_%H%M')}_{end_time.strftime('%H%M')}.mp4"
            save_path = save_dir / filename
            
            if self.download_reolink_recording(start_time, end_time, save_path):
                downloaded_files.append(save_path)
        else:
            # Download each individual recording
            save_dir.mkdir(parents=True, exist_ok=True)
            
            for i, recording in enumerate(recordings):
                # Parse recording times
                try:
                    rec_start = datetime.strptime(recording['start_time'], "%Y-%m-%d %H:%M:%S")
                    rec_end = datetime.strptime(recording['end_time'], "%Y-%m-%d %H:%M:%S")
                except:
                    self.logger.warning(f"Could not parse recording times for recording {i+1}")
                    continue
                
                # Generate filename
                filename = f"reolink_{rec_start.strftime('%Y%m%d_%H%M%S')}_{rec_end.strftime('%H%M%S')}.mp4"
                save_path = save_dir / filename
                
                if self.download_reolink_recording(rec_start, rec_end, save_path):
                    downloaded_files.append(save_path)
                else:
                    self.logger.warning(f"Failed to download recording {i+1}")
        
        self.logger.info(f"Downloaded {len(downloaded_files)} Reolink recordings")
        return downloaded_files

    def download_reolink_recordings_official_api(self, start_time: datetime, end_time: datetime, 
                                                save_dir: Path) -> List[Path]:
        """
        Download recordings using official Reolink API library
        
        Args:
            start_time: Start time for recordings
            end_time: End time for recordings  
            save_dir: Directory to save downloaded videos
            
        Returns:
            List of paths to downloaded video files
        """
        downloaded_files = []
        
        try:
            from reolinkapi import Camera as ReoCamera
            
            # Create Reolink API camera instance
            reo_camera = ReoCamera(self.ip, self.username, self.password)
            
            self.logger.info(f"Searching for recordings from {start_time} to {end_time} using Reolink API")
            
            # Get motion files in the time range
            motion_files = reo_camera.get_motion_files(
                start=start_time, 
                end=end_time, 
                streamtype='main',
                channel=0
            )
            
            if not motion_files:
                self.logger.warning("No motion files found in specified time range")
                return downloaded_files
                
            self.logger.info(f"Found {len(motion_files)} motion recording files")
            
            # Create save directory
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Download each file
            for i, file_info in enumerate(motion_files):
                if isinstance(file_info, dict) and 'filename' in file_info:
                    filename = file_info['filename']
                    start_dt = file_info.get('start', datetime.now())
                    end_dt = file_info.get('end', datetime.now())
                    
                    # Generate local filename
                    safe_filename = filename.replace('/', '_').replace('\\', '_')
                    local_filename = f"{start_dt.strftime('%Y%m%d_%H%M%S')}_{safe_filename.split('_')[-1]}"
                    save_path = save_dir / local_filename
                    
                    self.logger.info(f"Downloading file {i+1}/{len(motion_files)}: {filename}")
                    
                    try:
                        success = reo_camera.get_file(
                            filename=filename,
                            output_path=str(save_path),
                            method='Playback'
                        )
                        
                        if success and save_path.exists():
                            file_size = save_path.stat().st_size
                            self.logger.info(f"✓ Downloaded {file_size // (1024*1024)}MB: {save_path.name}")
                            downloaded_files.append(save_path)
                        else:
                            self.logger.warning(f"Failed to download {filename}")
                            
                    except Exception as download_error:
                        self.logger.error(f"Download error for {filename}: {download_error}")
                        
            self.logger.info(f"Successfully downloaded {len(downloaded_files)}/{len(motion_files)} files")
            return downloaded_files
            
        except ImportError:
            self.logger.error("Reolink API library not installed. Install with: pip install git+https://github.com/ReolinkCameraAPI/reolinkapipy.git")
            return downloaded_files
        except Exception as e:
            self.logger.error(f"Failed to download recordings via Reolink API: {e}")
            return downloaded_files