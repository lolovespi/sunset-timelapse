"""
Video Processor
Creates timelapse videos from sequences of images using ffmpeg
"""

import logging
import subprocess
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Tuple
import ffmpeg
from PIL import Image
import json

from config_manager import get_config


class VideoProcessor:
    """Processes images into timelapse videos"""
    
    def __init__(self):
        """Initialize video processor"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Video settings from config
        self.fps = self.config.get('video.fps', 12)
        self.output_format = self.config.get('video.output_format', 'mp4')
        self.codec = self.config.get('video.codec', 'libx264')
        self.bitrate = self.config.get('video.bitrate', '2M')
        self.width = self.config.get('video.resolution.width', 1920)
        self.height = self.config.get('video.resolution.height', 1080)
        self.threads = self.config.get('advanced.ffmpeg_threads', 4)
        
        self.logger.info("Video processor initialized")
        self.logger.info(f"Output settings: {self.width}x{self.height} @ {self.fps}fps, {self.codec}")
        
    def validate_ffmpeg(self) -> bool:
        """
        Check if ffmpeg is available and working
        
        Returns:
            True if ffmpeg is available, False otherwise
        """
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.logger.info("FFmpeg validation passed")
                return True
            else:
                self.logger.error("FFmpeg validation failed")
                return False
        except Exception as e:
            self.logger.error(f"FFmpeg not available: {e}")
            return False
            
    def analyze_images(self, image_paths: List[Path]) -> dict:
        """
        Analyze image sequence for consistency
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary with analysis results
        """
        if not image_paths:
            return {'error': 'No images provided'}
            
        analysis = {
            'total_images': len(image_paths),
            'valid_images': 0,
            'resolutions': {},
            'file_sizes': [],
            'timestamps': [],
            'errors': []
        }
        
        self.logger.info(f"Analyzing {len(image_paths)} images...")
        
        for i, img_path in enumerate(image_paths):
            try:
                if not img_path.exists():
                    analysis['errors'].append(f"File not found: {img_path}")
                    continue
                    
                # Get file info
                file_size = img_path.stat().st_size
                analysis['file_sizes'].append(file_size)
                
                # Get image creation time from filename or file stat
                try:
                    # Try to parse timestamp from filename (format: img_YYYYMMDD_HHMMSS.jpg)
                    name_parts = img_path.stem.split('_')
                    if len(name_parts) >= 3:
                        date_str = name_parts[1] + '_' + name_parts[2]
                        timestamp = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
                        analysis['timestamps'].append(timestamp)
                except:
                    # Fallback to file modification time
                    timestamp = datetime.fromtimestamp(img_path.stat().st_mtime)
                    analysis['timestamps'].append(timestamp)
                    
                # Get image dimensions
                with Image.open(img_path) as img:
                    resolution = f"{img.width}x{img.height}"
                    if resolution in analysis['resolutions']:
                        analysis['resolutions'][resolution] += 1
                    else:
                        analysis['resolutions'][resolution] = 1
                        
                analysis['valid_images'] += 1
                
            except Exception as e:
                analysis['errors'].append(f"Error analyzing {img_path}: {e}")
                
        # Calculate statistics
        if analysis['file_sizes']:
            analysis['avg_file_size'] = sum(analysis['file_sizes']) / len(analysis['file_sizes'])
            analysis['min_file_size'] = min(analysis['file_sizes'])
            analysis['max_file_size'] = max(analysis['file_sizes'])
            
        if analysis['timestamps']:
            analysis['start_time'] = min(analysis['timestamps'])
            analysis['end_time'] = max(analysis['timestamps'])
            analysis['duration'] = analysis['end_time'] - analysis['start_time']
            
        # Find most common resolution
        if analysis['resolutions']:
            analysis['primary_resolution'] = max(analysis['resolutions'].items(), 
                                               key=lambda x: x[1])[0]
            
        self.logger.info(f"Analysis complete: {analysis['valid_images']}/{analysis['total_images']} valid images")
        
        return analysis
        
    def prepare_image_list(self, image_paths: List[Path], output_dir: Path) -> Optional[Path]:
        """
        Create a text file listing images for ffmpeg
        
        Args:
            image_paths: List of image paths in order
            output_dir: Directory to save the list file
            
        Returns:
            Path to the image list file or None if failed
        """
        try:
            list_file = output_dir / 'image_list.txt'
            
            with open(list_file, 'w') as f:
                for img_path in sorted(image_paths):
                    if img_path.exists():
                        # Use absolute path and escape special characters
                        escaped_path = str(img_path.absolute()).replace("'", "'\\''")
                        f.write(f"file '{escaped_path}'\n")
                        
            self.logger.debug(f"Created image list file with {len(image_paths)} entries")
            return list_file
            
        except Exception as e:
            self.logger.error(f"Failed to create image list file: {e}")
            return None
            
    def create_timelapse(self, image_paths: List[Path], output_path: Path,
                        custom_fps: Optional[int] = None) -> bool:
        """
        Create timelapse video from image sequence
        
        Args:
            image_paths: List of image paths in chronological order
            output_path: Path for output video file
            custom_fps: Override default FPS if provided
            
        Returns:
            True if successful, False otherwise
        """
        if not image_paths:
            self.logger.error("No images provided for timelapse creation")
            return False
            
        # Analyze images first
        analysis = self.analyze_images(image_paths)
        if analysis['valid_images'] == 0:
            self.logger.error("No valid images found for timelapse")
            return False
            
        fps = custom_fps if custom_fps else self.fps
        
        self.logger.info(f"Creating timelapse from {analysis['valid_images']} images")
        self.logger.info(f"Output: {output_path}")
        
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sort images by timestamp/filename
            sorted_images = sorted(image_paths, key=lambda x: x.name)
            
            # Create temporary image list file
            temp_dir = self.config.get_storage_paths()['temp']
            list_file = self.prepare_image_list(sorted_images, temp_dir)
            
            if not list_file:
                return False
                
            try:
                # Build ffmpeg command using ffmpeg-python
                input_stream = ffmpeg.input(str(list_file), format='concat', safe=0)
                
                # Apply video settings
                output_stream = ffmpeg.output(
                    input_stream,
                    str(output_path),
                    vcodec=self.codec,
                    r=fps,  # Frame rate
                    pix_fmt='yuv420p',  # Pixel format for compatibility
                    s=f'{self.width}x{self.height}',  # Resolution
                    b=self.bitrate,  # Bitrate
                    threads=self.threads,
                    **{'preset': 'medium', 'crf': '23'}  # Quality settings
                )
                
                # Run ffmpeg with timeout to prevent hanging
                self.logger.info("Starting video encoding...")
                # Calculate reasonable timeout: base time + per-frame processing time
                base_timeout = 300  # 5 minutes base
                frame_timeout = analysis['valid_images'] * 2  # 2 seconds per frame
                total_timeout = base_timeout + frame_timeout
                
                try:
                    import subprocess
                    # Convert ffmpeg-python to subprocess call with timeout
                    args = ffmpeg.compile(output_stream, overwrite_output=True)
                    result = subprocess.run(args, capture_output=True, text=True, timeout=total_timeout)
                    
                    if result.returncode != 0:
                        self.logger.error(f"FFmpeg failed with return code {result.returncode}")
                        if result.stderr:
                            self.logger.error(f"FFmpeg stderr: {result.stderr}")
                        return False
                        
                except subprocess.TimeoutExpired:
                    self.logger.error(f"FFmpeg timed out after {total_timeout} seconds")
                    self.logger.error("This usually indicates corrupted input files or insufficient system resources")
                    return False
                
                # Verify output file was created
                if output_path.exists() and output_path.stat().st_size > 0:
                    duration = analysis.get('duration', 'unknown')
                    self.logger.info(f"Timelapse created successfully: {output_path}")
                    self.logger.info(f"Source duration: {duration}, Video length: {analysis['valid_images']/fps:.1f}s")
                    return True
                else:
                    self.logger.error("Output file was not created or is empty")
                    return False
                    
            finally:
                # Clean up temporary files
                if list_file and list_file.exists():
                    list_file.unlink()
                    
        except ffmpeg.Error as e:
            self.logger.error(f"FFmpeg error: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                self.logger.error(f"FFmpeg stderr: {e.stderr.decode()}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to create timelapse: {e}")
            return False
            
    def create_timelapse_from_directory(self, image_directory: Path, 
                                      output_path: Path,
                                      pattern: str = "*.jpg") -> bool:
        """
        Create timelapse from all images in a directory
        
        Args:
            image_directory: Directory containing images
            output_path: Path for output video
            pattern: Filename pattern to match (default: *.jpg)
            
        Returns:
            True if successful, False otherwise
        """
        if not image_directory.exists():
            self.logger.error(f"Image directory not found: {image_directory}")
            return False
            
        # Find all matching images
        image_paths = list(image_directory.glob(pattern))
        
        if not image_paths:
            self.logger.error(f"No images found in {image_directory} matching {pattern}")
            return False
            
        return self.create_timelapse(image_paths, output_path)
        
    def create_date_timelapse(self, target_date: date, custom_name: Optional[str] = None) -> Optional[Path]:
        """
        Create timelapse for a specific date using captured images
        
        Args:
            target_date: Date to create timelapse for
            custom_name: Custom output filename (without extension)
            
        Returns:
            Path to created video or None if failed
        """
        # Get paths
        paths = self.config.get_storage_paths()
        date_str = target_date.strftime('%Y-%m-%d')
        base_image_dir = paths['images'] / date_str
        
        # Check for images in multiple locations
        image_dir = None
        
        # First check for historical images (from downloaded recordings)
        historical_dir = base_image_dir / 'historical'
        if historical_dir.exists() and list(historical_dir.glob('*.jpg')):
            image_dir = historical_dir
            self.logger.info(f"Using historical images from {historical_dir}")
        
        # Fallback to regular date directory (from live captures)
        elif base_image_dir.exists() and list(base_image_dir.glob('*.jpg')):
            image_dir = base_image_dir
            self.logger.info(f"Using live capture images from {base_image_dir}")
        
        if not image_dir:
            self.logger.error(f"No images found for date {date_str} (checked both live and historical)")
            return None
            
        # Generate output filename
        if custom_name:
            output_filename = f"{custom_name}.{self.output_format}"
        else:
            date_formatted = target_date.strftime('%m/%d/%y').replace('/', '_')
            output_filename = f"Sunset_{date_formatted}.{self.output_format}"
            
        output_path = paths['videos'] / output_filename
        
        # Create timelapse
        if self.create_timelapse_from_directory(image_dir, output_path):
            return output_path
        else:
            return None
            
    def get_video_info(self, video_path: Path) -> Optional[dict]:
        """
        Get information about a video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information or None if failed
        """
        try:
            probe = ffmpeg.probe(str(video_path))
            
            # Extract video stream info
            video_stream = next((stream for stream in probe['streams'] 
                               if stream['codec_type'] == 'video'), None)
            
            if not video_stream:
                return None

            # Safely parse FPS fraction without eval() to prevent code injection
            def parse_fps_fraction(fraction_str: str) -> float:
                """Safely parse FPS fraction like '12/1' without using eval()"""
                try:
                    if '/' in fraction_str:
                        numerator, denominator = fraction_str.split('/', 1)
                        return float(numerator) / float(denominator)
                    else:
                        return float(fraction_str)
                except (ValueError, ZeroDivisionError, AttributeError):
                    return 0.0

            info = {
                'duration': float(probe['format']['duration']),
                'size': int(probe['format']['size']),
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': parse_fps_fraction(video_stream['r_frame_rate']),
                'codec': video_stream['codec_name'],
                'bitrate': int(probe['format']['bit_rate'])
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get video info for {video_path}: {e}")
            return None