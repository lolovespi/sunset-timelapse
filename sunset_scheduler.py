"""
Sunset Scheduler
Main orchestrator for daily sunset timelapse capture and processing
"""

import logging
import time
import signal
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, List, Dict
import schedule
import threading
import json

from config_manager import get_config
from sunset_calculator import SunsetCalculator
from camera_interface import CameraInterface
from video_processor import VideoProcessor
from youtube_uploader import YouTubeUploader
from drive_uploader import DriveUploader
from email_notifier import EmailNotifier
from sbs_reporter import SBSReporter


class SunsetScheduler:
    """Main scheduler for sunset timelapse operations"""
    
    def __init__(self):
        """Initialize the scheduler"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.sunset_calc = SunsetCalculator()
        self.camera = CameraInterface()
        self.video_processor = VideoProcessor()
        self.youtube_uploader = YouTubeUploader()
        self.drive_uploader = DriveUploader()
        self.email_notifier = EmailNotifier()
        self.sbs_reporter = SBSReporter()
        
        # State tracking
        self.running = False
        self.current_capture_thread = None
        self.shutdown_requested = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Sunset scheduler initialized")
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        if self.shutdown_requested:
            # Force exit if already shutting down
            self.logger.info("Force exit requested")
            import sys
            sys.exit(1)

        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self.stop()
        
    def validate_system(self) -> bool:
        """
        Validate all system components are working
        
        Returns:
            True if all components are valid, False otherwise
        """
        self.logger.info("Validating system components...")
        
        validation_results = []
        
        # Validate sunset calculator
        try:
            if self.sunset_calc.validate_location():
                self.logger.info("✓ Sunset calculator validation passed")
                validation_results.append(True)
            else:
                self.logger.error("✗ Sunset calculator validation failed")
                validation_results.append(False)
        except Exception as e:
            self.logger.error(f"✗ Sunset calculator error: {e}")
            validation_results.append(False)
            
        # Validate camera
        try:
            if self.camera.test_connection():
                self.logger.info("✓ Camera validation passed")
                validation_results.append(True)
            else:
                self.logger.error("✗ Camera validation failed")
                validation_results.append(False)
        except Exception as e:
            self.logger.error(f"✗ Camera error: {e}")
            validation_results.append(False)
            
        # Validate video processor
        try:
            if self.video_processor.validate_ffmpeg():
                self.logger.info("✓ Video processor validation passed")
                validation_results.append(True)
            else:
                self.logger.error("✗ Video processor validation failed")
                validation_results.append(False)
        except Exception as e:
            self.logger.error(f"✗ Video processor error: {e}")
            validation_results.append(False)
            
        # Validate YouTube uploader (optional)
        try:
            if self.youtube_uploader.test_authentication():
                self.logger.info("✓ YouTube uploader validation passed")
                validation_results.append(True)
            else:
                self.logger.warning("⚠ YouTube uploader validation failed (uploads will be skipped)")
                validation_results.append(True)  # Non-critical failure
        except Exception as e:
            self.logger.warning(f"⚠ YouTube uploader error: {e} (uploads will be skipped)")
            validation_results.append(True)  # Non-critical failure
            
        # Validate Drive uploader (optional)
        try:
            drive_enabled = self.config.get('drive.enabled', False)
            if drive_enabled:
                if self.drive_uploader.test_authentication():
                    self.logger.info("✓ Google Drive uploader validation passed")
                    validation_results.append(True)
                else:
                    self.logger.warning("⚠ Google Drive uploader validation failed (cloud storage will be skipped)")
                    validation_results.append(True)  # Non-critical failure
            else:
                self.logger.info("○ Google Drive uploader disabled in config")
        except Exception as e:
            self.logger.warning(f"⚠ Google Drive uploader error: {e} (cloud storage will be skipped)")
            validation_results.append(True)  # Non-critical failure
            
        success = all(validation_results)
        
        if success:
            self.logger.info("✓ System validation completed successfully")
        else:
            self.logger.error("✗ System validation failed")
            
        return success
        
    def capture_sunset_sequence(self, target_date: Optional[date] = None) -> Optional[List[Path]]:
        """
        Capture a complete sunset sequence for the specified date
        
        Args:
            target_date: Date to capture (default: today)
            
        Returns:
            List of captured image paths or None if failed
        """
        if target_date is None:
            target_date = date.today()
            
        self.logger.info(f"Starting sunset capture sequence for {target_date}")
        
        try:
            # Calculate capture window
            start_time, end_time = self.sunset_calc.get_capture_window(target_date)
            interval = self.config.get('capture.interval_seconds', 5)
            
            # Convert timezone-aware to naive datetimes for camera interface
            if hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
                start_time = start_time.replace(tzinfo=None)
            if hasattr(end_time, 'tzinfo') and end_time.tzinfo is not None:
                end_time = end_time.replace(tzinfo=None)
            
            self.logger.info(f"Capture window: {start_time} to {end_time}")
            
            # Capture sequence using RTSP video recording method (no ONVIF connection needed)
            captured_images = self.camera.capture_video_sequence(start_time, end_time, interval)
            
            if captured_images:
                self.logger.info(f"Capture sequence completed: {len(captured_images)} images")
                return captured_images
            else:
                self.logger.error("No images captured")
                # Send email notification for capture failure
                self.email_notifier.send_capture_failure(
                    "Camera failed to capture any images during the sunset window", 
                    datetime.combine(target_date, datetime.min.time())
                )
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to capture sunset sequence: {e}")
            self.camera.disconnect()
            # Send email notification for capture exception
            self.email_notifier.send_capture_failure(
                f"Exception during sunset capture: {str(e)}", 
                datetime.combine(target_date, datetime.min.time())
            )
            return None
            
    def process_captured_images(self, target_date: date) -> Optional[Path]:
        """
        Process captured images into a timelapse video
        
        Args:
            target_date: Date of the images to process
            
        Returns:
            Path to created video or None if failed
        """
        self.logger.info(f"Processing images for {target_date}")
        
        try:
            video_path = self.video_processor.create_date_timelapse(target_date)
            
            if video_path and video_path.exists():
                self.logger.info(f"Timelapse created: {video_path}")
                return video_path
            else:
                self.logger.error("Failed to create timelapse")
                # Send email notification for video processing failure
                self.email_notifier.send_processing_failure(
                    "Failed to create timelapse video from captured images", 
                    target_date, 
                    "video_creation"
                )
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to process images: {e}")
            # Send email notification for video processing exception
            self.email_notifier.send_processing_failure(
                f"Exception during video processing: {str(e)}", 
                target_date, 
                "video_creation"
            )
            return None
            
    def upload_to_youtube(self, video_path: Path, target_date: date, 
                         actual_start_time: datetime = None, actual_end_time: datetime = None, is_test: bool = False) -> bool:
        """
        Upload video to YouTube
        
        Args:
            video_path: Path to video file
            target_date: Date the video was captured
            actual_start_time: Actual start time of capture (optional, defaults to calculated sunset window)
            actual_end_time: Actual end time of capture (optional, defaults to calculated sunset window)
            is_test: If True, use test video title and description
            
        Returns:
            True if upload successful, False otherwise
        """
        try:
            # Use actual capture times if provided, otherwise use calculated sunset window
            if actual_start_time and actual_end_time:
                start_time, end_time = actual_start_time, actual_end_time
            else:
                start_time, end_time = self.sunset_calc.get_capture_window(target_date)
            
            # Upload video
            video_id = self.youtube_uploader.upload_video(
                video_path, target_date, start_time, end_time, is_test
            )
            
            if video_id:
                self.logger.info(f"Video uploaded to YouTube: {video_id}")
                return True
            else:
                self.logger.error("Failed to upload video to YouTube")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to upload video: {e}")
            return False
    
    def upload_to_youtube_with_sbs(self, video_path: Path, target_date: date, 
                                  sbs_report: Optional[Dict] = None, 
                                  actual_start_time: datetime = None, 
                                  actual_end_time: datetime = None, 
                                  is_test: bool = False) -> bool:
        """
        Upload video to YouTube with SBS-enhanced title and description
        
        Args:
            video_path: Path to video file
            target_date: Date the video was captured
            sbs_report: SBS analysis report (optional)
            actual_start_time: Actual start time of capture (optional)
            actual_end_time: Actual end time of capture (optional)  
            is_test: If True, use test video title and description
            
        Returns:
            True if upload successful, False otherwise
        """
        try:
            # Use actual capture times if provided, otherwise use calculated sunset window
            if actual_start_time and actual_end_time:
                start_time, end_time = actual_start_time, actual_end_time
            else:
                start_time, end_time = self.sunset_calc.get_capture_window(target_date)
            
            # Get SBS enhancements if report available
            title_enhancement = ""
            description_enhancement = ""
            
            if sbs_report:
                title_enhancement = self.sbs_reporter.get_video_title_enhancement(target_date)
                description_enhancement = self.sbs_reporter.get_video_description_enhancement(target_date)
            
            # Upload video with enhancements
            video_id = self.youtube_uploader.upload_video_with_sbs_enhancements(
                video_path, target_date, start_time, end_time, 
                title_enhancement, description_enhancement, is_test
            )
            
            if video_id:
                self.logger.info(f"Video uploaded to YouTube with SBS enhancements: {video_id}")
                if sbs_report:
                    score = sbs_report['summary']['daily_brilliance_score']
                    grade = sbs_report['summary']['quality_grade']
                    self.logger.info(f"SBS-enhanced upload: Score {score:.1f} (Grade {grade})")
                return True
            else:
                self.logger.error("Failed to upload video to YouTube")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to upload video with SBS enhancements: {e}")
            # Fallback to regular upload
            self.logger.info("Attempting fallback upload without SBS enhancements")
            return self.upload_to_youtube(video_path, target_date, actual_start_time, actual_end_time, is_test)
            
    def daily_maintenance(self):
        """Perform daily maintenance tasks including token refresh and file cleanup"""
        self.logger.info("Starting daily maintenance tasks...")
        
        try:
            # Task 1: Check and refresh YouTube token proactively
            self.logger.info("Checking YouTube token health...")
            token_healthy = self.youtube_uploader.check_token_health_and_alert()
            
            if token_healthy:
                # Try to refresh token if it needs it
                refresh_success = self.youtube_uploader.refresh_token_proactively()
                if refresh_success:
                    self.logger.info("✓ YouTube token is healthy and refreshed")
                else:
                    self.logger.warning("⚠ YouTube token refresh failed - manual intervention may be needed")
            else:
                self.logger.error("✗ YouTube token has issues - check logs and email notifications")
            
            # Task 2: Clean up old SBS analysis data
            self.logger.info("Cleaning up old SBS analysis data...")
            sbs_retention_days = self.config.get('sbs.retention_days', 30)
            self.sbs_reporter.cleanup_old_sbs_data(sbs_retention_days)
            
            # Task 3: Clean up old Google Drive files
            drive_enabled = self.config.get('drive.enabled', False)
            if drive_enabled:
                self.logger.info("Cleaning up old Google Drive files...")
                try:
                    deleted_count = self.drive_uploader.cleanup_old_files()
                    self.logger.info(f"Cleaned up {deleted_count} old files from Google Drive")
                except Exception as e:
                    self.logger.warning(f"Drive cleanup failed: {e}")
            
            # Task 4: Clean up local old files
            self.cleanup_old_files()
            
            self.logger.info("Daily maintenance completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during daily maintenance: {e}")
    
    def cleanup_old_files(self):
        """Clean up old files based on configuration"""
        self.logger.info("Starting cleanup of old files...")
        
        try:
            paths = self.config.get_storage_paths()
            current_date = date.today()
            
            # Clean up old images
            image_retention_days = self.config.get('storage.keep_images_days', 7)
            if image_retention_days > 0:
                cutoff_date = current_date - timedelta(days=image_retention_days)
                
                for date_dir in paths['images'].iterdir():
                    if date_dir.is_dir():
                        try:
                            dir_date = datetime.strptime(date_dir.name, '%Y-%m-%d').date()
                            if dir_date < cutoff_date:
                                self.logger.info(f"Removing old image directory: {date_dir}")
                                import shutil
                                shutil.rmtree(date_dir)
                        except ValueError:
                            # Skip directories that don't match date format
                            continue
                            
            # Clean up old videos
            video_retention_days = self.config.get('storage.keep_videos_locally_days', 1)
            if video_retention_days > 0:
                cutoff_date = current_date - timedelta(days=video_retention_days)
                cutoff_timestamp = cutoff_date.strftime('%m_%d_%y')
                
                for video_file in paths['videos'].glob('*.mp4'):
                    try:
                        # Extract date from filename (format: Sunset_MM_DD_YY.mp4)
                        if video_file.stem.startswith('Sunset_'):
                            date_part = video_file.stem.replace('Sunset_', '')
                            if date_part < cutoff_timestamp:
                                self.logger.info(f"Removing old video: {video_file}")
                                video_file.unlink()
                    except Exception as e:
                        self.logger.warning(f"Error checking video date for {video_file}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
    def complete_daily_workflow(self, target_date: Optional[date] = None) -> bool:
        """
        Execute complete daily workflow: capture -> process -> upload -> cleanup
        
        Args:
            target_date: Date to process (default: today)
            
        Returns:
            True if workflow completed successfully, False otherwise
        """
        if target_date is None:
            target_date = date.today()
            
        self.logger.info(f"Starting complete daily workflow for {target_date}")
        
        try:
            # Step 1: Capture images
            captured_images = self.capture_sunset_sequence(target_date)
            if not captured_images:
                self.logger.error("Image capture failed, aborting workflow")
                # Email notification already sent in capture_sunset_sequence
                return False
                
            # Step 2: Process into video
            video_path = self.process_captured_images(target_date)
            if not video_path:
                self.logger.error("Video processing failed, aborting workflow")
                # Email notification already sent in process_captured_images
                return False
                
            # Step 3: Generate SBS report and enhance video metadata
            sbs_report = self.sbs_reporter.generate_daily_report(target_date)
            if sbs_report:
                self.logger.info(f"SBS Analysis completed: Score {sbs_report['summary']['daily_brilliance_score']:.1f} "
                               f"(Grade {sbs_report['summary']['quality_grade']})")
            
            # Step 4: Upload to Google Drive (for Claude bot processing)
            drive_enabled = self.config.get('drive.enabled', False)
            if drive_enabled:
                try:
                    sunset_start, sunset_end = self.sunset_calc.get_capture_window(target_date)
                    drive_metadata = {
                        'sunset_start': sunset_start.isoformat(),
                        'sunset_end': sunset_end.isoformat(),
                        'capture_date': target_date.isoformat()
                    }
                    
                    # Add SBS data to metadata for Claude bot
                    if sbs_report:
                        drive_metadata.update({
                            'sbs_score': sbs_report['summary']['daily_brilliance_score'],
                            'sbs_grade': sbs_report['summary']['quality_grade'],
                            'sbs_analysis': sbs_report['summary']
                        })
                    
                    drive_result = self.drive_uploader.upload_video(video_path, target_date, drive_metadata)
                    if drive_result:
                        self.logger.info(f"Video uploaded to Google Drive: {drive_result['filename']}")
                    else:
                        self.logger.warning("Google Drive upload failed")
                except Exception as e:
                    self.logger.warning(f"Google Drive upload error: {e}")
            
            # Step 5: Upload to YouTube with SBS enhancements
            upload_success = self.upload_to_youtube_with_sbs(video_path, target_date, sbs_report)
            if not upload_success:
                self.logger.warning("YouTube upload failed, but workflow will continue")
                # Send email notification for YouTube upload failure (with SBS data)
                failure_msg = "YouTube upload failed"
                if sbs_report:
                    failure_msg += f" (SBS Score: {sbs_report['summary']['daily_brilliance_score']:.1f})"
                
                self.email_notifier.send_processing_failure(
                    failure_msg, 
                    target_date, 
                    "youtube_upload"
                )
                
            # Step 5: Daily maintenance (includes token refresh, SBS cleanup, and file cleanup)
            self.daily_maintenance()
            
            self.logger.info(f"Daily workflow completed for {target_date}")
            return True
            
        except Exception as e:
            self.logger.error(f"Daily workflow failed: {e}")
            return False
            
    def schedule_daily_capture(self):
        """Schedule daily capture based on sunset times"""
        self.logger.info("Setting up daily capture schedule...")
        
        # Clear existing schedules
        schedule.clear()
        
        # Calculate next capture time
        start_time, _ = self.sunset_calc.get_next_capture_window()
        
        # Schedule capture to start 5 minutes before the calculated start time
        # This ensures the system is ready
        schedule_time = start_time - timedelta(minutes=5)
        schedule_time_str = schedule_time.strftime('%H:%M')
        
        self.logger.info(f"Scheduling daily capture at {schedule_time_str}")
        
        # Schedule the job
        schedule.every().day.at(schedule_time_str).do(self.complete_daily_workflow)
        
        # Schedule daily maintenance at 6 AM (includes token refresh and cleanup)
        schedule.every().day.at("06:00").do(self.daily_maintenance)
        
        # Also schedule a backup weekly cleanup (in case daily maintenance fails)
        schedule.every().sunday.at("02:00").do(self.cleanup_old_files)
        
    def run_scheduler(self):
        """Run the main scheduler loop"""
        self.logger.info("Starting sunset scheduler...")
        self.running = True
        
        # Set up initial schedule
        self.schedule_daily_capture()
        
        # Display next scheduled jobs
        jobs = schedule.get_jobs()
        if jobs:
            next_job = min(jobs, key=lambda x: x.next_run)
            self.logger.info(f"Next scheduled job: {next_job.next_run}")
            
        # Main scheduler loop
        while self.running and not self.shutdown_requested:
            try:
                # Run scheduled jobs
                schedule.run_pending()
                
                # Check if we need to reschedule (daily at midnight)
                now = datetime.now()
                if now.hour == 0 and now.minute == 0:
                    self.schedule_daily_capture()
                    
                # Sleep for a minute before checking again
                time.sleep(60)
                
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(60)  # Wait a minute before retrying
                
        self.logger.info("Scheduler stopped")
        
    def stop(self):
        """Stop the scheduler"""
        self.logger.info("Stopping scheduler...")
        self.running = False
        
        # Wait for current capture to complete if running
        if self.current_capture_thread and self.current_capture_thread.is_alive():
            self.logger.info("Waiting for current capture to complete...")
            self.current_capture_thread.join(timeout=30)
            
    def run_immediate_capture(self, duration_minutes=5):
        """Run immediate capture for testing purposes
        
        Args:
            duration_minutes: Duration of capture in minutes (default: 5)
        """
        self.logger.info(f"Running immediate capture for {duration_minutes} minutes...")
        
        if not self.validate_system():
            self.logger.error("System validation failed, aborting immediate capture")
            return False
            
        # Run immediate capture with specified duration
        from datetime import datetime, timedelta
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        self.logger.info(f"Starting immediate test capture from {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')}")
        
        # Manual workflow for immediate testing
        target_date = start_time.date()
        
        # Capture images
        captured_images = self.capture_sunset_sequence_manual(start_time, end_time)
        if not captured_images:
            self.logger.error("Image capture failed, aborting workflow")
            return False
            
        # Process images into video
        video_path = self.process_captured_images(target_date)
        if not video_path:
            self.logger.error("Video processing failed")
            return False
            
        # Upload to YouTube with actual capture times (mark as test)
        upload_success = self.upload_to_youtube(video_path, target_date, start_time, end_time, is_test=True)
        if not upload_success:
            self.logger.warning("YouTube upload failed, but workflow completed")
            
        self.logger.info("Immediate capture workflow completed successfully")
        return True
        
    def capture_sunset_sequence_manual(self, start_time, end_time):
        """Manual capture sequence with specified start/end times"""
        try:
            interval = self.config.get('capture.interval_seconds', 5)
            
            self.logger.info(f"Manual capture window: {start_time} to {end_time}")
            
            # Capture sequence using RTSP video recording method (no ONVIF connection needed)
            captured_images = self.camera.capture_video_sequence(start_time, end_time, interval)
            
            if captured_images:
                self.logger.info(f"Manual capture sequence completed: {len(captured_images)} images")
                return captured_images
            else:
                self.logger.error("No images captured")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed manual capture sequence: {e}")
            self.camera.disconnect()
            return None
        
    def get_status(self) -> dict:
        """
        Get current system status
        
        Returns:
            Dictionary with system status information
        """
        now = datetime.now()
        
        status = {
            'timestamp': now.isoformat(),
            'running': self.running,
            'next_sunset': None,
            'next_capture_window': None,
            'system_health': {},
            'recent_activity': {}
        }
        
        try:
            # Get next sunset info
            start_time, end_time = self.sunset_calc.get_next_capture_window()
            sunset_time = self.sunset_calc.get_sunset_time(start_time.date())
            
            status['next_sunset'] = sunset_time.isoformat()
            status['next_capture_window'] = {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            }
            
            # System health checks
            status['system_health'] = {
                'camera_connected': self.camera.is_connected(),
                'youtube_authenticated': self.youtube_uploader.is_authenticated(),
                'ffmpeg_available': self.video_processor.validate_ffmpeg(),
                'storage_accessible': True  # TODO: Add storage health check
            }
            
            # Recent activity (check for recent files)
            paths = self.config.get_storage_paths()
            recent_images = list(paths['images'].glob('**/*.jpg'))[-5:]  # Last 5 images
            recent_videos = list(paths['videos'].glob('*.mp4'))[-3:]  # Last 3 videos
            
            status['recent_activity'] = {
                'recent_images': [str(p) for p in recent_images],
                'recent_videos': [str(p) for p in recent_videos]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            status['error'] = str(e)
            
        return status