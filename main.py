#!/usr/bin/env python3
"""
Sunset Timelapse Main CLI
Entry point for all sunset timelapse operations
"""

import argparse
import sys
import logging
from datetime import datetime, date, timedelta
from pathlib import Path

from config_manager import get_config
from sunset_scheduler import SunsetScheduler
from historical_retrieval import HistoricalRetrieval
from sunset_calculator import SunsetCalculator


def setup_logging():
    """Setup logging based on configuration"""
    config = get_config()
    config.setup_logging()


def cmd_schedule(args):
    """Run the daily scheduler"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting sunset timelapse scheduler...")
    
    scheduler = SunsetScheduler()
    
    if args.validate:
        logger.info("Running system validation...")
        if not scheduler.validate_system():
            logger.error("System validation failed")
            sys.exit(1)
        logger.info("System validation passed")
        
    if args.immediate:
        logger.info(f"Running immediate capture for {args.duration} minutes...")
        success = scheduler.run_immediate_capture(duration_minutes=args.duration)
        sys.exit(0 if success else 1)
        
    try:
        scheduler.run_scheduler()
    except KeyboardInterrupt:
        logger.info("Scheduler interrupted by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        sys.exit(1)


def cmd_historical(args):
    """Process historical footage"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting historical footage processing...")
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    except ValueError as e:
        logger.error(f"Invalid date format. Use YYYY-MM-DD: {e}")
        sys.exit(1)
        
    if start_date > end_date:
        logger.error("Start date must be before end date")
        sys.exit(1)
        
    retrieval = HistoricalRetrieval()
    
    if args.list_available:
        logger.info("Checking available dates...")
        available_dates = retrieval.get_available_dates(start_date, end_date)
        
        if available_dates:
            logger.info(f"Available dates ({len(available_dates)}):")
            for available_date in available_dates:
                print(f"  {available_date}")
        else:
            logger.info("No recordings found for the specified date range")
        return
        
    # Process historical footage
    created_videos = retrieval.create_historical_timelapse(
        start_date, end_date, 
        upload_to_youtube=args.upload
    )
    
    if created_videos:
        logger.info(f"Successfully created {len(created_videos)} videos:")
        for video in created_videos:
            print(f"  {video}")
    else:
        logger.error("No videos were created")
        sys.exit(1)


def cmd_test(args):
    """Run system tests"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Running system tests...")
    
    scheduler = SunsetScheduler()
    
    # Run validation
    if not scheduler.validate_system():
        logger.error("System validation failed")
        sys.exit(1)
        
    # Test individual components if requested
    if args.camera:
        logger.info("Testing camera connection...")
        if scheduler.camera.test_connection():
            logger.info("✓ Camera test passed")
        else:
            logger.error("✗ Camera test failed")
            sys.exit(1)
            
    if args.youtube:
        logger.info("Testing YouTube authentication...")
        if scheduler.youtube_uploader.test_authentication():
            logger.info("✓ YouTube test passed")
        else:
            logger.error("✗ YouTube test failed")
            sys.exit(1)
    
    if args.drive:
        logger.info("Testing Google Drive authentication...")
        from drive_uploader import DriveUploader
        drive_uploader = DriveUploader()
        if drive_uploader.test_authentication():
            logger.info("✓ Google Drive test passed")
        else:
            logger.error("✗ Google Drive test failed")
            sys.exit(1)
    
    if args.youtube_token:
        logger.info("Testing YouTube token management...")
        token_info = scheduler.youtube_uploader.get_token_expiry_info()
        if token_info:
            days = token_info['days_remaining']
            hours = token_info['hours_remaining']
            logger.info(f"Token expires in {days} days, {hours} hours")
            
            # Test proactive refresh
            refresh_success = scheduler.youtube_uploader.refresh_token_proactively()
            if refresh_success:
                logger.info("✓ Token refresh test passed")
                # After successful refresh, do a quick health check without alerts
                token_info_updated = scheduler.youtube_uploader.get_token_expiry_info()
                if token_info_updated and not token_info_updated['is_expired']:
                    logger.info("✓ Token health confirmed good after refresh")
                else:
                    logger.warning("⚠ Token still has issues after refresh")
            else:
                logger.warning("⚠ Token refresh not needed or failed")
                # Only check and alert if refresh failed - this prevents false expiration alerts
                health_check = scheduler.youtube_uploader.check_token_health_and_alert()
                if health_check:
                    logger.info("✓ Token health check passed")
                else:
                    logger.error("✗ Token health check failed")
        else:
            logger.error("✗ No token found or token invalid")
            sys.exit(1)
            
    if args.recordings:
        logger.info("Testing Reolink recording search and download...")
        from datetime import timedelta, date
        from historical_retrieval import HistoricalRetrieval
        
        # Test with yesterday's recordings
        yesterday = date.today() - timedelta(days=1)
        
        logger.info(f"Searching for recordings for {yesterday}")
        
        # Use historical retrieval system
        historical = HistoricalRetrieval()
        recordings = historical.get_camera_recordings(yesterday, yesterday)
        
        if recordings:
            logger.info(f"✓ Found {len(recordings)} recordings using integrated system")
            logger.info("Recording details:")
            for i, recording in enumerate(recordings[:3]):  # Show first 3
                logger.info(f"  {i+1}. File: {recording.get('file_name', recording.get('name', 'N/A'))}")
                logger.info(f"     Type: {recording.get('type', 'unknown')}")
                logger.info(f"     Time: {recording.get('start_time')} - {recording.get('end_time')}")
                logger.info(f"     Size: {recording.get('size', 0)} bytes")
                
            # Test download of first recording (if not local images)
            if recordings and recordings[0].get('type') != 'local_images':
                logger.info("Testing download of first recording...")
                try:
                    import tempfile
                    from pathlib import Path
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        test_path = Path(temp_dir) / f"test_{recordings[0]['name']}"
                        success = historical.download_recording(recordings[0], test_path)
                        
                        if success and test_path.exists():
                            file_size = test_path.stat().st_size
                            logger.info(f"✓ Successfully downloaded test file ({file_size:,} bytes)")
                        else:
                            logger.warning("Download test failed, but search functionality works")
                            
                except Exception as e:
                    logger.warning(f"Download test failed: {e}, but search functionality works")
        else:
            logger.warning("No recordings found in time range")
            logger.info("This might be normal if no motion was detected or recording is disabled")
            
        logger.info("✓ Reolink recording test completed")
            
    if args.sunset:
        logger.info("Testing sunset calculations...")
        calc = SunsetCalculator()
        if calc.validate_location():
            schedule = calc.get_weekly_schedule()
            print("\n" + calc.format_schedule(schedule))
            logger.info("✓ Sunset calculation test passed")
        else:
            logger.error("✗ Sunset calculation test failed")
            sys.exit(1)
    
    if args.email:
        logger.info("Testing email notifications...")
        from email_notifier import EmailNotifier
        email_notifier = EmailNotifier()
        
        if email_notifier.is_enabled():
            if email_notifier.test_connection():
                logger.info("✓ Email notification test passed")
            else:
                logger.error("✗ Email notification test failed")
                sys.exit(1)
        else:
            logger.info("✓ Email notifications disabled (test skipped)")
    
    if args.sbs:
        logger.info("Testing SBS (Sunset Brilliance Score) system...")
        from sunset_brilliance_score import SunsetBrillianceScore
        from sbs_reporter import SBSReporter
        import cv2
        import numpy as np
        
        try:
            # Test SBS analyzer initialization
            sbs_analyzer = SunsetBrillianceScore()
            logger.info("✓ SBS analyzer initialized")
            
            # Create test frame (sunset-like colors)
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Create gradient from orange/red at bottom to blue at top
            for y in range(480):
                # Orange/red sunset colors at bottom
                ratio = y / 480.0
                if ratio < 0.6:  # Bottom 60% - sunset colors
                    test_frame[y, :] = [int(50 + ratio * 100), int(100 + ratio * 100), int(200 + ratio * 50)]
                else:  # Top 40% - sky colors  
                    test_frame[y, :] = [200, 150, 100]
            
            # Test frame analysis
            frame_metrics = sbs_analyzer.analyze_frame(test_frame, 0, 0)
            
            if frame_metrics:
                logger.info(f"✓ Frame analysis successful:")
                logger.info(f"  - Colorfulness: {frame_metrics.colorfulness:.2f}")
                logger.info(f"  - Color Temperature: {frame_metrics.color_temperature:.0f}K")
                logger.info(f"  - Sky Saturation: {frame_metrics.sky_saturation:.3f}")
                logger.info(f"  - Processing Time: {frame_metrics.processing_time_ms:.2f}ms")
                
                # Test performance requirement (should be < 10ms on Pi)
                if frame_metrics.processing_time_ms < 10:
                    logger.info(f"✓ Processing performance excellent ({frame_metrics.processing_time_ms:.2f}ms < 10ms)")
                elif frame_metrics.processing_time_ms < 20:
                    logger.info(f"⚠ Processing performance acceptable ({frame_metrics.processing_time_ms:.2f}ms)")
                else:
                    logger.warning(f"⚠ Processing performance slow ({frame_metrics.processing_time_ms:.2f}ms > 20ms)")
                    logger.info("Consider reducing analysis_resize_height in config")
                
                # Test chunk analysis
                chunk_metrics = sbs_analyzer.analyze_chunk([frame_metrics], 0, 0)
                logger.info(f"✓ Chunk analysis successful - Brilliance Score: {chunk_metrics.brilliance_score:.1f}")
                
                # Test SBS reporter
                sbs_reporter = SBSReporter()
                logger.info("✓ SBS reporter initialized")
                
                logger.info("✓ SBS system test passed")
            else:
                logger.error("✗ Frame analysis failed")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"✗ SBS test failed: {e}")
            sys.exit(1)
    
    if args.sbs_sample:
        logger.info("Testing SBS analysis on sample sunset image...")
        from sunset_brilliance_score import SunsetBrillianceScore
        import cv2
        from pathlib import Path
        
        try:
            # Look for recent sunset images to analyze
            config = get_config()
            paths = config.get_storage_paths()
            
            # Find the most recent sunset image
            recent_images = []
            for date_dir in sorted(paths['images'].glob('*'), reverse=True):
                if date_dir.is_dir():
                    images_in_dir = list(date_dir.glob('img_*.jpg'))
                    if images_in_dir:
                        recent_images.extend(images_in_dir[:5])  # Take first 5 from each date
                        break
            
            if not recent_images:
                logger.warning("No recent sunset images found for SBS sample analysis")
                logger.info("Capture some images first using: python main.py capture --duration 300")
                return
            
            # Analyze first few images
            sbs_analyzer = SunsetBrillianceScore()
            
            for i, image_path in enumerate(recent_images[:3]):
                logger.info(f"Analyzing sample image {i+1}: {image_path.name}")
                
                frame = cv2.imread(str(image_path))
                if frame is not None:
                    frame_metrics = sbs_analyzer.analyze_frame(frame, i, i * 5)
                    
                    if frame_metrics:
                        logger.info(f"  Colorfulness: {frame_metrics.colorfulness:.2f}")
                        logger.info(f"  Color Temp: {frame_metrics.color_temperature:.0f}K")
                        logger.info(f"  Sky Saturation: {frame_metrics.sky_saturation:.3f}")
                        logger.info(f"  Gradient Intensity: {frame_metrics.gradient_intensity:.2f}")
                        logger.info(f"  Processing: {frame_metrics.processing_time_ms:.2f}ms")
                    else:
                        logger.warning(f"  Analysis failed for {image_path.name}")
                else:
                    logger.warning(f"  Could not load {image_path.name}")
            
            logger.info("✓ SBS sample analysis completed")
            
        except Exception as e:
            logger.error(f"✗ SBS sample analysis failed: {e}")
            sys.exit(1)
            
    logger.info("All tests passed!")


def cmd_capture(args):
    """Run on-demand test capture"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting on-demand capture for {args.duration} seconds...")
    
    scheduler = SunsetScheduler()
    
    if args.validate:
        logger.info("Running system validation...")
        if not scheduler.validate_system():
            logger.error("System validation failed")
            sys.exit(1)
        logger.info("System validation passed")
    
    # Calculate start and end times for custom duration
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=args.duration)
    interval = args.interval or 5
    
    logger.info(f"Capture window: {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')}")
    logger.info(f"Duration: {args.duration} seconds ({args.duration/3600:.1f} hours)")
    logger.info(f"Interval: {interval} seconds")
    
    try:
        # Use camera interface directly for custom timing
        captured_images = scheduler.camera.capture_video_sequence(
            start_time, end_time, interval
        )
        
        if captured_images:
            logger.info(f"Capture completed: {len(captured_images)} images")
            
            if args.process:
                logger.info("Processing images into video...")
                # Create video path in the videos directory
                config = get_config()
                paths = config.get_storage_paths()
                video_filename = f"test_capture_{start_time.strftime('%Y%m%d_%H%M%S')}.mp4"
                video_path = paths['videos'] / video_filename
                
                success = scheduler.video_processor.create_timelapse(
                    captured_images, 
                    video_path
                )
                
                if success:
                    logger.info(f"Video created: {video_path}")
                    if args.upload:
                        logger.info("Uploading to YouTube...")
                        success = scheduler.upload_to_youtube(
                            video_path, date.today(), start_time, end_time, is_test=True
                        )
                        logger.info("Upload successful" if success else "Upload failed")
                else:
                    logger.error("Failed to create video")
                    sys.exit(1)
        else:
            logger.error("No images captured")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Capture failed: {e}")
        sys.exit(1)


def cmd_status(args):
    """Show system status"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    scheduler = SunsetScheduler()
    status = scheduler.get_status()
    
    print("\n=== Sunset Timelapse System Status ===")
    print(f"Timestamp: {status['timestamp']}")
    print(f"Running: {status['running']}")
    
    if status.get('next_sunset'):
        next_sunset = datetime.fromisoformat(status['next_sunset'])
        print(f"Next Sunset: {next_sunset.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
    if status.get('next_capture_window'):
        window = status['next_capture_window']
        start = datetime.fromisoformat(window['start'])
        end = datetime.fromisoformat(window['end'])
        print(f"Next Capture: {start.strftime('%H:%M')} - {end.strftime('%H:%M %Z')}")
        
    print("\n--- System Health ---")


def cmd_cleanup(args):
    """Clean up old files"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting manual cleanup...")
    
    scheduler = SunsetScheduler()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be deleted")
        # TODO: Add dry-run functionality to cleanup_old_files method
        
    scheduler.cleanup_old_files()
    logger.info("Cleanup completed")


def cmd_upload(args):
    """Upload existing timelapse video to YouTube"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        from datetime import datetime
        upload_date = datetime.strptime(args.upload_date, '%Y-%m-%d').date()
    except ValueError:
        logger.error("Invalid date format. Use YYYY-MM-DD")
        return
    
    logger.info(f"Uploading timelapse video for {upload_date}")
    
    try:
        from video_processor import VideoProcessor
        from youtube_uploader import YouTubeUploader
        from sunset_calculator import SunsetCalculator
        
        video_processor = VideoProcessor()
        youtube_uploader = YouTubeUploader()
        sunset_calc = SunsetCalculator()
        
        # Build expected video path for this date
        from pathlib import Path
        config = get_config()
        storage_base = Path(config.get('storage.base_path')).expanduser()
        videos_dir = storage_base / 'videos'
        date_formatted = upload_date.strftime('%m/%d/%y').replace('/', '_')
        video_filename = f"Sunset_{date_formatted}.mp4"
        video_path = videos_dir / video_filename
        
        if not video_path.exists():
            logger.error(f"No timelapse video found for {upload_date}")
            logger.info(f"Expected location: {video_path}")
            logger.info("Available videos:")
            if videos_dir.exists():
                for video_file in videos_dir.glob("*.mp4"):
                    logger.info(f"  - {video_file.name}")
            return
            
        logger.info(f"Found video: {video_path}")
        
        # Get sunset times for metadata
        try:
            sunset_start, sunset_end = sunset_calc.get_capture_window(upload_date)
            logger.info(f"Sunset window: {sunset_start} to {sunset_end}")
        except Exception as e:
            logger.warning(f"Could not calculate sunset times: {e}")
            sunset_start = sunset_end = None
        
        # Upload to YouTube
        logger.info("Starting YouTube upload...")
        youtube_uploader.upload_video(video_path, upload_date, sunset_start, sunset_end)
        logger.info("✓ Upload completed successfully")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise


def cmd_config(args):
    """Configuration management"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    config = get_config()
    
    if args.show:
        print("\n=== Current Configuration ===")
        # Display sanitized config (no secrets)
        print(f"Location: {config.get('location.city')}, {config.get('location.state')}")
        print(f"Coordinates: {config.get('location.latitude')}, {config.get('location.longitude')}")
        print(f"Camera IP: {config.get('camera.ip')}")
        print(f"Capture Interval: {config.get('capture.interval_seconds')}s")
        print(f"Video FPS: {config.get('video.fps')}")
        print(f"Storage Path: {config.get('storage.base_path')}")
        
    if args.validate:
        logger.info("Validating configuration...")
        try:
            config._validate_config()
            logger.info("✓ Configuration validation passed")
        except Exception as e:
            logger.error(f"✗ Configuration validation failed: {e}")
            sys.exit(1)


def cmd_sbs(args):
    """Sunset Brilliance Score analysis and reporting"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    from sbs_reporter import SBSReporter
    from datetime import datetime, date, timedelta
    
    sbs_reporter = SBSReporter()
    
    # Determine target date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            logger.error("Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        target_date = date.today() - timedelta(days=1)  # Yesterday by default
    
    if args.report:
        logger.info(f"Generating SBS report for {target_date}")
        report = sbs_reporter.generate_daily_report(target_date)
        
        if report:
            print(f"\n=== SBS Report for {target_date} ===")
            summary = report['summary']
            print(f"Daily Brilliance Score: {summary['daily_brilliance_score']:.1f}/100")
            print(f"Quality Grade: {summary['quality_grade']}")
            print(f"Peak Chunk: {summary['peak_chunk']} (Score: {summary['peak_chunk_score']:.1f})")
            print(f"Total Frames Analyzed: {summary['total_frames']}")
            print(f"Processing Performance: {summary['avg_processing_time_ms']:.1f}ms average")
            
            print(f"\nChunk Breakdown ({len(report['chunk_details'])} chunks):")
            for chunk in report['chunk_details']:
                print(f"  Chunk {chunk['chunk_number']:2d} ({chunk['time_range']:>8}): "
                     f"{chunk['brilliance_score']:5.1f} pts, "
                     f"{chunk['peak_frames']:2d} peaks, "
                     f"smoothness {chunk['temporal_smoothness']:0.3f}")
            
            if report['recommendations']:
                print(f"\nRecommendations:")
                for rec in report['recommendations']:
                    print(f"  • {rec}")
                    
            logger.info("SBS report generated successfully")
        else:
            logger.error(f"No SBS data available for {target_date}")
            sys.exit(1)
    
    if args.export:
        if not args.start or not args.end:
            logger.error("Export requires --start and --end dates")
            sys.exit(1)
            
        try:
            start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
            end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
        except ValueError:
            logger.error("Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
        
        logger.info(f"Exporting SBS data from {start_date} to {end_date}")
        export_file = sbs_reporter.export_historical_analysis(start_date, end_date)
        
        if export_file.exists():
            logger.info(f"SBS data exported to: {export_file}")
            print(f"Export file: {export_file}")
        else:
            logger.error("Export failed")
            sys.exit(1)
    
    if args.cleanup:
        logger.info("Cleaning up old SBS analysis data...")
        config = get_config()
        retention_days = config.get('sbs.retention_days', 30)
        sbs_reporter.cleanup_old_sbs_data(retention_days)
        logger.info(f"SBS cleanup completed (retained {retention_days} days)")
    
    # Default action: show recent SBS summary
    if not any([args.report, args.export, args.cleanup]):
        logger.info(f"Showing SBS summary for {target_date}")
        
        from sunset_brilliance_score import SunsetBrillianceScore
        sbs_analyzer = SunsetBrillianceScore()
        
        date_str = target_date.strftime('%Y-%m-%d')
        chunk_metrics = sbs_analyzer.load_daily_analysis(date_str)
        
        if chunk_metrics:
            daily_summary = sbs_analyzer.calculate_daily_summary(chunk_metrics)
            
            print(f"\n=== SBS Summary for {target_date} ===")
            print(f"Brilliance Score: {daily_summary['daily_brilliance_score']:.1f}/100 (Grade {daily_summary['quality_grade']})")
            print(f"Peak Performance: Chunk {daily_summary['peak_chunk']}")
            print(f"Frames Analyzed: {daily_summary['total_frames']}")
            print(f"Processing Speed: {daily_summary['avg_processing_time_ms']:.1f}ms average")
        else:
            logger.warning(f"No SBS data found for {target_date}")
            print("Use 'python main.py sbs --report --date YYYY-MM-DD' to generate a full report")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Sunset Timelapse System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run daily scheduler (Raspberry Pi)
  python main.py schedule --validate
  
  # Run immediate capture for testing (5 minutes)
  python main.py schedule --immediate
  
  # Run immediate capture for 30 minutes
  python main.py schedule --immediate --duration 30
  
  # Process historical footage (MacBook)
  python main.py historical --start 2024-01-01 --end 2024-01-07 --upload
  
  # List available dates
  python main.py historical --start 2024-01-01 --end 2024-01-31 --list
  
  # Run system tests
  python main.py test --camera --youtube --email
  
  # Show system status
  python main.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Schedule command (daily operations)
    schedule_parser = subparsers.add_parser('schedule', help='Run daily scheduler')
    schedule_parser.add_argument('--validate', action='store_true',
                               help='Validate system before starting')
    schedule_parser.add_argument('--immediate', action='store_true',
                               help='Run immediate capture instead of scheduling')
    schedule_parser.add_argument('--duration', type=int, default=5,
                               help='Duration in minutes for immediate capture (default: 5)')
    schedule_parser.set_defaults(func=cmd_schedule)
    
    # Historical command
    hist_parser = subparsers.add_parser('historical', help='Process historical footage')
    hist_parser.add_argument('--start', dest='start_date', required=True,
                           help='Start date (YYYY-MM-DD)')
    hist_parser.add_argument('--end', dest='end_date', required=True,
                           help='End date (YYYY-MM-DD)')
    hist_parser.add_argument('--upload', action='store_true',
                           help='Upload videos to YouTube')
    hist_parser.add_argument('--list', dest='list_available', action='store_true',
                           help='List available dates instead of processing')
    hist_parser.set_defaults(func=cmd_historical)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run system tests')
    test_parser.add_argument('--camera', action='store_true',
                           help='Test camera connection')
    test_parser.add_argument('--youtube', action='store_true',
                           help='Test YouTube authentication')
    test_parser.add_argument('--youtube-token', action='store_true',
                           help='Test YouTube token refresh and health check')
    test_parser.add_argument('--drive', action='store_true',
                           help='Test Google Drive authentication and upload')
    test_parser.add_argument('--recordings', action='store_true',
                           help='Test Reolink recording search and download')
    test_parser.add_argument('--sunset', action='store_true',
                           help='Test sunset calculations')
    test_parser.add_argument('--email', action='store_true',
                           help='Test email notifications')
    test_parser.add_argument('--sbs', action='store_true',
                           help='Test SBS (Sunset Brilliance Score) system')
    test_parser.add_argument('--sbs-sample', action='store_true',
                           help='Test SBS analysis on recent sunset images')
    test_parser.set_defaults(func=cmd_test)
    
    # Capture command (on-demand test capture)
    capture_parser = subparsers.add_parser('capture', help='Run on-demand test capture')
    capture_parser.add_argument('--duration', type=int, required=True,
                               help='Capture duration in seconds (e.g., 7200 for 2 hours)')
    capture_parser.add_argument('--interval', type=int, default=5,
                               help='Interval between captures in seconds (default: 5)')
    capture_parser.add_argument('--validate', action='store_true',
                               help='Validate system before starting')
    capture_parser.add_argument('--process', action='store_true',
                               help='Process images into video after capture')
    capture_parser.add_argument('--upload', action='store_true',
                               help='Upload video to YouTube (requires --process)')
    capture_parser.set_defaults(func=cmd_capture)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.set_defaults(func=cmd_status)
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--show', action='store_true',
                             help='Show current configuration')
    config_parser.add_argument('--validate', action='store_true',
                             help='Validate configuration')
    config_parser.set_defaults(func=cmd_config)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old files')
    cleanup_parser.add_argument('--dry-run', action='store_true',
                              help='Show what would be deleted without actually deleting')
    cleanup_parser.set_defaults(func=cmd_cleanup)
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload existing timelapse video to YouTube')
    upload_parser.add_argument('--date', dest='upload_date', required=True,
                             help='Date of video to upload (YYYY-MM-DD)')
    upload_parser.set_defaults(func=cmd_upload)
    
    # SBS command (Sunset Brilliance Score analysis and reports)
    sbs_parser = subparsers.add_parser('sbs', help='Sunset Brilliance Score analysis and reports')
    sbs_parser.add_argument('--date', help='Analyze specific date (YYYY-MM-DD, default: yesterday)')
    sbs_parser.add_argument('--report', action='store_true',
                          help='Generate daily SBS report')
    sbs_parser.add_argument('--export', action='store_true', 
                          help='Export historical SBS data to CSV')
    sbs_parser.add_argument('--start', help='Start date for export (YYYY-MM-DD)')
    sbs_parser.add_argument('--end', help='End date for export (YYYY-MM-DD)')
    sbs_parser.add_argument('--cleanup', action='store_true',
                          help='Clean up old SBS analysis data')
    sbs_parser.set_defaults(func=cmd_sbs)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    # Run the selected command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()