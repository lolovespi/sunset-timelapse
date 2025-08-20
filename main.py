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
        logger.info("Running immediate capture...")
        success = scheduler.run_immediate_capture()
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
            
    logger.info("All tests passed!")


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
    health = status.get('system_health', {})
    for component, status_val in health.items():
        icon = "✓" if status_val else "✗"
        print(f"{icon} {component.replace('_', ' ').title()}: {status_val}")
        
    print("\n--- Recent Activity ---")
    activity = status.get('recent_activity', {})
    
    recent_images = activity.get('recent_images', [])
    print(f"Recent Images ({len(recent_images)}):")
    for img in recent_images[-3:]:  # Show last 3
        print(f"  {Path(img).name}")
        
    recent_videos = activity.get('recent_videos', [])
    print(f"Recent Videos ({len(recent_videos)}):")
    for vid in recent_videos:
        print(f"  {Path(vid).name}")


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


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Sunset Timelapse System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run daily scheduler (Raspberry Pi)
  python main.py schedule --validate
  
  # Run immediate capture for testing
  python main.py schedule --immediate
  
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
    test_parser.add_argument('--sunset', action='store_true',
                           help='Test sunset calculations')
    test_parser.add_argument('--email', action='store_true',
                           help='Test email notifications')
    test_parser.set_defaults(func=cmd_test)
    
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