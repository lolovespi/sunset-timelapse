"""
Sunset Calculator
Calculates sunset times and capture windows using astral library
Uses zoneinfo for modern timezone handling
"""

import logging
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from typing import Tuple, Optional
from astral import LocationInfo
from astral.sun import sun

from config_manager import get_config


class SunsetCalculator:
    """Calculates sunset times and capture windows"""
    
    def __init__(self):
        """Initialize sunset calculator with location from config"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Get location info from config
        self.city = self.config.get('location.city')
        self.region = self.config.get('location.state', self.config.get('location.country', 'USA'))
        self.timezone_str = self.config.get('location.timezone')
        self.latitude = self.config.get('location.latitude')
        self.longitude = self.config.get('location.longitude')
        
        # Validate required location data
        if not all([self.city, self.timezone_str, 
                   self.latitude is not None, self.longitude is not None]):
            raise ValueError("Incomplete location configuration. Please check config.yaml")
            
        # Create timezone object
        try:
            self.timezone = ZoneInfo(self.timezone_str)
        except Exception as e:
            raise ValueError(f"Invalid timezone '{self.timezone_str}': {e}")
            
        # Create location info for astral
        self.location = LocationInfo(
            name=self.city,
            region=self.region,
            timezone=self.timezone_str,
            latitude=self.latitude,
            longitude=self.longitude
        )
        
        self.logger.info(f"Sunset calculator initialized for {self.city}, {self.region}")
        self.logger.info(f"Coordinates: {self.latitude:.4f}, {self.longitude:.4f}")
        
    def get_sunset_time(self, target_date: date) -> datetime:
        """
        Get sunset time for a specific date
        
        Args:
            target_date: Date to calculate sunset for
            
        Returns:
            Sunset datetime in local timezone
        """
        try:
            # Create a datetime at midnight in the local timezone for the target date
            # This ensures Astral calculates for the correct local date
            local_midnight = datetime.combine(target_date, datetime.min.time())
            local_midnight = local_midnight.replace(tzinfo=self.timezone)
            
            s = sun(self.location.observer, date=local_midnight)
            sunset_utc = s['sunset']
            
            # Convert to local timezone
            sunset_local = sunset_utc.astimezone(self.timezone)
            
            self.logger.debug(f"Sunset on {target_date}: {sunset_local.strftime('%H:%M:%S %Z')}")
            return sunset_local
            
        except Exception as e:
            self.logger.error(f"Failed to calculate sunset for {target_date}: {e}")
            raise
            
    def get_capture_window(self, target_date: date) -> Tuple[datetime, datetime]:
        """
        Get capture window (start and end times) for a specific date
        
        Args:
            target_date: Date to calculate capture window for
            
        Returns:
            Tuple of (start_time, end_time) in local timezone
        """
        sunset_time = self.get_sunset_time(target_date)
        
        # Get time offsets from config
        before_minutes = self.config.get('capture.time_before_sunset_minutes', 60)
        after_minutes = self.config.get('capture.time_after_sunset_minutes', 60)
        
        start_time = sunset_time - timedelta(minutes=before_minutes)
        end_time = sunset_time + timedelta(minutes=after_minutes)
        
        self.logger.info(f"Capture window for {target_date}:")
        self.logger.info(f"  Start: {start_time.strftime('%H:%M:%S %Z')}")
        self.logger.info(f"  Sunset: {sunset_time.strftime('%H:%M:%S %Z')}")
        self.logger.info(f"  End: {end_time.strftime('%H:%M:%S %Z')}")
        
        return start_time, end_time
        
    def get_today_capture_window(self) -> Tuple[datetime, datetime]:
        """
        Get capture window for today
        
        Returns:
            Tuple of (start_time, end_time) in local timezone
        """
        today = date.today()
        return self.get_capture_window(today)
        
    def get_next_capture_window(self) -> Tuple[datetime, datetime]:
        """
        Get the next capture window (today if not passed, tomorrow otherwise)
        
        Returns:
            Tuple of (start_time, end_time) in local timezone
        """
        now = datetime.now(self.timezone)
        today = now.date()
        
        start_time, end_time = self.get_capture_window(today)
        
        # If today's capture window has passed, get tomorrow's
        if now > end_time:
            tomorrow = today + timedelta(days=1)
            start_time, end_time = self.get_capture_window(tomorrow)
            self.logger.info("Today's capture window has passed, using tomorrow's window")
            
        return start_time, end_time
        
    def time_until_next_capture(self) -> timedelta:
        """
        Get time remaining until next capture window starts
        
        Returns:
            Timedelta until next capture starts
        """
        now = datetime.now(self.timezone)
        start_time, _ = self.get_next_capture_window()
        
        time_until = start_time - now
        
        if time_until.total_seconds() <= 0:
            # We're in the capture window now
            return timedelta(0)
            
        return time_until
        
    def is_capture_time(self) -> bool:
        """
        Check if we're currently in a capture window
        
        Returns:
            True if we should be capturing now, False otherwise
        """
        now = datetime.now(self.timezone)
        start_time, end_time = self.get_today_capture_window()
        
        return start_time <= now <= end_time
        
    def get_capture_duration(self) -> timedelta:
        """
        Get total duration of capture window
        
        Returns:
            Timedelta representing total capture duration
        """
        before_minutes = self.config.get('capture.time_before_sunset_minutes', 60)
        after_minutes = self.config.get('capture.time_after_sunset_minutes', 60)
        
        return timedelta(minutes=before_minutes + after_minutes)
        
    def get_expected_image_count(self) -> int:
        """
        Get expected number of images in a capture session
        
        Returns:
            Expected number of images
        """
        duration = self.get_capture_duration()
        interval = self.config.get('capture.interval_seconds', 5)
        
        total_seconds = duration.total_seconds()
        image_count = int(total_seconds / interval) + 1  # +1 for the first image
        
        return image_count
        
    def get_weekly_schedule(self, start_date: Optional[date] = None) -> list:
        """
        Get sunset times for the next 7 days
        
        Args:
            start_date: Start date for schedule (default: today)
            
        Returns:
            List of (date, sunset_time, start_time, end_time) tuples
        """
        if start_date is None:
            start_date = date.today()
            
        schedule = []
        
        for i in range(7):
            current_date = start_date + timedelta(days=i)
            sunset_time = self.get_sunset_time(current_date)
            start_time, end_time = self.get_capture_window(current_date)
            
            schedule.append((current_date, sunset_time, start_time, end_time))
            
        return schedule
        
    def format_schedule(self, schedule: list = None) -> str:
        """
        Format schedule for display
        
        Args:
            schedule: Schedule from get_weekly_schedule() (default: next 7 days)
            
        Returns:
            Formatted schedule string
        """
        if schedule is None:
            schedule = self.get_weekly_schedule()
            
        lines = ["Sunset Schedule:"]
        lines.append("-" * 60)
        lines.append(f"{'Date':<12} {'Sunset':<10} {'Start':<10} {'End':<10} Duration")
        lines.append("-" * 60)
        
        for date_obj, sunset, start, end in schedule:
            duration = end - start
            duration_str = f"{duration.total_seconds()/3600:.1f}h"
            
            lines.append(
                f"{date_obj.strftime('%m/%d/%Y'):<12} "
                f"{sunset.strftime('%H:%M'):<10} "
                f"{start.strftime('%H:%M'):<10} "
                f"{end.strftime('%H:%M'):<10} "
                f"{duration_str}"
            )
            
        return "\n".join(lines)
        
    def validate_location(self) -> bool:
        """
        Validate location configuration by testing sunset calculation
        
        Returns:
            True if location is valid, False otherwise
        """
        try:
            today = date.today()
            sunset_time = self.get_sunset_time(today)
            
            # Basic sanity checks
            if not isinstance(sunset_time, datetime):
                return False
                
            # Check if sunset time is reasonable (between 4 PM and 9 PM)
            hour = sunset_time.hour
            if not (16 <= hour <= 21):
                self.logger.warning(f"Sunset time seems unusual: {sunset_time.strftime('%H:%M')}")
                
            self.logger.info("Location validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Location validation failed: {e}")
            return False