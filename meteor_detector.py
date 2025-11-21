#!/usr/bin/env python3
"""
Meteor Detection Module for Sunset Timelapse System
Analyzes video footage to detect and extract meteor events
"""

import cv2
import numpy as np
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import tempfile
from zoneinfo import ZoneInfo
import platform
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from config_manager import get_config
from historical_retrieval import HistoricalRetrieval
from sunset_calculator import SunsetCalculator


class MeteorCandidate:
    """Represents a potential meteor detection"""
    def __init__(self, start_frame: int, start_time: datetime):
        self.start_frame = start_frame
        self.end_frame = start_frame
        self.start_time = start_time
        self.end_time = start_time
        self.positions: List[Tuple[int, int]] = []  # (x, y) centroid positions
        self.brightness_values: List[float] = []
        self.frame_count = 0
        self.max_brightness = 0.0
        self.is_valid = False

    def add_detection(self, frame_num: int, centroid: Tuple[int, int], brightness: float, timestamp: datetime):
        """Add a detection to this candidate"""
        self.end_frame = frame_num
        self.end_time = timestamp
        self.positions.append(centroid)
        self.brightness_values.append(brightness)
        self.frame_count += 1
        self.max_brightness = max(self.max_brightness, brightness)

    def get_linearity_score(self) -> float:
        """Calculate how linear the path is (0-1, higher is more linear)"""
        if len(self.positions) < 3:
            return 0.0

        # Fit line to points and calculate R-squared
        x_coords = np.array([p[0] for p in self.positions])
        y_coords = np.array([p[1] for p in self.positions])

        # Calculate correlation coefficient
        if np.std(x_coords) == 0 or np.std(y_coords) == 0:
            return 1.0  # Perfectly vertical or horizontal line

        correlation = np.corrcoef(x_coords, y_coords)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0

    def get_velocity(self) -> float:
        """Calculate average velocity in pixels per frame"""
        if len(self.positions) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i-1][0]
            dy = self.positions[i][1] - self.positions[i-1][1]
            total_distance += np.sqrt(dx*dx + dy*dy)

        return total_distance / (len(self.positions) - 1)

    def has_consistent_motion(self) -> bool:
        """Check if motion is consistent (no blinking/stopping)"""
        if len(self.positions) < 3:
            return False

        # Check for large gaps in brightness (blinking = aircraft)
        brightness_diffs = np.diff(self.brightness_values)
        max_diff = np.max(np.abs(brightness_diffs))
        avg_brightness = np.mean(self.brightness_values)

        # If brightness drops by more than 50%, it's likely blinking
        if max_diff > 0.5 * avg_brightness:
            return False

        return True


class MeteorDetector:
    """Detects meteors in video footage using frame-by-frame analysis"""

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.historical = HistoricalRetrieval()

        # Detection parameters (configurable)
        self.min_brightness_threshold = self.config.get('meteor.min_brightness_threshold', 200)
        self.min_area = self.config.get('meteor.min_area', 10)
        self.max_area = self.config.get('meteor.max_area', 5000)
        self.min_frames = self.config.get('meteor.min_frames', 3)
        self.max_frames = self.config.get('meteor.max_frames', 30)
        self.min_linearity = self.config.get('meteor.min_linearity', 0.85)
        self.min_velocity = self.config.get('meteor.min_velocity', 5.0)
        self.max_velocity = self.config.get('meteor.max_velocity', 25.0)
        self.max_gap_frames = self.config.get('meteor.max_gap_frames', 2)
        self.clip_padding_seconds = self.config.get('meteor.clip_padding_seconds', 3)

        # Sky region filtering (to ignore ground/horizon with car lights)
        self.sky_region_top = self.config.get('meteor.sky_region_top', 0.0)
        self.sky_region_bottom = self.config.get('meteor.sky_region_bottom', 0.5)
        self.logger.info(f"Sky region: top {self.sky_region_top*100:.0f}% to {self.sky_region_bottom*100:.0f}% of frame")

        # Storage paths
        paths = self.config.get_storage_paths()
        self.meteor_dir = paths['base'] / 'meteors'
        self.meteor_dir.mkdir(parents=True, exist_ok=True)

        # Timezone from config for localizing timestamps
        self.timezone_str = self.config.get('location.timezone', 'America/Chicago')
        self.sunset_calc = SunsetCalculator()

    def _to_local_timezone(self, dt: datetime) -> datetime:
        """Convert datetime to local timezone (CST/CDT)"""
        # Get timezone from config
        local_tz = ZoneInfo(self.timezone_str)

        # If datetime is naive (no timezone), assume it's already in local time
        # Camera recordings come from Reolink API in local time without timezone info
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=local_tz)
            return dt

        # Convert to local timezone if it has timezone info
        return dt.astimezone(local_tz)

    def search_date_range(self, start_date: date, end_date: date,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> List[Dict]:
        """
        Search for meteors in video recordings from camera

        Args:
            start_date: Start date for search
            end_date: End date for search
            start_time: Optional specific start time (default: dusk)
            end_time: Optional specific end time (default: dawn)

        Returns:
            List of detected meteor events with metadata
        """
        self.logger.info(f"Starting meteor search from {start_date} to {end_date}")

        detected_meteors = []
        current_date = start_date

        while current_date <= end_date:
            self.logger.info(f"Searching {current_date} for meteor events...")

            # Get videos for this date
            date_meteors = self._search_single_date(current_date, start_time, end_time)
            detected_meteors.extend(date_meteors)

            self.logger.info(f"Found {len(date_meteors)} potential meteors on {current_date}")
            current_date += timedelta(days=1)

        self.logger.info(f"Total meteors detected: {len(detected_meteors)}")
        return detected_meteors

    def _search_single_date(self, target_date: date,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None) -> List[Dict]:
        """Search a single date for meteor events"""

        # Default to nighttime hours if not specified
        # Use astronomical twilight (sun 18° below horizon) for truly dark sky
        if start_time is None:
            use_astronomical = self.config.get('meteor.use_astronomical_twilight', True)
            if use_astronomical:
                # Use astronomical dusk - when sky is truly dark
                start_time = self.sunset_calc.get_astronomical_dusk(target_date)
                self.logger.info(f"Using astronomical dusk start time: {start_time.strftime('%H:%M')} (truly dark sky)")
            else:
                # Fall back to sunset offset
                sunset_offset = self.config.get('meteor.sunset_offset_minutes', 90)
                sunset_time = self.sunset_calc.get_sunset_time(target_date)
                start_time = sunset_time + timedelta(minutes=sunset_offset)
                self.logger.info(f"Using sunset-based start time: {start_time.strftime('%H:%M')} (sunset at {sunset_time.strftime('%H:%M')}, +{sunset_offset}min)")

        if end_time is None:
            use_astronomical = self.config.get('meteor.use_astronomical_twilight', True)
            if use_astronomical:
                # Use astronomical dawn next day - when sky starts to brighten
                next_day = target_date + timedelta(days=1)
                end_time = self.sunset_calc.get_astronomical_dawn(next_day)
                self.logger.info(f"Using astronomical dawn end time: {end_time.strftime('%H:%M')} (sky starts brightening)")
            else:
                # Fall back to sunrise offset
                sunrise_offset = self.config.get('meteor.sunrise_offset_minutes', 90)
                next_day = target_date + timedelta(days=1)
                sunrise_time = self.sunset_calc.get_sunrise_time(next_day)
                end_time = sunrise_time - timedelta(minutes=sunrise_offset)
                self.logger.info(f"Using sunrise-based end time: {end_time.strftime('%H:%M')} (sunrise at {sunrise_time.strftime('%H:%M')}, -{sunrise_offset}min)")

        # Get recordings from camera for this time window
        recordings = self.historical.get_camera_recordings(target_date, target_date)

        if not recordings:
            self.logger.warning(f"No recordings found for {target_date}")
            return []

        # Filter to nighttime recordings only
        nighttime_recordings = []
        for recording in recordings:
            rec_start = recording.get('start_time')
            rec_end = recording.get('end_time')

            if rec_start and rec_end:
                # Make recording times timezone-aware to match start_time/end_time
                # Camera recordings are in local time without timezone info
                if rec_start.tzinfo is None:
                    rec_start = rec_start.replace(tzinfo=start_time.tzinfo)
                if rec_end.tzinfo is None:
                    rec_end = rec_end.replace(tzinfo=start_time.tzinfo)

                # Check overlap with search window
                if rec_end < start_time or rec_start > end_time:
                    continue

            nighttime_recordings.append(recording)

        if not nighttime_recordings:
            self.logger.warning(f"No nighttime recordings found for {target_date}")
            return []

        self.logger.info(f"Processing {len(nighttime_recordings)} nighttime recordings")

        # Use parallel processing on Mac/desktop, sequential on Raspberry Pi
        # Check for ARM/low-power systems via multiple methods
        machine = platform.machine().lower()
        is_arm = 'arm' in machine or 'aarch' in machine
        is_raspberry_pi = os.path.exists('/proc/device-tree/model')

        # Only enable parallel processing on Mac/Windows or powerful Linux desktops
        is_desktop = (platform.system() in ['Darwin', 'Windows']) and not is_raspberry_pi

        # For Linux, only enable parallel if NOT ARM and NOT Raspberry Pi
        if platform.system() == 'Linux':
            is_desktop = not is_arm and not is_raspberry_pi

        if is_desktop and len(nighttime_recordings) > 1:
            # Parallel processing for desktop (Mac/Windows/Linux x86)
            # Limit to 3 workers to avoid overwhelming camera's connection limit
            max_workers = min(os.cpu_count() or 2, 3)  # Cap at 3 workers (camera session limit)
            self.logger.info(f"Using parallel processing with {max_workers} workers")
            detected_meteors = self._process_recordings_parallel(
                nighttime_recordings, target_date, max_workers
            )
        else:
            # Sequential processing for Raspberry Pi or single recording
            if not is_desktop:
                self.logger.info("Using sequential processing (Raspberry Pi detected)")
            detected_meteors = self._process_recordings_sequential(
                nighttime_recordings, target_date
            )

        return detected_meteors

    def _process_recordings_sequential(self, recordings: List[Dict],
                                      target_date: date) -> List[Dict]:
        """Process recordings sequentially (for Raspberry Pi)"""
        detected_meteors = []

        for recording in recordings:
            self.logger.debug(f"Analyzing recording: {recording.get('name', 'unknown')}")

            # Download recording to temp location
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / recording['name']

                if self.historical.download_recording(recording, temp_path):
                    # Analyze video for meteors
                    recording_start = recording.get('start_time')
                    meteors = self._analyze_video(temp_path, target_date, recording_start)

                    for meteor in meteors:
                        meteor['source_recording'] = recording.get('name', 'unknown')
                        meteor['date'] = target_date.isoformat()
                        detected_meteors.append(meteor)
                else:
                    self.logger.warning(f"Failed to download {recording.get('name')}")

        return detected_meteors

    def _process_recordings_parallel(self, recordings: List[Dict],
                                    target_date: date, max_workers: int) -> List[Dict]:
        """Process recordings in parallel (for Mac/desktop)"""
        detected_meteors = []

        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all recording analysis tasks
            future_to_recording = {
                executor.submit(
                    _analyze_recording_worker,
                    recording,
                    target_date
                ): recording for recording in recordings
            }

            # Collect results as they complete
            for future in as_completed(future_to_recording):
                recording = future_to_recording[future]
                try:
                    meteors = future.result()
                    if meteors:
                        detected_meteors.extend(meteors)
                        self.logger.info(f"Found {len(meteors)} meteor(s) in {recording.get('name', 'unknown')}")
                except Exception as e:
                    self.logger.error(f"Error processing {recording.get('name', 'unknown')}: {e}")

        return detected_meteors

    def _analyze_video(self, video_path: Path, target_date: date,
                       recording_start: Optional[datetime] = None) -> List[Dict]:
        """
        Analyze a video file for meteor events

        Args:
            video_path: Path to video file
            target_date: Date of recording
            recording_start: Actual start time of the recording (for accurate timestamps)

        Returns:
            List of meteor detection results
        """
        self.logger.info(f"Analyzing video: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"Could not open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.debug(f"Video: {total_frames} frames at {fps:.2f} FPS")

        # Background subtractor for motion detection
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=50, detectShadows=False
        )

        # Track active meteor candidates
        active_candidates: List[MeteorCandidate] = []
        confirmed_meteors: List[Dict] = []
        single_frame_detections: List[Dict] = []

        frame_num = 0
        prev_gray = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate timestamp for this frame
                if recording_start:
                    # Use actual recording start time + frame offset
                    timestamp = recording_start + timedelta(seconds=frame_num / fps)
                else:
                    # Fallback to midnight + frame offset (for local video analysis)
                    timestamp = datetime.combine(target_date, datetime.min.time()) + \
                               timedelta(seconds=frame_num / fps)

                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Apply background subtraction
                fg_mask = bg_subtractor.apply(frame)

                # Find bright spots that are moving
                detections = self._detect_bright_moving_objects(gray, fg_mask, prev_gray)

                # Also check for single-frame elongated streaks (fast meteors)
                single_frame = self._detect_single_frame_meteors(gray, fg_mask, frame_num, timestamp)
                single_frame_detections.extend(single_frame)

                # Update active candidates with new detections
                active_candidates = self._update_candidates(
                    active_candidates, detections, frame_num, timestamp
                )

                # Check for completed meteor tracks
                for candidate in list(active_candidates):
                    if frame_num - candidate.end_frame > self.max_gap_frames:
                        # Candidate has ended, validate it
                        if self._validate_meteor_candidate(candidate):
                            meteor_info = self._extract_meteor_clip(
                                video_path, candidate, fps, target_date
                            )
                            if meteor_info:
                                confirmed_meteors.append(meteor_info)
                        active_candidates.remove(candidate)

                prev_gray = gray
                frame_num += 1

                # Progress logging
                if frame_num % 1000 == 0:
                    self.logger.debug(f"Processed {frame_num}/{total_frames} frames")

        finally:
            cap.release()

        # Check any remaining candidates
        for candidate in active_candidates:
            if self._validate_meteor_candidate(candidate):
                meteor_info = self._extract_meteor_clip(video_path, candidate, fps, target_date)
                if meteor_info:
                    confirmed_meteors.append(meteor_info)

        # Deduplicate multi-frame detections (remove overlapping/consecutive detections of same meteor)
        if len(confirmed_meteors) > 1:
            original_count = len(confirmed_meteors)
            confirmed_meteors = self._deduplicate_multiframe_detections(confirmed_meteors)
            if len(confirmed_meteors) < original_count:
                self.logger.info(f"Deduplicated {original_count - len(confirmed_meteors)} overlapping multi-frame detections")

        # Consolidate single-frame detections (group nearby frames as same event)
        consolidated_sf = self._consolidate_single_frame_detections(single_frame_detections)
        self.logger.info(f"Consolidated {len(single_frame_detections)} single-frame detections into {len(consolidated_sf)} events")

        # Filter out single-frame detections that overlap with multi-frame detections
        # (same meteor detected by both methods)
        non_overlapping_sf = []
        for sf_meteor in consolidated_sf:
            sf_frame = sf_meteor['frame_num']
            overlaps = False

            for mf_meteor in confirmed_meteors:
                # Check if single-frame detection is within the frame range of multi-frame detection
                # Extract frame info from multi-frame meteor
                mf_start = mf_meteor.get('start_frame', 0)
                mf_end = mf_meteor.get('end_frame', 0)

                # If single-frame is within 30 frames of multi-frame detection, it's the same meteor
                if mf_start > 0 and mf_end > 0:
                    if mf_start - 30 <= sf_frame <= mf_end + 30:
                        overlaps = True
                        self.logger.debug(f"Single-frame meteor at frame {sf_frame} overlaps with multi-frame detection (frames {mf_start}-{mf_end})")
                        break
                else:
                    # Fallback: check by Y-coordinate proximity and time
                    mf_y = mf_meteor['path_start'][1] if mf_meteor.get('path_start') else 0
                    sf_y = sf_meteor['centroid'][1]
                    if abs(mf_y - sf_y) < 150:  # Within 150px vertically = likely same meteor
                        overlaps = True
                        self.logger.debug(f"Single-frame meteor at y={sf_y} overlaps with multi-frame at y={mf_y}")
                        break

            if not overlaps:
                non_overlapping_sf.append(sf_meteor)

        if len(non_overlapping_sf) < len(consolidated_sf):
            self.logger.info(f"Filtered {len(consolidated_sf) - len(non_overlapping_sf)} single-frame detections that overlap with multi-frame detections")

        # Extract clips for non-overlapping single-frame detections
        for sf_meteor in non_overlapping_sf:
            meteor_info = self._extract_single_frame_clip(video_path, sf_meteor, fps, target_date)
            if meteor_info:
                confirmed_meteors.append(meteor_info)

        # Aircraft filter: detect multiple detections that form a linear path across the frame
        # Aircraft produce multiple single-frame detections as they cross the field of view
        confirmed_meteors = self._filter_aircraft_detections(confirmed_meteors)

        self.logger.info(f"Found {len(confirmed_meteors)} meteors in {video_path.name}")
        return confirmed_meteors

    def _detect_bright_moving_objects(self, gray: np.ndarray, fg_mask: np.ndarray,
                                       prev_gray: Optional[np.ndarray]) -> List[Dict]:
        """
        Detect bright objects that are moving

        Returns list of detections with centroid and brightness
        """
        detections = []

        # Calculate sky region boundaries in pixels
        frame_height = gray.shape[0]
        sky_top_px = int(frame_height * self.sky_region_top)
        sky_bottom_px = int(frame_height * self.sky_region_bottom)

        # Find bright regions in current frame
        _, bright_mask = cv2.threshold(gray, self.min_brightness_threshold, 255, cv2.THRESH_BINARY)

        # Combine with motion mask
        combined_mask = cv2.bitwise_and(bright_mask, fg_mask)

        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if self.min_area <= area <= self.max_area:
                # Calculate centroid
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # Filter by sky region - ignore bottom portion (ground/cars)
                    if cy < sky_top_px or cy > sky_bottom_px:
                        continue  # Outside sky region, skip (probably car lights)

                    # Calculate brightness at centroid
                    brightness = float(gray[cy, cx])

                    detections.append({
                        'centroid': (cx, cy),
                        'brightness': brightness,
                        'area': area
                    })

        return detections

    def _update_candidates(self, candidates: List[MeteorCandidate],
                           detections: List[Dict], frame_num: int,
                           timestamp: datetime) -> List[MeteorCandidate]:
        """Update meteor candidates with new frame detections"""

        # Match detections to existing candidates
        used_detections = set()

        for candidate in candidates:
            if not candidate.positions:
                continue

            last_pos = candidate.positions[-1]
            best_match = None
            best_distance = float('inf')

            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue

                # Calculate distance to last known position
                dx = detection['centroid'][0] - last_pos[0]
                dy = detection['centroid'][1] - last_pos[1]
                distance = np.sqrt(dx*dx + dy*dy)

                # Match if reasonably close (meteor shouldn't jump too far)
                max_distance = self.min_velocity * 3 * (frame_num - candidate.end_frame)
                if distance < max_distance and distance < best_distance:
                    best_match = i
                    best_distance = distance

            if best_match is not None:
                detection = detections[best_match]
                candidate.add_detection(
                    frame_num, detection['centroid'],
                    detection['brightness'], timestamp
                )
                used_detections.add(best_match)

        # Start new candidates for unmatched detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                new_candidate = MeteorCandidate(frame_num, timestamp)
                new_candidate.add_detection(
                    frame_num, detection['centroid'],
                    detection['brightness'], timestamp
                )
                candidates.append(new_candidate)

        return candidates

    def _validate_meteor_candidate(self, candidate: MeteorCandidate) -> bool:
        """
        Validate if a candidate is actually a meteor

        Criteria:
        - Minimum number of frames (distinguishes from noise)
        - Maximum number of frames (meteors are brief)
        - Linear path (no erratic movement)
        - Consistent brightness (no blinking = not aircraft)
        - Minimum velocity (stationary objects eliminated)
        - Motion angle (reject mostly horizontal motion = planes)
        - Brightness variation (reject steady lights = planes)
        """

        # Check frame count
        if candidate.frame_count < self.min_frames:
            self.logger.debug(f"Rejected: too few frames ({candidate.frame_count})")
            return False

        if candidate.frame_count > self.max_frames:
            self.logger.debug(f"Rejected: too many frames ({candidate.frame_count})")
            return False

        # Check linearity
        linearity = candidate.get_linearity_score()
        if linearity < self.min_linearity:
            self.logger.debug(f"Rejected: path not linear enough ({linearity:.2f})")
            return False

        # Check velocity
        velocity = candidate.get_velocity()
        if velocity < self.min_velocity:
            self.logger.debug(f"Rejected: too slow ({velocity:.2f} px/frame)")
            return False

        if velocity > self.max_velocity:
            self.logger.debug(f"Rejected: too fast ({velocity:.2f} px/frame, likely aircraft)")
            return False

        # Check for consistent motion (no blinking)
        if not candidate.has_consistent_motion():
            self.logger.debug("Rejected: inconsistent motion (likely aircraft)")
            return False

        # Additional airplane rejection: check motion angle
        # Planes typically move horizontally, meteors have more vertical component
        if len(candidate.positions) >= 2:
            start_pos = candidate.positions[0]
            end_pos = candidate.positions[-1]
            dx = abs(end_pos[0] - start_pos[0])
            dy = abs(end_pos[1] - start_pos[1])

            # Calculate angle from horizontal (0 = horizontal, 90 = vertical)
            if dx > 0:
                angle_from_horizontal = np.degrees(np.arctan(dy / dx))

                # Reject if motion is too horizontal (< 30 degrees from horizontal = likely plane)
                # Increased from 15° to 30° to catch more diagonal plane paths
                if angle_from_horizontal < 30:
                    self.logger.debug(f"Rejected: too horizontal ({angle_from_horizontal:.1f}° from horizontal, likely aircraft)")
                    return False

        # Brightness variance check DISABLED
        # This check was rejecting legitimate meteors with steady brightness
        # (confirmed Nov 15 meteor had CV=0.006, well below the 0.25 threshold)
        # Airplanes are already filtered by: max_frames (30), motion angle (>30°), min_velocity
        #
        # Original logic (now disabled):
        # if len(candidate.brightness_values) >= 5:
        #     brightness_array = np.array(candidate.brightness_values)
        #     brightness_std = np.std(brightness_array)
        #     brightness_mean = np.mean(brightness_array)
        #     if brightness_mean > 0:
        #         brightness_cv = brightness_std / brightness_mean
        #         if brightness_cv < 0.25:
        #             return False

        # Additional check: frame duration vs distance traveled (acceleration test)
        # Meteors maintain constant velocity, planes can appear to accelerate/decelerate
        if len(candidate.positions) >= 5:
            # Calculate velocity for first half vs second half
            mid_point = len(candidate.positions) // 2

            first_half_start = candidate.positions[0]
            first_half_end = candidate.positions[mid_point]
            first_half_dist = np.sqrt((first_half_end[0] - first_half_start[0])**2 +
                                     (first_half_end[1] - first_half_start[1])**2)
            first_half_frames = mid_point

            second_half_start = candidate.positions[mid_point]
            second_half_end = candidate.positions[-1]
            second_half_dist = np.sqrt((second_half_end[0] - second_half_start[0])**2 +
                                       (second_half_end[1] - second_half_start[1])**2)
            second_half_frames = len(candidate.positions) - mid_point

            if first_half_frames > 0 and second_half_frames > 0:
                first_velocity = first_half_dist / first_half_frames
                second_velocity = second_half_dist / second_half_frames

                # Calculate velocity ratio (should be close to 1.0 for meteors)
                if first_velocity > 0:
                    velocity_ratio = second_velocity / first_velocity

                    # Reject if velocity changes too much (< 0.7 or > 1.3 = likely plane)
                    if velocity_ratio < 0.7 or velocity_ratio > 1.3:
                        self.logger.debug(f"Rejected: inconsistent velocity (ratio={velocity_ratio:.2f}, likely aircraft)")
                        return False

        self.logger.info(f"Valid meteor candidate: {candidate.frame_count} frames, "
                        f"linearity={linearity:.2f}, velocity={velocity:.2f}")
        return True

    def _filter_aircraft_detections(self, meteors: List[Dict]) -> List[Dict]:
        """
        Filter out aircraft detections from meteor list.

        Aircraft produce multiple single-frame detections that form a linear path
        across the field of view over the duration of a video (~2 minutes).

        Detection criteria:
        - 3+ single-frame detections in same video
        - X positions form a monotonic sequence (increasing or decreasing)
        - Detections span significant portion of frame width

        Args:
            meteors: List of meteor detection dictionaries

        Returns:
            Filtered list with aircraft detections removed
        """
        if len(meteors) < 3:
            return meteors

        # Get single-frame detections only (aircraft appear as single-frame streaks)
        single_frame = [m for m in meteors if m.get('detection_type') == 'single_frame']
        multi_frame = [m for m in meteors if m.get('detection_type') != 'single_frame']

        if len(single_frame) < 3:
            return meteors

        # Sort by timestamp/frame number to check spatial progression
        single_frame_sorted = sorted(single_frame, key=lambda m: m.get('timestamp', ''))

        # Extract X positions (path_start[0] for single-frame detections)
        x_positions = []
        for m in single_frame_sorted:
            if m.get('path_start'):
                x_positions.append(m['path_start'][0])
            elif m.get('centroid'):
                x_positions.append(m['centroid'][0])
            else:
                x_positions.append(None)

        # Filter out None values
        valid_positions = [(i, x) for i, x in enumerate(x_positions) if x is not None]

        if len(valid_positions) < 3:
            return meteors

        # Check if positions are monotonically increasing or decreasing (aircraft pattern)
        x_vals = [x for _, x in valid_positions]

        # Check monotonic increasing (west to east)
        is_increasing = all(x_vals[i] <= x_vals[i+1] for i in range(len(x_vals)-1))
        # Check monotonic decreasing (east to west)
        is_decreasing = all(x_vals[i] >= x_vals[i+1] for i in range(len(x_vals)-1))

        # Check if span is significant (at least 30% of frame width, assume 1920px)
        x_span = max(x_vals) - min(x_vals)
        frame_width = 1920  # Assumed frame width
        span_ratio = x_span / frame_width

        if (is_increasing or is_decreasing) and span_ratio > 0.3:
            # This looks like an aircraft traversing the frame
            self.logger.warning(
                f"Aircraft detected: {len(single_frame)} single-frame detections "
                f"forming {'west-to-east' if is_increasing else 'east-to-west'} path "
                f"(span: {span_ratio*100:.0f}% of frame). Rejecting all."
            )
            # Return only multi-frame detections (aircraft rarely produce these)
            return multi_frame

        return meteors

    def _detect_single_frame_meteors(self, gray: np.ndarray, fg_mask: np.ndarray,
                                      frame_num: int, timestamp: datetime) -> List[Dict]:
        """
        Detect single-frame meteor streaks based on elongation

        Fast meteors appear as elongated bright streaks in a single frame
        """
        single_frame_meteors = []

        # Calculate sky region boundaries in pixels
        frame_height = gray.shape[0]
        sky_top_px = int(frame_height * self.sky_region_top)
        sky_bottom_px = int(frame_height * self.sky_region_bottom)

        # Find bright regions
        _, bright_mask = cv2.threshold(gray, self.min_brightness_threshold, 255, cv2.THRESH_BINARY)

        # Combine with motion mask
        combined_mask = cv2.bitwise_and(bright_mask, fg_mask)

        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            # Check size bounds
            if area < self.min_area or area > self.max_area:
                continue

            # Check for elongation (meteor streaks are long and thin)
            if len(contour) >= 5:
                # Fit ellipse to get aspect ratio
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (center, axes, angle) = ellipse
                    major_axis = max(axes)
                    minor_axis = min(axes)

                    if minor_axis > 0:
                        aspect_ratio = major_axis / minor_axis

                        # Meteor streaks should be elongated (aspect ratio > 3)
                        # and bright
                        if aspect_ratio > 3.0 and major_axis > 20:
                            # Get centroid first to check sky region
                            M = cv2.moments(contour)
                            if M['m00'] > 0:
                                cx = int(M['m10'] / M['m00'])
                                cy = int(M['m01'] / M['m00'])

                                # Filter by sky region - ignore bottom portion (ground/cars)
                                if cy < sky_top_px or cy > sky_bottom_px:
                                    continue  # Outside sky region, skip (probably car lights)

                                # Calculate average brightness along the streak
                                mask_roi = np.zeros_like(gray)
                                cv2.drawContours(mask_roi, [contour], 0, 255, -1)
                                mean_brightness = cv2.mean(gray, mask=mask_roi)[0]

                                if mean_brightness > self.min_brightness_threshold:
                                    # Single-frame detections re-enabled with strict filters
                                    # Key insight: Meteors are BRIEF (1-6 frames), airplanes are LONG (30+ frames)
                                    # Let max_frames filter reject long airplane tracks

                                    # 1. Reject extremely bright objects (insects very close to IR)
                                    if mean_brightness > 245:
                                        self.logger.debug(f"Rejected single-frame: too bright ({mean_brightness:.0f}, likely insect)")
                                        continue

                                    # 2. Reject extreme aspect ratios (motion blur artifacts)
                                    if aspect_ratio > 20 or aspect_ratio < 3:
                                        self.logger.debug(f"Rejected single-frame: aspect ratio {aspect_ratio:.1f}")
                                        continue

                                    # 3. Require minimum streak length (reject noise/hot pixels)
                                    if major_axis < 15:
                                        self.logger.debug(f"Rejected single-frame: too short ({major_axis:.0f}px)")
                                        continue

                                    # 4. Reject extremely long streaks (likely aircraft)
                                    # For single-frame, streak length ~= velocity in px/frame
                                    if major_axis > self.max_velocity:
                                        self.logger.debug(f"Rejected single-frame: too fast ({major_axis:.0f}px, likely aircraft)")
                                        continue

                                    # Calculate angle from horizontal for logging
                                    # cv2.fitEllipse returns angle 0-180°, where 0° and 180° are horizontal
                                    # We want angle from horizontal (0° = horizontal, 90° = vertical)
                                    angle_from_horizontal = min(angle, 180 - angle)

                                    self.logger.info(f"Single-frame meteor detected! "
                                                   f"Frame {frame_num}, y={cy}, aspect_ratio={aspect_ratio:.1f}, "
                                                   f"length={major_axis:.0f}px, brightness={mean_brightness:.0f}, "
                                                   f"angle={angle_from_horizontal:.1f}° from horizontal")

                                    single_frame_meteors.append({
                                        'frame_num': frame_num,
                                        'timestamp': timestamp,
                                        'centroid': (cx, cy),
                                        'brightness': mean_brightness,
                                        'aspect_ratio': aspect_ratio,
                                        'length': major_axis,
                                        'angle': angle,
                                        'area': area
                                    })
                except cv2.error:
                    pass  # fitEllipse can fail on some contours

        return single_frame_meteors

    def _consolidate_single_frame_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Group single-frame detections that are close together in time/space as the same meteor event.

        This prevents multiple clips from the same meteor (e.g., frames 332-343 all showing same event).
        Returns the best detection from each group (longest streak, highest brightness).
        """
        if not detections:
            return []

        # Sort by frame number
        sorted_dets = sorted(detections, key=lambda x: x['frame_num'])

        # Group detections that are within 15 frames of each other and spatially close
        groups = []
        current_group = [sorted_dets[0]]

        for det in sorted_dets[1:]:
            last_det = current_group[-1]
            frame_gap = det['frame_num'] - last_det['frame_num']

            # Check spatial distance (Y-coordinate mainly, since meteor moves)
            cy_diff = abs(det['centroid'][1] - last_det['centroid'][1])

            # If within 15 frames and within 100px vertically, likely same meteor
            if frame_gap <= 15 and cy_diff < 100:
                current_group.append(det)
            else:
                groups.append(current_group)
                current_group = [det]

        groups.append(current_group)

        # Convert grouped single-frame detections into multi-frame meteor representations
        consolidated = []
        for group in groups:
            if len(group) == 1:
                # Single isolated frame - keep as single-frame detection
                self.logger.debug(f"Single-frame detection at frame {group[0]['frame_num']}")
                consolidated.append(group[0])
            else:
                # Multiple frames - create multi-frame meteor representation
                # This allows us to track meteors detected across multiple frames
                start_frame = group[0]['frame_num']
                end_frame = group[-1]['frame_num']

                # Store all frame data for multi-frame clip extraction
                meteor_data = {
                    'frame_num': start_frame,  # Use first frame as reference
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'frame_count': len(group),
                    'timestamp': group[0]['timestamp'],
                    'frames': group,  # Store all individual frame detections
                    'is_multi_frame_group': True,  # Flag to indicate this is grouped
                }

                self.logger.debug(f"Consolidated {len(group)} detections (frames {start_frame}-{end_frame}) "
                                f"into multi-frame meteor ({len(group)} frames)")
                consolidated.append(meteor_data)

        return consolidated

    def _deduplicate_multiframe_detections(self, meteors: List[Dict]) -> List[Dict]:
        """
        Remove duplicate multi-frame detections of the same meteor event.

        When the tracker loses and re-acquires the same meteor (due to frame gaps),
        we get multiple clips. Keep only the best one (longest duration, most frames).
        """
        if len(meteors) <= 1:
            return meteors

        # Sort by start frame
        sorted_meteors = sorted(meteors, key=lambda m: m.get('start_frame', 0))

        deduplicated = []
        current_group = [sorted_meteors[0]]

        for meteor in sorted_meteors[1:]:
            last_meteor = current_group[-1]

            # Check if this meteor is close enough to be the same event
            # If start frame is within 30 frames of last meteor's end frame = same event
            last_end = last_meteor.get('end_frame', 0)
            current_start = meteor.get('start_frame', 0)

            if current_start - last_end <= 30:
                # Same meteor event, add to group
                current_group.append(meteor)
            else:
                # Different event, keep best from current group
                best = max(current_group, key=lambda m: m['duration_frames'])
                deduplicated.append(best)
                # Delete the other clips
                for m in current_group:
                    if m != best:
                        clip_path = Path(m['clip_path'])
                        if clip_path.exists():
                            clip_path.unlink()
                            self.logger.debug(f"Removed duplicate clip: {clip_path.name}")
                        json_path = clip_path.with_suffix('.json')
                        if json_path.exists():
                            json_path.unlink()
                current_group = [meteor]

        # Handle last group
        if current_group:
            best = max(current_group, key=lambda m: m['duration_frames'])
            deduplicated.append(best)
            for m in current_group:
                if m != best:
                    clip_path = Path(m['clip_path'])
                    if clip_path.exists():
                        clip_path.unlink()
                    json_path = clip_path.with_suffix('.json')
                    if json_path.exists():
                        json_path.unlink()

        return deduplicated

    def _extract_meteor_clip(self, source_video: Path, candidate: MeteorCandidate,
                             fps: float, target_date: date) -> Optional[Dict]:
        """
        Extract a video clip containing the meteor event

        Returns meteor metadata including clip path
        """

        # Calculate clip boundaries with padding
        padding_frames = int(self.clip_padding_seconds * fps)
        start_frame = max(0, candidate.start_frame - padding_frames)
        end_frame = candidate.end_frame + padding_frames

        # Generate output filename with timezone-aware timestamp: meteor-MM-DD-YYYY-HH-MM-SS-AM-CST.mp4
        clip_time = candidate.start_time
        local_time = self._to_local_timezone(clip_time)
        tz_abbr = local_time.strftime('%Z')  # CST or CDT
        filename = f"meteor-{local_time.strftime('%m-%d-%Y-%I-%M-%S-%p')}-{tz_abbr}.mp4"
        output_path = self.meteor_dir / filename

        # Extract clip using OpenCV
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            self.logger.error(f"Could not open source video for clip extraction")
            return None

        try:
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Write frames
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()

            self.logger.info(f"Extracted meteor clip: {filename}")

            # Create metadata
            meteor_info = {
                'clip_path': str(output_path),
                'filename': filename,
                'timestamp': candidate.start_time.isoformat(),
                'duration_frames': candidate.frame_count,
                'duration_seconds': candidate.frame_count / fps,
                'max_brightness': candidate.max_brightness,
                'linearity_score': candidate.get_linearity_score(),
                'velocity': candidate.get_velocity(),
                'path_start': candidate.positions[0] if candidate.positions else None,
                'path_end': candidate.positions[-1] if candidate.positions else None,
                'start_frame': candidate.start_frame,
                'end_frame': candidate.end_frame,
                'detection_type': 'multi_frame'
            }

            # Save metadata alongside clip
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(meteor_info, f, indent=2, default=str)

            return meteor_info

        except Exception as e:
            self.logger.error(f"Error extracting meteor clip: {e}")
            return None
        finally:
            cap.release()

    def _extract_single_frame_clip(self, source_video: Path, sf_meteor: Dict,
                                    fps: float, target_date: date) -> Optional[Dict]:
        """
        Extract a video clip containing a single-frame meteor streak OR
        a multi-frame meteor created from grouped single-frame detections

        Returns meteor metadata including clip path
        """

        # Check if this is a multi-frame grouped detection
        is_multi_frame = sf_meteor.get('is_multi_frame_group', False)

        # Calculate clip boundaries with padding
        padding_frames = int(self.clip_padding_seconds * fps)

        if is_multi_frame:
            # Multi-frame meteor: use actual start/end frames
            frame_num = sf_meteor['start_frame']
            meteor_start = sf_meteor['start_frame']
            meteor_end = sf_meteor['end_frame']
            start_frame = max(0, meteor_start - padding_frames)
            end_frame = meteor_end + padding_frames
        else:
            # Single-frame meteor
            frame_num = sf_meteor['frame_num']
            start_frame = max(0, frame_num - padding_frames)
            end_frame = frame_num + padding_frames

        # Generate output filename with timezone-aware timestamp
        clip_time = sf_meteor['timestamp']
        local_time = self._to_local_timezone(clip_time)
        tz_abbr = local_time.strftime('%Z')  # CST or CDT
        filename = f"meteor-{local_time.strftime('%m-%d-%Y-%I-%M-%S-%p')}-{tz_abbr}.mp4"
        output_path = self.meteor_dir / filename

        # Check for duplicate (same timestamp)
        if output_path.exists():
            # Append frame number to make unique
            filename = f"meteor-{local_time.strftime('%m-%d-%Y-%I-%M-%S-%p')}-{tz_abbr}-f{frame_num}.mp4"
            output_path = self.meteor_dir / filename

        # Extract clip using OpenCV
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            self.logger.error(f"Could not open source video for clip extraction")
            return None

        try:
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Write frames
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()

            # Log extraction type
            if is_multi_frame:
                self.logger.info(f"Extracted multi-frame grouped meteor clip: {filename}")
            else:
                self.logger.info(f"Extracted single-frame meteor clip: {filename}")

            # Create metadata based on detection type
            if is_multi_frame:
                # Calculate proper multi-frame metadata
                frames = sf_meteor['frames']
                frame_count = len(frames)

                # Calculate velocity and linearity from path
                positions = [f['centroid'] for f in frames]
                max_brightness = max(f['brightness'] for f in frames)

                # Calculate linearity (how straight the path is)
                if len(positions) >= 2:
                    # Use numpy polyfit to measure linearity
                    x_coords = np.array([p[0] for p in positions])
                    y_coords = np.array([p[1] for p in positions])

                    # Fit line and calculate R² score
                    if np.std(x_coords) > 0:
                        coeffs = np.polyfit(x_coords, y_coords, 1)
                        y_fit = np.polyval(coeffs, x_coords)
                        ss_res = np.sum((y_coords - y_fit) ** 2)
                        ss_tot = np.sum((y_coords - np.mean(y_coords)) ** 2)
                        linearity = 1 - (ss_res / (ss_tot + 1e-6))  # R²
                    else:
                        linearity = 1.0  # Vertical line

                    # Calculate velocity (average px/frame)
                    total_distance = np.sqrt((positions[-1][0] - positions[0][0])**2 +
                                            (positions[-1][1] - positions[0][1])**2)
                    # Use actual number of detected frames, not frame number span
                    # (there may be gaps where meteor wasn't detected)
                    frame_span = len(positions) - 1
                    velocity = total_distance / max(frame_span, 1)
                else:
                    linearity = 1.0
                    velocity = frames[0]['length']

                meteor_info = {
                    'clip_path': str(output_path),
                    'filename': filename,
                    'timestamp': sf_meteor['timestamp'].isoformat(),
                    'duration_frames': frame_count,
                    'duration_seconds': frame_count / fps,
                    'max_brightness': max_brightness,
                    'linearity_score': linearity,
                    'velocity': velocity,
                    'path_start': positions[0],
                    'path_end': positions[-1],
                    'start_frame': meteor_start,
                    'end_frame': meteor_end,
                    'detection_type': 'multi_frame_grouped'
                }
            else:
                # Single-frame meteor metadata
                meteor_info = {
                    'clip_path': str(output_path),
                    'filename': filename,
                    'timestamp': sf_meteor['timestamp'].isoformat(),
                    'duration_frames': 1,
                    'duration_seconds': 1 / fps,
                    'max_brightness': sf_meteor['brightness'],
                    'linearity_score': 1.0,  # Single frame = perfect linearity
                    'velocity': sf_meteor['length'],  # Length is velocity proxy for single frame
                    'aspect_ratio': sf_meteor['aspect_ratio'],
                    'streak_length_px': sf_meteor['length'],
                    'streak_angle': sf_meteor['angle'],
                    'path_start': sf_meteor['centroid'],
                    'path_end': sf_meteor['centroid'],
                    'detection_type': 'single_frame'
                }

            # Save metadata alongside clip
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(meteor_info, f, indent=2, default=str)

            return meteor_info

        except Exception as e:
            self.logger.error(f"Error extracting single-frame meteor clip: {e}")
            return None
        finally:
            cap.release()

    def analyze_local_video(self, video_path: Path,
                           detection_date: Optional[date] = None) -> List[Dict]:
        """
        Analyze a local video file for meteors (without downloading from camera)

        Args:
            video_path: Path to local video file
            detection_date: Date to use for naming (default: today)

        Returns:
            List of detected meteor events
        """
        if detection_date is None:
            detection_date = date.today()

        if not video_path.exists():
            self.logger.error(f"Video file not found: {video_path}")
            return []

        return self._analyze_video(video_path, detection_date)

    def get_detection_stats(self) -> Dict:
        """Get statistics about meteor detections"""

        stats = {
            'total_detections': 0,
            'detections_by_date': {},
            'avg_brightness': 0.0,
            'avg_velocity': 0.0,
            'clips_directory': str(self.meteor_dir)
        }

        # Count clips and gather stats
        meteor_clips = list(self.meteor_dir.glob('meteor-*.mp4'))
        stats['total_detections'] = len(meteor_clips)

        brightness_values = []
        velocity_values = []

        for clip in meteor_clips:
            metadata_file = clip.with_suffix('.json')
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    # Extract date from filename
                    date_str = clip.stem.split('-')[1:4]
                    date_key = f"{date_str[0]}-{date_str[1]}-{date_str[2]}"

                    if date_key not in stats['detections_by_date']:
                        stats['detections_by_date'][date_key] = 0
                    stats['detections_by_date'][date_key] += 1

                    brightness_values.append(metadata.get('max_brightness', 0))
                    velocity_values.append(metadata.get('velocity', 0))

                except Exception as e:
                    self.logger.warning(f"Could not read metadata for {clip}: {e}")

        if brightness_values:
            stats['avg_brightness'] = np.mean(brightness_values)
        if velocity_values:
            stats['avg_velocity'] = np.mean(velocity_values)

        return stats


# Worker function for parallel processing (must be at module level for multiprocessing)
def _analyze_recording_worker(recording: Dict, target_date: date) -> List[Dict]:
    """
    Worker function for parallel video analysis

    This must be at module level for ProcessPoolExecutor to pickle it.
    Creates its own MeteorDetector instance to avoid sharing state.
    """
    import time

    # Each worker creates its own detector instance
    detector = MeteorDetector()
    detected_meteors = []

    # Download recording to temp location with retry logic for 503 errors
    max_retries = 3
    retry_delay = 2  # seconds

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / recording['name']

        download_success = False
        for attempt in range(max_retries):
            try:
                if detector.historical.download_recording(recording, temp_path):
                    download_success = True
                    break
                else:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            except Exception as e:
                if '503' in str(e) or 'connection' in str(e).lower():
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    break  # Don't retry non-503 errors

        if download_success:
            # Analyze video for meteors
            recording_start = recording.get('start_time')
            meteors = detector._analyze_video(temp_path, target_date, recording_start)

            for meteor in meteors:
                meteor['source_recording'] = recording.get('name', 'unknown')
                meteor['date'] = target_date.isoformat()
                detected_meteors.append(meteor)

    return detected_meteors


def search_meteors(start_date: date, end_date: date,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[Dict]:
    """
    Convenience function to search for meteors

    Args:
        start_date: Start date of search
        end_date: End date of search
        start_time: Optional specific start time
        end_time: Optional specific end time

    Returns:
        List of detected meteor events
    """
    detector = MeteorDetector()
    return detector.search_date_range(start_date, end_date, start_time, end_time)


def analyze_video_file(video_path: str, detection_date: Optional[date] = None) -> List[Dict]:
    """
    Convenience function to analyze a local video file

    Args:
        video_path: Path to video file
        detection_date: Optional date for the detection

    Returns:
        List of detected meteor events
    """
    detector = MeteorDetector()
    return detector.analyze_local_video(Path(video_path), detection_date)
