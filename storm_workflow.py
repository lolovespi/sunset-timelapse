"""
Storm Workflow

Storm-specific capture and upload pipeline.
Mirrors the sunset workflow inside sunset_scheduler.complete_daily_workflow,
but with storm-specific scoring, captions, and orchestration.

See docs/superpowers/specs/2026-05-22-storm-capture-design.md for design.
"""

import json
import logging
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from config_manager import get_config


def sis_score_to_grade(score: float) -> str:
    """Map numeric Storm Intensity Score (0-100) to letter grade."""
    if score >= 80:
        return 'A'
    if score >= 60:
        return 'B'
    if score >= 40:
        return 'C'
    if score >= 20:
        return 'D'
    return 'F'


def compute_storm_intensity_score(
    strikes: list,
    observations: list,
    weights: Optional[Dict] = None,
) -> Dict:
    """
    Compute Storm Intensity Score from Tempest sensor data only.

    No video analysis. Pure function over inputs.

    Args:
        strikes: list of LightningStrike from tempest_monitor
        observations: list of WeatherObservation from tempest_monitor
        weights: optional dict overriding default weights (from config)

    Returns:
        {
            'score': float (0-100),
            'grade': str ('A'-'F'),
            'components': {
                'lightning': float,
                'distance': float,
                'rain': float,
                'wind': float,
                'pressure': float,
            },
            'metrics': {
                'lightning_count': int,
                'lightning_avg_distance_km': float | None,
                'rain_max_mm_hr': float,
                'wind_gust_max_mph': float,
                'pressure_drop_hpa': float,
            },
        }
    """
    if weights is None:
        config = get_config()
        weights = {
            'lightning': config.get('storm_analysis.scoring.lightning_weight', 5.0),
            'distance': config.get('storm_analysis.scoring.distance_weight', 20.0),
            'rain': config.get('storm_analysis.scoring.rain_weight', 2.0),
            'wind': config.get('storm_analysis.scoring.wind_weight', 0.5),
            'pressure': config.get('storm_analysis.scoring.pressure_weight', 10.0),
            'max_score': config.get('storm_analysis.scoring.max_score', 100),
        }

    # Lightning metrics
    lightning_count = len(strikes)
    avg_distance = (
        sum(s.distance_km for s in strikes) / lightning_count
        if lightning_count > 0 else None
    )

    # Observation-derived metrics
    rain_max = max((o.rain_rate for o in observations), default=0.0)
    wind_gust_max = max((o.wind_gust for o in observations), default=0.0)

    pressure_drop = 0.0
    if len(observations) >= 2:
        # Drop = max - min over the observation window
        pressures = [o.pressure for o in observations]
        pressure_drop = max(pressures) - min(pressures)

    # Component scores
    lightning_pts = lightning_count * weights['lightning']
    if avg_distance is not None:
        distance_pts = (40.0 / max(avg_distance, 5.0)) * weights['distance']
    else:
        distance_pts = 0.0
    rain_pts = rain_max * weights['rain']
    wind_pts = wind_gust_max * weights['wind']
    pressure_pts = abs(pressure_drop) * weights['pressure']

    total = lightning_pts + distance_pts + rain_pts + wind_pts + pressure_pts
    score = min(total, float(weights['max_score']))

    return {
        'score': round(score, 1),
        'grade': sis_score_to_grade(score),
        'components': {
            'lightning': round(lightning_pts, 2),
            'distance': round(distance_pts, 2),
            'rain': round(rain_pts, 2),
            'wind': round(wind_pts, 2),
            'pressure': round(pressure_pts, 2),
        },
        'metrics': {
            'lightning_count': lightning_count,
            'lightning_avg_distance_km': avg_distance,
            'rain_max_mm_hr': round(rain_max, 2),
            'wind_gust_max_mph': round(wind_gust_max, 2),
            'pressure_drop_hpa': round(pressure_drop, 2),
        },
    }


class StormWorkflow:
    """Orchestrates storm capture: capture → process → SIS → caption → social posts.

    Mirrors SunsetScheduler.complete_daily_workflow but storm-flavored.
    Reuses (does not duplicate) camera, video processor, drive uploader, social uploaders.
    """

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # Reuse the same shared components as SunsetScheduler.
        # Lazy import avoids circular dependency at module load time.
        from camera_interface import CameraInterface
        from video_processor import VideoProcessor
        from drive_uploader import DriveUploader
        from facebook_uploader import FacebookUploader
        from youtube_uploader import YouTubeUploader
        from email_notifier import EmailNotifier

        self.camera = CameraInterface()
        self.video_processor = VideoProcessor()
        self.drive_uploader = DriveUploader()
        self.facebook_uploader = FacebookUploader()
        self.youtube_uploader = YouTubeUploader()
        self.email_notifier = EmailNotifier()

        # Storms directory under data/storms/
        paths = self.config.get_storage_paths()
        self.storms_dir = paths['base'] / 'storms'
        self.storms_dir.mkdir(parents=True, exist_ok=True)

    def _compute_storm_duration_seconds(self) -> int:
        """Capture duration: min(max_duration_hours, post_storm_minutes)."""
        post_minutes = self.config.get('tempest.capture.post_storm_minutes', 60)
        max_hours = self.config.get('tempest.capture.max_duration_hours', 4)
        return int(min(max_hours * 3600, post_minutes * 60))

    def _storm_metadata_path(self, target_date: date) -> Path:
        """Where to persist storm metadata for a given date."""
        d = self.storms_dir / target_date.isoformat()
        d.mkdir(parents=True, exist_ok=True)
        return d / 'storm_metadata.json'

    def _persist_storm_metadata(self, target_date: date, metadata: Dict):
        """Write storm metadata JSON (survives across restarts for deferred recovery)."""
        path = self._storm_metadata_path(target_date)
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        self.logger.info(f"Storm metadata persisted to {path}")

    def complete_storm_workflow(self, *args, **kwargs):
        """Stub — full implementation lands in Task 12."""
        raise NotImplementedError("complete_storm_workflow lands in Task 12")
