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

    def _capture_storm_sequence(
        self,
        start_time: datetime,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[Path]:
        """
        Capture storm frames using the existing camera capture path.

        Duration is fixed (no continuous Tempest polling during capture) — see
        spec non-goals. Cancellation supported between camera chunks (~15 min).

        Args:
            start_time: when capture starts (naive datetime, local time)
            cancel_event: optional threading.Event; capture exits early when set

        Returns:
            List of captured frame paths (possibly empty if camera failed)
        """
        duration_seconds = self._compute_storm_duration_seconds()
        end_time = start_time + timedelta(seconds=duration_seconds)
        interval = self.config.get('capture.interval_seconds', 5)

        self.logger.info(
            f"Storm capture window: {start_time} → {end_time} "
            f"({duration_seconds // 60} min, interval {interval}s)"
        )

        try:
            return self.camera.capture_video_sequence(
                start_time, end_time, interval, cancel_event=cancel_event,
            )
        except Exception as e:
            self.logger.error(f"Storm capture failed: {e}")
            return []

    def _expected_frame_count(self) -> int:
        """How many frames a complete storm capture should yield."""
        duration = self._compute_storm_duration_seconds()
        interval = self.config.get('capture.interval_seconds', 5)
        return duration // interval

    def _should_attempt_recovery(
        self, captured: List[Path], expected_frame_count: int,
    ) -> bool:
        """Recovery triggered when captured < 50% of expected."""
        if expected_frame_count <= 0:
            return False
        return len(captured) < (expected_frame_count // 2)

    def _attempt_immediate_recovery(
        self, target_date: date, start_time: datetime, end_time: datetime,
    ) -> List[Path]:
        """
        Pull camera SD-card recordings for the storm window via historical_retrieval.
        Returns recovered frames (possibly empty if recovery also fails).
        """
        from historical_retrieval import HistoricalRetrieval
        self.logger.info(
            f"Attempting immediate recovery from camera recordings: "
            f"{start_time} → {end_time}"
        )
        try:
            hr = HistoricalRetrieval()
            # HistoricalRetrieval's interface returns processed video paths per-day.
            # For storm windows we need the underlying frame extraction.
            # If the existing API doesn't expose frame-extraction for arbitrary
            # time windows, the simplest path is to use create_historical_timelapse
            # for the day and then point storm_workflow at the resulting video.
            videos = hr.create_historical_timelapse(
                target_date, target_date, upload_to_youtube=False,
            )
            # Recovery succeeded if any video was created for the date.
            if videos:
                self.logger.info(f"Recovery produced {len(videos)} video(s) for {target_date}")
                return [Path(v) for v in videos]
            return []
        except Exception as e:
            self.logger.error(f"Immediate recovery failed: {e}")
            return []

    def _queue_deferred_recovery(
        self, target_date: date, start_time: datetime, end_time: datetime,
        sis: Dict, storm_conditions_summary: Dict,
    ):
        """Queue this storm window for retry during morning maintenance."""
        queue_path = self.storms_dir / 'pending_recovery.json'
        queue: List[Dict] = []
        if queue_path.exists():
            try:
                with open(queue_path) as f:
                    queue = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Could not read pending_recovery.json: {e}")
                queue = []

        queue.append({
            'date': target_date.isoformat(),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'sis': sis,
            'conditions': storm_conditions_summary,
            'queued_at': datetime.now().isoformat(),
        })

        with open(queue_path, 'w') as f:
            json.dump(queue, f, indent=2, default=str)
        self.logger.info(f"Storm queued for deferred recovery in {queue_path}")

    def _collect_tempest_data(
        self, tempest_monitor, window_minutes: int = 90,
    ) -> tuple:
        """Snapshot strikes + observations from the monitor's ring buffers."""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        strikes = [s for s in tempest_monitor.lightning_strikes if s.timestamp >= cutoff]
        observations = [o for o in tempest_monitor.observations if o.timestamp >= cutoff]
        return strikes, observations

    def complete_storm_workflow(
        self,
        conditions,                 # StormConditions
        start_time: datetime,
        tempest_monitor,            # TempestMonitor instance
        cancel_event: Optional[threading.Event] = None,
    ) -> bool:
        """
        Full storm capture pipeline: capture → process → SIS → caption → post.

        Returns True on overall success (some upload failures are non-fatal and
        emit emails; only capture+process failure with no recovery returns False).
        """
        target_date = start_time.date()
        self.logger.info(f"[STORM] Starting workflow for {target_date} at {start_time.time()}")

        try:
            # Step 1: Capture
            captured = self._capture_storm_sequence(start_time, cancel_event=cancel_event)

            # Step 2: Recovery if needed
            expected = self._expected_frame_count()
            recovered_videos: List[Path] = []
            recovered = False
            if self._should_attempt_recovery(captured, expected):
                self.logger.warning(
                    f"[STORM] Only {len(captured)}/{expected} frames captured; "
                    "attempting immediate recovery from camera SD"
                )
                duration = self._compute_storm_duration_seconds()
                end_time = start_time + timedelta(seconds=duration)
                recovered_videos = self._attempt_immediate_recovery(
                    target_date, start_time, end_time,
                )
                recovered = bool(recovered_videos)

            # Step 3: Determine working video path
            if recovered and recovered_videos:
                video_path = recovered_videos[0]
                source_tag = '[RECOVERED]'
            elif captured:
                paths = self.config.get_storage_paths()
                video_output = paths['videos'] / f"storm_{target_date.isoformat()}.mp4"
                video_path = self.video_processor.create_timelapse(
                    captured, video_output,
                )
                if not video_path:
                    self.logger.error("[STORM] Video processing failed")
                    self._handle_capture_failure(
                        target_date, start_time, conditions, tempest_monitor,
                        reason='video_processing_failed',
                    )
                    return False
                source_tag = ''
            else:
                # No frames and no recovery succeeded — queue deferred recovery
                self.logger.error("[STORM] Capture and immediate recovery both failed")
                self._handle_capture_failure(
                    target_date, start_time, conditions, tempest_monitor,
                    reason='capture_and_recovery_failed',
                )
                return False

            # Step 4: Compute SIS from Tempest data
            strikes, observations = self._collect_tempest_data(tempest_monitor, 90)
            sis = compute_storm_intensity_score(strikes, observations)
            self.logger.info(
                f"[STORM] SIS {sis['score']} (Grade {sis['grade']}), "
                f"lightning={sis['metrics']['lightning_count']}, "
                f"gust={sis['metrics']['wind_gust_max_mph']}"
            )

            # Step 5: Build metadata for downstream consumers
            metadata = {
                'event_type': 'storm',
                'date': target_date.isoformat(),
                'capture_date': target_date.isoformat(),
                'start_time': start_time.isoformat(),
                'sis_score': sis['score'],
                'sis_grade': sis['grade'],
                'storm_metrics': sis['metrics'],
                'storm_components': sis['components'],
                'trigger_reasons': list(conditions.trigger_reasons),
                'recovered': recovered,
            }
            self._persist_storm_metadata(target_date, metadata)

            # Step 6: Drive backup
            try:
                drive_result = self.drive_uploader.upload_video(
                    video_path, target_date, metadata,
                )
                if drive_result:
                    self.logger.info(f"[STORM] Drive upload OK: {drive_result.get('filename')}")
            except Exception as e:
                self.logger.warning(f"[STORM] Drive upload failed (non-fatal): {e}")

            # Step 7: Generate one caption for all socials
            try:
                shared_caption = self.facebook_uploader.generate_caption(
                    metadata, video_path=str(video_path),
                )
                self.logger.info(f"[STORM] Caption: {shared_caption[:100]}...")
            except Exception as e:
                self.logger.warning(f"[STORM] Caption generation failed; using fallback: {e}")
                shared_caption = (
                    f"Thunderstorm timelapse from Pelham, AL on "
                    f"{target_date.strftime('%B %d, %Y')}. "
                    f"SIS {sis['score']:.0f}/100 (grade {sis['grade']}). "
                    f"Camera: Reolink RLC810-WA"
                )

            # Step 8: Post to FB+IG+Reels
            try:
                self.facebook_uploader.post_sunset(
                    video_path, metadata, caption_override=shared_caption,
                )
            except Exception as e:
                self.logger.warning(f"[STORM] Facebook/Instagram post failed: {e}")
                self.email_notifier.send_notification(
                    f"{source_tag} Storm FB/IG Post Failed - {target_date}",
                    f"Storm video {video_path} created but FB/IG post failed:\n{e}",
                )

            # Step 9: YouTube
            try:
                youtube_title = f"Storm {target_date.strftime('%m/%d/%y')}"
                description = self._build_youtube_description(
                    shared_caption, sis, metadata,
                )
                self.youtube_uploader.upload_video(
                    video_path, youtube_title, description,
                )
            except Exception as e:
                self.logger.warning(f"[STORM] YouTube upload failed: {e}")
                self.email_notifier.send_notification(
                    f"{source_tag} Storm YouTube Upload Failed - {target_date}",
                    f"Storm video {video_path} created but YouTube upload failed:\n{e}",
                )

            # Step 10: Success email
            subject_prefix = source_tag if source_tag else ''
            self.email_notifier.send_notification(
                f"{subject_prefix} Storm captured - {target_date} (SIS {sis['score']:.0f}/{sis['grade']})".strip(),
                f"Storm timelapse captured and posted.\n\n"
                f"SIS: {sis['score']:.0f}/100 (grade {sis['grade']})\n"
                f"Lightning: {sis['metrics']['lightning_count']} strikes\n"
                f"Peak gust: {sis['metrics']['wind_gust_max_mph']:.0f} mph\n"
                f"Peak rain: {sis['metrics']['rain_max_mm_hr']:.1f} mm/hr\n"
                f"Pressure drop: {sis['metrics']['pressure_drop_hpa']:.1f} hPa\n"
                f"\nVideo: {video_path}",
            )

            self.logger.info(f"[STORM] Workflow completed for {target_date}")
            return True

        except Exception as e:
            self.logger.error(f"[STORM] Workflow exception: {e}", exc_info=True)
            self.email_notifier.send_notification(
                f"Storm Workflow Error - {start_time.date()}",
                f"Unhandled exception in storm workflow:\n\n{str(e)}",
            )
            return False

    def _handle_capture_failure(
        self, target_date: date, start_time: datetime,
        conditions, tempest_monitor, reason: str,
    ):
        """Queue deferred recovery + email notification."""
        try:
            strikes, observations = self._collect_tempest_data(tempest_monitor, 90)
            sis = compute_storm_intensity_score(strikes, observations)
        except Exception:
            sis = {'score': 0.0, 'grade': 'F', 'metrics': {}, 'components': {}}

        duration = self._compute_storm_duration_seconds()
        end_time = start_time + timedelta(seconds=duration)
        self._queue_deferred_recovery(
            target_date, start_time, end_time, sis,
            {'reasons': list(conditions.trigger_reasons), 'confidence': conditions.confidence},
        )
        self.email_notifier.send_notification(
            f"Storm Capture Failed - {target_date}",
            f"Storm capture failed (reason: {reason}). "
            f"Queued for deferred recovery in morning maintenance.\n\n"
            f"SIS estimate from Tempest data: {sis['score']:.0f}/100",
        )

    def _build_youtube_description(
        self, caption: str, sis: Dict, metadata: Dict,
    ) -> str:
        """Compose YouTube description: caption + structured sensor block + hashtags."""
        m = sis['metrics']
        avg_dist = m.get('lightning_avg_distance_km')
        avg_dist_str = f"{avg_dist:.1f}km" if avg_dist is not None else "n/a"
        block = (
            f"\n\n---\n"
            f"Lightning: {m['lightning_count']} strikes, avg distance {avg_dist_str}\n"
            f"Peak wind gust: {m['wind_gust_max_mph']:.0f} mph\n"
            f"Peak rain rate: {m['rain_max_mm_hr']:.1f} mm/hr\n"
            f"Pressure drop: {m['pressure_drop_hpa']:.1f} hPa\n"
            f"Storm Intensity Score: {sis['score']:.0f} (Grade {sis['grade']})\n\n"
            f"#thunderstorm #lightning #timelapse #alabama #alabamawx #weather #pelham"
        )
        return caption + block
