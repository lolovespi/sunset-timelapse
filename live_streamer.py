"""
Live stream coordinator.

Wires together platform broadcast creation (Facebook Live, YouTube Live)
and the FFmpeg RTMP streaming pipeline into a single managed lifecycle.

Phase 1: Facebook only, manual trigger.
"""

import logging
import signal
import time
from datetime import datetime
from typing import List, Optional

from config_manager import get_config
from email_notifier import EmailNotifier
from live_facebook import FacebookLive
from live_youtube import YouTubeLive
from rtmp_streamer import RTMPStreamer


class LiveStreamCoordinator:
    """Orchestrates a live streaming session across platforms."""

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.email = EmailNotifier()

        self.streamer = RTMPStreamer()
        self.facebook: Optional[FacebookLive] = None
        self.youtube: Optional[YouTubeLive] = None
        self._shutdown_requested = False

    def stream(self, duration_minutes: int, platforms: List[str],
               title: Optional[str] = None, description: Optional[str] = None,
               test_mode: bool = False) -> bool:
        """
        Start a live stream to the requested platforms for a given duration.

        Args:
            duration_minutes: How long to stream (capped by stream.max_duration_minutes)
            platforms: List of platform names, e.g. ['facebook']
            title: Broadcast title (default: auto-generated)
            description: Broadcast description
            test_mode: If True, create unpublished broadcast (no public viewers)

        Returns:
            True if stream completed without critical error.
        """
        max_duration = self.config.get('stream.max_duration_minutes', 240)
        duration_minutes = min(duration_minutes, max_duration)

        if not title:
            title = f"Live from Pelham, AL — {datetime.now().strftime('%b %d %I:%M %p')}"
        if not description:
            description = ("Live stream from the Pelham, AL sunset camera. "
                           "Automatically triggered by weather conditions.")

        rtmp_urls: List[str] = []

        # 1. Create broadcast on each platform and collect RTMP URLs
        if 'facebook' in platforms:
            try:
                self.facebook = FacebookLive()
                fb_status = 'UNPUBLISHED' if test_mode else 'LIVE_NOW'
                fb_url = self.facebook.create_broadcast(title, description, status=fb_status)
                if fb_url:
                    rtmp_urls.append(fb_url)
                    self.logger.info(f"Facebook broadcast ready (test_mode={test_mode})")
                else:
                    self.logger.error("Facebook broadcast creation failed")
                    self.email.send_notification(
                        "Live Stream Failed - Facebook Setup",
                        f"Could not create Facebook Live broadcast at {datetime.now().isoformat()}"
                    )
            except Exception as e:
                self.logger.error(f"Facebook setup error: {e}")

        if 'youtube' in platforms:
            try:
                self.youtube = YouTubeLive()
                privacy = 'unlisted' if test_mode else 'public'
                yt_url = self.youtube.create_broadcast(
                    title=title, description=description,
                    privacy_status=privacy, made_for_kids=False,
                )
                if yt_url:
                    rtmp_urls.append(yt_url)
                    self.logger.info(f"YouTube broadcast ready (privacy={privacy})")
                else:
                    self.logger.error("YouTube broadcast creation failed")
                    self.email.send_notification(
                        "Live Stream Failed - YouTube Setup",
                        f"Could not create YouTube Live broadcast at {datetime.now().isoformat()}"
                    )
            except Exception as e:
                self.logger.error(f"YouTube setup error: {e}")

        if not rtmp_urls:
            self.logger.error("No RTMP endpoints available, aborting stream")
            return False

        # 2. Install signal handlers for graceful shutdown (Ctrl-C, systemd stop)
        def _handle_signal(signum, frame):
            self.logger.info(f"Signal {signum} received, stopping stream")
            self._shutdown_requested = True

        old_sigint = signal.signal(signal.SIGINT, _handle_signal)
        old_sigterm = signal.signal(signal.SIGTERM, _handle_signal)

        # 3. Start FFmpeg streaming to all RTMP endpoints
        try:
            if not self.streamer.start(rtmp_urls):
                self.logger.error("FFmpeg streaming failed to start")
                self._cleanup_broadcasts()
                return False

            self.logger.info(f"Streaming for up to {duration_minutes} minutes...")
            self._notify_started(title, duration_minutes)

            # 4. Run for the requested duration, checking for early termination
            deadline = time.time() + duration_minutes * 60
            while time.time() < deadline and not self._shutdown_requested:
                if not self.streamer.is_running():
                    self.logger.error("FFmpeg exited unexpectedly")
                    self.email.send_notification(
                        "Live Stream Ended Unexpectedly",
                        f"FFmpeg exited before {duration_minutes}min duration was reached."
                    )
                    break
                time.sleep(5)

            return True

        finally:
            # 5. Always stop streaming and end broadcasts
            self.streamer.stop()
            self._cleanup_broadcasts()
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)

    def _cleanup_broadcasts(self):
        """End all platform broadcasts."""
        if self.facebook:
            try:
                self.facebook.end_broadcast()
                permalink = self.facebook.permalink()
                if permalink:
                    self.logger.info(f"Facebook Live VOD: {permalink}")
            except Exception as e:
                self.logger.warning(f"Facebook end broadcast error: {e}")

        if self.youtube:
            try:
                self.youtube.end_broadcast()
                watch = self.youtube.watch_url()
                if watch:
                    self.logger.info(f"YouTube Live: {watch}")
            except Exception as e:
                self.logger.warning(f"YouTube end broadcast error: {e}")

    def _notify_started(self, title: str, duration_minutes: int):
        """Send email notification that stream started."""
        body_parts = [
            f"Live stream started at {datetime.now().strftime('%Y-%m-%d %I:%M %p')}",
            f"Title: {title}",
            f"Duration: up to {duration_minutes} minutes",
        ]
        if self.facebook:
            fb = self.facebook.permalink()
            if fb:
                body_parts.append(f"Facebook: {fb}")
        if self.youtube:
            yt = self.youtube.watch_url()
            if yt:
                body_parts.append(f"YouTube: {yt}")
        try:
            self.email.send_notification("Live Stream Started", "\n".join(body_parts))
        except Exception:
            pass
