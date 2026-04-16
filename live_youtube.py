"""
YouTube Live broadcast lifecycle via the Data API v3.

Uses the same OAuth credentials as uploads (scope 'youtube' covers both).
Broadcasts use enableAutoStart/enableAutoStop so the stream auto-transitions
to live when FFmpeg sends data and auto-completes when it disconnects.
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from youtube_uploader import YouTubeUploader


class YouTubeLive:
    """Manages a single YouTube Live broadcast."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.uploader = YouTubeUploader()
        if not self.uploader.authenticate():
            raise RuntimeError("YouTube authentication failed")
        self.service = self.uploader.service

        # Broadcast state
        self.broadcast_id: Optional[str] = None
        self.stream_id: Optional[str] = None
        self.rtmp_url: Optional[str] = None
        self.stream_key: Optional[str] = None

    def create_broadcast(self, title: str, description: str = "",
                         privacy_status: str = "public",
                         made_for_kids: bool = False) -> Optional[str]:
        """
        Create a live broadcast + stream, bind them together, and return the
        RTMP ingestion URL that FFmpeg should push to.

        Broadcast uses auto-start/auto-stop so it transitions to live when
        RTMP data begins flowing, and completes when the stream disconnects.

        Args:
            title: Broadcast title
            description: Broadcast description
            privacy_status: "public", "unlisted", or "private"
            made_for_kids: True if the content is aimed at children (usually False)

        Returns:
            Full RTMP URL (ingestion_address + stream_key) for FFmpeg, or None
            on failure.
        """
        now = datetime.now(timezone.utc)
        scheduled_start = (now + timedelta(seconds=30)).isoformat()

        # 1. Create the broadcast
        broadcast_body = {
            "snippet": {
                "title": title,
                "description": description,
                "scheduledStartTime": scheduled_start,
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": made_for_kids,
            },
            "contentDetails": {
                "enableAutoStart": True,
                "enableAutoStop": True,
                "enableDvr": True,
                "recordFromStart": True,
                "startWithSlate": False,
                "monitorStream": {"enableMonitorStream": False},
            },
        }

        try:
            broadcast = self.service.liveBroadcasts().insert(
                part="snippet,status,contentDetails",
                body=broadcast_body,
            ).execute()
        except Exception as e:
            self.logger.error(f"YouTube liveBroadcasts.insert failed: {e}")
            return None

        self.broadcast_id = broadcast["id"]
        self.logger.info(f"YouTube broadcast created: {self.broadcast_id}")

        # 2. Create the stream resource (1080p variable bitrate)
        stream_body = {
            "snippet": {"title": f"Stream for {self.broadcast_id}"},
            "cdn": {
                "frameRate": "variable",
                "ingestionType": "rtmp",
                "resolution": "variable",
            },
            "contentDetails": {"isReusable": False},
        }

        try:
            stream = self.service.liveStreams().insert(
                part="snippet,cdn,contentDetails",
                body=stream_body,
            ).execute()
        except Exception as e:
            self.logger.error(f"YouTube liveStreams.insert failed: {e}")
            self._cleanup_broadcast_only()
            return None

        self.stream_id = stream["id"]
        ingest = stream["cdn"]["ingestionInfo"]
        self.rtmp_url = ingest["ingestionAddress"]
        self.stream_key = ingest["streamName"]

        # 3. Bind broadcast ↔ stream
        try:
            self.service.liveBroadcasts().bind(
                part="id,contentDetails",
                id=self.broadcast_id,
                streamId=self.stream_id,
            ).execute()
        except Exception as e:
            self.logger.error(f"YouTube liveBroadcasts.bind failed: {e}")
            self._cleanup()
            return None

        full_rtmp = f"{self.rtmp_url}/{self.stream_key}"
        self.logger.info(f"YouTube Live ready: broadcast={self.broadcast_id} stream={self.stream_id}")
        return full_rtmp

    def watch_url(self) -> Optional[str]:
        """Return the public watch URL for this broadcast."""
        if not self.broadcast_id:
            return None
        return f"https://www.youtube.com/watch?v={self.broadcast_id}"

    def end_broadcast(self) -> bool:
        """
        End the broadcast. With enableAutoStop, this usually happens automatically
        when FFmpeg disconnects, but we explicitly transition to complete as a
        fallback in case auto-stop didn't fire.
        """
        if not self.broadcast_id:
            return False

        # If auto-stop already moved it to 'complete', this will return a 4xx -
        # that's fine, the broadcast is already done.
        try:
            self.service.liveBroadcasts().transition(
                broadcastStatus="complete",
                id=self.broadcast_id,
                part="id,status",
            ).execute()
            self.logger.info(f"YouTube broadcast transitioned to complete: {self.broadcast_id}")
            return True
        except Exception as e:
            # Commonly: "Invalid transition" because auto-stop already fired
            msg = str(e)
            if "Invalid transition" in msg or "redundantTransition" in msg:
                self.logger.info(f"YouTube broadcast already ended (auto-stop): {self.broadcast_id}")
                return True
            self.logger.warning(f"YouTube broadcast end failed: {e}")
            return False

    def _cleanup_broadcast_only(self):
        """Delete just the broadcast (used when stream creation failed)."""
        if self.broadcast_id:
            try:
                self.service.liveBroadcasts().delete(id=self.broadcast_id).execute()
                self.logger.info(f"Cleaned up broadcast: {self.broadcast_id}")
            except Exception as e:
                self.logger.warning(f"Broadcast cleanup failed: {e}")
            self.broadcast_id = None

    def _cleanup(self):
        """Delete both broadcast and stream (used on bind failure)."""
        self._cleanup_broadcast_only()
        if self.stream_id:
            try:
                self.service.liveStreams().delete(id=self.stream_id).execute()
                self.logger.info(f"Cleaned up stream: {self.stream_id}")
            except Exception as e:
                self.logger.warning(f"Stream cleanup failed: {e}")
            self.stream_id = None
