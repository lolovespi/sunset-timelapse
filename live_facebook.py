"""
Facebook Live broadcast lifecycle via the Graph API.

Uses the existing page_access_token from facebook_config.json.
Requires the `publish_video` permission to be granted on the app.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import requests

from config_manager import get_config


class FacebookLive:
    """Manages a single Facebook Live broadcast."""

    GRAPH_API_VERSION = "v19.0"
    GRAPH_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # Load page credentials from facebook_config.json
        config_path = Path(__file__).parent / 'facebook_config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"facebook_config.json not found at {config_path}")
        with open(config_path) as f:
            fb_config = json.load(f)

        self.page_id = fb_config.get('page_id')
        self.page_access_token = fb_config.get('page_access_token')
        if not self.page_id or not self.page_access_token:
            raise ValueError("facebook_config.json missing page_id or page_access_token")

        # Broadcast state
        self.video_id: Optional[str] = None
        self.stream_url: Optional[str] = None
        self.secure_stream_url: Optional[str] = None
        self.title: Optional[str] = None
        self.description: Optional[str] = None

    def create_broadcast(self, title: str, description: str = "",
                         status: str = "LIVE_NOW") -> Optional[str]:
        """
        Create a live broadcast and get its RTMP stream URL.

        Args:
            title: Broadcast title
            description: Broadcast description
            status: LIVE_NOW (immediate) or UNPUBLISHED (test mode, no viewers)

        Returns:
            Secure RTMP URL for FFmpeg, or None on failure.
        """
        url = f"{self.GRAPH_URL}/{self.page_id}/live_videos"
        data = {
            'status': status,
            'title': title,
            'description': description,
            'access_token': self.page_access_token,
        }

        try:
            r = requests.post(url, data=data, timeout=30)
        except requests.RequestException as e:
            self.logger.error(f"Facebook Live create failed: {e}")
            return None

        if r.status_code != 200:
            self.logger.error(f"Facebook Live create failed [{r.status_code}]: {r.text}")
            return None

        resp = r.json()
        self.video_id = resp.get('id')
        self.stream_url = resp.get('stream_url')
        self.secure_stream_url = resp.get('secure_stream_url') or self.stream_url
        self.title = title
        self.description = description

        self.logger.info(f"Facebook Live broadcast created: id={self.video_id} status={status}")
        return self.secure_stream_url

    def get_status(self) -> Optional[str]:
        """Fetch current broadcast status (LIVE, VOD, UNPUBLISHED, etc.)."""
        if not self.video_id:
            return None
        try:
            r = requests.get(
                f"{self.GRAPH_URL}/{self.video_id}",
                params={'fields': 'status', 'access_token': self.page_access_token},
                timeout=15,
            )
            if r.status_code == 200:
                return r.json().get('status')
        except requests.RequestException as e:
            self.logger.warning(f"Facebook Live status fetch failed: {e}")
        return None

    def end_broadcast(self) -> bool:
        """Gracefully end the broadcast. Video becomes a VOD on the page."""
        if not self.video_id:
            return False
        try:
            r = requests.post(
                f"{self.GRAPH_URL}/{self.video_id}",
                data={'end_live_video': 'true', 'access_token': self.page_access_token},
                timeout=30,
            )
            if r.status_code == 200:
                self.logger.info(f"Facebook Live broadcast ended: {self.video_id}")
                return True
            self.logger.warning(f"Facebook Live end failed [{r.status_code}]: {r.text}")
        except requests.RequestException as e:
            self.logger.error(f"Facebook Live end request failed: {e}")
        return False

    def permalink(self) -> Optional[str]:
        """Return a URL users can watch the live (or VOD) at."""
        if not self.video_id:
            return None
        try:
            r = requests.get(
                f"{self.GRAPH_URL}/{self.video_id}",
                params={'fields': 'permalink_url', 'access_token': self.page_access_token},
                timeout=10,
            )
            if r.status_code == 200:
                path = r.json().get('permalink_url')
                if path:
                    return f"https://www.facebook.com{path}"
        except requests.RequestException:
            pass
        return None
