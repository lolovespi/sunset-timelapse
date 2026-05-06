"""
Facebook Uploader
Handles posting timelapse videos to Facebook with AI-generated captions
"""

import logging
import os
import json
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    logging.warning("Anthropic library not available. Facebook caption generation will be disabled.")
    ANTHROPIC_AVAILABLE = False

from config_manager import get_config
from email_notifier import EmailNotifier


class FacebookUploader:
    """Handles Facebook video uploads with AI-generated captions"""

    # Facebook Graph API version
    GRAPH_API_VERSION = "v19.0"
    GRAPH_VIDEO_URL = f"https://graph-video.facebook.com/{GRAPH_API_VERSION}"

    # Hashtags to append to all posts
    HASHTAGS = "#Pelham #Alabama #SunsetTimelapse #Sunset #BirminghamAL #AlabamaSky #SunsetLovers #Timelapse #GoldenHour #NaturePhotography"

    def __init__(self):
        """Initialize Facebook uploader"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.email_notifier = EmailNotifier()
        self.anthropic_client = None

        # Load Facebook configuration
        self.facebook_config = self._load_facebook_config()
        self.page_id = self.facebook_config.get('page_id')
        self.page_access_token = self.facebook_config.get('page_access_token')
        self.instagram_account_id = self.facebook_config.get('instagram_account_id')

        # Load Anthropic API key
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

        # Initialize Anthropic client
        if ANTHROPIC_AVAILABLE and self.anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
            self.logger.info("Anthropic client initialized")
        else:
            if not ANTHROPIC_AVAILABLE:
                self.logger.warning("Anthropic library not available")
            if not self.anthropic_api_key:
                self.logger.warning("ANTHROPIC_API_KEY not set in environment")

        # Tracking database path
        self.tracking_db_path = self._get_tracking_db_path()

        self.logger.info("Facebook uploader initialized")

    def _load_facebook_config(self) -> Dict[str, str]:
        """Load Facebook configuration from facebook_config.json"""
        config_path = Path(__file__).parent / 'facebook_config.json'

        if not config_path.exists():
            self.logger.warning(f"Facebook config not found at {config_path}")
            return {}

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info("Facebook configuration loaded")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load Facebook config: {e}")
            return {}

    def _get_tracking_db_path(self) -> Path:
        """Get path to tracking database"""
        # Store in project root for simplicity
        return Path(__file__).parent / 'facebook_posts.json'

    def load_tracking_db(self) -> Dict[str, Any]:
        """Load the tracking database"""
        if not self.tracking_db_path.exists():
            return {}

        try:
            with open(self.tracking_db_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load tracking database: {e}")
            return {}

    def save_tracking_db(self, data: Dict[str, Any]):
        """Save the tracking database"""
        try:
            with open(self.tracking_db_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.debug("Tracking database saved")
        except Exception as e:
            self.logger.error(f"Failed to save tracking database: {e}")

    def update_tracking_db(self, date_str: str, status: str, facebook_id: Optional[str] = None,
                           instagram_id: Optional[str] = None):
        """Update tracking database with posting status.

        Merges with existing entry so Facebook and Instagram tracking are independent.
        """
        db = self.load_tracking_db()

        entry = db.get(date_str, {})
        entry['status'] = status
        entry['updated_at'] = datetime.now(timezone.utc).isoformat()

        if facebook_id:
            entry['facebook_id'] = facebook_id
            entry['facebook_posted_at'] = datetime.now(timezone.utc).isoformat()

        if instagram_id:
            entry['instagram_id'] = instagram_id
            entry['instagram_posted_at'] = datetime.now(timezone.utc).isoformat()

        if status == 'posted':
            entry['posted_at'] = datetime.now(timezone.utc).isoformat()

        db[date_str] = entry
        self.save_tracking_db(db)
        self.logger.info(f"Updated tracking database: {date_str} -> {status}")

    def _build_fallback_caption(self, metadata: Dict[str, Any]) -> str:
        """Build a caption from metadata when AI generation is unavailable"""
        date_str = metadata.get('capture_date', metadata.get('date', ''))
        try:
            from datetime import datetime as dt
            d = dt.fromisoformat(date_str) if date_str else None
            date_display = d.strftime('%B %d, %Y') if d else 'today'
        except (ValueError, TypeError):
            date_display = date_str or 'today'

        weather = metadata.get('weather', {})
        parts = []
        if weather.get('conditions') and weather['conditions'] not in ('unknown', 'Unknown'):
            parts.append(weather['conditions'])
        if weather.get('temperature_f') is not None:
            parts.append(f"{weather['temperature_f']}°F")
        if weather.get('humidity_pct') is not None:
            parts.append(f"{weather['humidity_pct']}% humidity")
        if weather.get('wind_speed_mph') is not None and weather['wind_speed_mph'] > 3:
            parts.append(f"wind {weather['wind_speed_mph']} mph")

        sbs_score = metadata.get('sbs_score')
        sbs_part = ""
        if sbs_score is not None:
            try:
                sbs_part = f" SBS: {float(sbs_score):.0f}."
            except (ValueError, TypeError):
                pass

        weather_str = f" {', '.join(parts)}." if parts else ""
        caption = f"{date_display}, Pelham, AL.{weather_str}{sbs_part} Camera: Reolink RLC810-WA"

        return caption

    def generate_caption(self, metadata: Dict[str, Any]) -> str:
        """
        Generate a casual Facebook caption using Anthropic API

        Args:
            metadata: Dictionary containing weather data, visual analysis, SBS score, etc.

        Returns:
            Generated caption (2-3 sentences, casual tone)
        """
        if not self.anthropic_client:
            self.logger.error("Anthropic client not initialized. Cannot generate caption.")
            return self._build_fallback_caption(metadata)

        try:
            # Build context for the AI
            prompt = self._build_caption_prompt(metadata)

            # Call Anthropic API
            message = self.anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            caption = message.content[0].text.strip()
            self.logger.info(f"Generated caption: {caption}")
            return caption

        except Exception as e:
            self.logger.error(f"Failed to generate caption with Anthropic API: {e}")
            return self._build_fallback_caption(metadata)

    def generate_youtube_title(self, metadata: Dict[str, Any]) -> str:
        """
        Generate a short AI title for YouTube (max ~70 chars).

        Always includes the date. Falls back to "Sunset MM/DD/YY" if AI fails.
        """
        try:
            from datetime import datetime as dt
            date_str = metadata.get('date') or metadata.get('capture_date', '')
            d = dt.fromisoformat(date_str) if date_str else dt.now()
            date_display = d.strftime('%m/%d/%y')
        except (ValueError, TypeError):
            from datetime import datetime as dt
            d = dt.now()
            date_display = d.strftime('%m/%d/%y')

        fallback = f"Sunset {date_display}"

        if not self.anthropic_client:
            return fallback

        weather = metadata.get('weather') or {}
        visual = metadata.get('visual_analysis') or {}
        sbs_score = metadata.get('sbs_score')

        # Build sky summary from SBS (reliable) + weather + dominant colors
        sky_parts = []
        if weather.get('conditions') and weather['conditions'] not in ('unknown', 'Unknown'):
            sky_parts.append(weather['conditions'])
        if sbs_score is not None:
            try:
                s = float(sbs_score)
                if s >= 80:
                    sky_parts.append("vivid color")
                elif s >= 70:
                    sky_parts.append("good color")
                elif s >= 60:
                    sky_parts.append("some color")
                elif s >= 50:
                    sky_parts.append("limited color")
                else:
                    sky_parts.append("flat/gray")
            except (ValueError, TypeError):
                pass
        # Add dominant colors from visual analysis if available
        dominant_hex = self._get_dominant_colors_from_visual(visual)
        if dominant_hex:
            sky_parts.append(f"dominant colors: {', '.join(dominant_hex)}")
        sky_line = '; '.join(sky_parts) if sky_parts else 'no data'

        prompt = f"""Write a YouTube title for a sunset timelapse video.

Data:
- Date: {date_display}
- Sky: {sky_line}

Format: "Sunset {date_display} - " followed by 2-5 words about the sky.
Direct and plain. No hype. Vary the phrasing.

Examples:
  "Sunset {date_display} - Flat Gray, No Color"
  "Sunset {date_display} - Clouds Broke at the Horizon"
  "Sunset {date_display} - Clear Sky, Quick Fade"
  "Sunset {date_display} - Low Overcast"
  "Sunset {date_display} - Good Color Tonight"

Return only the title.

Title:"""

        try:
            message = self.anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )
            title = message.content[0].text.strip().strip('"').strip()
            # Enforce prefix and length
            if not title.startswith(f"Sunset {date_display}"):
                title = fallback
            if len(title) > 100:
                title = title[:97] + "..."
            self.logger.info(f"Generated YouTube title: {title}")
            return title
        except Exception as e:
            self.logger.warning(f"YouTube title generation failed: {e}")
            return fallback

    def _build_caption_prompt(self, metadata: Dict[str, Any]) -> str:
        """Build the prompt for Anthropic API"""
        # Extract metadata fields
        date_str = metadata.get('date', 'today')
        weather = metadata.get('weather', {})
        visual = metadata.get('visual_analysis', {})
        sbs_score = metadata.get('sbs_score')
        sbs_grade = metadata.get('sbs_grade')

        # Format date nicely
        try:
            from datetime import datetime as dt
            d = dt.fromisoformat(date_str)
            date_display = d.strftime('%B %d, %Y')
        except (ValueError, TypeError):
            date_display = date_str

        # Build weather line with all Tempest data
        weather_parts = []
        temp = weather.get('temperature_f')
        conditions = weather.get('conditions', '')
        wind = weather.get('wind_speed_mph')
        humidity = weather.get('humidity_pct')
        cloud_cover = weather.get('cloud_cover_pct')
        if temp is not None:
            weather_parts.append(f"{temp}°F")
        if conditions and conditions not in ('unknown', 'Unknown'):
            weather_parts.append(conditions)
        if humidity is not None:
            weather_parts.append(f"{humidity}% humidity")
        if wind is not None and wind > 3:
            weather_parts.append(f"wind {wind} mph")
        if cloud_cover is not None:
            weather_parts.append(f"{cloud_cover}% cloud cover")
        weather_line = ', '.join(weather_parts) if weather_parts else 'no weather data'

        # SBS score and grade — the primary quality signal
        sbs_line = "Sunset Brilliance Score: unknown"
        if sbs_score is not None:
            try:
                s = float(sbs_score)
                grade = sbs_grade or self._sbs_score_to_grade(s)
                sbs_line = f"Sunset Brilliance Score: {s:.1f}/100 (grade: {grade})"
            except (ValueError, TypeError):
                pass

        # Dominant colors from visual analysis (objective, useful)
        dominant_hex = self._get_dominant_colors_from_visual(visual)
        colors_line = f"Dominant sky colors: {', '.join(dominant_hex)}" if dominant_hex else ""

        prompt = f"""Write a short social media caption for a daily sunset timelapse video from Pelham, Alabama.

Data for this sunset:
- Date: {date_display}
- Weather at sunset: {weather_line}
- {sbs_line}
{f'- {colors_line}' if colors_line else ''}

The Sunset Brilliance Score (SBS) is a 0-100 measure of how colorful/dramatic
the sunset was. Use it to calibrate your tone: 80+ is exceptional, 70+ is good,
60+ is decent, 50+ is ordinary, below 50 is flat. Include the score in the caption.

VOICE:
- Direct, dry, observational. No performative enthusiasm.
- Short sentences. No filler.
- Never apologize for an unremarkable sunset. Just describe it honestly.
- Sound like a real person who watches sunsets regularly and knows the difference between a good one and a fine one.

RULES:
- 1-3 sentences max
- Include the date and "Pelham, AL" naturally in the text
- Include the weather conditions and temperature
- Include the SBS score naturally (e.g., "SBS: 72" or "Brilliance score: 72")
- End with "Camera: Reolink RLC810-WA" on its own line
- Vary the structure. Don't start with "Sunset timelapse from..."
- Do NOT claim weather events (rain, storms) unless the data explicitly confirms it
- No emojis, no hashtags
- Do NOT use: "hit different", "still worth watching", "in their own way", "nothing dramatic", "low-contrast", "without incident"
- Do NOT call a 60+ SBS sunset flat/gray/washed out
- Do NOT oversell a sub-50 SBS sunset

Examples:
  "{date_display}, Pelham, AL. Partly cloudy, 74°F. Clouds caught real color tonight — orange bled into pink for a good ten minutes. SBS: 78. Camera: Reolink RLC810-WA"
  "Clear sky, 68°F. Sun dropped fast with nothing to catch the light. {date_display}, Pelham, AL. Brilliance score: 41. Camera: Reolink RLC810-WA"
  "Overcast broke just enough to let some warm light through. {date_display}, Pelham, AL, 71°F. SBS: 63. Camera: Reolink RLC810-WA"
  "Thick clouds over Pelham tonight. {date_display}, 65°F. Flat light, no color. SBS: 32. Camera: Reolink RLC810-WA"

Caption:"""

        return prompt

    @staticmethod
    def _sbs_score_to_grade(score: float) -> str:
        """Convert numeric SBS score to letter grade."""
        if score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        return "F"

    def _get_season(self, month: int) -> str:
        """Get season from month"""
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Fall"
        return ""

    @staticmethod
    def _get_dominant_colors_from_visual(visual: Dict) -> list:
        """Extract dominant hex colors from visual analysis frames."""
        if not visual or 'frames' not in visual:
            return []
        # Collect the top dominant color from each frame, deduplicate
        colors = []
        seen = set()
        for frame in visual.get('frames', []):
            for c in frame.get('dominant_colors', [])[:1]:
                if c not in seen:
                    colors.append(c)
                    seen.add(c)
        return colors[:3]

    def append_hashtags(self, caption: str) -> str:
        """Append hashtags to caption"""
        return f"{caption}\n\n{self.HASHTAGS}"

    def post_video(self, video_path: str, caption: str, date_str: str) -> Optional[str]:
        """
        Post video to Facebook

        Args:
            video_path: Path to local .mp4 file
            caption: Complete caption with hashtags
            date_str: Date string (YYYY-MM-DD) for tracking

        Returns:
            Facebook post ID if successful, None otherwise
        """
        if not self.page_id or not self.page_access_token:
            self.logger.error("Facebook credentials not configured")
            return None

        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return None

        import time
        url = f"{self.GRAPH_VIDEO_URL}/{self.page_id}/videos"
        last_error = None

        # Retry up to 3 times for transient Meta errors (code 1 / subcode 99)
        for attempt in range(3):
            if attempt > 0:
                wait = 30 * attempt
                self.logger.info(f"Facebook upload retry {attempt}/3 after {wait}s wait...")
                time.sleep(wait)

            try:
                self.logger.info(f"Uploading video to Facebook: {video_path}")
                with open(video_path, 'rb') as video_file:
                    response = requests.post(
                        url,
                        files={'source': video_file},
                        data={'description': caption, 'access_token': self.page_access_token},
                        timeout=300,
                    )

                if response.status_code == 200:
                    result = response.json()
                    facebook_id = result.get('id')
                    if facebook_id:
                        self.logger.info(f"Successfully posted to Facebook. Post ID: {facebook_id}")
                        self.update_tracking_db(date_str, 'posted', facebook_id)
                        return facebook_id
                    self.logger.error(f"Facebook response missing 'id': {result}")
                    return None

                last_error = f"status {response.status_code}: {response.text}"
                # Check for transient Meta errors
                try:
                    err = response.json().get('error', {})
                    is_transient = err.get('error_subcode') == 99 or err.get('code') == 1
                except (ValueError, AttributeError):
                    is_transient = False

                self.logger.warning(f"Facebook upload attempt {attempt + 1} failed: {last_error}")
                if not is_transient:
                    break

            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Facebook upload attempt {attempt + 1} exception: {e}")

        # All retries exhausted
        self.logger.error(f"Facebook upload failed after retries: {last_error}")
        self.email_notifier.send_notification(
            f"Facebook Upload Failed - {date_str}",
            f"Failed to upload sunset timelapse to Facebook after 3 retries.\n\n{last_error}"
        )
        return None

    def post_to_instagram(self, video_path: str, caption: str) -> Optional[str]:
        """
        Post video as a Reel to Instagram via the Content Publishing API.

        Instagram requires a publicly accessible URL for video uploads.
        We upload to Facebook first, then use that URL for Instagram.

        Args:
            video_path: Path to local .mp4 file
            caption: Caption text with hashtags

        Returns:
            Instagram media ID if successful, None otherwise
        """
        if not self.instagram_account_id:
            self.logger.info("Instagram account ID not configured, skipping Instagram post")
            return None

        if not self.page_access_token:
            self.logger.error("Page access token required for Instagram posting")
            return None

        try:
            import time

            graph_url = f"https://graph.facebook.com/{self.GRAPH_API_VERSION}"

            # Step 1: Upload video to get a hosted URL via Facebook
            # Use the resumable upload protocol for Instagram
            self.logger.info("Uploading video for Instagram Reel...")

            # First, initialize the upload
            init_url = f"{graph_url}/{self.instagram_account_id}/media"
            init_data = {
                'media_type': 'REELS',
                'caption': caption,
                'access_token': self.page_access_token,
                'upload_type': 'resumable',
            }

            init_response = requests.post(init_url, data=init_data, timeout=60)
            if init_response.status_code != 200:
                self.logger.error(f"Instagram init failed: {init_response.text}")
                return None

            init_result = init_response.json()
            container_id = init_result.get('id')
            upload_url = init_result.get('uri')

            if not container_id or not upload_url:
                self.logger.error(f"Instagram init missing id/uri: {init_result}")
                return None

            self.logger.info(f"Instagram container created: {container_id}")

            # Step 2: Upload the video file
            file_size = os.path.getsize(video_path)
            headers = {
                'Authorization': f'OAuth {self.page_access_token}',
                'offset': '0',
                'file_size': str(file_size),
            }

            with open(video_path, 'rb') as video_file:
                upload_response = requests.post(
                    upload_url,
                    headers=headers,
                    data=video_file,
                    timeout=300
                )

            if upload_response.status_code != 200:
                self.logger.error(f"Instagram upload failed: {upload_response.text}")
                return None

            self.logger.info("Video uploaded, waiting for Instagram to process...")

            # Step 3: Poll for container status
            status_url = f"{graph_url}/{container_id}"
            for attempt in range(30):  # Wait up to 5 minutes
                time.sleep(10)
                status_response = requests.get(
                    status_url,
                    params={
                        'fields': 'status_code,status',
                        'access_token': self.page_access_token
                    },
                    timeout=30
                )

                if status_response.status_code == 200:
                    status = status_response.json()
                    status_code = status.get('status_code')
                    self.logger.debug(f"Instagram container status: {status_code}")

                    if status_code == 'FINISHED':
                        break
                    elif status_code == 'ERROR':
                        self.logger.error(f"Instagram processing failed: {status}")
                        return None
            else:
                self.logger.error("Instagram processing timed out")
                return None

            # Step 4: Publish the container (with retry for transient Meta errors)
            publish_url = f"{graph_url}/{self.instagram_account_id}/media_publish"
            for publish_attempt in range(3):
                if publish_attempt > 0:
                    # Wait between retries - Instagram container sometimes needs more settle time
                    wait = 30 * publish_attempt
                    self.logger.info(f"Instagram publish retry {publish_attempt}/3 after {wait}s wait...")
                    time.sleep(wait)

                publish_response = requests.post(
                    publish_url,
                    data={
                        'creation_id': container_id,
                        'access_token': self.page_access_token
                    },
                    timeout=60
                )

                if publish_response.status_code == 200:
                    media_id = publish_response.json().get('id')
                    self.logger.info(f"Successfully posted to Instagram. Media ID: {media_id}")
                    return media_id
                else:
                    # Check if it's the transient "unknown error" that usually resolves with retry
                    try:
                        err = publish_response.json().get('error', {})
                        err_subcode = err.get('error_subcode')
                        is_transient = err_subcode == 99 or err.get('code') == 1
                    except (ValueError, AttributeError):
                        is_transient = False

                    self.logger.warning(f"Instagram publish attempt {publish_attempt + 1} failed: {publish_response.text}")
                    if not is_transient:
                        # Non-transient error, don't retry
                        break

            self.logger.error(f"Instagram publish failed after retries: {publish_response.text}")
            return None

        except Exception as e:
            self.logger.error(f"Instagram posting error: {e}")
            return None

    def validate(self) -> bool:
        """Validate Facebook uploader configuration"""
        if not self.page_id:
            self.logger.error("Facebook page_id not configured")
            return False

        if not self.page_access_token:
            self.logger.error("Facebook page_access_token not configured")
            return False

        if not self.anthropic_api_key:
            self.logger.warning("ANTHROPIC_API_KEY not set - will use fallback captions")

        if not ANTHROPIC_AVAILABLE:
            self.logger.warning("Anthropic library not installed - will use fallback captions")

        if self.instagram_account_id:
            self.logger.info("Instagram posting enabled")
        else:
            self.logger.info("Instagram posting disabled (no instagram_account_id)")

        self.logger.info("Facebook uploader validation passed")
        return True

    def post_sunset(self, video_path: str, metadata: Dict[str, Any],
                    caption_override: Optional[str] = None) -> bool:
        """
        Complete Facebook + Instagram posting workflow

        Args:
            video_path: Path to sunset video file
            metadata: Complete metadata including weather, visual analysis, SBS score
            caption_override: Pre-generated caption to use (ensures same caption across all platforms)

        Returns:
            True if successful, False otherwise
        """
        date_str = metadata.get('date', datetime.now().strftime('%Y-%m-%d'))

        # Check what's already been posted (per-platform tracking)
        db = self.load_tracking_db()
        entry = db.get(date_str, {})
        fb_already_posted = bool(entry.get('facebook_id'))
        ig_already_posted = bool(entry.get('instagram_id'))

        if fb_already_posted and ig_already_posted:
            self.logger.info(f"Sunset for {date_str} already posted to both Facebook and Instagram")
            return True

        try:
            # Use provided caption or generate a new one
            if caption_override:
                caption = caption_override
            else:
                self.logger.info("Generating caption with Anthropic API...")
                caption = self.generate_caption(metadata)
            full_caption = self.append_hashtags(caption)

            # Post to Facebook if not already done
            facebook_id = None
            if fb_already_posted:
                self.logger.info(f"Facebook post for {date_str} already exists, skipping")
                facebook_id = entry.get('facebook_id')
            else:
                facebook_id = self.post_video(video_path, full_caption, date_str)

            # Post to Instagram if not already done
            if ig_already_posted:
                self.logger.info(f"Instagram post for {date_str} already exists, skipping")
            else:
                try:
                    instagram_id = self.post_to_instagram(str(video_path), full_caption)
                    if instagram_id:
                        self.logger.info(f"Instagram post successful: {instagram_id}")
                        self.update_tracking_db(date_str, 'posted', instagram_id=instagram_id)
                except Exception as e:
                    self.logger.warning(f"Instagram posting failed (non-critical): {e}")

            return facebook_id is not None

        except Exception as e:
            self.logger.error(f"Failed to post sunset to Facebook: {e}")
            return False
