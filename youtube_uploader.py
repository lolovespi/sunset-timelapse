"""
YouTube Uploader
Handles uploading timelapse videos to YouTube using Google API
"""

import logging
import os
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    GOOGLE_AVAILABLE = True
except ImportError:
    logging.warning("Google API libraries not available. YouTube upload will be disabled.")
    GOOGLE_AVAILABLE = False

from config_manager import get_config
from email_notifier import EmailNotifier


class YouTubeUploader:
    """Handles YouTube video uploads"""
    
    # OAuth 2.0 scopes for YouTube uploads and channel access
    SCOPES = ['https://www.googleapis.com/auth/youtube.upload', 'https://www.googleapis.com/auth/youtube.readonly']
    
    def __init__(self):
        """Initialize YouTube uploader"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.service = None
        self._authenticated = False
        self.email_notifier = EmailNotifier()
        
        if not GOOGLE_AVAILABLE:
            self.logger.error("Google API libraries not available. Install google-api-python-client")
            return
            
        # YouTube settings from config
        self.channel_name = self.config.get('youtube.channel_name', 'Unknown Channel')
        self.title_format = self.config.get('youtube.video_title_format', 'Sunset {date}')
        self.description_template = self.config.get('youtube.description_template', '')
        self.privacy_status = self.config.get('youtube.privacy_status', 'public')
        self.category_id = self.config.get('youtube.category_id', 22)
        
        self.logger.info("YouTube uploader initialized")
    
    def _get_token_file_path(self) -> str:
        """Get the secure path for the YouTube token file"""
        credentials_dir = os.path.expanduser('~/.config/sunset-timelapse/credentials')
        return os.path.join(credentials_dir, 'youtube_token.json')
        
    def authenticate(self) -> bool:
        """
        Authenticate with YouTube API using OAuth
        
        Returns:
            True if authentication successful, False otherwise
        """
        if not GOOGLE_AVAILABLE:
            return False
            
        try:
            creds = None
            token_file = self._get_token_file_path()
            
            # Check for existing OAuth token
            if os.path.exists(token_file):
                creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)
                
            # If there are no valid credentials, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    # Refresh existing credentials
                    self.logger.info("Refreshing YouTube credentials")
                    creds.refresh(Request())
                else:
                    # Get new credentials via OAuth flow
                    credentials_path = self.config.get_google_credentials_path()
                    
                    if not credentials_path or not os.path.exists(credentials_path):
                        self.logger.error("Google credentials file not found. Set GOOGLE_APPLICATION_CREDENTIALS")
                        return False
                        
                    self.logger.info("Starting OAuth flow for YouTube authentication")
                    flow = InstalledAppFlow.from_client_secrets_file(credentials_path, self.SCOPES)
                    # Force offline access to get refresh token
                    creds = flow.run_local_server(port=0, access_type='offline', prompt='consent')
                    
                # Save credentials for next run
                with open(token_file, 'w') as token:
                    token.write(creds.to_json())
                    
            # Build YouTube service
            self.service = build('youtube', 'v3', credentials=creds)
            self._authenticated = True
            
            # Test authentication by getting channel info
            channel_response = self.service.channels().list(
                part='snippet',
                mine=True
            ).execute()
            
            if 'items' in channel_response and channel_response['items']:
                channel = channel_response['items'][0]
                channel_title = channel['snippet']['title']
                self.logger.info(f"Successfully authenticated with YouTube channel: {channel_title}")
                return True
            else:
                self.logger.error(f"No YouTube channel found for authenticated account. Response: {channel_response}")
                return False
                
        except Exception as e:
            self.logger.error(f"YouTube authentication failed: {e}")
            self._authenticated = False
            return False
            
    def is_authenticated(self) -> bool:
        """Check if authenticated with YouTube"""
        return self._authenticated and self.service is not None
        
    def format_video_metadata(self, video_date: date, start_time: datetime, 
                            end_time: datetime, is_test: bool = False) -> Dict[str, Any]:
        """
        Format video metadata based on configuration
        
        Args:
            video_date: Date the video was captured
            start_time: Capture start time
            end_time: Capture end time
            is_test: If True, use test title format
            
        Returns:
            Dictionary with formatted metadata
        """
        # Format date for title
        date_str = video_date.strftime('%m/%d/%y')
        
        if is_test:
            title = f"Test Capture - {datetime.now().strftime('%H:%M')} - {date_str}"
        else:
            title = self.title_format.format(date=date_str)
        
        # Format description
        if is_test:
            description = f"""
System test capture from Pelham, Alabama
Captured on {date_str} from {start_time.strftime('%I:%M %p')} to {end_time.strftime('%I:%M %p')}

This is an automated system test to verify camera and upload functionality.

Camera: Reolink RLC810-WA
Interval: 5 seconds

#test #timelapse #alabama #pelham #systemtest
            """.strip()
        else:
            description = self.description_template.format(
                date=date_str,
                start_time=start_time.strftime('%I:%M %p'),
                end_time=end_time.strftime('%I:%M %p')
            )
        
        # Create tags
        if is_test:
            tags = [
                'test',
                'timelapse',
                'alabama', 
                'pelham',
                'systemtest',
                'automated',
                f'{video_date.strftime("%B").lower()}',
                f'{video_date.year}'
            ]
        else:
            tags = [
                'sunset',
                'timelapse',
                'alabama',
                'pelham',
                f'{video_date.strftime("%B").lower()}',
                f'{video_date.year}',
                'daily',
                'automated'
            ]
        
        metadata = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags,
                'categoryId': str(self.category_id)
            },
            'status': {
                'privacyStatus': self.privacy_status
            }
        }
        
        return metadata
        
    def upload_video(self, video_path: Path, video_date: date,
                    start_time: datetime, end_time: datetime, is_test: bool = False) -> Optional[str]:
        """
        Upload video to YouTube
        
        Args:
            video_path: Path to video file
            video_date: Date the video was captured
            start_time: Capture start time
            end_time: Capture end time
            is_test: If True, use test title format
            
        Returns:
            YouTube video ID if successful, None otherwise
        """
        if not self.is_authenticated():
            self.logger.info("Not authenticated, attempting to authenticate...")
            if not self.authenticate():
                self.logger.error("Failed to authenticate with YouTube")
                return None
            
        if not video_path.exists():
            self.logger.error(f"Video file not found: {video_path}")
            return None
            
        try:
            # Format metadata
            metadata = self.format_video_metadata(video_date, start_time, end_time, is_test)
            
            self.logger.info(f"Uploading video to YouTube: {metadata['snippet']['title']}")
            self.logger.info(f"File: {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")
            
            # Create media upload object
            media = MediaFileUpload(
                str(video_path),
                chunksize=-1,  # Upload in a single request
                resumable=True
            )
            
            # Create upload request
            request = self.service.videos().insert(
                part=','.join(metadata.keys()),
                body=metadata,
                media_body=media
            )
            
            # Execute upload with progress tracking
            response = None
            error = None
            retry = 0
            max_retries = self.config.get('advanced.max_retries', 3)
            
            while response is None and retry < max_retries:
                try:
                    self.logger.info(f"Upload attempt {retry + 1}/{max_retries}")
                    status, response = request.next_chunk()
                    
                    if status:
                        progress = status.progress() * 100
                        self.logger.info(f"Upload progress: {progress:.1f}%")
                        
                except HttpError as e:
                    if e.resp.status in [500, 502, 503, 504]:
                        # Recoverable errors - retry
                        self.logger.warning(f"Recoverable error {e.resp.status}, retrying...")
                        retry += 1
                    else:
                        # Non-recoverable error
                        self.logger.error(f"HTTP error {e.resp.status}: {e}")
                        return None
                        
                except Exception as e:
                    self.logger.error(f"Unexpected error during upload: {e}")
                    retry += 1
                    
            if response is not None:
                video_id = response['id']
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                
                self.logger.info(f"Video uploaded successfully!")
                self.logger.info(f"Video ID: {video_id}")
                self.logger.info(f"Video URL: {video_url}")
                
                # Send success email notification
                stats = {
                    "Video ID": video_id,
                    "File Size": f"{video_path.stat().st_size / 1024 / 1024:.1f} MB",
                    "Privacy Status": metadata['status']['privacyStatus'],
                    "Upload Time": datetime.now().strftime('%I:%M %p')
                }

                # Add token expiry information
                try:
                    token_info = self.get_token_expiry_info()
                    if token_info and token_info['expiry_time']:
                        expiry_str = token_info['expiry_time'].strftime('%Y-%m-%d %H:%M UTC')
                        days_remaining = token_info['days_remaining']
                        if days_remaining > 0:
                            stats["YouTube Token Expires"] = f"{expiry_str} ({days_remaining} days)"
                        else:
                            stats["YouTube Token Expires"] = f"{expiry_str} (EXPIRED)"
                except Exception:
                    stats["YouTube Token Expires"] = "Unable to determine"
                
                self.email_notifier.send_upload_success(
                    metadata['snippet']['title'],
                    video_url,
                    datetime.combine(video_date, datetime.min.time()),
                    stats
                )
                
                return video_id
            else:
                self.logger.error("Upload failed after all retry attempts")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to upload video: {e}")
            return None
    
    def upload_video_with_sbs_enhancements(self, video_path: Path, video_date: date,
                                         start_time: datetime, end_time: datetime,
                                         title_enhancement: str = "", description_enhancement: str = "",
                                         is_test: bool = False) -> Optional[str]:
        """
        Upload video to YouTube with SBS enhancements to title and description
        
        Args:
            video_path: Path to video file
            video_date: Date the video was captured
            start_time: Capture start time
            end_time: Capture end time
            title_enhancement: SBS-based title enhancement (e.g., " ✨ SPECTACULAR")
            description_enhancement: SBS-based description enhancement
            is_test: If True, use test title format
            
        Returns:
            YouTube video ID if successful, None otherwise
        """
        if not self.is_authenticated():
            self.logger.info("Not authenticated, attempting to authenticate...")
            if not self.authenticate():
                self.logger.error("Failed to authenticate with YouTube")
                return None
            
        if not video_path.exists():
            self.logger.error(f"Video file not found: {video_path}")
            return None
            
        try:
            # Format metadata with SBS enhancements
            metadata = self.format_video_metadata(video_date, start_time, end_time, is_test)
            
            # Add SBS enhancements
            if title_enhancement:
                metadata['snippet']['title'] += title_enhancement
                
            if description_enhancement:
                metadata['snippet']['description'] += description_enhancement
            
            self.logger.info(f"Uploading video to YouTube with SBS enhancements: {metadata['snippet']['title']}")
            self.logger.info(f"File: {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")
            
            # Create media upload object
            media = MediaFileUpload(
                str(video_path),
                chunksize=-1,  # Upload in a single request
                resumable=True
            )
            
            # Create upload request
            request = self.service.videos().insert(
                part=','.join(metadata.keys()),
                body=metadata,
                media_body=media
            )
            
            # Execute upload with progress tracking
            response = None
            error = None
            retry = 0
            
            while response is None and retry < 3:
                try:
                    status, response = request.next_chunk()
                    if response is not None:
                        if 'id' in response:
                            video_id = response['id']
                            video_url = f"https://www.youtube.com/watch?v={video_id}"
                            
                            self.logger.info(f"YouTube upload successful: {video_id}")
                            self.logger.info(f"Video URL: {video_url}")
                            
                            # Send success notification
                            stats = {
                                "Video ID": video_id,
                                "Title": metadata['snippet']['title'],
                                "Duration": f"{(end_time - start_time).total_seconds() / 60:.1f} minutes",
                                "File Size": f"{video_path.stat().st_size / 1024 / 1024:.1f} MB",
                                "Upload Time": datetime.now().strftime('%I:%M %p')
                            }

                            # Add token expiry information
                            try:
                                token_info = self.get_token_expiry_info()
                                if token_info and token_info['expiry_time']:
                                    expiry_str = token_info['expiry_time'].strftime('%Y-%m-%d %H:%M UTC')
                                    days_remaining = token_info['days_remaining']
                                    if days_remaining > 0:
                                        stats["YouTube Token Expires"] = f"{expiry_str} ({days_remaining} days)"
                                    else:
                                        stats["YouTube Token Expires"] = f"{expiry_str} (EXPIRED)"
                            except Exception:
                                stats["YouTube Token Expires"] = "Unable to determine"
                            
                            self.email_notifier.send_upload_success(
                                metadata['snippet']['title'],
                                video_url,
                                datetime.combine(video_date, datetime.min.time()),
                                stats
                            )
                            
                            return video_id
                        else:
                            self.logger.error(f"Upload failed with response: {response}")
                            return None
                            
                except HttpError as e:
                    if e.resp.status in [500, 502, 503, 504]:
                        self.logger.warning(f"Retriable error occurred: {e}. Retrying in 5 seconds...")
                        time.sleep(5)
                        retry += 1
                    else:
                        self.logger.error(f"Non-retriable error occurred: {e}")
                        return None
                        
                except Exception as e:
                    self.logger.error(f"Upload error: {e}")
                    retry += 1
                    if retry < 3:
                        self.logger.info("Retrying upload...")
                        time.sleep(5)
                
            if response is None:
                self.logger.error("Upload failed after all retry attempts")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to upload video with SBS enhancements: {e}")
            return None
            
    def get_channel_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the authenticated YouTube channel
        
        Returns:
            Channel information dictionary or None if failed
        """
        if not self.is_authenticated():
            return None
            
        try:
            response = self.service.channels().list(
                part='snippet,statistics',
                mine=True
            ).execute()
            
            if response['items']:
                return response['items'][0]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get channel info: {e}")
            return None
            
    def get_video_info(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific video
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Video information dictionary or None if failed
        """
        if not self.is_authenticated():
            return None
            
        try:
            response = self.service.videos().list(
                part='snippet,statistics,status',
                id=video_id
            ).execute()
            
            if response['items']:
                return response['items'][0]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get video info for {video_id}: {e}")
            return None
            
    def list_recent_videos(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        List recent videos from the channel
        
        Args:
            max_results: Maximum number of videos to return
            
        Returns:
            List of video information dictionaries
        """
        if not self.is_authenticated():
            return []
            
        try:
            # Get channel ID first
            channel_info = self.get_channel_info()
            if not channel_info:
                return []
                
            channel_id = channel_info['id']
            
            # Search for videos from this channel
            search_response = self.service.search().list(
                part='snippet',
                channelId=channel_id,
                type='video',
                order='date',
                maxResults=max_results
            ).execute()
            
            return search_response.get('items', [])
            
        except Exception as e:
            self.logger.error(f"Failed to list recent videos: {e}")
            return []
    
    def get_token_expiry_info(self) -> Optional[dict]:
        """
        Get information about current OAuth token expiry
        
        Returns:
            Dictionary with token expiry information or None if no valid credentials
        """
        try:
            token_file = self._get_token_file_path()
            if not os.path.exists(token_file):
                return None
                
            creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)
            
            if not creds or not creds.expiry:
                return None
                
            from datetime import timezone
            now = datetime.now(timezone.utc)
            expiry = creds.expiry
            
            # Ensure both datetimes are timezone-aware for comparison
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
                
            time_until_expiry = expiry - now
            days_remaining = time_until_expiry.days
            hours_remaining = time_until_expiry.seconds // 3600
            
            return {
                'expiry_time': expiry,
                'days_remaining': days_remaining,
                'hours_remaining': hours_remaining,
                'total_seconds_remaining': time_until_expiry.total_seconds(),
                'is_expired': time_until_expiry.total_seconds() <= 0,
                'needs_refresh': time_until_expiry.total_seconds() < 3600  # Less than 1 hour
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get token expiry info: {e}")
            return None
    
    def refresh_token_proactively(self) -> bool:
        """
        Refresh OAuth token before it expires to prevent upload failures
        
        Returns:
            True if token is valid (refreshed or still good), False if refresh failed
        """
        try:
            token_info = self.get_token_expiry_info()
            
            if not token_info:
                self.logger.warning("No OAuth token information available - may need re-authentication")
                return False
                
            if token_info['is_expired']:
                self.logger.warning("OAuth token is already expired - re-authentication required")
                return False
                
            if not token_info['needs_refresh']:
                days = token_info['days_remaining']
                hours = token_info['hours_remaining']
                self.logger.info(f"OAuth token is healthy - expires in {days} days, {hours} hours")
                return True
                
            # Token needs refresh
            self.logger.info("OAuth token expires soon, attempting proactive refresh...")
            
            token_file = self._get_token_file_path()
            creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)
            
            if creds and creds.refresh_token:
                # Refresh the credentials (works for both expired and soon-to-expire tokens)
                creds.refresh(Request())
                
                # Save refreshed credentials
                with open(token_file, 'w') as token:
                    token.write(creds.to_json())
                    
                self.logger.info("OAuth token refreshed successfully")
                
                # Update our service instance with new credentials
                self.service = build('youtube', 'v3', credentials=creds)
                self._authenticated = True
                
                return True
            else:
                self.logger.warning("Cannot refresh OAuth token - refresh_token not available")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to refresh OAuth token proactively: {e}")
            return False
    
    def check_token_health_and_alert(self) -> bool:
        """
        Check OAuth token health and send email alerts if expiring soon
        
        Returns:
            True if token is healthy, False if there are issues
        """
        try:
            token_info = self.get_token_expiry_info()
            
            if not token_info:
                self.logger.warning("No OAuth token information - authentication may be required")
                return False
                
            if token_info['is_expired']:
                expiry_date_str = token_info['expiry_time'].strftime('%Y-%m-%d %H:%M UTC')
                self.logger.error(f"YouTube OAuth token is expired! (Expired on {expiry_date_str})")
                # Send alert email about expired token
                try:
                    self.email_notifier.send_notification(
                        f"🚨 YouTube Token EXPIRED (since {expiry_date_str})",
                        f"The YouTube OAuth token expired on {expiry_date_str}. Manual re-authentication required.",
                        {
                            "Status": "EXPIRED",
                            "Expired On": expiry_date_str,
                            "Action Required": "Run: ./sync_credentials.sh OR python main.py test --youtube"
                        }
                    )
                except:
                    pass  # Don't fail if email fails
                return False
                
            days_remaining = token_info['days_remaining']
            
            # Send warning email if token expires in less than 1 day (24 hours)
            if days_remaining < 1 and days_remaining > 0:
                expiry_date_str = token_info['expiry_time'].strftime('%Y-%m-%d %H:%M UTC')
                self.logger.warning(f"YouTube OAuth token expires in {days_remaining} days on {expiry_date_str}")
                try:
                    self.email_notifier.send_notification(
                        f"⚠️ YouTube Token Expires in {days_remaining} Days ({expiry_date_str})",
                        f"YouTube OAuth token will expire in {days_remaining} days on {expiry_date_str}. System will attempt automatic refresh.",
                        {
                            "Days Remaining": str(days_remaining),
                            "Expiry Date": expiry_date_str,
                            "Current Status": "Warning - expires soon",
                            "Action": "Monitor for successful refresh or re-authenticate if needed"
                        }
                    )
                except:
                    pass  # Don't fail if email fails
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to check OAuth token health: {e}")
            return False
            
    def test_authentication(self) -> bool:
        """
        Test YouTube API authentication and permissions
        
        Returns:
            True if all tests pass, False otherwise
        """
        self.logger.info("Testing YouTube authentication...")
        
        if not self.authenticate():
            return False
            
        try:
            # Test getting channel info
            channel_info = self.get_channel_info()
            if not channel_info:
                self.logger.error("Failed to get channel information")
                return False
                
            channel_name = channel_info['snippet']['title']
            subscriber_count = channel_info['statistics'].get('subscriberCount', 'Unknown')
            
            self.logger.info(f"Authentication: OAuth")
            self.logger.info(f"Channel: {channel_name}")
            self.logger.info(f"Subscribers: {subscriber_count}")
            
            # Show token health info
            token_info = self.get_token_expiry_info()
            if token_info:
                days = token_info['days_remaining']
                hours = token_info['hours_remaining']
                self.logger.info(f"Token expires in {days} days, {hours} hours")
            
            # Test listing videos
            recent_videos = self.list_recent_videos(5)
            self.logger.info(f"Found {len(recent_videos)} recent videos")
            
            self.logger.info("YouTube authentication test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"YouTube authentication test failed: {e}")
            return False