"""
Google Drive Uploader
Handles uploading timelapse videos to Google Drive for cloud storage
"""

import logging
import os
import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    GOOGLE_AVAILABLE = True
except ImportError:
    logging.warning("Google API libraries not available. Drive upload will be disabled.")
    GOOGLE_AVAILABLE = False

from config_manager import get_config
from email_notifier import EmailNotifier


class DriveUploader:
    """Handles Google Drive video uploads"""
    
    # Google Drive API scopes
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    
    def __init__(self):
        """Initialize Drive uploader"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.service = None
        self._authenticated = False
        self.email_notifier = EmailNotifier()
        
        if not GOOGLE_AVAILABLE:
            self.logger.error("Google API libraries not available. Install google-api-python-client")
            return
            
        # Drive settings from config
        self.folder_name = self.config.get('drive.folder_name', 'Sunset Timelapses')
        self.folder_id = None  # Will be found/created dynamically
        self.cleanup_after_days = self.config.get('drive.cleanup_after_days', 7)
        
        self.logger.info("Google Drive uploader initialized")
    
    def get_token_file_path(self) -> str:
        """Get path to Drive token file"""
        credentials_dir = os.path.expanduser('~/.config/sunset-timelapse/credentials')
        os.makedirs(credentials_dir, exist_ok=True)
        return os.path.join(credentials_dir, 'drive_token.json')
    
    def authenticate(self) -> bool:
        """
        Authenticate with Google Drive API
        Returns True if authentication successful
        """
        if not GOOGLE_AVAILABLE:
            return False
            
        creds = None
        token_file = self.get_token_file_path()
        
        # Load existing token
        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)
        
        # If there are no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                # Refresh existing credentials
                self.logger.info("Refreshing Google Drive credentials")
                creds.refresh(Request())
            else:
                # Get new credentials via OAuth flow
                credentials_path = self.config.get_google_credentials_path()
                
                if not credentials_path or not os.path.exists(credentials_path):
                    self.logger.error("Google credentials file not found. Set GOOGLE_APPLICATION_CREDENTIALS")
                    return False
                
                self.logger.info("Starting OAuth flow for Google Drive authentication")
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
            
            # Set secure permissions
            os.chmod(token_file, 0o600)
        
        try:
            self.service = build('drive', 'v3', credentials=creds)
            self._authenticated = True
            self.logger.info("Google Drive authentication successful")
            
            # Ensure upload folder exists
            self._ensure_upload_folder()
            return True
            
        except Exception as e:
            self.logger.error(f"Google Drive authentication failed: {e}")
            return False
    
    def _ensure_upload_folder(self) -> bool:
        """Ensure the upload folder exists in Drive"""
        try:
            # Search for existing folder
            query = f"name='{self.folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(q=query, fields='files(id, name)').execute()
            files = results.get('files', [])
            
            if files:
                self.folder_id = files[0]['id']
                self.logger.info(f"Found existing Drive folder: {self.folder_name} ({self.folder_id})")
            else:
                # Create new folder
                folder_metadata = {
                    'name': self.folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                
                folder = self.service.files().create(body=folder_metadata, fields='id').execute()
                self.folder_id = folder.get('id')
                self.logger.info(f"Created new Drive folder: {self.folder_name} ({self.folder_id})")
            
            return True
            
        except HttpError as e:
            self.logger.error(f"Failed to ensure upload folder: {e}")
            return False
    
    def upload_video(self, video_path: Path, target_date: date, 
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, str]]:
        """
        Upload video to Google Drive
        
        Args:
            video_path: Path to video file
            target_date: Date the video was captured
            metadata: Optional metadata dict (sunset times, location, etc.)
            
        Returns:
            Dict with file_id and public_url if successful, None if failed
        """
        if not self.authenticate():
            self.logger.error("Drive authentication failed")
            return None
        
        try:
            # Generate filename
            date_str = target_date.strftime('%Y-%m-%d')
            filename = f"sunset_{date_str}.mp4"
            
            # Prepare file metadata
            file_metadata = {
                'name': filename,
                'parents': [self.folder_id] if self.folder_id else [],
                'description': self._generate_description(target_date, metadata)
            }
            
            # Upload file
            self.logger.info(f"Uploading {video_path.name} to Google Drive...")
            media = MediaFileUpload(str(video_path), mimetype='video/mp4', resumable=True)
            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, size, webViewLink'
            ).execute()
            
            file_id = file.get('id')
            file_size = file.get('size', 0)
            web_link = file.get('webViewLink')
            
            self.logger.info(f"Upload successful: {filename}")
            self.logger.info(f"File ID: {file_id}")
            self.logger.info(f"File size: {int(file_size) / (1024*1024):.1f} MB")
            
            # Make file publicly accessible for Claude bot
            public_url = self._make_file_public(file_id)
            
            # Create metadata file for Claude bot
            self._create_metadata_file(file_id, target_date, metadata, public_url)
            
            return {
                'file_id': file_id,
                'public_url': public_url,
                'web_link': web_link,
                'filename': filename
            }
            
        except Exception as e:
            self.logger.error(f"Drive upload failed: {e}")
            return None
    
    def _make_file_public(self, file_id: str) -> Optional[str]:
        """Make file publicly accessible and return direct download URL"""
        try:
            # Set public permissions
            permission = {
                'role': 'reader',
                'type': 'anyone'
            }
            
            self.service.permissions().create(
                fileId=file_id,
                body=permission,
                fields='id'
            ).execute()
            
            # Generate direct download URL
            public_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            self.logger.info("File made publicly accessible")
            return public_url
            
        except HttpError as e:
            self.logger.warning(f"Failed to make file public: {e}")
            return None
    
    def _create_metadata_file(self, file_id: str, target_date: date, 
                            metadata: Optional[Dict], public_url: Optional[str]):
        """Create metadata file for Claude bot"""
        try:
            meta_data = {
                'date': target_date.isoformat(),
                'file_id': file_id,
                'public_url': public_url,
                'location': f"{self.config.get('location.city')}, {self.config.get('location.state')}",
                'coordinates': {
                    'latitude': self.config.get('location.latitude'),
                    'longitude': self.config.get('location.longitude')
                },
                'status': 'ready_for_posting',
                'created_at': datetime.now().isoformat()
            }
            
            # Add sunset metadata if provided
            if metadata:
                meta_data.update(metadata)
            
            # Upload metadata as JSON file
            meta_filename = f"sunset_{target_date.isoformat()}_metadata.json"
            meta_content = json.dumps(meta_data, indent=2)
            
            file_metadata = {
                'name': meta_filename,
                'parents': [self.folder_id] if self.folder_id else []
            }
            
            # Create temporary file for metadata upload
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                tmp_file.write(meta_content)
                tmp_file_path = tmp_file.name
            
            try:
                media = MediaFileUpload(tmp_file_path, mimetype='application/json')
                
                self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                
                self.logger.info(f"Metadata file created: {meta_filename}")
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to create metadata file: {e}")
    
    def _generate_description(self, target_date: date, metadata: Optional[Dict]) -> str:
        """Generate file description"""
        description = f"Sunset timelapse captured on {target_date.strftime('%B %d, %Y')}"
        
        if metadata:
            if 'sunset_start' in metadata and 'sunset_end' in metadata:
                description += f"\nSunset window: {metadata['sunset_start']} - {metadata['sunset_end']}"
            
        description += f"\nLocation: {self.config.get('location.city')}, {self.config.get('location.state')}"
        description += "\nAutomatically uploaded by Sunset Timelapse System"
        
        return description
    
    def cleanup_old_files(self, days_old: Optional[int] = None) -> int:
        """
        Clean up old files from Drive
        
        Args:
            days_old: Delete files older than this many days (uses config default if None)
            
        Returns:
            Number of files deleted
        """
        if not self.authenticate():
            return 0
        
        days_old = days_old or self.cleanup_after_days
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_old)
        
        try:
            # Find old files in the upload folder
            query = f"parents='{self.folder_id}' and trashed=false"
            results = self.service.files().list(q=query, fields='files(id, name, createdTime)').execute()
            files = results.get('files', [])
            
            deleted_count = 0
            
            for file in files:
                file_date = datetime.fromisoformat(file['createdTime'].replace('Z', '+00:00'))
                
                if file_date < cutoff_date:
                    try:
                        self.service.files().delete(fileId=file['id']).execute()
                        self.logger.info(f"Deleted old file: {file['name']}")
                        deleted_count += 1
                    except HttpError as e:
                        self.logger.warning(f"Failed to delete {file['name']}: {e}")
            
            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old files from Drive")
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Drive cleanup failed: {e}")
            return 0
    
    def test_authentication(self) -> bool:
        """Test Drive authentication and show folder info"""
        try:
            if not self.authenticate():
                return False
            
            # Test API access
            about = self.service.about().get(fields='user, storageQuota').execute()
            user_name = about.get('user', {}).get('displayName', 'Unknown')
            
            storage_quota = about.get('storageQuota', {})
            total_gb = int(storage_quota.get('limit', 0)) / (1024**3)
            used_gb = int(storage_quota.get('usage', 0)) / (1024**3)
            
            self.logger.info(f"Drive user: {user_name}")
            self.logger.info(f"Storage: {used_gb:.1f}GB used of {total_gb:.0f}GB")
            self.logger.info(f"Upload folder: {self.folder_name} ({self.folder_id})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Drive authentication test failed: {e}")
            return False
    
    def list_recent_files(self, limit: int = 10) -> List[Dict]:
        """List recent files in the upload folder"""
        if not self.authenticate():
            return []
        
        try:
            query = f"parents='{self.folder_id}' and trashed=false"
            results = self.service.files().list(
                q=query,
                orderBy='createdTime desc',
                pageSize=limit,
                fields='files(id, name, size, createdTime, webViewLink)'
            ).execute()
            
            return results.get('files', [])
            
        except Exception as e:
            self.logger.error(f"Failed to list files: {e}")
            return []