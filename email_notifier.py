"""
Email notification system for sunset timelapse operations
Sends notifications for failures and successful YouTube uploads
"""

import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional, List

from config_manager import get_config


class EmailNotifier:
    """Email notification system with SMTP support"""
    
    def __init__(self):
        """Initialize email notifier"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.enabled = self.config.get('email.enabled', False)
        
        if self.enabled:
            self.smtp_server = self.config.get('email.smtp_server')
            self.smtp_port = self.config.get('email.smtp_port', 587)
            self.use_tls = self.config.get('email.use_tls', True)
            self.from_email = self.config.get('email.from_email')
            self.to_emails = self.config.get('email.to_emails', [])
            self.username = self.config.get('email.username')
            self.password = self.config.get('email.password')
            
            # Validate required settings
            if not all([self.smtp_server, self.from_email, self.to_emails, self.username, self.password]):
                self.logger.warning("Email notifications enabled but missing required configuration")
                self.enabled = False
        else:
            self.logger.info("Email notifications disabled")
    
    def is_enabled(self) -> bool:
        """Check if email notifications are enabled"""
        return self.enabled
    
    def send_notification(self, subject: str, body: str, is_html: bool = False) -> bool:
        """
        Send email notification
        
        Args:
            subject: Email subject line
            body: Email body content
            is_html: Whether body contains HTML content
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.enabled:
            self.logger.debug("Email notifications disabled, skipping send")
            return False
            
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = subject
            
            # Attach body
            body_type = 'html' if is_html else 'plain'
            msg.attach(MIMEText(body, body_type))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                
            self.logger.info(f"Email notification sent successfully: {subject}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False
    
    def send_capture_failure(self, error_message: str, capture_date: datetime) -> bool:
        """
        Send notification for capture failure
        
        Args:
            error_message: Description of the error
            capture_date: Date when capture was attempted
            
        Returns:
            True if notification sent successfully
        """
        subject = f"🚨 Sunset Timelapse Capture Failed - {capture_date.strftime('%Y-%m-%d')}"
        
        body = f"""
        <h2>Sunset Timelapse Capture Failed</h2>
        <p><strong>Date:</strong> {capture_date.strftime('%Y-%m-%d')}</p>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Error:</strong></p>
        <pre>{error_message}</pre>
        
        <p>Please check the system logs and camera connectivity.</p>
        
        <hr>
        <p><small>Sent from Sunset Timelapse System</small></p>
        """
        
        return self.send_notification(subject, body, is_html=True)
    
    def send_upload_success(self, video_title: str, youtube_url: str, 
                          capture_date: datetime, stats: dict) -> bool:
        """
        Send notification for successful YouTube upload
        
        Args:
            video_title: Title of uploaded video
            youtube_url: YouTube video URL
            capture_date: Date of the sunset capture
            stats: Dictionary with capture statistics
            
        Returns:
            True if notification sent successfully
        """
        subject = f"✅ Sunset Timelapse Uploaded - {capture_date.strftime('%Y-%m-%d')}"
        
        # Format statistics
        stats_html = ""
        if stats:
            stats_html = "<h3>Capture Statistics</h3><ul>"
            for key, value in stats.items():
                stats_html += f"<li><strong>{key}:</strong> {value}</li>"
            stats_html += "</ul>"
        
        body = f"""
        <h2>Sunset Timelapse Successfully Uploaded</h2>
        <p><strong>Date:</strong> {capture_date.strftime('%Y-%m-%d')}</p>
        <p><strong>Upload Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Video Title:</strong> {video_title}</p>
        <p><strong>YouTube URL:</strong> <a href="{youtube_url}">{youtube_url}</a></p>
        
        {stats_html}
        
        <p>🌅 Another beautiful sunset captured and shared!</p>
        
        <hr>
        <p><small>Sent from Sunset Timelapse System</small></p>
        """
        
        return self.send_notification(subject, body, is_html=True)
    
    def send_processing_failure(self, error_message: str, capture_date: datetime, 
                               stage: str) -> bool:
        """
        Send notification for processing failure (video creation, upload, etc.)
        
        Args:
            error_message: Description of the error
            capture_date: Date of the capture
            stage: Which stage failed (e.g., "video_creation", "youtube_upload")
            
        Returns:
            True if notification sent successfully
        """
        stage_names = {
            "video_creation": "Video Creation",
            "youtube_upload": "YouTube Upload",
            "image_processing": "Image Processing",
            "cleanup": "File Cleanup"
        }
        
        stage_display = stage_names.get(stage, stage.replace('_', ' ').title())
        subject = f"⚠️ Sunset Timelapse {stage_display} Failed - {capture_date.strftime('%Y-%m-%d')}"
        
        body = f"""
        <h2>Sunset Timelapse {stage_display} Failed</h2>
        <p><strong>Date:</strong> {capture_date.strftime('%Y-%m-%d')}</p>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Failed Stage:</strong> {stage_display}</p>
        <p><strong>Error:</strong></p>
        <pre>{error_message}</pre>
        
        <p>Images may have been captured successfully, but processing failed.</p>
        <p>Please check the system logs for more details.</p>
        
        <hr>
        <p><small>Sent from Sunset Timelapse System</small></p>
        """
        
        return self.send_notification(subject, body, is_html=True)
    
    def send_system_status(self, status_info: dict) -> bool:
        """
        Send system status notification (for debugging/monitoring)
        
        Args:
            status_info: Dictionary with system status information
            
        Returns:
            True if notification sent successfully
        """
        subject = f"📊 Sunset Timelapse System Status - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Format status information
        status_html = "<h3>System Status</h3><ul>"
        for key, value in status_info.items():
            status_html += f"<li><strong>{key}:</strong> {value}</li>"
        status_html += "</ul>"
        
        body = f"""
        <h2>Sunset Timelapse System Status</h2>
        <p><strong>Report Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        {status_html}
        
        <hr>
        <p><small>Sent from Sunset Timelapse System</small></p>
        """
        
        return self.send_notification(subject, body, is_html=True)
    
    def test_connection(self) -> bool:
        """
        Test email configuration and connection
        
        Returns:
            True if test email sent successfully
        """
        if not self.enabled:
            self.logger.info("Email notifications disabled, cannot test")
            return False
            
        subject = "🧪 Sunset Timelapse Email Test"
        body = f"""
        <h2>Email Test Successful</h2>
        <p>This is a test email from your Sunset Timelapse system.</p>
        <p><strong>Test Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>If you received this email, your email notifications are working correctly!</p>
        
        <hr>
        <p><small>Sent from Sunset Timelapse System</small></p>
        """
        
        success = self.send_notification(subject, body, is_html=True)
        if success:
            self.logger.info("Email test completed successfully")
        else:
            self.logger.error("Email test failed")
        
        return success