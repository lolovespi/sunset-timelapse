"""
RTMP Streamer
FFmpeg pipeline for streaming RTSP camera feed to RTMP endpoints
(Facebook Live, YouTube Live, etc.) with hardware H.264 encoding on Pi 5.
"""

import logging
import os
import shlex
import signal
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional

from config_manager import get_config


class RTMPStreamer:
    """Stream camera RTSP feed to one or more RTMP endpoints via FFmpeg."""

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.process: Optional[subprocess.Popen] = None
        self._monitor_thread: Optional[threading.Thread] = None

        # Camera settings
        self.camera_ip = self.config.get('camera.ip')
        self.rtsp_port = self.config.get('camera.rtsp_port', 554)
        self.camera_username, self.camera_password = self.config.get_camera_credentials()

        # Streaming settings
        stream_cfg = self.config.get('stream', {}) or {}
        # Mode: "copy" (default, zero CPU, keeps camera resolution) or "reencode"
        self.mode = stream_cfg.get('mode', 'copy')
        self.output_width = stream_cfg.get('width', 1920)
        self.output_height = stream_cfg.get('height', 1080)
        self.video_bitrate = stream_cfg.get('video_bitrate_kbps', 4000)
        self.fps = stream_cfg.get('fps', 20)
        self.rtsp_path = stream_cfg.get('rtsp_path', 'h264Preview_01_main')
        # libx264 preset: ultrafast/superfast/veryfast/faster/fast/medium
        self.x264_preset = stream_cfg.get('x264_preset', 'veryfast')

    def _build_rtsp_url(self) -> str:
        """Build RTSP URL with properly-escaped credentials."""
        user = urllib.parse.quote(self.camera_username)
        pwd = urllib.parse.quote(self.camera_password, safe="")
        return f"rtsp://{user}:{pwd}@{self.camera_ip}:{self.rtsp_port}/{self.rtsp_path}"

    def _build_ffmpeg_cmd(self, rtmp_urls: List[str]) -> List[str]:
        """Build FFmpeg command to stream RTSP → one or more RTMP endpoints."""
        if not rtmp_urls:
            raise ValueError("At least one RTMP URL required")

        rtsp_url = self._build_rtsp_url()

        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'warning',
            # Input
            '-rtsp_transport', 'tcp',
            '-i', rtsp_url,
        ]

        if self.mode == 'copy':
            # Stream-copy: no re-encode. Zero CPU, keeps camera resolution/fps.
            # Camera already outputs H.264 so this just repackages into FLV.
            cmd.extend(['-c:v', 'copy'])
        else:
            # Software re-encode with libx264 (Pi 5 has no real HW encoder)
            cmd.extend([
                '-vf', f'scale={self.output_width}:{self.output_height}',
                '-c:v', 'libx264',
                '-preset', self.x264_preset,
                '-tune', 'zerolatency',
                '-pix_fmt', 'yuv420p',
                '-b:v', f'{self.video_bitrate}k',
                '-maxrate', f'{self.video_bitrate}k',
                '-bufsize', f'{self.video_bitrate * 2}k',
                '-g', str(self.fps * 2),  # Keyframe every 2s
                '-r', str(self.fps),
            ])

        # No audio
        cmd.append('-an')

        if len(rtmp_urls) == 1:
            cmd.extend(['-f', 'flv', rtmp_urls[0]])
        else:
            # Tee muxer for multiple RTMP outputs
            tee_targets = '|'.join(f'[f=flv:onfail=ignore]{u}' for u in rtmp_urls)
            cmd.extend(['-f', 'tee', '-map', '0:v', tee_targets])

        return cmd

    def start(self, rtmp_urls: List[str]) -> bool:
        """
        Start streaming to the given RTMP URLs. Returns True on successful start.

        Blocks briefly to verify FFmpeg started. Non-blocking after that —
        use is_running() or wait() to check status.
        """
        if self.process is not None:
            self.logger.warning("Stream already running")
            return False

        cmd = self._build_ffmpeg_cmd(rtmp_urls)
        sanitized = [self._sanitize(arg) for arg in cmd]
        self.logger.info(f"Starting FFmpeg: {' '.join(sanitized)}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            self.logger.error(f"Failed to start FFmpeg: {e}")
            return False

        # Give it a moment to fail fast on bad input
        time.sleep(2)
        if self.process.poll() is not None:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            self.logger.error(f"FFmpeg exited immediately. stderr: {stderr}")
            self.process = None
            return False

        # Start background thread to consume stderr (prevents blocking)
        self._monitor_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._monitor_thread.start()

        self.logger.info(f"Streaming started (PID {self.process.pid})")
        return True

    def _drain_stderr(self):
        """Background task: read FFmpeg stderr so pipe doesn't fill up."""
        if not self.process or not self.process.stderr:
            return
        for line in self.process.stderr:
            line = line.rstrip()
            if line:
                # FFmpeg emits progress to stderr; log only warnings/errors
                if 'error' in line.lower() or 'warning' in line.lower():
                    self.logger.warning(f"ffmpeg: {line}")

    def is_running(self) -> bool:
        """Check if streaming process is still alive."""
        return self.process is not None and self.process.poll() is None

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        """Wait for stream to finish. Returns exit code or None on timeout."""
        if self.process is None:
            return None
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def stop(self, timeout: float = 10.0):
        """Gracefully stop the stream. Sends SIGINT, falls back to SIGKILL."""
        if self.process is None:
            return

        self.logger.info("Stopping stream...")
        try:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.logger.warning("FFmpeg didn't exit gracefully, killing")
            self.process.kill()
            self.process.wait(timeout=5)
        finally:
            self.process = None
        self.logger.info("Stream stopped")

    @staticmethod
    def _sanitize(arg: str) -> str:
        """Strip credentials from an argument for safe logging."""
        if arg.startswith('rtsp://') and '@' in arg:
            before_at = arg.split('@')[0]
            # rtsp://user:pass — keep rtsp:// prefix, mask user:pass
            parts = before_at.split('//', 1)
            if len(parts) == 2:
                return f"{parts[0]}//***:***@{arg.split('@', 1)[1]}"
        if arg.startswith('rtmps://') or arg.startswith('rtmp://'):
            # Mask the stream key at the end of RTMP URL
            return arg.rsplit('/', 1)[0] + '/***'
        return arg
