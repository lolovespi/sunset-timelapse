"""
Visual Analyzer
Extracts frames from sunset timelapse videos and analyzes sky characteristics
(dominant colors, sunset type classification, intensity).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config_manager import get_config


class VisualAnalyzer:
    """Analyze sunset video frames for color and sky characteristics"""

    # Sunset type classification thresholds (HSV-based)
    SUNSET_TYPES = {
        "golden":           {"hue_range": (15, 35),  "sat_min": 100, "val_min": 150},
        "pink/magenta":     {"hue_range": (140, 175), "sat_min": 60,  "val_min": 100},
        "fiery orange":     {"hue_range": (5, 20),   "sat_min": 150, "val_min": 150},
        "blue hour":        {"hue_range": (95, 135),  "sat_min": 40,  "val_min": 40},
        "dramatic/stormy":  {"hue_range": (0, 180),  "sat_max": 50,  "val_max": 100},
    }

    # Sample more positions to better catch peak sunset color (which can
    # land anywhere in the capture window, not always the middle).
    SAMPLE_POSITIONS = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

    def analyze_video(self, video_path: str | Path) -> Optional[Dict]:
        """
        Analyze a sunset timelapse video.

        Extracts frames at 25%, 50%, and 75% through the video, analyzes
        sky region of each frame, and returns a visual analysis block.

        Args:
            video_path: Path to the MP4 file.

        Returns:
            Dict suitable for the 'visual_analysis' key in metadata,
            or None on failure.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            self.logger.error(f"Video not found: {video_path}")
            return None

        try:
            frames = self._extract_frames(video_path)
            if not frames:
                return None

            frame_analyses = []
            for position, frame in frames:
                analysis = self._analyze_frame(frame, position)
                frame_analyses.append(analysis)

            # Aggregate across all frames
            sunset_type = self._classify_sunset(frame_analyses)
            intensity = self._assess_intensity(frame_analyses)

            return {
                "frames_analyzed": len(frame_analyses),
                "sunset_type": sunset_type,
                "intensity": intensity,
                "frames": frame_analyses,
            }

        except Exception as e:
            self.logger.error(f"Visual analysis failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(
        self, video_path: Path
    ) -> List[Tuple[float, np.ndarray]]:
        """Extract frames at 25%, 50%, 75% of video duration."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 3:
            self.logger.warning(f"Video too short ({total_frames} frames)")
            cap.release()
            return []

        results = []
        for pos in self.SAMPLE_POSITIONS:
            frame_idx = int(total_frames * pos)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                results.append((pos, frame))
            else:
                self.logger.warning(f"Failed to read frame at {pos:.0%}")

        cap.release()
        return results

    # ------------------------------------------------------------------
    # Single-frame analysis
    # ------------------------------------------------------------------

    def _analyze_frame(self, frame: np.ndarray, position: float) -> Dict:
        """Analyze a single frame's sky region."""
        sky = self._extract_sky_region(frame)
        dominant_colors = self._get_dominant_colors(sky, k=3)
        avg_hsv = self._get_avg_hsv(sky)

        return {
            "position": f"{int(position * 100)}%",
            "dominant_colors": dominant_colors,
            "avg_saturation": round(float(avg_hsv[1]), 1),
            "avg_brightness": round(float(avg_hsv[2]), 1),
        }

    def _extract_sky_region(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract the sky portion of the frame.
        Uses the same convention as SBS: top 60% of the frame by default,
        excluding bottom foreground (buildings/trees).
        """
        h = frame.shape[0]
        sky_ratio = self.config.get('sbs.sky_region_ratio', 0.6)
        sky_bottom = int(h * sky_ratio)
        return frame[0:sky_bottom, :]

    def _get_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[str]:
        """
        Find the top-k dominant colors via k-means clustering.
        Returns hex color strings sorted by cluster size (most dominant first).
        """
        # Resize for speed
        small = cv2.resize(image, (80, 60))
        pixels = small.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        # Sort clusters by size (largest first)
        counts = np.bincount(labels.flatten(), minlength=k)
        order = np.argsort(-counts)

        hex_colors = []
        for idx in order:
            bgr = centers[idx].astype(int)
            # OpenCV uses BGR; convert to RGB for hex
            r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
            hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")

        return hex_colors

    @staticmethod
    def _get_avg_hsv(image: np.ndarray) -> Tuple[float, float, float]:
        """Return mean H, S, V of the image."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return tuple(hsv.mean(axis=(0, 1)))

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_sunset(self, frame_analyses: List[Dict]) -> str:
        """
        Classify the sunset type from aggregated frame data.
        Uses the most saturated (most colorful) frame as primary, since
        peak sunset color can occur anywhere in the capture window.
        """
        if not frame_analyses:
            return "muted/overcast"

        # Pick the most saturated frame — that's where the real color is
        primary = max(frame_analyses, key=lambda f: f["avg_saturation"])
        avg_sat = primary["avg_saturation"]
        avg_val = primary["avg_brightness"]

        # Parse the most-dominant color to get its hue
        hex_color = primary["dominant_colors"][0]
        r, g, b = self._hex_to_rgb(hex_color)
        hsv_pixel = cv2.cvtColor(
            np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV
        )[0][0]
        hue = int(hsv_pixel[0])

        # Dramatic/stormy: low saturation AND low brightness
        if avg_sat < 50 and avg_val < 100:
            return "dramatic/stormy"

        # Muted/overcast: low saturation overall
        if avg_sat < 60:
            return "muted/overcast"

        # Blue hour: blue-ish hues with moderate saturation
        if 95 <= hue <= 135 and avg_val < 140:
            return "blue hour"

        # Pink/magenta
        if 140 <= hue <= 175:
            return "pink/magenta"

        # Fiery orange: narrow warm hue with high saturation
        if 5 <= hue <= 20 and avg_sat >= 150:
            return "fiery orange"

        # Golden: warm hue with good brightness
        if 15 <= hue <= 35:
            return "golden"

        # Fiery can also appear in 0-5 range (red wrap-around)
        if hue < 5 and avg_sat >= 120:
            return "fiery orange"

        return "muted/overcast"

    def _assess_intensity(self, frame_analyses: List[Dict]) -> str:
        """
        Assess overall intensity from saturation and brightness.
        Uses peak saturation rather than mean — a sustained vibrant
        moment matters more than the average across pre/post-sunset.
        """
        if not frame_analyses:
            return "low"

        peak_sat = max(f["avg_saturation"] for f in frame_analyses)
        peak_val = max(f["avg_brightness"] for f in frame_analyses)

        # High: vivid colors + good brightness at peak
        if peak_sat >= 90 and peak_val >= 120:
            return "high"

        # Medium: moderate color presence
        if peak_sat >= 55 or peak_val >= 130:
            return "medium"

        return "low"

    @staticmethod
    def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
        """Convert '#rrggbb' to (r, g, b)."""
        h = hex_str.lstrip('#')
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
