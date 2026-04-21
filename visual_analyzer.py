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

    SAMPLE_POSITIONS = (0.25, 0.50, 0.75)

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
        stats = self._get_sky_stats(sky)

        return {
            "position": f"{int(position * 100)}%",
            "dominant_colors": dominant_colors,
            "avg_saturation": round(stats["avg_saturation"], 1),
            "avg_brightness": round(stats["avg_brightness"], 1),
            "peak_saturation": round(stats["peak_saturation"], 1),
            "lit_saturation": round(stats["lit_saturation"], 1),
            "warm_pixel_ratio": round(stats["warm_pixel_ratio"], 3),
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
    def _get_sky_stats(image: np.ndarray) -> Dict[str, float]:
        """
        Compute saturation/brightness stats for the sky region.

        A flat mean over the whole sky misclassifies vibrant sunsets: the
        bright, saturated horizon band is a small fraction of pixels, so its
        signal is washed out by the larger area of dim, desaturated upper sky.
        We return several views so the classifier can pick the right one:

        - avg_saturation / avg_brightness: original flat means (kept for
          backward-compatible logging and metadata).
        - peak_saturation: 90th-percentile saturation, robust to a small
          population of vivid pixels.
        - lit_saturation: mean saturation restricted to pixels above a
          brightness floor, so dark/silhouette pixels (whose HSV saturation
          is meaningless) don't drag the average down.
        - warm_pixel_ratio: fraction of pixels that are both bright and
          saturated in the warm hue range (orange/pink/magenta). A non-trivial
          value is direct evidence of sunset color regardless of the mean.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_chan = hsv[:, :, 0]
        s_chan = hsv[:, :, 1]
        v_chan = hsv[:, :, 2]

        avg_hsv = hsv.mean(axis=(0, 1))

        # Restrict to pixels with enough luminance that hue/saturation are
        # meaningful. Very dark pixels (silhouettes, deep upper sky) sit near
        # black where saturation is undefined or noisy.
        lit_mask = v_chan >= 60
        if lit_mask.any():
            lit_saturation = float(s_chan[lit_mask].mean())
        else:
            lit_saturation = float(s_chan.mean())

        # 95th percentile catches a bright horizon band that's only a small
        # fraction of total sky pixels.
        peak_saturation = float(np.percentile(s_chan, 95))

        # Warm hues in OpenCV HSV: red wraps around 0/180, orange ~5-25,
        # pink/magenta ~140-179. Require both saturation and brightness so
        # dim warm noise doesn't count.
        warm_hue = ((h_chan <= 25) | (h_chan >= 140))
        warm_mask = warm_hue & (s_chan >= 80) & (v_chan >= 90)
        warm_pixel_ratio = float(warm_mask.mean())

        return {
            "avg_hue": float(avg_hsv[0]),
            "avg_saturation": float(avg_hsv[1]),
            "avg_brightness": float(avg_hsv[2]),
            "peak_saturation": peak_saturation,
            "lit_saturation": lit_saturation,
            "warm_pixel_ratio": warm_pixel_ratio,
        }

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_sunset(self, frame_analyses: List[Dict]) -> str:
        """
        Classify the sunset type from aggregated frame data.

        Selects the most vivid of the sampled frames (the peak of color often
        falls at 25% or 75%, not the middle) and uses brightness-weighted /
        percentile saturation so a bright horizon band isn't washed out by
        the larger area of dim upper sky.
        """
        if not frame_analyses:
            return "muted/overcast"

        # Pick the frame with the strongest evidence of color. Warm pixel
        # ratio is the most direct signal; peak saturation breaks ties.
        primary = max(
            frame_analyses,
            key=lambda f: (
                f.get("warm_pixel_ratio", 0.0),
                f.get("peak_saturation", f["avg_saturation"]),
            ),
        )

        peak_sat = primary.get("peak_saturation", primary["avg_saturation"])
        lit_sat = primary.get("lit_saturation", primary["avg_saturation"])
        warm_ratio = primary.get("warm_pixel_ratio", 0.0)
        avg_val = primary["avg_brightness"]

        # Parse the most-dominant color to get its hue
        hex_color = primary["dominant_colors"][0]
        r, g, b = self._hex_to_rgb(hex_color)
        hsv_pixel = cv2.cvtColor(
            np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV
        )[0][0]
        hue = int(hsv_pixel[0])

        # Any one of these is enough. warm_ratio is the strongest signal
        # because it requires both warm hue AND saturation AND brightness;
        # even a 2% slice of the sky lit up that way is a real sunset.
        vivid = peak_sat >= 130 or lit_sat >= 85 or warm_ratio >= 0.02

        # Dramatic/stormy: dim and desaturated everywhere — but only if no
        # frame shows real color. A dark, washed-out scene with a colorful
        # horizon is still a sunset, not "stormy".
        if not vivid and lit_sat < 55 and avg_val < 100:
            return "dramatic/stormy"

        # Muted/overcast: no frame carries vivid color anywhere.
        if not vivid:
            return "muted/overcast"

        # Pink/magenta
        if 140 <= hue <= 175:
            return "pink/magenta"

        # Fiery orange: narrow warm hue with strong saturation
        if 5 <= hue <= 20 and peak_sat >= 170:
            return "fiery orange"

        # Golden: warm hue with good brightness
        if 15 <= hue <= 35:
            return "golden"

        # Fiery can also appear in 0-5 range (red wrap-around)
        if hue < 5 and peak_sat >= 150:
            return "fiery orange"

        # Blue hour: blue-ish dominant hue with limited brightness, and only
        # if there isn't strong warm color elsewhere in the frame.
        if 95 <= hue <= 135 and avg_val < 140 and warm_ratio < 0.02:
            return "blue hour"

        # Vivid but hue didn't fit a named bucket — fall back to golden,
        # which the caption layer treats as a generic warm sunset.
        return "golden"

    def _assess_intensity(self, frame_analyses: List[Dict]) -> str:
        """
        Assess overall intensity from saturation and brightness across frames.

        Uses the most-vivid frame's peak/lit saturation and warm pixel ratio
        rather than a flat mean, so a brilliant horizon band isn't diluted by
        the larger area of dim upper sky or by less-vivid samples in the run.
        """
        if not frame_analyses:
            return "low"

        peak_sat = max(
            f.get("peak_saturation", f["avg_saturation"]) for f in frame_analyses
        )
        lit_sat = max(
            f.get("lit_saturation", f["avg_saturation"]) for f in frame_analyses
        )
        warm_ratio = max(f.get("warm_pixel_ratio", 0.0) for f in frame_analyses)
        avg_val = float(np.mean([f["avg_brightness"] for f in frame_analyses]))

        # High: clearly vivid color across a non-trivial area, with usable light.
        if (peak_sat >= 170 or warm_ratio >= 0.08) and avg_val >= 80:
            return "high"

        # Medium: visible color somewhere in the sky.
        if peak_sat >= 120 or lit_sat >= 75 or warm_ratio >= 0.02:
            return "medium"

        return "low"

    @staticmethod
    def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
        """Convert '#rrggbb' to (r, g, b)."""
        h = hex_str.lstrip('#')
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
