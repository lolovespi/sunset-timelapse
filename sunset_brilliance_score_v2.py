"""
Enhanced Sunset Brilliance Score (SBS) System v2.0
Human-perception-optimized algorithms for sunset quality analysis
"""

import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
from dataclasses import dataclass, asdict

from config_manager import get_config


@dataclass
class FrameMetricsV2:
    """Enhanced container for per-frame SBS metrics"""
    frame_number: int
    timestamp_offset: float
    
    # Core sunset-specific metrics
    sunset_warmth_score: float      # NEW: Weighted warm color detection
    atmospheric_brilliance: float   # NEW: Sky-based color intensity
    horizon_glow_intensity: float   # NEW: Horizon region analysis
    sky_color_richness: float       # Enhanced color richness (sky-only)
    
    # Supporting metrics
    color_temperature: float
    sky_saturation: float
    gradient_intensity: float
    brightness_balance: float       # NEW: Optimal exposure detection
    cloud_enhancement: float        # NEW: Cloud color enhancement detection
    
    # Legacy compatibility
    colorfulness: float            # For comparison with v1
    brightness_mean: float
    processing_time_ms: float


@dataclass
class ChunkMetricsV2:
    """Enhanced container for chunk-level SBS analysis"""
    chunk_number: int
    start_offset: float
    end_offset: float
    brilliance_score_v2: float      # NEW: Human-perception-tuned score
    sunset_quality_grade: str       # NEW: A+ to F grading
    
    # Peak detection
    peak_sunset_moments: List[int]  # NEW: True sunset peak frames
    warmth_progression: float      # NEW: Temperature transition quality
    atmospheric_drama: float       # NEW: Sky drama/intensity
    
    # Legacy metrics for comparison
    brilliance_score: float        # v1 compatibility
    color_variation_range: float
    temporal_smoothness: float
    golden_hour_bonus: float
    frame_count: int
    avg_processing_time_ms: float


class SunsetBrillianceScoreV2:
    """
    Enhanced Sunset Brilliance Score system with human-perception optimization
    
    Key Improvements:
    - Warm color detection and weighting
    - Sky-focused analysis regions
    - Atmospheric brilliance metrics
    - Human-validated sunset quality correlation
    """
    
    def __init__(self):
        """Initialize enhanced SBS analyzer"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Enhanced configuration
        self.sbs_config = self.config.get('sbs', {})
        self.enabled = self.sbs_config.get('enabled', True)
        self.analysis_resize_height = self.sbs_config.get('analysis_resize_height', 240)
        
        # Sky analysis regions (more sophisticated)
        self.horizon_region_ratio = self.sbs_config.get('horizon_region_ratio', 0.7)  # Bottom 70% to top 85%
        self.zenith_region_ratio = self.sbs_config.get('zenith_region_ratio', 0.3)    # Top 30% 
        self.foreground_exclude_ratio = self.sbs_config.get('foreground_exclude_ratio', 0.15)  # Bottom 15% excluded
        
        # Sunset-specific parameters
        self.warm_color_weight = self.sbs_config.get('warm_color_weight', 2.5)        # Boost warm colors
        self.cool_color_penalty = self.sbs_config.get('cool_color_penalty', 0.3)      # Reduce cool colors
        self.atmosphere_sensitivity = self.sbs_config.get('atmosphere_sensitivity', 1.2)  # Atmospheric detection
        
        # Golden hour parameters
        self.golden_hour_minutes = self.sbs_config.get('golden_hour_minutes', 30)
        
        # Performance monitoring
        self.processing_times = []
        self.chunk_data = []
        
        self.logger.info("Enhanced SBS v2.0 system initialized with human-perception optimization")
    
    def analyze_frame_v2(self, frame: np.ndarray, frame_number: int, 
                        timestamp_offset: float) -> Optional[FrameMetricsV2]:
        """
        Enhanced frame analysis optimized for human sunset perception
        
        Args:
            frame: OpenCV image frame (BGR)
            frame_number: Frame sequence number
            timestamp_offset: Seconds from capture start
            
        Returns:
            FrameMetricsV2 object with enhanced metrics
        """
        if not self.enabled:
            return None
            
        start_time = time.perf_counter()
        
        try:
            # Resize for performance (maintain aspect ratio)
            height, width = frame.shape[:2]
            if height > self.analysis_resize_height:
                scale = self.analysis_resize_height / height
                new_width = int(width * scale)
                frame_resized = cv2.resize(frame, (new_width, self.analysis_resize_height))
            else:
                frame_resized = frame
                
            # Extract sky regions for focused analysis
            sky_regions = self._extract_sky_regions(frame_resized)
            
            # Enhanced sunset-specific metrics
            sunset_warmth_score = self._calculate_sunset_warmth(sky_regions['horizon'])
            atmospheric_brilliance = self._calculate_atmospheric_brilliance(sky_regions['full_sky'])
            horizon_glow_intensity = self._calculate_horizon_glow(sky_regions['horizon'])
            sky_color_richness = self._calculate_sky_color_richness(sky_regions['full_sky'])
            cloud_enhancement = self._detect_cloud_enhancement(sky_regions['full_sky'])
            
            # Supporting metrics
            color_temperature = self._estimate_color_temperature_enhanced(sky_regions['full_sky'])
            sky_saturation = self._analyze_sky_saturation_enhanced(sky_regions['full_sky'])
            gradient_intensity = self._calculate_atmospheric_gradient(sky_regions['full_sky'])
            brightness_balance = self._calculate_brightness_balance(sky_regions['full_sky'])
            
            # Legacy compatibility metrics
            colorfulness = self._hasler_susstrunk_colorfulness(frame_resized)
            brightness_mean = np.mean(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY))
            
            processing_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            self.processing_times.append(processing_time)
            
            return FrameMetricsV2(
                frame_number=frame_number,
                timestamp_offset=timestamp_offset,
                sunset_warmth_score=sunset_warmth_score,
                atmospheric_brilliance=atmospheric_brilliance,
                horizon_glow_intensity=horizon_glow_intensity,
                sky_color_richness=sky_color_richness,
                color_temperature=color_temperature,
                sky_saturation=sky_saturation,
                gradient_intensity=gradient_intensity,
                brightness_balance=brightness_balance,
                cloud_enhancement=cloud_enhancement,
                colorfulness=colorfulness,
                brightness_mean=brightness_mean,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced frame analysis failed: {e}")
            return None
    
    def _extract_sky_regions(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract different sky regions for focused analysis"""
        height, width = frame.shape[:2]
        
        # Define region boundaries
        foreground_cutoff = int(height * self.foreground_exclude_ratio)  # Exclude bottom 15%
        horizon_start = foreground_cutoff
        horizon_end = int(height * self.horizon_region_ratio)           # Horizon region: 15%-70%
        zenith_start = int(height * (1 - self.zenith_region_ratio))     # Zenith region: top 30%
        
        return {
            'full_sky': frame[foreground_cutoff:, :],                   # All sky (no foreground)
            'horizon': frame[horizon_start:horizon_end, :],             # Horizon region
            'zenith': frame[zenith_start:, :],                          # Zenith region
            'middle_sky': frame[horizon_end:zenith_start, :]            # Middle sky region
        }
    
    def _calculate_sunset_warmth(self, horizon_region: np.ndarray) -> float:
        """
        Calculate sunset warmth score based on warm color presence
        HUMAN-CALIBRATED v2.1: Much more selective based on your feedback
        """
        # Convert to HSV for hue analysis
        hsv = cv2.cvtColor(horizon_region, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        
        # Define sunset color ranges in HSV (OpenCV: H=0-179, S/V=0-255)
        warm_ranges = [
            (0, 15),    # Red/orange (0-15°)
            (165, 179), # Red wraparound (330-360° -> 165-179°)
            (15, 35),   # Orange/yellow (15-35°)
        ]
        
        # Create mask for warm sunset colors
        warm_mask = np.zeros_like(hue, dtype=bool)
        for h_min, h_max in warm_ranges:
            warm_mask |= (hue >= h_min) & (hue <= h_max)
        
        # HUMAN-CALIBRATED: Much stricter thresholds based on your feedback
        # Your Sept 1st had sat ~0.25, brightness ~0.74 → you said "not spectacular"
        # So we need much higher thresholds for "spectacular"
        sunset_mask = warm_mask & (sat > 120) & (val > 100) & (val < 200)  # Much more selective
        
        # Calculate warm color statistics
        if np.sum(sunset_mask) == 0:
            return 0.0
        
        # Weighted scoring based on color intensity and coverage
        warm_pixels = np.sum(sunset_mask)
        total_pixels = horizon_region.shape[0] * horizon_region.shape[1]
        coverage_ratio = warm_pixels / total_pixels
        
        # Average saturation and value of warm pixels
        avg_warm_saturation = np.mean(sat[sunset_mask]) / 255.0
        avg_warm_brightness = np.mean(val[sunset_mask]) / 255.0
        
        # HUMAN-CALIBRATED penalties based on your feedback
        # Coverage penalty: Sept 1st had 1.5% coverage → you said "not spectacular"
        coverage_penalty = 1.0
        if coverage_ratio < 0.05:  # Less than 5% coverage
            coverage_penalty = coverage_ratio / 0.05  # Linearly reduce score
        
        # Saturation penalty: Sept 1st had 0.25 saturation → you said "not spectacular"
        saturation_penalty = 1.0
        if avg_warm_saturation < 0.5:  # Less than 50% saturation
            saturation_penalty = avg_warm_saturation / 0.5  # Linearly reduce score
        
        # RECALIBRATED FORMULA: Much more conservative
        base_score = (
            coverage_ratio * 25 +           # Reduced from 40
            avg_warm_saturation * 35 +      # Keep high saturation weight  
            avg_warm_brightness * 20        # Reduced from 25
        ) * 100
        
        # Apply penalties
        final_score = base_score * coverage_penalty * saturation_penalty
        
        # HUMAN-CALIBRATED: Scale to realistic range
        # Sept 1st should score ~15-25, not 85-100
        realistic_score = final_score * 0.3  # Conservative scaling factor
        
        return float(np.clip(realistic_score, 0, 100))
    
    def _calculate_atmospheric_brilliance(self, sky_region: np.ndarray) -> float:
        """
        Calculate atmospheric brilliance - the drama and intensity of sky colors
        """
        # Convert to LAB color space for perceptual analysis
        lab = cv2.cvtColor(sky_region, cv2.COLOR_BGR2LAB)
        L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]
        
        # Calculate color intensity in perceptual space
        chroma = np.sqrt(A.astype(float)**2 + B.astype(float)**2)
        
        # Atmospheric brilliance factors:
        # 1. Color intensity variation (dramatic vs. flat)
        chroma_variation = np.std(chroma)
        
        # 2. Luminance gradient (atmospheric layering)
        luminance_gradient = np.std(L.astype(float))
        
        # 3. Color purity in high-luminance areas (bright colored vs. washed out)
        bright_mask = L > 100  # Bright areas
        if np.sum(bright_mask) > 0:
            bright_chroma = np.mean(chroma[bright_mask])
        else:
            bright_chroma = 0
        
        # Combine factors for atmospheric brilliance
        brilliance = (
            chroma_variation * 0.4 +      # Color drama
            luminance_gradient * 0.3 +    # Atmospheric layers
            bright_chroma * 0.3           # Bright color intensity
        )
        
        # Normalize to 0-100 scale
        return float(np.clip(brilliance / 50 * 100, 0, 100))
    
    def _calculate_horizon_glow(self, horizon_region: np.ndarray) -> float:
        """
        Detect the characteristic horizon glow of spectacular sunsets
        """
        height, width = horizon_region.shape[:2]
        
        # Focus on bottom portion of horizon region (where glow appears)
        glow_region = horizon_region[int(height*0.6):, :]  # Bottom 40% of horizon
        
        # Convert to HSV
        hsv = cv2.cvtColor(glow_region, cv2.COLOR_BGR2HSV)
        
        # Detect bright warm colors (sunset glow)
        warm_bright_mask = (
            (hsv[:,:,0] <= 30) |           # Orange/red hues
            (hsv[:,:,0] >= 150)            # Red wraparound
        ) & (hsv[:,:,1] > 80) & (hsv[:,:,2] > 120)  # High saturation and brightness
        
        # Calculate glow intensity
        if np.sum(warm_bright_mask) == 0:
            return 0.0
        
        glow_coverage = np.sum(warm_bright_mask) / warm_bright_mask.size
        glow_intensity = np.mean(hsv[:,:,2][warm_bright_mask]) / 255.0
        
        # Horizontal spread of glow (wider glow = more spectacular)
        glow_rows = np.any(warm_bright_mask, axis=1)
        glow_width_ratio = np.sum(glow_rows) / len(glow_rows)
        
        horizon_glow = (glow_coverage * 40 + glow_intensity * 35 + glow_width_ratio * 25) * 100
        return float(np.clip(horizon_glow, 0, 100))
    
    def _calculate_sky_color_richness(self, sky_region: np.ndarray) -> float:
        """
        Calculate color richness specifically for sky colors (not general colorfulness)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)
        
        # Focus on sky-relevant colors and exclude very dark or very bright pixels
        valid_mask = (hsv[:,:,2] > 20) & (hsv[:,:,2] < 240)  # Exclude very dark/bright
        
        if np.sum(valid_mask) == 0:
            return 0.0
        
        # Calculate color diversity in sky region
        valid_hues = hsv[:,:,0][valid_mask]
        valid_sats = hsv[:,:,1][valid_mask]
        valid_vals = hsv[:,:,2][valid_mask]
        
        # Color richness factors:
        hue_diversity = np.std(valid_hues)          # Variety of colors
        saturation_quality = np.mean(valid_sats)    # Color purity
        brightness_balance = 255 - np.std(valid_vals)  # Even lighting
        
        # Normalize and combine
        richness = (
            hue_diversity / 50 * 35 +           # Hue variety (up to 35 points)
            saturation_quality / 255 * 40 +     # Saturation (up to 40 points)
            brightness_balance / 255 * 25       # Lighting quality (up to 25 points)
        ) * 100
        
        return float(np.clip(richness, 0, 100))
    
    def _detect_cloud_enhancement(self, sky_region: np.ndarray) -> float:
        """
        Detect when clouds enhance sunset colors (vs. obscuring them)
        """
        # Convert to grayscale for cloud detection
        gray = cv2.cvtColor(sky_region, cv2.COLOR_BGR2GRAY)
        
        # Detect cloud patterns using edge detection
        edges = cv2.Canny(gray, 50, 150)
        cloud_structure = np.sum(edges) / edges.size
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)
        
        # Check if cloud edges have enhanced colors
        kernel = np.ones((5,5), np.uint8)
        cloud_edges = cv2.dilate(edges, kernel, iterations=1)
        
        if np.sum(cloud_edges) == 0:
            return 0.0
        
        # Analyze colors near cloud edges
        edge_mask = cloud_edges > 0
        edge_saturations = hsv[:,:,1][edge_mask]
        edge_values = hsv[:,:,2][edge_mask]
        
        # Enhanced clouds have higher saturation and interesting brightness patterns
        avg_edge_saturation = np.mean(edge_saturations) / 255.0
        brightness_variation = np.std(edge_values) / 255.0
        
        # Cloud enhancement score
        enhancement = (
            cloud_structure * 30 +           # Cloud structure presence
            avg_edge_saturation * 40 +       # Color saturation at edges
            brightness_variation * 30        # Dramatic lighting variation
        ) * 100
        
        return float(np.clip(enhancement, 0, 100))
    
    def _estimate_color_temperature_enhanced(self, sky_region: np.ndarray) -> float:
        """Enhanced color temperature estimation focused on sky colors"""
        # Convert to float for precision
        sky_float = sky_region.astype(np.float32)
        
        # Focus on mid-brightness pixels (avoid extremes)
        gray = cv2.cvtColor(sky_region, cv2.COLOR_BGR2GRAY)
        mid_brightness_mask = (gray > 40) & (gray < 200)
        
        if np.sum(mid_brightness_mask) == 0:
            return 6500.0  # Default daylight
        
        # Calculate average RGB for mid-brightness sky pixels
        mean_rgb = np.mean(sky_float[mid_brightness_mask], axis=0)
        mean_b, mean_g, mean_r = mean_rgb
        
        # Avoid division by zero
        if mean_b == 0:
            mean_b = 1
            
        # Enhanced color temperature estimation
        rb_ratio = mean_r / mean_b
        
        # Sunset-calibrated temperature mapping
        if rb_ratio > 2.5:
            color_temp = 1800 + (500 * (1.0 / rb_ratio))    # Deep sunset
        elif rb_ratio > 1.8:
            color_temp = 2200 + (800 * (2.5 - rb_ratio) / 0.7)  # Golden hour
        elif rb_ratio > 1.3:
            color_temp = 3500 + (1500 * (1.8 - rb_ratio) / 0.5)  # Warm
        elif rb_ratio > 1.0:
            color_temp = 5000 + (1500 * (1.3 - rb_ratio) / 0.3)  # Neutral
        else:
            color_temp = 6500  # Cool
            
        return float(np.clip(color_temp, 1500, 8000))
    
    def _analyze_sky_saturation_enhanced(self, sky_region: np.ndarray) -> float:
        """Enhanced sky saturation analysis"""
        hsv_sky = cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)
        
        # Focus on well-lit areas of sky
        value_mask = (hsv_sky[:,:,2] > 30) & (hsv_sky[:,:,2] < 220)
        
        if np.sum(value_mask) == 0:
            return 0.0
        
        # Weight saturation by brightness for better perception correlation
        saturation = hsv_sky[:,:,1][value_mask].astype(float)
        brightness = hsv_sky[:,:,2][value_mask].astype(float)
        
        # Brightness-weighted saturation
        weights = brightness / 255.0
        weighted_saturation = np.average(saturation, weights=weights)
        
        return float(weighted_saturation / 255.0)
    
    def _calculate_atmospheric_gradient(self, sky_region: np.ndarray) -> float:
        """Calculate atmospheric gradient (sky color transitions)"""
        # Convert to LAB for perceptual gradients
        lab = cv2.cvtColor(sky_region, cv2.COLOR_BGR2LAB)
        
        # Calculate gradients in perceptual color space
        grad_L = cv2.Sobel(lab[:,:,0], cv2.CV_64F, 0, 1, ksize=3)
        grad_A = cv2.Sobel(lab[:,:,1], cv2.CV_64F, 0, 1, ksize=3)
        grad_B = cv2.Sobel(lab[:,:,2], cv2.CV_64F, 0, 1, ksize=3)
        
        # Combined gradient magnitude
        gradient_magnitude = np.sqrt(grad_L**2 + grad_A**2 + grad_B**2)
        
        return float(np.mean(gradient_magnitude))
    
    def _calculate_brightness_balance(self, sky_region: np.ndarray) -> float:
        """Calculate optimal brightness balance (avoid over/under exposure)"""
        gray = cv2.cvtColor(sky_region, cv2.COLOR_BGR2GRAY)
        
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Optimal range: 70-180, with good dynamic range
        if 70 <= mean_brightness <= 180:
            brightness_score = 1.0
        elif mean_brightness < 40 or mean_brightness > 220:
            brightness_score = 0.1  # Poor exposure
        else:
            brightness_score = 0.6  # Acceptable
        
        # Good dynamic range bonus
        if 20 < brightness_std < 80:
            dynamic_range_bonus = 0.3
        else:
            dynamic_range_bonus = 0.0
        
        return float((brightness_score + dynamic_range_bonus) * 100)
    
    def _hasler_susstrunk_colorfulness(self, frame: np.ndarray) -> float:
        """Legacy colorfulness metric for comparison"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        
        rg = R - G
        yb = 0.5 * (R + G) - B
        
        rg_std = np.std(rg)
        yb_std = np.std(yb)
        rg_mean = np.mean(rg)
        yb_mean = np.mean(yb)
        
        std_rgyb = np.sqrt(rg_std**2 + yb_std**2)
        mean_rgyb = np.sqrt(rg_mean**2 + yb_mean**2)
        
        colorfulness = std_rgyb + 0.3 * mean_rgyb
        return float(colorfulness)
    
    def analyze_chunk_v2(self, frame_metrics: List[FrameMetricsV2], 
                        chunk_number: int, sunset_offset_minutes: float = 0) -> ChunkMetricsV2:
        """
        Enhanced chunk analysis with human-perception-tuned scoring
        """
        if not frame_metrics:
            return ChunkMetricsV2(
                chunk_number=chunk_number, start_offset=0, end_offset=0,
                brilliance_score_v2=0.0, sunset_quality_grade='F',
                peak_sunset_moments=[], warmth_progression=0.0, atmospheric_drama=0.0,
                brilliance_score=0.0, color_variation_range=0.0,
                temporal_smoothness=0.0, golden_hour_bonus=0.0,
                frame_count=0, avg_processing_time_ms=0.0
            )
        
        try:
            # Extract enhanced metrics arrays
            warmth_scores = [m.sunset_warmth_score for m in frame_metrics]
            atmospheric_scores = [m.atmospheric_brilliance for m in frame_metrics]
            horizon_glows = [m.horizon_glow_intensity for m in frame_metrics]
            sky_richness = [m.sky_color_richness for m in frame_metrics]
            cloud_enhancements = [m.cloud_enhancement for m in frame_metrics]
            color_temps = [m.color_temperature for m in frame_metrics]
            brightness_balances = [m.brightness_balance for m in frame_metrics]
            
            # Calculate enhanced brilliance score
            brilliance_score_v2 = self._calculate_brilliance_score_v2(
                warmth_scores, atmospheric_scores, horizon_glows, 
                sky_richness, cloud_enhancements, brightness_balances
            )
            
            # Determine quality grade
            sunset_quality_grade = self._get_quality_grade_v2(brilliance_score_v2)
            
            # Identify peak sunset moments (top 10% warmth + atmosphere)
            combined_peak_scores = [
                w * 0.6 + a * 0.4 for w, a in zip(warmth_scores, atmospheric_scores)
            ]
            peak_threshold = np.percentile(combined_peak_scores, 90)
            peak_sunset_moments = [
                m.frame_number for m in frame_metrics 
                if (m.sunset_warmth_score * 0.6 + m.atmospheric_brilliance * 0.4) >= peak_threshold
            ]
            
            # Warmth progression (temperature transition quality)
            warmth_progression = self._calculate_warmth_progression(color_temps, warmth_scores)
            
            # Atmospheric drama
            atmospheric_drama = np.mean(atmospheric_scores) * (1 + np.std(atmospheric_scores) / 50)
            
            # Golden hour bonus (enhanced)
            golden_hour_bonus = self._calculate_golden_hour_bonus_v2(
                sunset_offset_minutes, warmth_scores
            )
            
            # Apply golden hour bonus to final score
            final_brilliance_score = brilliance_score_v2 * (1.0 + golden_hour_bonus)
            
            # Legacy metrics for compatibility
            legacy_colorfulness = [m.colorfulness for m in frame_metrics]
            legacy_score = self._calculate_legacy_score(legacy_colorfulness)
            color_variation_range = np.max(legacy_colorfulness) - np.min(legacy_colorfulness)
            temporal_smoothness = self._calculate_temporal_smoothness(legacy_colorfulness)
            processing_times = [m.processing_time_ms for m in frame_metrics]
            
            return ChunkMetricsV2(
                chunk_number=chunk_number,
                start_offset=frame_metrics[0].timestamp_offset,
                end_offset=frame_metrics[-1].timestamp_offset,
                brilliance_score_v2=final_brilliance_score,
                sunset_quality_grade=sunset_quality_grade,
                peak_sunset_moments=peak_sunset_moments,
                warmth_progression=warmth_progression,
                atmospheric_drama=atmospheric_drama,
                brilliance_score=legacy_score,
                color_variation_range=color_variation_range,
                temporal_smoothness=temporal_smoothness,
                golden_hour_bonus=golden_hour_bonus,
                frame_count=len(frame_metrics),
                avg_processing_time_ms=np.mean(processing_times)
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced chunk analysis failed: {e}")
            return ChunkMetricsV2(
                chunk_number=chunk_number, start_offset=0, end_offset=0,
                brilliance_score_v2=0.0, sunset_quality_grade='F',
                peak_sunset_moments=[], warmth_progression=0.0, atmospheric_drama=0.0,
                brilliance_score=0.0, color_variation_range=0.0,
                temporal_smoothness=0.0, golden_hour_bonus=0.0,
                frame_count=len(frame_metrics), avg_processing_time_ms=0.0
            )
    
    def _calculate_brilliance_score_v2(self, warmth_scores: List[float],
                                     atmospheric_scores: List[float],
                                     horizon_glows: List[float],
                                     sky_richness: List[float],
                                     cloud_enhancements: List[float],
                                     brightness_balances: List[float]) -> float:
        """
        Enhanced brilliance score optimized for human sunset perception
        """
        # Human-perception-tuned weights
        weights = {
            'sunset_warmth': 0.35,      # Most important: warm sunset colors
            'atmospheric_drama': 0.25,   # Sky drama and intensity
            'horizon_glow': 0.20,       # Characteristic sunset glow
            'sky_richness': 0.15,       # Color richness in sky
            'cloud_enhancement': 0.03,   # Clouds enhancing colors
            'brightness_balance': 0.02   # Proper exposure
        }
        
        # Component scores (0-100 each)
        warmth_score = np.mean(warmth_scores)
        atmosphere_score = np.mean(atmospheric_scores) 
        glow_score = np.mean(horizon_glows)
        richness_score = np.mean(sky_richness)
        cloud_score = np.mean(cloud_enhancements)
        brightness_score = np.mean(brightness_balances)
        
        # Weighted combination
        brilliance_v2 = (
            weights['sunset_warmth'] * warmth_score +
            weights['atmospheric_drama'] * atmosphere_score +
            weights['horizon_glow'] * glow_score +
            weights['sky_richness'] * richness_score +
            weights['cloud_enhancement'] * cloud_score +
            weights['brightness_balance'] * brightness_score
        )
        
        return float(np.clip(brilliance_v2, 0, 100))
    
    def _get_quality_grade_v2(self, score: float) -> str:
        """Enhanced quality grading"""
        if score >= 85:
            return 'A+'
        elif score >= 75:
            return 'A'
        elif score >= 65:
            return 'B+'
        elif score >= 55:
            return 'B'
        elif score >= 45:
            return 'C+'
        elif score >= 35:
            return 'C'
        elif score >= 25:
            return 'D'
        else:
            return 'F'
    
    def _calculate_warmth_progression(self, color_temps: List[float], 
                                    warmth_scores: List[float]) -> float:
        """Calculate quality of temperature transition during sunset"""
        if len(color_temps) < 2:
            return 0.0
        
        # Good sunset should show progression from cooler to warmer
        temp_trend = np.polyfit(range(len(color_temps)), color_temps, 1)[0]  # Slope
        warmth_trend = np.polyfit(range(len(warmth_scores)), warmth_scores, 1)[0]
        
        # Ideal: decreasing temperature, increasing warmth
        temp_progression = max(0, -temp_trend / 1000)  # Normalize temperature decrease
        warmth_progression_val = max(0, warmth_trend / 10)  # Normalize warmth increase
        
        return float(np.clip((temp_progression + warmth_progression_val) * 50, 0, 100))
    
    def _calculate_golden_hour_bonus_v2(self, sunset_offset_minutes: float, 
                                       warmth_scores: List[float]) -> float:
        """Enhanced golden hour bonus that considers actual warm color presence"""
        # Base time proximity bonus
        time_bonus = 0.0
        if abs(sunset_offset_minutes) <= self.golden_hour_minutes:
            time_bonus = 0.2 * (1.0 - abs(sunset_offset_minutes) / self.golden_hour_minutes)
        
        # Actual warmth performance bonus
        avg_warmth = np.mean(warmth_scores)
        warmth_bonus = 0.0
        if avg_warmth > 30:  # Only bonus if actually warm
            warmth_bonus = 0.15 * (avg_warmth / 100)
        
        return float(time_bonus + warmth_bonus)
    
    def _calculate_legacy_score(self, colorfulness_values: List[float]) -> float:
        """Legacy scoring for comparison"""
        return float(np.mean(colorfulness_values) / 100.0 * 100)
    
    def _calculate_temporal_smoothness(self, values: List[float]) -> float:
        """Calculate temporal smoothness score"""
        if len(values) < 2:
            return 1.0
            
        mean_val = np.mean(values)
        if mean_val == 0:
            return 1.0
            
        cv = np.std(values) / mean_val
        smoothness = np.exp(-cv)
        
        return float(np.clip(smoothness, 0, 1))
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.processing_times:
            return {'status': 'no_data'}
            
        return {
            'avg_processing_time_ms': float(np.mean(self.processing_times)),
            'max_processing_time_ms': float(np.max(self.processing_times)),
            'min_processing_time_ms': float(np.min(self.processing_times)),
            'frames_processed': len(self.processing_times),
            'total_processing_time_s': float(np.sum(self.processing_times) / 1000),
            'pi_performance_status': 'good' if np.mean(self.processing_times) < 15 else 'slow',
            'version': 'v2.0_human_optimized'
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking"""
        self.processing_times = []
        self.chunk_data = []