"""
Sunset Brilliance Score (SBS) System
Pi-optimized lightweight algorithms for real-time sunset quality analysis
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
class FrameMetrics:
    """Container for per-frame SBS metrics"""
    frame_number: int
    timestamp_offset: float  # seconds from start
    colorfulness: float
    color_temperature: float
    sky_saturation: float
    gradient_intensity: float
    brightness_mean: float
    processing_time_ms: float


@dataclass
class ChunkMetrics:
    """Container for chunk-level SBS analysis"""
    chunk_number: int
    start_offset: float
    end_offset: float
    brilliance_score: float
    peak_frames: List[int]
    color_variation_range: float
    temporal_smoothness: float
    golden_hour_bonus: float
    frame_count: int
    avg_processing_time_ms: float


class SunsetBrillianceScore:
    """
    Pi-optimized Sunset Brilliance Score system using lightweight mathematical algorithms
    
    Core Algorithm: Hasler-Süsstrunk Colorfulness Metric
    - 95.3% correlation with human perception
    - ~1ms processing time per frame
    - Pure NumPy/OpenCV calculations
    """
    
    def __init__(self):
        """Initialize SBS analyzer with Pi-optimized settings"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Pi-optimized parameters
        self.sbs_config = self.config.get('sbs', {})
        self.enabled = self.sbs_config.get('enabled', True)
        self.analysis_resize_height = self.sbs_config.get('analysis_resize_height', 480)  # Reduce processing load
        self.sky_region_ratio = self.sbs_config.get('sky_region_ratio', 0.6)  # Top 60% is sky
        self.golden_hour_minutes = self.sbs_config.get('golden_hour_minutes', 30)  # ±30min from sunset
        
        # Performance monitoring
        self.processing_times = []
        self.chunk_data = []
        
        self.logger.info("SBS system initialized with Pi-optimized settings")
    
    def analyze_frame(self, frame: np.ndarray, frame_number: int, 
                     timestamp_offset: float) -> Optional[FrameMetrics]:
        """
        Analyze single frame for sunset brilliance metrics
        Pi-optimized: ~2-3ms total processing time
        
        Args:
            frame: OpenCV image frame (BGR)
            frame_number: Frame sequence number
            timestamp_offset: Seconds from capture start
            
        Returns:
            FrameMetrics object or None if analysis disabled
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
                
            # Core metrics calculation
            colorfulness = self._hasler_susstrunk_colorfulness(frame_resized)
            color_temperature = self._estimate_color_temperature(frame_resized)
            sky_saturation = self._analyze_sky_saturation(frame_resized)
            gradient_intensity = self._calculate_gradient_intensity(frame_resized)
            brightness_mean = np.mean(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY))
            
            processing_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            self.processing_times.append(processing_time)
            
            return FrameMetrics(
                frame_number=frame_number,
                timestamp_offset=timestamp_offset,
                colorfulness=colorfulness,
                color_temperature=color_temperature,
                sky_saturation=sky_saturation,
                gradient_intensity=gradient_intensity,
                brightness_mean=brightness_mean,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Frame analysis failed: {e}")
            return None
    
    def _hasler_susstrunk_colorfulness(self, frame: np.ndarray) -> float:
        """
        Hasler-Süsstrunk colorfulness metric - Pi optimized
        95.3% correlation with human perception, ~1ms processing time
        
        Reference: "Measuring colourfulness in natural images" (2003)
        """
        # Convert BGR to RGB for proper color space calculation
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        
        # Opponent color space (rg, yb)
        rg = R - G
        yb = 0.5 * (R + G) - B
        
        # Calculate standard deviations and means
        rg_std = np.std(rg)
        yb_std = np.std(yb)
        rg_mean = np.mean(rg)
        yb_mean = np.mean(yb)
        
        # Colorfulness formula
        std_rgyb = np.sqrt(rg_std**2 + yb_std**2)
        mean_rgyb = np.sqrt(rg_mean**2 + yb_mean**2)
        
        colorfulness = std_rgyb + 0.3 * mean_rgyb
        
        return float(colorfulness)
    
    def _estimate_color_temperature(self, frame: np.ndarray) -> float:
        """
        Estimate color temperature using RGB ratios
        Optimized for sunset analysis (warm/cool transitions)
        """
        # Convert to float for precision
        frame_float = frame.astype(np.float32)
        
        # Calculate average RGB values
        mean_rgb = np.mean(frame_float, axis=(0, 1))  # [B, G, R] in OpenCV
        mean_b, mean_g, mean_r = mean_rgb
        
        # Avoid division by zero
        if mean_b == 0:
            mean_b = 1
            
        # Color temperature estimation using R/B ratio
        # Higher R/B = warmer (lower K), Lower R/B = cooler (higher K)
        rb_ratio = mean_r / mean_b
        
        # Empirical mapping to Kelvin (calibrated for sunset conditions)
        # Sunset range: ~2000K (deep red) to ~6500K (daylight)
        if rb_ratio > 2.0:
            color_temp = 2000 + (4500 * (1.0 / rb_ratio))  # Very warm
        elif rb_ratio > 1.5:
            color_temp = 2500 + (2000 * (2.0 - rb_ratio) / 0.5)  # Warm
        elif rb_ratio > 1.0:
            color_temp = 4500 + (2000 * (1.5 - rb_ratio) / 0.5)  # Neutral-warm
        else:
            color_temp = 6500  # Cool/daylight
            
        return float(np.clip(color_temp, 1000, 10000))
    
    def _analyze_sky_saturation(self, frame: np.ndarray) -> float:
        """
        Analyze saturation in sky region using HSV color space
        Focus on upper portion of frame where sky dominates
        """
        height = frame.shape[0]
        sky_region = frame[:int(height * self.sky_region_ratio), :]
        
        # Convert to HSV for saturation analysis
        hsv_sky = cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)
        
        # Extract saturation channel (0-255)
        saturation = hsv_sky[:,:,1]
        
        # Calculate mean saturation, weighted by brightness to ignore dark areas
        value = hsv_sky[:,:,2]  # Brightness channel
        
        # Create brightness mask (ignore very dark pixels)
        brightness_mask = value > 30  # Threshold to ignore dark areas
        
        if np.sum(brightness_mask) > 0:
            weighted_saturation = np.mean(saturation[brightness_mask])
        else:
            weighted_saturation = np.mean(saturation)
            
        # Normalize to 0-1 range
        return float(weighted_saturation / 255.0)
    
    def _calculate_gradient_intensity(self, frame: np.ndarray) -> float:
        """
        Measure color gradient intensity from horizon to zenith
        Indicates dramatic color transitions typical of brilliant sunsets
        """
        # Convert to grayscale for gradient calculation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate vertical gradient (horizon to zenith)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate horizontal gradient (sky color variations)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        
        # Combine gradients
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Focus on sky region
        height = frame.shape[0]
        sky_gradient = gradient_magnitude[:int(height * self.sky_region_ratio), :]
        
        # Calculate mean gradient intensity
        gradient_intensity = np.mean(sky_gradient)
        
        return float(gradient_intensity)
    
    def analyze_chunk(self, frame_metrics: List[FrameMetrics], 
                     chunk_number: int, sunset_offset_minutes: float = 0) -> ChunkMetrics:
        """
        Analyze 15-minute chunk and calculate brilliance score
        
        Args:
            frame_metrics: List of frame metrics for this chunk
            chunk_number: Chunk identifier (0-7 for 2-hour capture)
            sunset_offset_minutes: Minutes from actual sunset time
            
        Returns:
            ChunkMetrics with aggregated brilliance analysis
        """
        if not frame_metrics:
            return ChunkMetrics(
                chunk_number=chunk_number,
                start_offset=0, end_offset=0,
                brilliance_score=0.0,
                peak_frames=[], color_variation_range=0.0,
                temporal_smoothness=0.0, golden_hour_bonus=0.0,
                frame_count=0, avg_processing_time_ms=0.0
            )
        
        try:
            # Extract metric arrays
            colorfulness_values = [m.colorfulness for m in frame_metrics]
            color_temps = [m.color_temperature for m in frame_metrics]
            saturations = [m.sky_saturation for m in frame_metrics]
            gradients = [m.gradient_intensity for m in frame_metrics]
            brightness_values = [m.brightness_mean for m in frame_metrics]
            processing_times = [m.processing_time_ms for m in frame_metrics]
            
            # Core brilliance calculation
            brilliance_score = self._calculate_brilliance_score(
                colorfulness_values, saturations, gradients, 
                color_temps, brightness_values
            )
            
            # Identify peak frames (top 10% colorfulness)
            colorfulness_threshold = np.percentile(colorfulness_values, 90)
            peak_frames = [
                m.frame_number for m in frame_metrics 
                if m.colorfulness >= colorfulness_threshold
            ]
            
            # Color variation analysis
            color_variation_range = np.max(colorfulness_values) - np.min(colorfulness_values)
            
            # Temporal smoothness (measure of consistent quality)
            temporal_smoothness = self._calculate_temporal_smoothness(colorfulness_values)
            
            # Golden hour bonus (proximity to sunset enhances score)
            golden_hour_bonus = self._calculate_golden_hour_bonus(sunset_offset_minutes)
            
            # Apply golden hour bonus to final score
            final_brilliance_score = brilliance_score * (1.0 + golden_hour_bonus)
            
            return ChunkMetrics(
                chunk_number=chunk_number,
                start_offset=frame_metrics[0].timestamp_offset,
                end_offset=frame_metrics[-1].timestamp_offset,
                brilliance_score=final_brilliance_score,
                peak_frames=peak_frames,
                color_variation_range=color_variation_range,
                temporal_smoothness=temporal_smoothness,
                golden_hour_bonus=golden_hour_bonus,
                frame_count=len(frame_metrics),
                avg_processing_time_ms=np.mean(processing_times)
            )
            
        except Exception as e:
            self.logger.error(f"Chunk analysis failed: {e}")
            return ChunkMetrics(
                chunk_number=chunk_number, start_offset=0, end_offset=0,
                brilliance_score=0.0, peak_frames=[], color_variation_range=0.0,
                temporal_smoothness=0.0, golden_hour_bonus=0.0,
                frame_count=len(frame_metrics), avg_processing_time_ms=0.0
            )
    
    def _calculate_brilliance_score(self, colorfulness_values: List[float],
                                  saturations: List[float], gradients: List[float],
                                  color_temps: List[float], brightness_values: List[float]) -> float:
        """
        Calculate weighted brilliance score from component metrics
        Tuned for sunset conditions with Pi-optimized weights
        """
        # Component weights (tuned for sunset brilliance)
        weights = {
            'colorfulness': 0.40,     # Primary metric - Hasler-Süsstrunk
            'saturation': 0.25,       # Sky color richness
            'gradient': 0.20,         # Color transition drama
            'color_temp': 0.10,       # Warmth bonus for golden hour
            'brightness': 0.05        # Avoid over/under exposure
        }
        
        # Normalize and score each component
        colorfulness_score = np.mean(colorfulness_values) / 100.0  # Typical range 0-150
        saturation_score = np.mean(saturations)  # Already 0-1 range
        gradient_score = np.clip(np.mean(gradients) / 50.0, 0, 1)  # Typical range 0-100
        
        # Color temperature scoring (warm colors get higher scores)
        avg_temp = np.mean(color_temps)
        if avg_temp < 3000:  # Very warm (sunset peak)
            temp_score = 1.0
        elif avg_temp < 4000:  # Warm
            temp_score = 0.8
        elif avg_temp < 5000:  # Neutral
            temp_score = 0.5
        else:  # Cool
            temp_score = 0.2
            
        # Brightness scoring (avoid extremes)
        avg_brightness = np.mean(brightness_values)
        if 50 <= avg_brightness <= 200:  # Good exposure range
            brightness_score = 1.0
        elif avg_brightness < 30 or avg_brightness > 220:  # Poor exposure
            brightness_score = 0.3
        else:
            brightness_score = 0.7
            
        # Weighted combination
        brilliance_score = (
            weights['colorfulness'] * colorfulness_score +
            weights['saturation'] * saturation_score +
            weights['gradient'] * gradient_score +
            weights['color_temp'] * temp_score +
            weights['brightness'] * brightness_score
        )
        
        # Scale to 0-100 range for intuitive scoring
        return float(np.clip(brilliance_score * 100, 0, 100))
    
    def _calculate_temporal_smoothness(self, values: List[float]) -> float:
        """Calculate temporal smoothness score (0-1, higher = more consistent)"""
        if len(values) < 2:
            return 1.0
            
        # Calculate coefficient of variation (std/mean)
        mean_val = np.mean(values)
        if mean_val == 0:
            return 1.0
            
        cv = np.std(values) / mean_val
        
        # Convert to smoothness score (lower CV = higher smoothness)
        smoothness = np.exp(-cv)  # Exponential decay
        
        return float(np.clip(smoothness, 0, 1))
    
    def _calculate_golden_hour_bonus(self, sunset_offset_minutes: float) -> float:
        """
        Calculate golden hour proximity bonus
        Peak bonus at sunset time, decaying with distance
        
        Args:
            sunset_offset_minutes: Minutes from sunset (negative=before, positive=after)
            
        Returns:
            Bonus multiplier (0-0.3 additional score)
        """
        # Maximum bonus within golden hour window
        if abs(sunset_offset_minutes) <= self.golden_hour_minutes:
            # Peak bonus at sunset (0 minutes)
            bonus = 0.3 * (1.0 - abs(sunset_offset_minutes) / self.golden_hour_minutes)
        else:
            bonus = 0.0
            
        return float(bonus)
    
    def save_chunk_analysis(self, chunk_metrics: ChunkMetrics, date_str: str):
        """Save chunk analysis to JSON for later aggregation"""
        paths = self.config.get_storage_paths()
        sbs_dir = paths['temp'] / 'sbs' / date_str
        sbs_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_file = sbs_dir / f"chunk_{chunk_metrics.chunk_number:02d}.json"
        
        with open(chunk_file, 'w') as f:
            json.dump(asdict(chunk_metrics), f, indent=2)
            
        self.logger.debug(f"Saved chunk analysis: {chunk_file}")
    
    def load_daily_analysis(self, date_str: str) -> List[ChunkMetrics]:
        """Load all chunk analyses for a given date"""
        paths = self.config.get_storage_paths()
        sbs_dir = paths['temp'] / 'sbs' / date_str
        
        chunk_metrics = []
        if sbs_dir.exists():
            for chunk_file in sorted(sbs_dir.glob("chunk_*.json")):
                try:
                    with open(chunk_file) as f:
                        data = json.load(f)
                        chunk_metrics.append(ChunkMetrics(**data))
                except Exception as e:
                    self.logger.warning(f"Failed to load {chunk_file}: {e}")
                    
        return chunk_metrics
    
    def calculate_daily_summary(self, chunk_metrics: List[ChunkMetrics]) -> Dict:
        """
        Calculate daily summary statistics from all chunks
        
        Returns:
            Dictionary with daily SBS summary
        """
        if not chunk_metrics:
            return {
                'daily_brilliance_score': 0.0,
                'peak_chunk': None,
                'total_frames': 0,
                'avg_processing_time_ms': 0.0,
                'quality_grade': 'F'
            }
        
        # Calculate daily metrics
        chunk_scores = [c.brilliance_score for c in chunk_metrics]
        daily_score = np.mean(chunk_scores)
        peak_chunk_idx = np.argmax(chunk_scores)
        total_frames = sum(c.frame_count for c in chunk_metrics)
        avg_processing_time = np.mean([c.avg_processing_time_ms for c in chunk_metrics])
        
        # Assign quality grade
        if daily_score >= 80:
            grade = 'A'
        elif daily_score >= 70:
            grade = 'B'
        elif daily_score >= 60:
            grade = 'C'
        elif daily_score >= 50:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'daily_brilliance_score': float(daily_score),
            'peak_chunk': chunk_metrics[peak_chunk_idx].chunk_number,
            'peak_chunk_score': chunk_scores[peak_chunk_idx],
            'chunk_scores': chunk_scores,
            'total_frames': total_frames,
            'avg_processing_time_ms': float(avg_processing_time),
            'quality_grade': grade,
            'chunks_analyzed': len(chunk_metrics)
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for Pi monitoring"""
        if not self.processing_times:
            return {'status': 'no_data'}
            
        return {
            'avg_processing_time_ms': float(np.mean(self.processing_times)),
            'max_processing_time_ms': float(np.max(self.processing_times)),
            'min_processing_time_ms': float(np.min(self.processing_times)),
            'frames_processed': len(self.processing_times),
            'total_processing_time_s': float(np.sum(self.processing_times) / 1000),
            'pi_performance_status': 'good' if np.mean(self.processing_times) < 10 else 'slow'
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking for new capture session"""
        self.processing_times = []
        self.chunk_data = []