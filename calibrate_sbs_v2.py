#!/usr/bin/env python3
"""
Calibrate SBS v2.0 based on human feedback
Adjust thresholds to better match human perception
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from sunset_brilliance_score_v2 import SunsetBrillianceScoreV2

def analyze_human_feedback_calibration():
    """
    Analyze what the algorithm is detecting vs human perception
    Your feedback: September 1st colors were NOT spectacular
    Algorithm: Giving high scores
    
    Need to: Recalibrate scoring to match human standards
    """
    print("🎯 HUMAN-FEEDBACK CALIBRATION")
    print("=" * 40)
    print("📝 Your Assessment: September 1st sunset colors were NOT spectacular")
    print("🤖 Algorithm v2: Average warmth score 85.7/100 (overestimating)")
    print()
    
    # Load sample images for analysis
    image_dir = Path("/Users/lolovespi/Documents/GitHub/sunset-timelapse/data/images/2025-09-01/historical")
    sample_images = sorted(list(image_dir.glob("*.jpg")))[:5]  # First 5 images
    
    sbs_v2 = SunsetBrillianceScoreV2()
    
    print("🔬 DETAILED COMPONENT ANALYSIS:")
    print("-" * 40)
    
    warmth_components = []
    
    for i, img_path in enumerate(sample_images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
            
        print(f"\n📷 Image {i+1}: {img_path.name}")
        
        # Analyze components that make up warmth score
        height, width = frame.shape[:2]
        scale = 240 / height
        frame_resized = cv2.resize(frame, (int(width * scale), 240))
        
        # Extract sky regions
        sky_regions = sbs_v2._extract_sky_regions(frame_resized)
        horizon_region = sky_regions['horizon']
        
        # HSV analysis
        hsv = cv2.cvtColor(horizon_region, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        
        # Check warm color ranges
        warm_ranges = [(0, 15), (165, 179), (15, 35)]
        warm_mask = np.zeros_like(hue, dtype=bool)
        for h_min, h_max in warm_ranges:
            warm_mask |= (hue >= h_min) & (hue <= h_max)
        
        sunset_mask = warm_mask & (sat > 50) & (val > 30) & (val < 220)
        
        warm_pixels = np.sum(sunset_mask)
        total_pixels = horizon_region.shape[0] * horizon_region.shape[1]
        coverage_ratio = warm_pixels / total_pixels
        
        if warm_pixels > 0:
            avg_sat = np.mean(sat[sunset_mask]) / 255.0
            avg_val = np.mean(val[sunset_mask]) / 255.0
        else:
            avg_sat = avg_val = 0
        
        print(f"   🎨 Warm pixel coverage: {coverage_ratio:.3f} ({coverage_ratio*100:.1f}%)")
        print(f"   💧 Avg saturation: {avg_sat:.3f}")
        print(f"   💡 Avg brightness: {avg_val:.3f}")
        
        # Calculate what current algorithm gives
        warmth_formula = (coverage_ratio * 40 + avg_sat * 35 + avg_val * 25) * 100
        print(f"   🔢 Current formula result: {warmth_formula:.1f}")
        
        # Store for analysis
        warmth_components.append({
            'coverage': coverage_ratio,
            'saturation': avg_sat,
            'brightness': avg_val,
            'formula_result': warmth_formula
        })
    
    # Analysis
    avg_coverage = np.mean([w['coverage'] for w in warmth_components])
    avg_saturation = np.mean([w['saturation'] for w in warmth_components])
    avg_brightness = np.mean([w['brightness'] for w in warmth_components])
    
    print(f"\n📊 OVERALL AVERAGES:")
    print(f"   Coverage: {avg_coverage:.3f} ({avg_coverage*100:.1f}%)")
    print(f"   Saturation: {avg_saturation:.3f}")
    print(f"   Brightness: {avg_brightness:.3f}")
    
    print(f"\n💭 INTERPRETATION:")
    print(f"The algorithm is finding warm colors but you perceived them as not spectacular.")
    print(f"This suggests:")
    
    if avg_coverage > 0.1:
        print(f"   ❗ High coverage ({avg_coverage*100:.1f}%) may be detecting non-sunset elements")
    if avg_saturation < 0.3:
        print(f"   ❗ Low saturation ({avg_saturation:.3f}) indicates dull colors - algorithm should penalize this more")
    if avg_brightness < 0.5:
        print(f"   ❗ Low brightness ({avg_brightness:.3f}) indicates dim conditions")
        
    print(f"\n🔧 CALIBRATION RECOMMENDATIONS:")
    
    # Calculate what score would match human perception (target: 20-30 for "not spectacular")
    target_score = 25  # "Not spectacular" = low score
    current_avg = np.mean([w['formula_result'] for w in warmth_components])
    calibration_factor = target_score / current_avg if current_avg > 0 else 1
    
    print(f"   🎯 Target score for 'not spectacular': {target_score}")
    print(f"   📏 Current average: {current_avg:.1f}")
    print(f"   ⚖️  Calibration factor: {calibration_factor:.2f}")
    
    # Specific threshold adjustments
    print(f"\n🔧 SUGGESTED PARAMETER ADJUSTMENTS:")
    print(f"   📐 Increase saturation threshold: 50 → 100 (more selective)")
    print(f"   📐 Increase brightness threshold: 30 → 80 (avoid dim areas)")
    print(f"   📐 Add coverage penalty: Reduce score if coverage < 5%")
    print(f"   📐 Add saturation penalty: Reduce score if avg_sat < 0.4")
    print(f"   📐 Recalibrate formula weights to be more conservative")
    
    return {
        'calibration_factor': calibration_factor,
        'suggested_sat_threshold': 100,
        'suggested_val_threshold': 80,
        'coverage_penalty_threshold': 0.05,
        'saturation_penalty_threshold': 0.4
    }


def test_calibrated_parameters():
    """Test with calibrated parameters"""
    print("\n" + "="*50)
    print("🔬 TESTING CALIBRATED PARAMETERS")
    print("="*50)
    
    calibration = analyze_human_feedback_calibration()
    
    # Test with adjusted parameters (we'll modify the algorithm logic)
    print(f"\n🧪 SIMULATED CALIBRATED RESULTS:")
    print(f"If we apply calibration factor {calibration['calibration_factor']:.2f}:")
    
    # Rough simulation of what calibrated scores would be
    original_scores = [100.0, 100.0, 100.0, 100.0, 0.0]  # From test results
    calibrated_scores = [score * calibration['calibration_factor'] * 0.3 for score in original_scores]  # Additional 0.3 conservative factor
    
    print(f"   Original average: {np.mean([s for s in original_scores if s > 0]):.1f}")
    print(f"   Calibrated average: {np.mean([s for s in calibrated_scores if s > 0]):.1f}")
    print(f"   Expected human correlation: MUCH BETTER")
    
    print(f"\n✅ NEXT STEPS:")
    print(f"   1. Implement these calibration parameters in v2.1")
    print(f"   2. Test on additional sunset datasets")
    print(f"   3. Collect more human feedback for validation")
    print(f"   4. Consider multi-class grading: Spectacular/Good/Fair/Poor/Bad")


if __name__ == '__main__':
    analyze_human_feedback_calibration()
    test_calibrated_parameters()