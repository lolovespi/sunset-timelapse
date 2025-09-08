#!/usr/bin/env python3
"""
Test Enhanced SBS v2.0 Algorithm
Compare v1 vs v2 performance on your September 1st sunset images
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import time
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from sunset_brilliance_score import SunsetBrillianceScore  # v1
from sunset_brilliance_score_v2 import SunsetBrillianceScoreV2  # v2

def test_both_algorithms(image_paths: List[Path], num_samples: int = 10) -> None:
    """Test both v1 and v2 algorithms on sample images"""
    
    print("🔬 SBS ALGORITHM COMPARISON: v1 vs v2")
    print("=" * 60)
    
    # Initialize both algorithms
    sbs_v1 = SunsetBrillianceScore()
    sbs_v2 = SunsetBrillianceScoreV2()
    
    # Select sample images (start, middle, end + random samples)
    if len(image_paths) < num_samples:
        sample_images = image_paths
    else:
        # Strategic sampling: start, end, and evenly distributed middle samples
        indices = [0, len(image_paths)-1]  # Start and end
        step = len(image_paths) // (num_samples - 2)
        indices.extend(range(step, len(image_paths)-1, step))
        indices = sorted(set(indices))[:num_samples]  # Remove duplicates and limit
        sample_images = [image_paths[i] for i in indices]
    
    print(f"📊 Testing {len(sample_images)} sample images...")
    print(f"Total dataset: {len(image_paths)} images")
    print()
    
    v1_scores = []
    v2_scores = []
    v1_times = []
    v2_times = []
    
    for i, img_path in enumerate(sample_images):
        print(f"🖼️  Image {i+1}/{len(sample_images)}: {img_path.name}")
        
        # Load image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print("   ❌ Could not load image")
            continue
        
        # Test v1 algorithm
        start_time = time.perf_counter()
        metrics_v1 = sbs_v1.analyze_frame(frame, i, i*300)
        v1_time = (time.perf_counter() - start_time) * 1000
        
        # Test v2 algorithm
        start_time = time.perf_counter()
        metrics_v2 = sbs_v2.analyze_frame_v2(frame, i, i*300)
        v2_time = (time.perf_counter() - start_time) * 1000
        
        if metrics_v1 and metrics_v2:
            v1_scores.append(metrics_v1.colorfulness)
            v2_scores.append(metrics_v2.sunset_warmth_score)
            v1_times.append(v1_time)
            v2_times.append(v2_time)
            
            print(f"   📈 v1 Colorfulness: {metrics_v1.colorfulness:6.1f} | Processing: {v1_time:5.2f}ms")
            print(f"   🌅 v2 Sunset Warmth: {metrics_v2.sunset_warmth_score:6.1f} | Processing: {v2_time:5.2f}ms")
            print(f"   🌡️  v1 Color Temp: {metrics_v1.color_temperature:4.0f}K | v2: {metrics_v2.color_temperature:4.0f}K")
            print(f"   💧 v1 Sky Sat: {metrics_v1.sky_saturation:5.3f} | v2: {metrics_v2.sky_saturation:5.3f}")
            print(f"   ✨ v2 Atmospheric: {metrics_v2.atmospheric_brilliance:6.1f}")
            print(f"   🔥 v2 Horizon Glow: {metrics_v2.horizon_glow_intensity:6.1f}")
            print()
        else:
            print("   ❌ Analysis failed")
            print()
    
    if not v1_scores:
        print("❌ No successful analyses")
        return
    
    # Statistical comparison
    print("📊 STATISTICAL COMPARISON")
    print("=" * 40)
    print(f"📈 v1 Colorfulness:")
    print(f"   Mean: {np.mean(v1_scores):6.1f} | Std: {np.std(v1_scores):6.1f}")
    print(f"   Range: {np.min(v1_scores):6.1f} - {np.max(v1_scores):6.1f}")
    print()
    print(f"🌅 v2 Sunset Warmth:")
    print(f"   Mean: {np.mean(v2_scores):6.1f} | Std: {np.std(v2_scores):6.1f}")
    print(f"   Range: {np.min(v2_scores):6.1f} - {np.max(v2_scores):6.1f}")
    print()
    print(f"⚡ Performance:")
    print(f"   v1 Processing: {np.mean(v1_times):5.2f}ms avg (max: {np.max(v1_times):5.2f}ms)")
    print(f"   v2 Processing: {np.mean(v2_times):5.2f}ms avg (max: {np.max(v2_times):5.2f}ms)")
    print(f"   v2 Overhead: {np.mean(v2_times) - np.mean(v1_times):+5.2f}ms ({(np.mean(v2_times)/np.mean(v1_times)-1)*100:+4.1f}%)")
    print()
    
    # Interpretation
    print("💭 INTERPRETATION:")
    print("=" * 20)
    
    # Check for warm color detection improvement
    v2_mean = np.mean(v2_scores)
    v1_mean = np.mean(v1_scores)
    
    if v2_mean < 30:
        print("✅ v2 correctly identifies LOW sunset warmth (matches your observation)")
        print(f"   v2 Warmth Score: {v2_mean:.1f}/100 (appropriately low)")
    else:
        print(f"⚠️  v2 Warmth Score: {v2_mean:.1f}/100 (may still be overestimating)")
    
    if v1_mean > 150:
        print("❌ v1 shows artificially high colorfulness (mathematical noise)")
        print(f"   v1 Colorfulness: {v1_mean:.1f} (detecting contrast, not beauty)")
    
    performance_overhead = (np.mean(v2_times) / np.mean(v1_times) - 1) * 100
    if performance_overhead < 50:
        print(f"✅ v2 performance acceptable ({performance_overhead:+.1f}% overhead)")
    else:
        print(f"⚠️  v2 performance overhead significant ({performance_overhead:+.1f}%)")
    
    print()
    print("🎯 RECOMMENDATIONS:")
    if v2_mean < 30:
        print("✅ Enhanced algorithm correctly correlates with human perception")
        print("✅ Ready to replace v1 in production")
    elif v2_mean < v1_mean / 3:
        print("✅ Significant improvement in sunset quality detection")
        print("📈 Consider fine-tuning warmth detection thresholds")
    else:
        print("📈 Algorithm needs further calibration")
        print("🔧 Consider adjusting warm color detection parameters")


def detailed_analysis_comparison(image_path: Path) -> None:
    """Detailed side-by-side analysis of a single image"""
    print(f"\n🔍 DETAILED ANALYSIS: {image_path.name}")
    print("=" * 50)
    
    frame = cv2.imread(str(image_path))
    if frame is None:
        print("❌ Could not load image")
        return
    
    sbs_v1 = SunsetBrillianceScore()
    sbs_v2 = SunsetBrillianceScoreV2()
    
    # Analyze with both versions
    metrics_v1 = sbs_v1.analyze_frame(frame, 0, 0)
    metrics_v2 = sbs_v2.analyze_frame_v2(frame, 0, 0)
    
    if not metrics_v1 or not metrics_v2:
        print("❌ Analysis failed")
        return
    
    print("📊 METRIC COMPARISON:")
    print(f"{'Metric':<25} {'v1':<12} {'v2':<12} {'Improvement':<12}")
    print("-" * 65)
    
    # Core metrics
    print(f"{'Primary Score':<25} {metrics_v1.colorfulness:<12.1f} {metrics_v2.sunset_warmth_score:<12.1f} {'SUNSET FOCUS':<12}")
    print(f"{'Color Temperature':<25} {metrics_v1.color_temperature:<12.0f} {metrics_v2.color_temperature:<12.0f} {'ENHANCED':<12}")
    print(f"{'Sky Saturation':<25} {metrics_v1.sky_saturation:<12.3f} {metrics_v2.sky_saturation:<12.3f} {'SKY-FOCUSED':<12}")
    print(f"{'Gradient Analysis':<25} {metrics_v1.gradient_intensity:<12.1f} {metrics_v2.gradient_intensity:<12.1f} {'ATMOSPHERIC':<12}")
    
    # New v2 metrics
    print()
    print("🆕 NEW v2 METRICS:")
    print(f"   Atmospheric Brilliance: {metrics_v2.atmospheric_brilliance:6.1f}/100")
    print(f"   Horizon Glow Intensity: {metrics_v2.horizon_glow_intensity:6.1f}/100")
    print(f"   Sky Color Richness:     {metrics_v2.sky_color_richness:6.1f}/100")
    print(f"   Cloud Enhancement:      {metrics_v2.cloud_enhancement:6.1f}/100")
    print(f"   Brightness Balance:     {metrics_v2.brightness_balance:6.1f}/100")
    
    # Performance
    print()
    print(f"⚡ PERFORMANCE:")
    print(f"   v1 Processing Time: {metrics_v1.processing_time_ms:5.2f}ms")
    print(f"   v2 Processing Time: {metrics_v2.processing_time_ms:5.2f}ms")
    print(f"   Overhead: {metrics_v2.processing_time_ms - metrics_v1.processing_time_ms:+5.2f}ms")


def main():
    """Main test execution"""
    image_dir = Path("/Users/lolovespi/Documents/GitHub/sunset-timelapse/data/images/2025-09-01/historical")
    
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return 1
    
    # Get all images
    image_files = sorted(list(image_dir.glob("*.jpg")))
    
    if not image_files:
        print("❌ No images found")
        return 1
    
    print(f"🌅 TESTING ENHANCED SBS v2.0 ALGORITHM")
    print(f"📁 Dataset: {len(image_files)} images from September 1st sunset")
    print(f"💭 Human Assessment: Colors were NOT spectacular")
    print()
    
    # Run comparison test
    test_both_algorithms(image_files, num_samples=8)
    
    # Detailed analysis of a middle image
    middle_image = image_files[len(image_files)//2]
    detailed_analysis_comparison(middle_image)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())