#!/usr/bin/env python3
"""
Debug SBS Analysis - Examine what the algorithm is actually detecting
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from sunset_brilliance_score import SunsetBrillianceScore

def analyze_sample_images():
    """Analyze a few sample images to see what SBS is detecting"""
    image_dir = Path("/Users/lolovespi/Documents/GitHub/sunset-timelapse/data/images/2025-09-01/historical")
    
    # Get a representative sample: start, middle, end
    all_images = sorted(list(image_dir.glob("*.jpg")))
    
    if not all_images:
        print("No images found!")
        return
    
    sample_images = [
        all_images[0],           # Start
        all_images[len(all_images)//2],  # Middle  
        all_images[-1]           # End
    ]
    
    sbs = SunsetBrillianceScore()
    
    print("🔍 SBS ALGORITHM DEBUG ANALYSIS")
    print("=" * 50)
    
    for i, img_path in enumerate(sample_images):
        print(f"\n📷 Image {i+1}: {img_path.name}")
        
        # Load image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print("   ❌ Could not load image")
            continue
        
        # Get basic image stats
        height, width = frame.shape[:2]
        mean_bgr = np.mean(frame, axis=(0,1))
        
        print(f"   📐 Dimensions: {width}x{height}")
        print(f"   🎨 Mean BGR: [{mean_bgr[0]:.1f}, {mean_bgr[1]:.1f}, {mean_bgr[2]:.1f}]")
        
        # Analyze with SBS
        metrics = sbs.analyze_frame(frame, i, i*300)
        
        if metrics:
            print(f"   🌅 Colorfulness: {metrics.colorfulness:.2f}")
            print(f"   🌡️  Color Temp: {metrics.color_temperature:.0f}K")
            print(f"   💧 Sky Saturation: {metrics.sky_saturation:.3f}")
            print(f"   📈 Gradient: {metrics.gradient_intensity:.2f}")
            print(f"   💡 Brightness: {metrics.brightness_mean:.1f}")
            
            # Manual colorfulness breakdown
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            
            # Hasler-Süsstrunk components
            rg = R - G  # Red-Green opponent
            yb = 0.5 * (R + G) - B  # Yellow-Blue opponent
            
            rg_std = np.std(rg)
            yb_std = np.std(yb)
            rg_mean = np.mean(rg)
            yb_mean = np.mean(yb)
            
            print(f"   🔴 RG std: {rg_std:.2f}, mean: {rg_mean:.2f}")
            print(f"   🔵 YB std: {yb_std:.2f}, mean: {yb_mean:.2f}")
            
            # Color analysis
            if metrics.color_temperature < 3000:
                temp_desc = "Very Warm (Deep sunset)"
            elif metrics.color_temperature < 4000:
                temp_desc = "Warm (Golden hour)"
            elif metrics.color_temperature < 5000:
                temp_desc = "Neutral"
            else:
                temp_desc = "Cool (Daylight)"
            
            print(f"   🌡️  Temperature: {temp_desc}")
            
            # Check for potential issues
            issues = []
            if metrics.colorfulness > 120:
                issues.append("Very high colorfulness - may be noise/artifacts")
            if metrics.brightness_mean < 50:
                issues.append("Very dark - may affect color perception")
            if metrics.brightness_mean > 200:
                issues.append("Overexposed - colors may be washed out")
            if metrics.sky_saturation < 0.2:
                issues.append("Low saturation - colors may appear dull")
            
            if issues:
                print(f"   ⚠️  Potential issues:")
                for issue in issues:
                    print(f"      • {issue}")
        else:
            print("   ❌ SBS analysis failed")
    
    print(f"\n💭 INTERPRETATION:")
    print(f"The high colorfulness score (105+) suggests the algorithm is detecting:")
    print(f"• High color variation/contrast in the image")
    print(f"• Strong opponent color differences (red-green, yellow-blue)")
    print(f"• This DOESN'T necessarily mean 'beautiful sunset colors'")
    print(f"• Could be detecting: clouds, foliage, buildings, or even noise")
    print(f"\n🔧 POSSIBLE IMPROVEMENTS:")
    print(f"• Focus analysis on sky region only (currently 60% of top)")
    print(f"• Add sunset-specific color detection (orange/red/pink hues)")
    print(f"• Weight warm colors more heavily")
    print(f"• Consider cloud/atmospheric features")
    print(f"• Add human-validated training data for calibration")

if __name__ == '__main__':
    analyze_sample_images()