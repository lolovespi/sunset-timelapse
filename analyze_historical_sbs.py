#!/usr/bin/env python3
"""
Historical SBS Analysis Script
Analyze existing sunset images and generate SBS reports
"""

import logging
import cv2
from datetime import datetime, timedelta
from pathlib import Path
import sys
import re
from typing import List, Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import get_config
from sunset_brilliance_score import SunsetBrillianceScore, FrameMetrics
from sbs_reporter import SBSReporter


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_image_timestamp(filename: str) -> Optional[datetime]:
    """Parse timestamp from image filename"""
    # Pattern: frame_20250901_181130.jpg -> 2025-09-01 18:11:30
    match = re.search(r'frame_(\d{8})_(\d{6})\.jpg', filename)
    if match:
        date_str, time_str = match.groups()
        # Parse date: YYYYMMDD
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        # Parse time: HHMMSS
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        
        return datetime(year, month, day, hour, minute, second)
    return None


def analyze_historical_images(image_dir: Path, target_date: str = "2025-09-01") -> bool:
    """
    Analyze historical sunset images and generate SBS report
    
    Args:
        image_dir: Directory containing the historical images
        target_date: Date string for organizing the analysis
    
    Returns:
        True if analysis completed successfully
    """
    logger = logging.getLogger(__name__)
    
    # Initialize SBS components
    sbs_analyzer = SunsetBrillianceScore()
    sbs_reporter = SBSReporter()
    
    # Find all images in the directory
    image_files = list(image_dir.glob("*.jpg"))
    image_files.sort()  # Sort by filename (chronological)
    
    if not image_files:
        logger.error(f"No JPG images found in {image_dir}")
        return False
    
    logger.info(f"Found {len(image_files)} images to analyze")
    
    # Parse timestamps and organize into chunks
    frame_data = []
    base_time = None
    
    for i, image_path in enumerate(image_files):
        # Parse timestamp from filename
        timestamp = parse_image_timestamp(image_path.name)
        if timestamp is None:
            logger.warning(f"Could not parse timestamp from {image_path.name}, using sequence number")
            if base_time is None:
                base_time = datetime(2025, 9, 1, 18, 11, 30)  # Estimated start time
            timestamp = base_time + timedelta(seconds=i * 5)  # Assume 5-second intervals
        else:
            if base_time is None:
                base_time = timestamp
        
        # Calculate offset from start
        offset_seconds = (timestamp - base_time).total_seconds()
        
        frame_data.append({
            'path': image_path,
            'timestamp': timestamp,
            'offset': offset_seconds,
            'frame_number': i
        })
    
    logger.info(f"Analyzing images from {base_time} to {frame_data[-1]['timestamp']}")
    logger.info(f"Total duration: {frame_data[-1]['offset']/3600:.1f} hours")
    
    # Analyze frames and organize into 15-minute chunks
    chunk_duration = 15 * 60  # 15 minutes in seconds
    current_chunk_metrics = []
    chunk_number = 0
    all_frame_metrics = []
    
    for frame_info in frame_data:
        try:
            # Load and analyze image
            frame = cv2.imread(str(frame_info['path']))
            if frame is None:
                logger.warning(f"Could not load {frame_info['path']}")
                continue
            
            # Perform SBS analysis
            frame_metrics = sbs_analyzer.analyze_frame(
                frame, 
                frame_info['frame_number'], 
                frame_info['offset']
            )
            
            if frame_metrics:
                current_chunk_metrics.append(frame_metrics)
                all_frame_metrics.append(frame_metrics)
                
                # Log progress every 100 frames
                if len(all_frame_metrics) % 100 == 0:
                    logger.info(f"Analyzed {len(all_frame_metrics)}/{len(frame_data)} frames "
                              f"({len(all_frame_metrics)/len(frame_data)*100:.1f}%)")
            
            # Check if we need to complete a chunk (every 15 minutes)
            if frame_info['offset'] > 0 and frame_info['offset'] % chunk_duration == 0:
                if current_chunk_metrics:
                    # Calculate sunset offset for golden hour bonus
                    sunset_time = datetime(2025, 9, 1, 19, 15)  # Approximate sunset time for Sept 1
                    chunk_time = base_time + timedelta(seconds=frame_info['offset'] - (chunk_duration/2))
                    sunset_offset_minutes = (chunk_time - sunset_time).total_seconds() / 60
                    
                    # Analyze chunk
                    chunk_metrics = sbs_analyzer.analyze_chunk(
                        current_chunk_metrics, 
                        chunk_number, 
                        sunset_offset_minutes
                    )
                    
                    # Save chunk analysis
                    sbs_analyzer.save_chunk_analysis(chunk_metrics, target_date)
                    
                    logger.info(f"Completed chunk {chunk_number}: Score {chunk_metrics.brilliance_score:.1f}, "
                              f"{len(current_chunk_metrics)} frames")
                    
                    # Reset for next chunk
                    current_chunk_metrics = []
                    chunk_number += 1
                    
        except Exception as e:
            logger.error(f"Error analyzing {frame_info['path']}: {e}")
            continue
    
    # Process final chunk if there are remaining metrics
    if current_chunk_metrics:
        # Calculate sunset offset
        sunset_time = datetime(2025, 9, 1, 19, 15)
        chunk_time = base_time + timedelta(seconds=frame_data[-1]['offset'])
        sunset_offset_minutes = (chunk_time - sunset_time).total_seconds() / 60
        
        chunk_metrics = sbs_analyzer.analyze_chunk(
            current_chunk_metrics, 
            chunk_number, 
            sunset_offset_minutes
        )
        
        sbs_analyzer.save_chunk_analysis(chunk_metrics, target_date)
        
        logger.info(f"Completed final chunk {chunk_number}: Score {chunk_metrics.brilliance_score:.1f}, "
                  f"{len(current_chunk_metrics)} frames")
    
    # Generate comprehensive report
    logger.info("Generating SBS report...")
    target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
    report = sbs_reporter.generate_daily_report(target_date_obj)
    
    if report:
        print(f"\n{'='*50}")
        print(f"SBS ANALYSIS REPORT FOR {target_date}")
        print(f"{'='*50}")
        
        summary = report['summary']
        print(f"📊 DAILY SUMMARY:")
        print(f"   Brilliance Score: {summary['daily_brilliance_score']:.1f}/100")
        print(f"   Quality Grade: {summary['quality_grade']}")
        print(f"   Peak Chunk: #{summary['peak_chunk']} (Score: {summary['peak_chunk_score']:.1f})")
        print(f"   Total Frames: {summary['total_frames']:,}")
        print(f"   Processing Speed: {summary['avg_processing_time_ms']:.1f}ms average")
        
        print(f"\n🕐 CHUNK BREAKDOWN ({len(report['chunk_details'])} chunks):")
        for chunk in report['chunk_details']:
            time_start = int(chunk['time_range'].split('-')[0])
            time_end = int(chunk['time_range'].split('-')[1].replace('min', ''))
            print(f"   Chunk {chunk['chunk_number']:2d} ({time_start:3d}-{time_end:3d}min): "
                 f"{chunk['brilliance_score']:5.1f} pts | "
                 f"{chunk['peak_frames']:2d} peaks | "
                 f"smooth {chunk['temporal_smoothness']:0.3f}")
        
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"   • {rec}")
        
        # Performance summary
        perf_stats = sbs_analyzer.get_performance_stats()
        if perf_stats.get('status') != 'no_data':
            print(f"\n⚡ PERFORMANCE STATS:")
            print(f"   Processing: {perf_stats['avg_processing_time_ms']:.1f}ms avg")
            print(f"   Total Time: {perf_stats['total_processing_time_s']:.1f}s")
            print(f"   Status: {perf_stats['pi_performance_status'].upper()}")
        
        print(f"\n{'='*50}")
        
        logger.info("✅ Historical SBS analysis completed successfully!")
        return True
    else:
        logger.error("❌ Failed to generate SBS report")
        return False


def main():
    """Main entry point"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Path to historical images
    image_dir = Path("/Users/lolovespi/Documents/GitHub/sunset-timelapse/data/images/2025-09-01/historical")
    
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return 1
    
    logger.info(f"Starting historical SBS analysis on {image_dir}")
    
    success = analyze_historical_images(image_dir, "2025-09-01")
    
    if success:
        print("\n🎉 Analysis complete! You can now use:")
        print("   python main.py sbs --report --date 2025-09-01")
        print("   python main.py sbs --date 2025-09-01")
        return 0
    else:
        logger.error("Analysis failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())