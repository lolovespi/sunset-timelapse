"""
SBS Reporter - Generate reports and integrate SBS data with existing systems
"""

import logging
import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv

from config_manager import get_config
from sunset_brilliance_score import SunsetBrillianceScore, ChunkMetrics


class SBSReporter:
    """Generate SBS reports and integrate with video processing and notifications"""
    
    def __init__(self):
        """Initialize SBS reporter"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.sbs_analyzer = SunsetBrillianceScore()
        
    def generate_daily_report(self, target_date: date) -> Optional[Dict]:
        """
        Generate comprehensive daily SBS report
        
        Args:
            target_date: Date to generate report for
            
        Returns:
            Dictionary with daily SBS analysis or None if no data
        """
        try:
            date_str = target_date.strftime('%Y-%m-%d')
            
            # Load all chunk analyses for the date
            chunk_metrics = self.sbs_analyzer.load_daily_analysis(date_str)
            
            if not chunk_metrics:
                self.logger.warning(f"No SBS data found for {date_str}")
                return None
            
            # Calculate daily summary
            daily_summary = self.sbs_analyzer.calculate_daily_summary(chunk_metrics)
            
            # Generate detailed report
            report = {
                'date': date_str,
                'analysis_timestamp': datetime.now().isoformat(),
                'summary': daily_summary,
                'chunk_details': [
                    {
                        'chunk_number': chunk.chunk_number,
                        'time_range': f"{chunk.start_offset/60:.0f}-{chunk.end_offset/60:.0f}min",
                        'brilliance_score': chunk.brilliance_score,
                        'peak_frames': len(chunk.peak_frames),
                        'color_variation': chunk.color_variation_range,
                        'temporal_smoothness': chunk.temporal_smoothness,
                        'golden_hour_bonus': chunk.golden_hour_bonus,
                        'frame_count': chunk.frame_count
                    } for chunk in chunk_metrics
                ],
                'recommendations': self._generate_recommendations(daily_summary, chunk_metrics)
            }
            
            # Save report
            self._save_daily_report(report, target_date)
            
            self.logger.info(f"Generated SBS report for {date_str}: "
                           f"Score {daily_summary['daily_brilliance_score']:.1f} "
                           f"(Grade {daily_summary['quality_grade']})")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate SBS report for {target_date}: {e}")
            return None
    
    def _generate_recommendations(self, daily_summary: Dict, chunk_metrics: List[ChunkMetrics]) -> List[str]:
        """Generate actionable recommendations based on SBS analysis"""
        recommendations = []
        
        score = daily_summary['daily_brilliance_score']
        grade = daily_summary['quality_grade']
        
        # Score-based recommendations
        if score >= 80:
            recommendations.append("Excellent sunset quality! Consider sharing highlights on social media.")
        elif score >= 70:
            recommendations.append("Good sunset quality. Video should perform well on YouTube.")
        elif score >= 60:
            recommendations.append("Average sunset. Consider enhancing video with music or effects.")
        elif score >= 50:
            recommendations.append("Below average sunset. Check camera positioning and settings.")
        else:
            recommendations.append("Poor sunset conditions. Consider weather-based scheduling adjustments.")
        
        # Chunk-specific recommendations
        if chunk_metrics:
            peak_chunk = max(chunk_metrics, key=lambda x: x.brilliance_score)
            if peak_chunk.brilliance_score > score + 20:
                start_min = int(peak_chunk.start_offset / 60)
                end_min = int(peak_chunk.end_offset / 60)
                recommendations.append(f"Best footage in minutes {start_min}-{end_min}. Consider creating highlight reel.")
            
            # Golden hour analysis
            golden_chunks = [c for c in chunk_metrics if c.golden_hour_bonus > 0.1]
            if golden_chunks:
                avg_golden_score = sum(c.brilliance_score for c in golden_chunks) / len(golden_chunks)
                if avg_golden_score > score + 10:
                    recommendations.append("Golden hour timing is optimal. Maintain current capture schedule.")
            else:
                recommendations.append("No significant golden hour enhancement. Consider adjusting capture timing.")
        
        # Performance recommendations
        avg_processing_time = daily_summary.get('avg_processing_time_ms', 0)
        if avg_processing_time > 10:
            recommendations.append("SBS processing is slow. Consider reducing analysis_resize_height in config.")
        elif avg_processing_time < 3:
            recommendations.append("SBS processing is fast. Could increase analysis_resize_height for better accuracy.")
        
        return recommendations
    
    def _save_daily_report(self, report: Dict, target_date: date):
        """Save daily report to JSON and CSV formats"""
        paths = self.config.get_storage_paths()
        reports_dir = paths['temp'] / 'sbs_reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        date_str = target_date.strftime('%Y-%m-%d')
        
        # Save JSON report (detailed)
        json_path = reports_dir / f"sbs_report_{date_str}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save CSV summary (for spreadsheet analysis)
        csv_path = reports_dir / "sbs_daily_summary.csv"
        csv_exists = csv_path.exists()
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if not csv_exists:
                # Write header
                writer.writerow([
                    'Date', 'Daily_Score', 'Quality_Grade', 'Peak_Chunk', 'Peak_Score',
                    'Total_Frames', 'Chunks_Analyzed', 'Avg_Processing_Time_ms'
                ])
            
            # Write data row
            summary = report['summary']
            writer.writerow([
                date_str,
                f"{summary['daily_brilliance_score']:.1f}",
                summary['quality_grade'],
                summary['peak_chunk'],
                f"{summary['peak_chunk_score']:.1f}",
                summary['total_frames'],
                summary['chunks_analyzed'],
                f"{summary['avg_processing_time_ms']:.2f}"
            ])
    
    def get_sbs_summary_for_email(self, target_date: date) -> str:
        """Generate SBS summary text for email notifications"""
        try:
            report = self.generate_daily_report(target_date)
            if not report:
                return "SBS analysis not available"
            
            summary = report['summary']
            recommendations = report['recommendations'][:2]  # Top 2 recommendations
            
            sbs_text = f"""
Sunset Brilliance Score Analysis:
• Daily Score: {summary['daily_brilliance_score']:.1f}/100 (Grade {summary['quality_grade']})
• Peak Performance: Chunk {summary['peak_chunk']} ({summary['peak_chunk_score']:.1f} points)
• Frames Analyzed: {summary['total_frames']} across {summary['chunks_analyzed']} chunks
• Processing Performance: {summary['avg_processing_time_ms']:.1f}ms average

Top Recommendations:
{chr(10).join(f"• {rec}" for rec in recommendations)}
            """
            
            return sbs_text.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to generate SBS email summary: {e}")
            return f"SBS analysis error: {str(e)}"
    
    def get_video_title_enhancement(self, target_date: date) -> str:
        """Generate SBS-based video title enhancement"""
        try:
            date_str = target_date.strftime('%Y-%m-%d')
            chunk_metrics = self.sbs_analyzer.load_daily_analysis(date_str)
            
            if not chunk_metrics:
                return ""
            
            daily_summary = self.sbs_analyzer.calculate_daily_summary(chunk_metrics)
            score = daily_summary['daily_brilliance_score']
            
            # Add quality indicators to video title
            if score >= 90:
                return " ✨ SPECTACULAR"
            elif score >= 80:
                return " 🌅 BRILLIANT" 
            elif score >= 70:
                return " 🔥 VIBRANT"
            elif score >= 60:
                return " 🎨 COLORFUL"
            else:
                return ""
                
        except Exception as e:
            self.logger.error(f"Failed to generate video title enhancement: {e}")
            return ""
    
    def get_video_description_enhancement(self, target_date: date) -> str:
        """Generate SBS-based video description enhancement"""
        try:
            date_str = target_date.strftime('%Y-%m-%d')
            chunk_metrics = self.sbs_analyzer.load_daily_analysis(date_str)
            
            if not chunk_metrics:
                return ""
            
            daily_summary = self.sbs_analyzer.calculate_daily_summary(chunk_metrics)
            score = daily_summary['daily_brilliance_score']
            grade = daily_summary['quality_grade']
            
            # Find best moments
            peak_chunk = daily_summary['peak_chunk']
            peak_time_start = int(peak_chunk * 15)  # 15 minutes per chunk
            peak_time_end = peak_time_start + 15
            
            enhancement = f"""

🌅 Sunset Analysis:
Quality Grade: {grade}
Peak Moments: Minutes {peak_time_start:02d}:00 - {peak_time_end:02d}:00
Captured with AI-powered sunset analysis

#SunsetAnalysis #QualityGrade{grade}"""
            
            return enhancement
            
        except Exception as e:
            self.logger.error(f"Failed to generate video description enhancement: {e}")
            return ""
    
    def cleanup_old_sbs_data(self, retention_days: int = 30):
        """Clean up old SBS analysis data"""
        try:
            paths = self.config.get_storage_paths()
            sbs_base_dir = paths['temp'] / 'sbs'
            reports_dir = paths['temp'] / 'sbs_reports'
            
            from datetime import timedelta
            cutoff_date = date.today() - timedelta(days=retention_days)
            cutoff_str = cutoff_date.strftime('%Y-%m-%d')
            
            cleaned_count = 0
            
            # Clean chunk data directories
            if sbs_base_dir.exists():
                for date_dir in sbs_base_dir.iterdir():
                    if date_dir.is_dir() and date_dir.name < cutoff_str:
                        import shutil
                        shutil.rmtree(date_dir)
                        cleaned_count += 1
                        self.logger.debug(f"Removed old SBS data: {date_dir}")
            
            # Clean old report files
            if reports_dir.exists():
                for report_file in reports_dir.glob(f"sbs_report_*.json"):
                    # Extract date from filename
                    try:
                        file_date_str = report_file.stem.replace('sbs_report_', '')
                        if file_date_str < cutoff_str:
                            report_file.unlink()
                            cleaned_count += 1
                            self.logger.debug(f"Removed old SBS report: {report_file}")
                    except Exception:
                        continue
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old SBS data items (older than {retention_days} days)")
                
        except Exception as e:
            self.logger.error(f"SBS cleanup failed: {e}")
    
    def export_historical_analysis(self, start_date: date, end_date: date, 
                                 output_file: Optional[Path] = None) -> Path:
        """
        Export historical SBS analysis to CSV for external analysis
        
        Args:
            start_date: Start date for export
            end_date: End date for export  
            output_file: Optional output file path
            
        Returns:
            Path to exported CSV file
        """
        if output_file is None:
            paths = self.config.get_storage_paths()
            output_file = paths['temp'] / f"sbs_historical_{start_date}_{end_date}.csv"
        
        try:
            current_date = start_date
            all_data = []
            
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                chunk_metrics = self.sbs_analyzer.load_daily_analysis(date_str)
                
                if chunk_metrics:
                    daily_summary = self.sbs_analyzer.calculate_daily_summary(chunk_metrics)
                    
                    # Add daily summary row
                    all_data.append({
                        'Date': date_str,
                        'Type': 'Daily',
                        'Score': daily_summary['daily_brilliance_score'],
                        'Grade': daily_summary['quality_grade'],
                        'Frames': daily_summary['total_frames'],
                        'Peak_Chunk': daily_summary['peak_chunk'],
                        'Avg_Processing_ms': daily_summary['avg_processing_time_ms']
                    })
                    
                    # Add chunk detail rows
                    for chunk in chunk_metrics:
                        all_data.append({
                            'Date': date_str,
                            'Type': f'Chunk_{chunk.chunk_number:02d}',
                            'Score': chunk.brilliance_score,
                            'Grade': '',
                            'Frames': chunk.frame_count,
                            'Peak_Chunk': len(chunk.peak_frames),
                            'Avg_Processing_ms': chunk.avg_processing_time_ms,
                            'Color_Variation': chunk.color_variation_range,
                            'Temporal_Smoothness': chunk.temporal_smoothness,
                            'Golden_Hour_Bonus': chunk.golden_hour_bonus
                        })
                
                current_date += timedelta(days=1)
            
            # Write to CSV
            if all_data:
                with open(output_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
                    writer.writeheader()
                    writer.writerows(all_data)
                
                self.logger.info(f"Exported {len(all_data)} SBS records to {output_file}")
            else:
                self.logger.warning(f"No SBS data found for period {start_date} to {end_date}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Historical SBS export failed: {e}")
            return output_file