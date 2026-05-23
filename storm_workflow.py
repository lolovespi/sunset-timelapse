"""
Storm Workflow

Storm-specific capture and upload pipeline.
Mirrors the sunset workflow inside sunset_scheduler.complete_daily_workflow,
but with storm-specific scoring, captions, and orchestration.

See docs/superpowers/specs/2026-05-22-storm-capture-design.md for design.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config_manager import get_config


def sis_score_to_grade(score: float) -> str:
    """Map numeric Storm Intensity Score (0-100) to letter grade."""
    if score >= 80:
        return 'A'
    if score >= 60:
        return 'B'
    if score >= 40:
        return 'C'
    if score >= 20:
        return 'D'
    return 'F'


def compute_storm_intensity_score(
    strikes: list,
    observations: list,
    weights: Optional[Dict] = None,
) -> Dict:
    """
    Compute Storm Intensity Score from Tempest sensor data only.

    No video analysis. Pure function over inputs.

    Args:
        strikes: list of LightningStrike from tempest_monitor
        observations: list of WeatherObservation from tempest_monitor
        weights: optional dict overriding default weights (from config)

    Returns:
        {
            'score': float (0-100),
            'grade': str ('A'-'F'),
            'components': {
                'lightning': float,
                'distance': float,
                'rain': float,
                'wind': float,
                'pressure': float,
            },
            'metrics': {
                'lightning_count': int,
                'lightning_avg_distance_km': float | None,
                'rain_max_mm_hr': float,
                'wind_gust_max_mph': float,
                'pressure_drop_hpa': float,
            },
        }
    """
    if weights is None:
        config = get_config()
        weights = {
            'lightning': config.get('storm_analysis.scoring.lightning_weight', 5.0),
            'distance': config.get('storm_analysis.scoring.distance_weight', 20.0),
            'rain': config.get('storm_analysis.scoring.rain_weight', 2.0),
            'wind': config.get('storm_analysis.scoring.wind_weight', 0.5),
            'pressure': config.get('storm_analysis.scoring.pressure_weight', 10.0),
            'max_score': config.get('storm_analysis.scoring.max_score', 100),
        }

    # Lightning metrics
    lightning_count = len(strikes)
    avg_distance = (
        sum(s.distance_km for s in strikes) / lightning_count
        if lightning_count > 0 else None
    )

    # Observation-derived metrics
    rain_max = max((o.rain_rate for o in observations), default=0.0)
    wind_gust_max = max((o.wind_gust for o in observations), default=0.0)

    pressure_drop = 0.0
    if len(observations) >= 2:
        # Drop = max - min over the observation window
        pressures = [o.pressure for o in observations]
        pressure_drop = max(pressures) - min(pressures)

    # Component scores
    lightning_pts = lightning_count * weights['lightning']
    if avg_distance is not None:
        distance_pts = (40.0 / max(avg_distance, 5.0)) * weights['distance']
    else:
        distance_pts = 0.0
    rain_pts = rain_max * weights['rain']
    wind_pts = wind_gust_max * weights['wind']
    pressure_pts = abs(pressure_drop) * weights['pressure']

    total = lightning_pts + distance_pts + rain_pts + wind_pts + pressure_pts
    score = min(total, float(weights['max_score']))

    return {
        'score': round(score, 1),
        'grade': sis_score_to_grade(score),
        'components': {
            'lightning': round(lightning_pts, 2),
            'distance': round(distance_pts, 2),
            'rain': round(rain_pts, 2),
            'wind': round(wind_pts, 2),
            'pressure': round(pressure_pts, 2),
        },
        'metrics': {
            'lightning_count': lightning_count,
            'lightning_avg_distance_km': avg_distance,
            'rain_max_mm_hr': round(rain_max, 2),
            'wind_gust_max_mph': round(wind_gust_max, 2),
            'pressure_drop_hpa': round(pressure_drop, 2),
        },
    }
