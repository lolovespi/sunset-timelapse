"""
Meteor Shower Database and Calculator

Uses solar longitude (Earth's position in orbit) to calculate meteor shower
activity periods for any year. Solar longitude values are constant - only
calendar dates change based on Earth's orbital position.

Solar longitude (λ☉) is measured in degrees from the vernal equinox (0° = ~March 20).
"""

import math
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Meteor shower database with solar longitude values (constant each year)
# Data from IMO (International Meteor Organization) and AMS (American Meteor Society)
METEOR_SHOWERS = {
    "quadrantids": {
        "name": "Quadrantids",
        "code": "QUA",
        "peak_solar_longitude": 283.16,  # ~Jan 3-4
        "start_solar_longitude": 276.0,   # ~Dec 28
        "end_solar_longitude": 293.0,     # ~Jan 12
        "velocity_kms": 40.4,
        "zhr": 120,
        "parent_body": "2003 EH1 (asteroid)",
        "description": "Brief but intense peak, best after midnight"
    },
    "lyrids": {
        "name": "Lyrids",
        "code": "LYR",
        "peak_solar_longitude": 32.32,    # ~Apr 22
        "start_solar_longitude": 24.0,     # ~Apr 14
        "end_solar_longitude": 40.0,       # ~Apr 30
        "velocity_kms": 49.0,
        "zhr": 18,
        "parent_body": "C/1861 G1 (Thatcher)",
        "description": "Medium-speed meteors, occasional fireballs"
    },
    "eta_aquariids": {
        "name": "Eta Aquariids",
        "code": "ETA",
        "peak_solar_longitude": 45.5,     # ~May 6
        "start_solar_longitude": 35.0,     # ~Apr 26
        "end_solar_longitude": 60.0,       # ~May 20
        "velocity_kms": 66.0,
        "zhr": 50,
        "parent_body": "1P/Halley",
        "description": "Fast meteors, best in Southern Hemisphere"
    },
    "delta_aquariids": {
        "name": "Southern Delta Aquariids",
        "code": "SDA",
        "peak_solar_longitude": 125.0,    # ~Jul 30
        "start_solar_longitude": 100.0,    # ~Jul 12
        "end_solar_longitude": 140.0,      # ~Aug 23
        "velocity_kms": 41.0,
        "zhr": 25,
        "parent_body": "96P/Machholz (suspected)",
        "description": "Overlaps with Perseids"
    },
    "perseids": {
        "name": "Perseids",
        "code": "PER",
        "peak_solar_longitude": 140.0,    # ~Aug 12-13
        "start_solar_longitude": 114.0,    # ~Jul 17
        "end_solar_longitude": 152.0,      # ~Aug 24
        "velocity_kms": 59.0,
        "zhr": 100,
        "parent_body": "109P/Swift-Tuttle",
        "description": "Reliable, bright meteors, many fireballs"
    },
    "orionids": {
        "name": "Orionids",
        "code": "ORI",
        "peak_solar_longitude": 208.0,    # ~Oct 21
        "start_solar_longitude": 189.0,    # ~Oct 2
        "end_solar_longitude": 225.0,      # ~Nov 7
        "velocity_kms": 66.0,
        "zhr": 20,
        "parent_body": "1P/Halley",
        "description": "Fast meteors with persistent trains"
    },
    "southern_taurids": {
        "name": "Southern Taurids",
        "code": "STA",
        "peak_solar_longitude": 223.0,    # ~Nov 5
        "start_solar_longitude": 187.0,    # ~Sep 28
        "end_solar_longitude": 245.0,      # ~Nov 25
        "velocity_kms": 27.0,
        "zhr": 5,
        "parent_body": "2P/Encke",
        "description": "Slow meteors, known for fireballs"
    },
    "northern_taurids": {
        "name": "Northern Taurids",
        "code": "NTA",
        "peak_solar_longitude": 230.0,    # ~Nov 12
        "start_solar_longitude": 207.0,    # ~Oct 20
        "end_solar_longitude": 260.0,      # ~Dec 10
        "velocity_kms": 29.0,
        "zhr": 5,
        "parent_body": "2P/Encke",
        "description": "Slow meteors, known for fireballs"
    },
    "leonids": {
        "name": "Leonids",
        "code": "LEO",
        "peak_solar_longitude": 235.27,   # ~Nov 17
        "start_solar_longitude": 224.0,    # ~Nov 6
        "end_solar_longitude": 248.0,      # ~Nov 30
        "velocity_kms": 70.0,
        "zhr": 15,
        "parent_body": "55P/Tempel-Tuttle",
        "description": "Very fast meteors, periodic storms"
    },
    "geminids": {
        "name": "Geminids",
        "code": "GEM",
        "peak_solar_longitude": 262.2,    # ~Dec 14
        "start_solar_longitude": 252.0,    # ~Dec 4
        "end_solar_longitude": 266.0,      # ~Dec 17
        "velocity_kms": 35.0,
        "zhr": 150,
        "parent_body": "3200 Phaethon (asteroid)",
        "description": "Best annual shower, multicolored meteors"
    },
    "ursids": {
        "name": "Ursids",
        "code": "URS",
        "peak_solar_longitude": 270.7,    # ~Dec 22
        "start_solar_longitude": 265.0,    # ~Dec 17
        "end_solar_longitude": 275.0,      # ~Dec 26
        "velocity_kms": 33.0,
        "zhr": 10,
        "parent_body": "8P/Tuttle",
        "description": "Often overlooked, occasional outbursts"
    }
}


def calculate_solar_longitude(dt: datetime) -> float:
    """
    Calculate solar longitude for a given datetime.

    Solar longitude is the ecliptic longitude of the Sun as seen from Earth.
    0° = vernal equinox (~March 20)

    Algorithm based on astronomical calculations, accurate to ~0.01°

    Args:
        dt: datetime object (timezone-aware or naive, treated as UTC if naive)

    Returns:
        Solar longitude in degrees (0-360)
    """
    # Convert to Julian Date
    if dt.tzinfo is not None:
        dt = dt.astimezone(ZoneInfo('UTC'))

    # Julian Date calculation
    year = dt.year
    month = dt.month
    day = dt.day + dt.hour/24 + dt.minute/1440 + dt.second/86400

    if month <= 2:
        year -= 1
        month += 12

    A = int(year / 100)
    B = 2 - A + int(A / 4)

    JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5

    # Julian centuries from J2000.0
    T = (JD - 2451545.0) / 36525.0

    # Mean longitude of the Sun (degrees)
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2

    # Mean anomaly of the Sun (degrees)
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2
    M_rad = math.radians(M)

    # Equation of center (degrees)
    C = (1.914602 - 0.004817 * T - 0.000014 * T**2) * math.sin(M_rad)
    C += (0.019993 - 0.000101 * T) * math.sin(2 * M_rad)
    C += 0.000289 * math.sin(3 * M_rad)

    # Sun's true longitude (degrees)
    sun_longitude = L0 + C

    # Normalize to 0-360
    sun_longitude = sun_longitude % 360

    return sun_longitude


def solar_longitude_to_date(solar_long: float, year: int) -> date:
    """
    Convert solar longitude to approximate calendar date for a given year.

    Args:
        solar_long: Solar longitude in degrees (0-360)
        year: Year to calculate for

    Returns:
        Approximate date when Sun reaches that solar longitude
    """
    # Start from vernal equinox (~March 20)
    # Solar longitude 0° = vernal equinox
    # Each degree ≈ 1.014 days (365.25 / 360)

    # Approximate vernal equinox for the year
    vernal_equinox = datetime(year, 3, 20, 12, 0, 0)

    # Days from vernal equinox
    days_offset = solar_long * (365.25 / 360.0)

    target_date = vernal_equinox + timedelta(days=days_offset)

    # Refine estimate by iterating
    for _ in range(5):
        actual_sl = calculate_solar_longitude(target_date)
        diff = solar_long - actual_sl
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        target_date += timedelta(days=diff * (365.25 / 360.0))

    return target_date.date()


def get_active_showers(target_date: date) -> List[Dict]:
    """
    Get all meteor showers active on a given date.

    Args:
        target_date: Date to check

    Returns:
        List of active shower dictionaries with calculated dates
    """
    # Calculate solar longitude for the target date at midnight
    dt = datetime.combine(target_date, datetime.min.time())
    dt = dt.replace(tzinfo=ZoneInfo('UTC'))
    current_sl = calculate_solar_longitude(dt)

    active = []
    year = target_date.year

    for shower_id, shower in METEOR_SHOWERS.items():
        start_sl = shower["start_solar_longitude"]
        end_sl = shower["end_solar_longitude"]

        # Handle wrap-around at 360°/0° (for Quadrantids)
        if start_sl > end_sl:
            # Shower spans year boundary
            is_active = current_sl >= start_sl or current_sl <= end_sl
        else:
            is_active = start_sl <= current_sl <= end_sl

        if is_active:
            # Calculate dates for this year
            peak_date = solar_longitude_to_date(shower["peak_solar_longitude"], year)
            start_date = solar_longitude_to_date(start_sl, year)
            end_date = solar_longitude_to_date(end_sl, year)

            # Adjust year for showers spanning year boundary
            if start_sl > end_sl:
                if current_sl >= start_sl:
                    end_date = solar_longitude_to_date(end_sl, year + 1)
                else:
                    start_date = solar_longitude_to_date(start_sl, year - 1)

            # Calculate days to/from peak
            days_to_peak = (peak_date - target_date).days

            active.append({
                "id": shower_id,
                "name": shower["name"],
                "code": shower["code"],
                "velocity_kms": shower["velocity_kms"],
                "zhr": shower["zhr"],
                "parent_body": shower["parent_body"],
                "description": shower["description"],
                "peak_date": peak_date,
                "start_date": start_date,
                "end_date": end_date,
                "days_to_peak": days_to_peak,
                "is_peak": abs(days_to_peak) <= 1,
                "solar_longitude": current_sl
            })

    # Sort by ZHR (most active first)
    active.sort(key=lambda x: x["zhr"], reverse=True)

    return active


def get_expected_velocity_range(target_date: date) -> Tuple[float, float]:
    """
    Get expected meteor velocity range based on active showers.

    Args:
        target_date: Date to check

    Returns:
        Tuple of (min_velocity_kms, max_velocity_kms) for active showers
    """
    active = get_active_showers(target_date)

    if not active:
        # Default range if no showers active
        return (20.0, 72.0)

    velocities = [s["velocity_kms"] for s in active]
    return (min(velocities), max(velocities))


def estimate_pixel_velocity(velocity_kms: float,
                           camera_fov_degrees: float = 90.0,
                           frame_width_px: int = 1920,
                           frame_rate: float = 13.3) -> Tuple[float, float]:
    """
    Estimate meteor pixel velocity from atmospheric velocity.

    This is a rough estimate - actual pixel velocity depends on:
    - Meteor altitude (typically 80-120 km)
    - Meteor angle to camera
    - Distance from camera

    Args:
        velocity_kms: Atmospheric velocity in km/s
        camera_fov_degrees: Camera horizontal field of view
        frame_width_px: Frame width in pixels
        frame_rate: Camera frame rate (fps)

    Returns:
        Tuple of (min_px_per_frame, max_px_per_frame) estimated range
    """
    # Typical meteor altitude range
    altitude_min_km = 80
    altitude_max_km = 120

    # Degrees per pixel
    deg_per_px = camera_fov_degrees / frame_width_px

    # At altitude h, angular velocity = linear_velocity / h (radians/s)
    # Convert to degrees/s, then to degrees/frame, then to pixels/frame

    results = []
    for altitude_km in [altitude_min_km, altitude_max_km]:
        # Angular velocity in degrees/second
        angular_vel_deg_s = math.degrees(velocity_kms / altitude_km)

        # Degrees per frame
        deg_per_frame = angular_vel_deg_s / frame_rate

        # Pixels per frame
        px_per_frame = deg_per_frame / deg_per_px

        results.append(px_per_frame)

    # Return range (higher altitude = slower apparent motion)
    return (min(results), max(results))


def format_active_showers(target_date: date) -> str:
    """
    Format active showers for display.

    Args:
        target_date: Date to check

    Returns:
        Formatted string describing active showers
    """
    active = get_active_showers(target_date)

    if not active:
        return f"No major meteor showers active on {target_date}"

    lines = [f"Active meteor showers on {target_date}:"]
    lines.append("-" * 60)

    for shower in active:
        peak_str = "** PEAK **" if shower["is_peak"] else f"{shower['days_to_peak']:+d} days from peak"
        lines.append(f"\n{shower['name']} ({shower['code']})")
        lines.append(f"  Velocity: {shower['velocity_kms']} km/s")
        lines.append(f"  ZHR: {shower['zhr']}")
        lines.append(f"  Peak: {shower['peak_date']} ({peak_str})")
        lines.append(f"  Active: {shower['start_date']} to {shower['end_date']}")
        lines.append(f"  Parent: {shower['parent_body']}")

        # Estimate pixel velocity
        px_min, px_max = estimate_pixel_velocity(shower['velocity_kms'])
        lines.append(f"  Est. pixel velocity: {px_min:.1f} - {px_max:.1f} px/frame")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test with today and Nov 17, 2025 (Leonid peak)
    from datetime import date

    print("=" * 60)
    print(format_active_showers(date.today()))

    print("\n" + "=" * 60)
    test_date = date(2025, 11, 17)
    print(format_active_showers(test_date))

    print("\n" + "=" * 60)
    print(f"\nSolar longitude on {test_date}:")
    dt = datetime.combine(test_date, datetime.min.time())
    sl = calculate_solar_longitude(dt)
    print(f"  λ☉ = {sl:.2f}°")
