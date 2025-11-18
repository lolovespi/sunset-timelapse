"""
Geography Calculator
Calculates camera field of view, horizon distance, and visibility
For determining if weather events are visible from camera position
"""

import logging
import math
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

from config_manager import get_config


@dataclass
class ViewingCone:
    """Represents the camera's viewing area"""
    azimuth_center: float  # Center direction in degrees (0=N, 90=E, 180=S, 270=W)
    azimuth_min: float  # Left edge of FOV
    azimuth_max: float  # Right edge of FOV
    horizontal_fov: float  # Total horizontal field of view in degrees
    vertical_fov: float  # Total vertical field of view in degrees
    elevation_angle: float  # Camera tilt (+ = up, - = down)


class GeographyCalculator:
    """Calculate geographic visibility and field of view for camera"""

    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0

    def __init__(self):
        """Initialize geography calculator"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # Camera position
        self.latitude = self.config.get('camera.location.latitude',
                                       self.config.get('location.latitude'))
        self.longitude = self.config.get('camera.location.longitude',
                                        self.config.get('location.longitude'))
        self.elevation_m = self.config.get('camera.location.elevation_meters', 140)

        # Camera orientation
        self.azimuth = self.config.get('camera.azimuth_degrees', 270)
        self.tilt = self.config.get('camera.tilt_degrees', 0)

        # Field of view
        self.h_fov = self.config.get('camera.horizontal_fov_degrees', 87)
        self.v_fov = self.config.get('camera.vertical_fov_degrees', 47)

        self.logger.info(f"Geography calculator initialized for camera at "
                        f"{self.latitude:.4f}°N, {self.longitude:.4f}°W")
        self.logger.info(f"Camera elevation: {self.elevation_m}m, "
                        f"pointing {self.azimuth}° ± {self.h_fov/2:.1f}°")

    def calculate_horizon_distance(self,
                                   observer_elevation_m: Optional[float] = None,
                                   target_altitude_m: float = 0) -> float:
        """
        Calculate visible distance to an object at a given altitude

        Uses the geometric horizon formula accounting for Earth's curvature.
        For objects above the surface (clouds, elevated terrain), the visible
        distance is greater than the horizon.

        Args:
            observer_elevation_m: Observer height above sea level (defaults to camera elevation)
            target_altitude_m: Target object height above sea level (0 = ground level)

        Returns:
            Visible distance in kilometers

        Examples:
            - Ground level horizon (target_altitude=0): ~14-15 km
            - Cloud base at 3000m altitude: ~60-70 km
            - Storm tops at 10000m altitude: ~100-120 km
        """
        if observer_elevation_m is None:
            observer_elevation_m = self.elevation_m

        # Convert meters to kilometers for calculation
        h1_km = observer_elevation_m / 1000.0
        h2_km = target_altitude_m / 1000.0

        # Distance from observer to horizon
        d1 = math.sqrt(2 * self.EARTH_RADIUS_KM * h1_km + h1_km * h1_km)

        # Distance from target to horizon
        d2 = math.sqrt(2 * self.EARTH_RADIUS_KM * h2_km + h2_km * h2_km)

        # Total visible distance
        total_distance = d1 + d2

        self.logger.debug(f"Horizon distance: observer at {observer_elevation_m}m, "
                         f"target at {target_altitude_m}m → {total_distance:.2f} km")

        return total_distance

    def get_viewing_cone(self) -> ViewingCone:
        """
        Get camera's viewing cone (field of view parameters)

        Returns:
            ViewingCone object with azimuth ranges and FOV dimensions
        """
        # Calculate azimuth range (handle wraparound at 0°/360°)
        azimuth_min = (self.azimuth - self.h_fov / 2.0) % 360
        azimuth_max = (self.azimuth + self.h_fov / 2.0) % 360

        return ViewingCone(
            azimuth_center=self.azimuth,
            azimuth_min=azimuth_min,
            azimuth_max=azimuth_max,
            horizontal_fov=self.h_fov,
            vertical_fov=self.v_fov,
            elevation_angle=self.tilt
        )

    def is_bearing_in_fov(self, bearing: float) -> bool:
        """
        Check if a compass bearing falls within camera's field of view

        Args:
            bearing: Compass bearing in degrees (0=N, 90=E, 180=S, 270=W)

        Returns:
            True if bearing is within camera's horizontal FOV
        """
        cone = self.get_viewing_cone()

        # Handle wraparound at 0°/360°
        if cone.azimuth_min > cone.azimuth_max:
            # FOV crosses 0° (e.g., 350° to 10°)
            in_fov = bearing >= cone.azimuth_min or bearing <= cone.azimuth_max
        else:
            # Normal case
            in_fov = cone.azimuth_min <= bearing <= cone.azimuth_max

        self.logger.debug(f"Bearing {bearing:.1f}° in FOV [{cone.azimuth_min:.1f}° - "
                         f"{cone.azimuth_max:.1f}°]: {in_fov}")

        return in_fov

    def is_point_visible(self,
                        distance_km: float,
                        bearing: float,
                        altitude_m: float = 3000) -> Tuple[bool, str]:
        """
        Determine if a point is visible from the camera

        Checks both distance (horizon limit) and bearing (FOV limit)

        Args:
            distance_km: Distance to point in kilometers
            bearing: Compass bearing to point in degrees
            altitude_m: Altitude of point above sea level (default: typical cloud base)

        Returns:
            Tuple of (visible: bool, reason: str)

        Examples:
            >>> calc.is_point_visible(10, 270, 3000)
            (True, "Visible")

            >>> calc.is_point_visible(100, 270, 0)
            (False, "Beyond horizon (max 14.6 km for ground level)")

            >>> calc.is_point_visible(10, 90, 3000)
            (False, "Outside camera FOV (camera points 270°)")
        """
        # Check distance against horizon
        max_visible_distance = self.calculate_horizon_distance(
            target_altitude_m=altitude_m
        )

        if distance_km > max_visible_distance:
            return False, (f"Beyond horizon (max {max_visible_distance:.1f} km "
                          f"for altitude {altitude_m}m)")

        # Check bearing against FOV
        if not self.is_bearing_in_fov(bearing):
            cone = self.get_viewing_cone()
            return False, (f"Outside camera FOV (camera points {self.azimuth}°, "
                          f"FOV: {cone.azimuth_min:.1f}° - {cone.azimuth_max:.1f}°)")

        return True, "Visible"

    def calculate_bearing(self,
                         target_lat: float,
                         target_lon: float) -> float:
        """
        Calculate compass bearing from camera to target coordinates

        Args:
            target_lat: Target latitude in degrees
            target_lon: Target longitude in degrees

        Returns:
            Bearing in degrees (0=N, 90=E, 180=S, 270=W)
        """
        # Convert to radians
        lat1 = math.radians(self.latitude)
        lat2 = math.radians(target_lat)
        lon_diff = math.radians(target_lon - self.longitude)

        # Calculate bearing using spherical trigonometry
        x = math.sin(lon_diff) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2) -
             math.sin(lat1) * math.cos(lat2) * math.cos(lon_diff))

        bearing_rad = math.atan2(x, y)
        bearing_deg = (math.degrees(bearing_rad) + 360) % 360

        return bearing_deg

    def calculate_distance(self,
                          target_lat: float,
                          target_lon: float) -> float:
        """
        Calculate great circle distance from camera to target coordinates

        Uses the Haversine formula for spherical distance calculation

        Args:
            target_lat: Target latitude in degrees
            target_lon: Target longitude in degrees

        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1 = math.radians(self.latitude)
        lat2 = math.radians(target_lat)
        lon1 = math.radians(self.longitude)
        lon2 = math.radians(target_lon)

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))

        distance_km = self.EARTH_RADIUS_KM * c

        return distance_km

    def get_visibility_stats(self) -> Dict:
        """
        Get comprehensive visibility statistics for camera position

        Returns:
            Dictionary with visibility ranges and viewing area information
        """
        cone = self.get_viewing_cone()

        # Calculate horizon distances for different altitudes
        ground_horizon = self.calculate_horizon_distance(target_altitude_m=0)
        cloud_horizon = self.calculate_horizon_distance(target_altitude_m=3000)
        storm_top_horizon = self.calculate_horizon_distance(target_altitude_m=10000)

        # Calculate viewing area (approximate)
        # Area of circular sector: A = r² × θ / 2 (where θ is in radians)
        theta_rad = math.radians(self.h_fov)
        ground_area_km2 = (ground_horizon ** 2) * theta_rad / 2

        return {
            'camera_position': {
                'latitude': self.latitude,
                'longitude': self.longitude,
                'elevation_m': self.elevation_m
            },
            'viewing_cone': {
                'azimuth_center': cone.azimuth_center,
                'azimuth_range': f"{cone.azimuth_min:.1f}° - {cone.azimuth_max:.1f}°",
                'horizontal_fov': cone.horizontal_fov,
                'vertical_fov': cone.vertical_fov,
                'elevation_angle': cone.elevation_angle
            },
            'horizon_distances_km': {
                'ground_level': round(ground_horizon, 2),
                'cloud_base_3000m': round(cloud_horizon, 2),
                'storm_tops_10000m': round(storm_top_horizon, 2)
            },
            'viewing_area': {
                'ground_area_km2': round(ground_area_km2, 2),
                'description': f"~{round(ground_area_km2)} km² ground coverage"
            }
        }

    def validate_configuration(self) -> bool:
        """
        Validate camera geographic configuration

        Returns:
            True if configuration is valid, False otherwise
        """
        errors = []

        # Check required fields
        if self.latitude is None or self.longitude is None:
            errors.append("Camera latitude/longitude not configured")

        # Validate ranges
        if not (-90 <= self.latitude <= 90):
            errors.append(f"Invalid latitude: {self.latitude}")

        if not (-180 <= self.longitude <= 180):
            errors.append(f"Invalid longitude: {self.longitude}")

        if not (0 <= self.azimuth < 360):
            errors.append(f"Invalid azimuth: {self.azimuth} (must be 0-359)")

        if not (1 <= self.h_fov <= 180):
            errors.append(f"Invalid horizontal FOV: {self.h_fov} (must be 1-180)")

        if not (1 <= self.v_fov <= 180):
            errors.append(f"Invalid vertical FOV: {self.v_fov} (must be 1-180)")

        if errors:
            for error in errors:
                self.logger.error(error)
            return False

        self.logger.info("Camera geographic configuration validated successfully")
        return True


# Convenience function for testing
def main():
    """Test geography calculator"""
    import json

    calc = GeographyCalculator()

    print("\n=== Camera Geography Calculator Test ===\n")

    # Validate configuration
    if not calc.validate_configuration():
        print("ERROR: Configuration validation failed")
        return

    # Get and display statistics
    stats = calc.get_visibility_stats()
    print("Camera Visibility Statistics:")
    print(json.dumps(stats, indent=2))

    print("\n=== Visibility Tests ===\n")

    # Test various points
    test_cases = [
        (10, 270, 3000, "Storm 10km directly west (in FOV)"),
        (20, 270, 0, "Ground level 20km west (likely beyond horizon)"),
        (15, 90, 3000, "Storm 15km east (outside FOV if camera points west)"),
        (5, 280, 3000, "Storm 5km slightly north of west"),
    ]

    for distance, bearing, altitude, description in test_cases:
        visible, reason = calc.is_point_visible(distance, bearing, altitude)
        status = "✓ VISIBLE" if visible else "✗ NOT VISIBLE"
        print(f"{status}: {description}")
        print(f"  → {reason}\n")


if __name__ == "__main__":
    main()
