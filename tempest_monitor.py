"""
Tempest Weather Station Monitor
Listens for real-time weather data from Tempest station via UDP
Detects storm conditions and triggers timelapse captures
"""

import logging
import socket
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable
from collections import deque
from dataclasses import dataclass, field

from config_manager import get_config


@dataclass
class WeatherObservation:
    """Single weather observation from Tempest station"""
    timestamp: datetime
    temperature: float  # °F
    humidity: float  # %
    pressure: float  # hPa (mb)
    wind_speed: float  # mph
    wind_gust: float  # mph
    wind_direction: int  # degrees
    rain_rate: float  # mm/hr
    rain_accumulation: float  # mm
    solar_radiation: float  # W/m²
    uv_index: float
    battery_voltage: float

    # Optional fields
    feels_like: Optional[float] = None
    dew_point: Optional[float] = None


@dataclass
class LightningStrike:
    """Lightning strike event from Tempest"""
    timestamp: datetime
    distance_km: float
    energy: int
    bearing: Optional[float] = None  # Not always provided by Tempest


@dataclass
class StormConditions:
    """Evaluated storm conditions for triggering capture"""
    storm_detected: bool = False
    confidence: float = 0.0  # 0-1 score
    trigger_reasons: List[str] = field(default_factory=list)

    # Individual condition flags
    lightning_active: bool = False
    heavy_rain: bool = False
    high_winds: bool = False
    rapid_pressure_drop: bool = False
    dramatic_sky_change: bool = False

    # Metrics
    lightning_strikes_recent: int = 0
    lightning_avg_distance: Optional[float] = None
    pressure_change_hpa: float = 0.0
    rain_rate_mm_hr: float = 0.0
    wind_gust_mph: float = 0.0


class TempestMonitor:
    """Monitor Tempest weather station for storm conditions"""

    def __init__(self):
        """Initialize Tempest monitor"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.enabled = self.config.get('tempest.enabled', False)
        self.station_id = self.config.get('tempest.station_id', '')
        self.udp_enabled = self.config.get('tempest.udp.enabled', True)
        self.udp_port = self.config.get('tempest.udp.port', 50222)
        self.udp_timeout = self.config.get('tempest.udp.timeout_seconds', 300)

        # Trigger thresholds
        self.lightning_enabled = self.config.get('tempest.triggers.lightning.enabled', True)
        self.lightning_max_distance = self.config.get('tempest.triggers.lightning.max_distance_km', 15)
        self.lightning_require_fov = self.config.get('tempest.triggers.lightning.require_in_camera_fov', True)
        self.lightning_min_strikes = self.config.get('tempest.triggers.lightning.min_strike_count', 3)
        self.lightning_time_window = self.config.get('tempest.triggers.lightning.strike_window_minutes', 10)

        self.pressure_drop_threshold = self.config.get('tempest.triggers.weather_change.pressure_drop_hpa', 2.0)
        self.rain_rate_threshold = self.config.get('tempest.triggers.rain.rate_threshold_mm_hr', 10.0)
        self.wind_gust_threshold = self.config.get('tempest.triggers.wind.gust_threshold_mph', 25)

        # Data storage (ring buffers for recent history)
        self.observations = deque(maxlen=100)  # Last ~100 minutes of observations
        self.lightning_strikes = deque(maxlen=50)  # Last 50 lightning strikes

        # UDP listener
        self.udp_socket = None
        self.udp_thread = None
        self.running = False
        self.last_message_time = None

        # Storm detection state
        self.storm_active = False
        self.last_storm_capture_time = None

        # Callbacks for storm events
        self.storm_callbacks: List[Callable[[StormConditions], None]] = []

        if not self.enabled:
            self.logger.info("Tempest monitor disabled in configuration")
        else:
            self.logger.info(f"Tempest monitor initialized for station {self.station_id}")

    def is_enabled(self) -> bool:
        """Check if Tempest monitoring is enabled"""
        return self.enabled

    def register_storm_callback(self, callback: Callable[[StormConditions], None]):
        """
        Register a callback to be called when storm conditions are detected

        Args:
            callback: Function to call with StormConditions when storm detected
        """
        self.storm_callbacks.append(callback)
        self.logger.info(f"Registered storm callback: {callback.__name__}")

    def start_udp_listener(self):
        """Start UDP listener in background thread"""
        if not self.enabled or not self.udp_enabled:
            self.logger.info("UDP listener not started (disabled in config)")
            return False

        if self.running:
            self.logger.warning("UDP listener already running")
            return False

        try:
            # Create UDP socket
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_socket.bind(('', self.udp_port))
            self.udp_socket.settimeout(5.0)  # 5 second timeout for checking stop condition

            # Start listener thread
            self.running = True
            self.udp_thread = threading.Thread(target=self._udp_listener_loop, daemon=True)
            self.udp_thread.start()

            self.logger.info(f"UDP listener started on port {self.udp_port}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start UDP listener: {e}")
            return False

    def stop_udp_listener(self):
        """Stop UDP listener"""
        if not self.running:
            return

        self.logger.info("Stopping UDP listener...")
        self.running = False

        if self.udp_thread:
            self.udp_thread.join(timeout=10)

        if self.udp_socket:
            self.udp_socket.close()
            self.udp_socket = None

        self.logger.info("UDP listener stopped")

    def _udp_listener_loop(self):
        """Main UDP listener loop (runs in background thread)"""
        self.logger.info("UDP listener loop started")

        while self.running:
            try:
                # Receive UDP message
                data, addr = self.udp_socket.recvfrom(4096)
                message = json.loads(data.decode('utf-8'))

                self.last_message_time = datetime.now()

                # Process message based on type
                msg_type = message.get('type')

                if msg_type == 'obs_st':
                    # Tempest observation (once per minute)
                    self._process_observation(message)

                elif msg_type == 'evt_strike':
                    # Lightning strike event
                    self._process_lightning_strike(message)

                elif msg_type == 'rapid_wind':
                    # Rapid wind update (every 3 seconds)
                    self._process_rapid_wind(message)

                elif msg_type == 'evt_precip':
                    # Rain start event
                    self.logger.info("Rain started")

                elif msg_type == 'device_status':
                    # Device status update
                    self._process_device_status(message)

            except socket.timeout:
                # Normal timeout, check if we should continue
                if self.last_message_time:
                    silence_duration = (datetime.now() - self.last_message_time).total_seconds()
                    if silence_duration > self.udp_timeout:
                        self.logger.warning(f"No messages from Tempest for {silence_duration:.0f} seconds")
                continue

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to decode UDP message: {e}")

            except Exception as e:
                self.logger.error(f"Error in UDP listener loop: {e}")
                time.sleep(1)  # Brief pause before retry

        self.logger.info("UDP listener loop ended")

    def _process_observation(self, message: Dict):
        """
        Process Tempest observation message (obs_st)

        Format: https://weatherflow.github.io/Tempest/api/udp/v171/
        """
        try:
            obs_data = message.get('obs', [[]])[0]  # Array of observations

            if len(obs_data) < 18:
                self.logger.warning(f"Incomplete observation data: {len(obs_data)} fields")
                return

            # Parse observation fields
            # See: https://weatherflow.github.io/Tempest/api/udp/v171/
            timestamp = datetime.fromtimestamp(obs_data[0])
            wind_lull = obs_data[1]  # m/s
            wind_avg = obs_data[2]  # m/s
            wind_gust = obs_data[3]  # m/s
            wind_direction = obs_data[4]  # degrees
            wind_sample_interval = obs_data[5]  # seconds
            pressure = obs_data[6]  # hPa
            temp_c = obs_data[7]  # °C
            humidity = obs_data[8]  # %
            lux = obs_data[9]  # lux
            uv = obs_data[10]  # UV index
            solar_radiation = obs_data[11]  # W/m²
            rain_accumulated = obs_data[12]  # mm
            precip_type = obs_data[13]  # 0=none, 1=rain, 2=hail
            lightning_avg_distance = obs_data[14]  # km
            lightning_strike_count = obs_data[15]  # count
            battery = obs_data[16]  # volts
            report_interval = obs_data[17]  # minutes

            # Convert to imperial units
            temp_f = temp_c * 9/5 + 32
            wind_speed_mph = wind_avg * 2.237
            wind_gust_mph = wind_gust * 2.237

            # Calculate rain rate (mm per hour based on accumulation)
            rain_rate = 0.0
            if len(self.observations) > 0:
                prev_obs = self.observations[-1]
                time_diff = (timestamp - prev_obs.timestamp).total_seconds() / 3600
                if time_diff > 0:
                    rain_diff = rain_accumulated - prev_obs.rain_accumulation
                    rain_rate = max(0, rain_diff / time_diff)

            # Create observation object
            observation = WeatherObservation(
                timestamp=timestamp,
                temperature=temp_f,
                humidity=humidity,
                pressure=pressure,
                wind_speed=wind_speed_mph,
                wind_gust=wind_gust_mph,
                wind_direction=wind_direction,
                rain_rate=rain_rate,
                rain_accumulation=rain_accumulated,
                solar_radiation=solar_radiation,
                uv_index=uv,
                battery_voltage=battery
            )

            # Store observation
            self.observations.append(observation)

            # Log observation
            self.logger.debug(f"Observation: {temp_f:.1f}°F, {humidity:.0f}% RH, "
                            f"{pressure:.1f} hPa, Wind {wind_speed_mph:.1f}/{wind_gust_mph:.1f} mph, "
                            f"Rain {rain_rate:.1f} mm/hr")

            # Check for storm conditions
            self._evaluate_storm_conditions()

        except Exception as e:
            self.logger.error(f"Failed to process observation: {e}")

    def _process_lightning_strike(self, message: Dict):
        """Process lightning strike event (evt_strike)"""
        try:
            evt_data = message.get('evt', [])

            if len(evt_data) < 3:
                self.logger.warning(f"Incomplete lightning data: {len(evt_data)} fields")
                return

            timestamp = datetime.fromtimestamp(evt_data[0])
            distance_km = evt_data[1]
            energy = evt_data[2]

            # Create lightning strike object
            strike = LightningStrike(
                timestamp=timestamp,
                distance_km=distance_km,
                energy=energy
            )

            # Store strike
            self.lightning_strikes.append(strike)

            self.logger.info(f"⚡ Lightning strike detected: {distance_km:.1f} km away, "
                           f"energy {energy}")

            # Check if this triggers storm capture
            self._evaluate_storm_conditions()

        except Exception as e:
            self.logger.error(f"Failed to process lightning strike: {e}")

    def _process_rapid_wind(self, message: Dict):
        """Process rapid wind update (every 3 seconds)"""
        try:
            obs_data = message.get('ob', [])

            if len(obs_data) < 3:
                return

            timestamp = datetime.fromtimestamp(obs_data[0])
            wind_speed = obs_data[1] * 2.237  # m/s to mph
            wind_direction = obs_data[2]  # degrees

            # We don't store all rapid wind updates (too many)
            # But we can use this for real-time wind monitoring

        except Exception as e:
            self.logger.debug(f"Failed to process rapid wind: {e}")

    def _process_device_status(self, message: Dict):
        """Process device status message"""
        try:
            timestamp = datetime.fromtimestamp(message.get('timestamp', 0))
            uptime = message.get('uptime', 0)
            voltage = message.get('voltage', 0)
            rssi = message.get('rssi', 0)

            self.logger.debug(f"Tempest status: uptime {uptime}s, battery {voltage:.2f}V, "
                            f"RSSI {rssi} dBm")

        except Exception as e:
            self.logger.debug(f"Failed to process device status: {e}")

    def _evaluate_storm_conditions(self):
        """
        Evaluate current weather conditions to determine if storm is present
        Triggers callbacks if storm conditions are met
        """
        conditions = StormConditions()

        if not self.observations:
            return

        current_obs = self.observations[-1]
        now = datetime.now()

        # 1. Check for recent lightning activity
        if self.lightning_enabled:
            recent_window = timedelta(minutes=self.lightning_time_window)
            recent_strikes = [s for s in self.lightning_strikes
                            if now - s.timestamp < recent_window]

            if recent_strikes:
                # Filter by distance
                nearby_strikes = [s for s in recent_strikes
                                if s.distance_km <= self.lightning_max_distance]

                conditions.lightning_strikes_recent = len(nearby_strikes)

                if nearby_strikes:
                    avg_distance = sum(s.distance_km for s in nearby_strikes) / len(nearby_strikes)
                    conditions.lightning_avg_distance = avg_distance

                    if len(nearby_strikes) >= self.lightning_min_strikes:
                        conditions.lightning_active = True
                        conditions.trigger_reasons.append(
                            f"Lightning: {len(nearby_strikes)} strikes within "
                            f"{self.lightning_max_distance}km in last {self.lightning_time_window} min"
                        )

        # 2. Check for rapid pressure drop
        if len(self.observations) >= 6:  # Need ~30 min of data
            pressure_window = timedelta(minutes=30)
            old_obs = [o for o in self.observations
                      if now - o.timestamp >= pressure_window]

            if old_obs:
                old_pressure = old_obs[-1].pressure
                pressure_change = current_obs.pressure - old_pressure
                conditions.pressure_change_hpa = pressure_change

                if pressure_change <= -self.pressure_drop_threshold:
                    conditions.rapid_pressure_drop = True
                    conditions.trigger_reasons.append(
                        f"Pressure drop: {abs(pressure_change):.1f} hPa in 30 min"
                    )

        # 3. Check for heavy rain
        if current_obs.rain_rate >= self.rain_rate_threshold:
            conditions.heavy_rain = True
            conditions.rain_rate_mm_hr = current_obs.rain_rate
            conditions.trigger_reasons.append(
                f"Heavy rain: {current_obs.rain_rate:.1f} mm/hr"
            )

        # 4. Check for high winds
        if current_obs.wind_gust >= self.wind_gust_threshold:
            conditions.high_winds = True
            conditions.wind_gust_mph = current_obs.wind_gust
            conditions.trigger_reasons.append(
                f"High winds: gust {current_obs.wind_gust:.1f} mph"
            )

        # Calculate overall confidence score
        score = 0.0
        if conditions.lightning_active:
            score += 0.4
        if conditions.rapid_pressure_drop:
            score += 0.2
        if conditions.heavy_rain:
            score += 0.2
        if conditions.high_winds:
            score += 0.2

        conditions.confidence = min(score, 1.0)

        # Determine if storm should be flagged
        if conditions.confidence >= 0.4:  # At least 2 conditions met
            conditions.storm_detected = True

            # Check cooldown period
            cooldown_hours = self.config.get('tempest.capture.cooldown_hours', 1)
            if self.last_storm_capture_time:
                time_since_last = (now - self.last_storm_capture_time).total_seconds() / 3600
                if time_since_last < cooldown_hours:
                    self.logger.info(f"Storm detected but in cooldown period "
                                   f"({time_since_last:.1f}h < {cooldown_hours}h)")
                    return

            # Trigger callbacks
            if not self.storm_active:
                self.storm_active = True
                self.logger.info(f"🌩️ STORM DETECTED! Confidence: {conditions.confidence:.1%}")
                for reason in conditions.trigger_reasons:
                    self.logger.info(f"  • {reason}")

                # Call registered callbacks
                for callback in self.storm_callbacks:
                    try:
                        callback(conditions)
                    except Exception as e:
                        self.logger.error(f"Error in storm callback {callback.__name__}: {e}")

                self.last_storm_capture_time = now
        else:
            # Clear storm flag if conditions improve
            if self.storm_active:
                self.logger.info("Storm conditions have cleared")
                self.storm_active = False

    def get_current_conditions(self) -> Optional[WeatherObservation]:
        """Get most recent weather observation"""
        if self.observations:
            return self.observations[-1]
        return None

    def get_recent_lightning(self, minutes: int = 10) -> List[LightningStrike]:
        """Get lightning strikes from last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [s for s in self.lightning_strikes if s.timestamp >= cutoff]

    def get_status(self) -> Dict:
        """Get monitor status"""
        status = {
            'enabled': self.enabled,
            'running': self.running,
            'last_message': self.last_message_time.isoformat() if self.last_message_time else None,
            'observations_cached': len(self.observations),
            'lightning_strikes_cached': len(self.lightning_strikes),
            'storm_active': self.storm_active
        }

        if self.observations:
            current = self.observations[-1]
            status['current_weather'] = {
                'temperature': current.temperature,
                'humidity': current.humidity,
                'pressure': current.pressure,
                'wind_speed': current.wind_speed,
                'wind_gust': current.wind_gust,
                'rain_rate': current.rain_rate
            }

        return status


# Test/demo function
def main():
    """Test Tempest monitor"""
    import signal
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    monitor = TempestMonitor()

    if not monitor.is_enabled():
        print("Tempest monitor is disabled in configuration")
        print("Set tempest.enabled=true in config.yaml to enable")
        return

    # Define storm callback
    def on_storm_detected(conditions: StormConditions):
        print(f"\n{'='*60}")
        print(f"🌩️  STORM DETECTED!")
        print(f"{'='*60}")
        print(f"Confidence: {conditions.confidence:.1%}")
        print(f"Reasons:")
        for reason in conditions.trigger_reasons:
            print(f"  • {reason}")
        print(f"{'='*60}\n")

    monitor.register_storm_callback(on_storm_detected)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nStopping monitor...")
        monitor.stop_udp_listener()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start monitoring
    print(f"Starting Tempest UDP listener on port {monitor.udp_port}...")
    print("Press Ctrl+C to stop\n")

    if monitor.start_udp_listener():
        # Keep main thread alive
        while monitor.running:
            time.sleep(1)

            # Print status every 60 seconds
            if int(time.time()) % 60 == 0:
                status = monitor.get_status()
                print(f"\nStatus: {status['observations_cached']} observations, "
                      f"{status['lightning_strikes_cached']} lightning strikes")

                current = monitor.get_current_conditions()
                if current:
                    print(f"Current: {current.temperature:.1f}°F, "
                          f"{current.pressure:.1f} hPa, "
                          f"Wind {current.wind_gust:.1f} mph")
    else:
        print("Failed to start UDP listener")


if __name__ == "__main__":
    main()
