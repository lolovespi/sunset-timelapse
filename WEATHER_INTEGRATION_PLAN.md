# Weather Analysis & Storm-Based Timelapse Capture - Implementation Plan

## Project Overview

Expand the sunset timelapse system to include **Tempest weather station integration** with **geographic visibility analysis** to automatically capture storms and weather events that are visible from the camera's position in Pelham, Alabama.

---

## ✅ COMPLETED WORK

### Phase 0: Camera & Geography Configuration (DONE)

#### Files Created:
- ✅ `geography_calculator.py` - Complete FOV and horizon distance calculations

#### Configuration Updated:
- ✅ `config.yaml` - Added comprehensive camera and Tempest sections

#### Achievements:

**1. Camera Geography Module (`geography_calculator.py`)**
- Horizon distance calculations using Earth curvature formula
- Field of view (FOV) cone calculations
- Bearing visibility checks
- Geographic distance and bearing calculations
- Validation and testing functions

**2. Camera Position Configured (Measured with Garmin)**
- **Location**: Pelham, AL (33.2856°N, 86.8097°W)
- **Elevation**: 1,240 feet (378 meters) above sea level
- **Direction**: 280° (West-Northwest)
- **Field of View**: 87° horizontal × 47° vertical
- **Viewing Cone**: 236.5° (WSW) to 323.5° (NW)

**3. Visibility Ranges Calculated**
- **Ground level horizon**: 69 km (43 miles)
- **Cloud base (3000m)**: 265 km (165 miles)
- **Storm tops (10km)**: 426 km (265 miles)
- **Viewing area**: ~3,657 km² ground coverage
- **Tempest lightning range** (40km) is fully within visible horizon ✓

**4. Configuration Sections Added to `config.yaml`**

##### Camera Configuration:
```yaml
camera:
  location:
    elevation_meters: 378  # 1240 feet
    latitude: 33.2856
    longitude: -86.8097

  # Optical properties
  lens_focal_length_mm: 4.0
  sensor_size: "1/2.8"
  horizontal_fov_degrees: 87
  vertical_fov_degrees: 47

  # Orientation (measured)
  azimuth_degrees: 280  # WNW
  tilt_degrees: 0

  # Viewing ranges
  visible_horizon_km: 69
  storm_visible_altitude_km: 426
```

##### Tempest Weather Station Configuration:
```yaml
tempest:
  enabled: false  # Set to true when configured
  station_id: ""  # Your station ID

  udp:
    enabled: true
    port: 50222

  triggers:
    lightning:
      enabled: true
      max_distance_km: 15
      require_in_camera_fov: true
      min_strike_count: 3
      strike_window_minutes: 10

    weather_change:
      pressure_drop_hpa: 2.0
      temp_drop_f: 5.0
      solar_radiation_drop_percent: 50

    rain:
      rate_threshold_mm_hr: 10.0

    wind:
      gust_threshold_mph: 25

  capture:
    start_on_first_strike: true
    pre_storm_minutes: 30
    post_storm_minutes: 60
    max_duration_hours: 4
    cooldown_hours: 1
```

##### Storm Analysis Configuration:
```yaml
storm_analysis:
  enabled: true

  lightning_detection:
    enabled: true
    brightness_threshold: 50

  cloud_analysis:
    enabled: true
    darkness_threshold: 0.3

  scoring:
    lightning_weight: 5.0
    distance_weight: 20.0
    rain_weight: 2.0
    wind_weight: 0.5
    max_score: 100
```

### Phase 1: Tempest Weather Monitor (DONE)

#### Files Created:
- ✅ `tempest_monitor.py` - Complete UDP listener and storm detection

#### Capabilities Implemented:

**1. Real-Time Data Collection via UDP**
- Listens on port 50222 for Tempest broadcasts
- Processes observation messages (obs_st) - temperature, pressure, wind, rain
- Processes lightning strike events (evt_strike) - distance, energy
- Processes rapid wind updates (rapid_wind) - 3-second intervals
- Processes device status messages

**2. Data Storage**
- Ring buffer of last 100 weather observations (~100 minutes)
- Ring buffer of last 50 lightning strikes
- Automatic data management (old data expires)

**3. Storm Detection Logic**
- **Lightning**: Tracks strikes within distance threshold and time window
- **Pressure**: Detects rapid pressure drops (≥2 hPa in 30 min)
- **Rain**: Monitors heavy rainfall rates (≥10 mm/hr)
- **Wind**: Tracks high wind gusts (≥25 mph)
- **Confidence scoring**: Combines multiple indicators for reliable detection

**4. Callback System**
- Registers storm detection callbacks
- Triggers callbacks when storm conditions met
- Includes cooldown period to prevent duplicate captures
- Thread-safe operation

**5. Status Monitoring**
- Current weather conditions
- Recent lightning strikes
- Storm active/inactive state
- Message reception health check

**6. Testing Capabilities**
- Standalone test mode (`python tempest_monitor.py`)
- Real-time console output
- Debug logging

---

## 🚧 REMAINING WORK

### Phase 2: Visibility Analyzer (NEW FILE)

**File to Create**: `visibility_analyzer.py`

**Purpose**: Integrate geography calculator with Tempest monitor to determine if detected weather events are actually visible from the camera.

**Key Functions Needed**:
1. `is_lightning_visible(distance_km, bearing)` → bool
   - Check if lightning strike is within horizon
   - Check if strike is within camera FOV
   - Return visibility status with reason

2. `is_storm_visible(storm_location)` → bool
   - Given lat/lon of storm, check visibility
   - Calculate distance and bearing
   - Apply horizon and FOV filters

3. `filter_visible_strikes(strikes)` → List[strikes]
   - Take list of lightning strikes
   - Return only those visible from camera

4. Integration with TempestMonitor:
   - Modify `_process_lightning_strike()` to check visibility
   - Only count visible strikes toward threshold
   - Log visibility decisions

**Estimated Time**: 2-3 hours

---

### Phase 3: Scheduler Integration (MODIFY EXISTING)

**File to Modify**: `sunset_scheduler.py`

**Changes Needed**:

1. **Import and Initialize TempestMonitor**
   ```python
   from tempest_monitor import TempestMonitor
   from visibility_analyzer import VisibilityAnalyzer

   def __init__(self):
       # ... existing code ...
       self.tempest_monitor = TempestMonitor()
       self.visibility_analyzer = VisibilityAnalyzer()
   ```

2. **Add Storm Capture State Management**
   ```python
   # New states
   self.capture_type = None  # 'sunset', 'storm', 'combined'
   self.storm_capture_active = False
   ```

3. **Register Storm Callback**
   ```python
   def on_storm_detected(self, conditions):
       # Check if visible
       # Check if already capturing
       # Schedule or extend capture
       # Send notification
   ```

4. **Add New Capture Methods**
   - `capture_storm_sequence(conditions)` - Similar to sunset but for storms
   - `handle_sunset_storm_overlap()` - Merge or extend captures
   - `complete_storm_workflow()` - Full storm capture → process → upload

5. **Update Main Scheduler Loop**
   - Start Tempest monitor in background
   - Handle concurrent sunset + storm events
   - Proper cleanup on shutdown

6. **Validation Updates**
   - Add Tempest to `validate_system()`
   - Test UDP connection
   - Verify configuration

**Estimated Time**: 6-8 hours

---

### Phase 4: Storm Brilliance Score (NEW FILE)

**File to Create**: `storm_brilliance_score.py`

**Purpose**: Analyze storm intensity using video footage + Tempest sensor data (analogous to Sunset Brilliance Score).

**Key Components**:

1. **Visual Lightning Detection**
   - Detect sudden brightness increases in frames
   - Compare frame-to-frame brightness changes
   - Count visual lightning flashes

2. **Cloud Darkness Analysis**
   - Measure overall frame darkness
   - Detect dramatic dark clouds
   - Calculate cloud contrast

3. **Rain Visibility Detection**
   - Detect rain streaks in video
   - Measure visibility degradation
   - Quantify atmospheric opacity

4. **Storm Intensity Score (SIS)**
   ```python
   score = (
       lightning_count × 5 +           # Sensor data
       (1 / avg_distance) × 20 +       # Proximity bonus
       rain_rate_max × 2 +              # Heavy rain
       wind_gust_max × 0.5 +            # Wind intensity
       pressure_drop × 10 +             # Dramatic change
       visual_lightning_flashes × 3     # Video confirmation
   )
   cap at 100
   ```

5. **Integration with Tempest Data**
   - Read Tempest observations during capture window
   - Combine visual + sensor metrics
   - Generate storm report (similar to SBS report)

**Estimated Time**: 8-10 hours

---

### Phase 5: Enhanced Video Metadata (MODIFY EXISTING)

**File to Modify**: `youtube_uploader.py`

**Changes Needed**:

1. **New Title Formats**
   ```python
   # Storm videos
   "Thunderstorm Timelapse {date} - {lightning_count} Strikes ⚡"

   # Stormy sunsets (combined)
   "Epic Stormy Sunset {date} - SBS {score} + {strikes} Lightning ⚡🌅"

   # Severe weather
   "Severe Storm {date} - SIS {score} ⚡🌩️"
   ```

2. **Enhanced Descriptions with Weather Data**
   ```python
   description = f"""
   Thunderstorm timelapse captured {date} from Pelham, Alabama.

   🌩️ Storm Statistics (from Tempest Weather Station):
   • Lightning strikes: {strike_count} within {distance}km
   • Closest strike: {closest}km to the {direction}
   • Peak wind gust: {wind_max} mph
   • Max rain rate: {rain_max} mm/hr
   • Pressure drop: {pressure_drop} hPa in {duration} minutes

   ⚡ Storm Intensity Score: {sis_score} ({grade})

   📍 Location: Pelham, AL (33.2856°N, 86.8097°W)
   📷 Camera: Reolink RLC810-WA facing {azimuth}° ({direction})
   ⏱️ Captured: {start_time} - {end_time}
   🔄 Interval: 5-second frames

   #thunderstorm #lightning #timelapse #alabama #weather #tempest
   """
   ```

3. **New Upload Methods**
   - `upload_storm_video()` - Storm-specific upload
   - `upload_combined_video()` - Sunset + storm combined

**Estimated Time**: 2-3 hours

---

### Phase 6: CLI Commands & Setup (MODIFY EXISTING)

**File to Modify**: `main.py`

**New Commands to Add**:

1. **Setup Command**
   ```bash
   python main.py setup tempest --discover --test
   ```
   - Discover Tempest on network
   - Test UDP reception
   - Validate configuration
   - Show viewing area visualization

2. **Weather Command**
   ```bash
   python main.py weather --monitor
   python main.py weather --status
   python main.py weather --test-storm
   ```
   - Live weather monitoring
   - Current conditions
   - Simulate storm detection

3. **Test Command Extensions**
   ```bash
   python main.py test --tempest
   python main.py test --visibility
   ```
   - Test Tempest connectivity
   - Test visibility calculations

4. **Historical Command Extension**
   ```bash
   python main.py historical --storms --start DATE --end DATE
   ```
   - Process storm footage from date range

**Estimated Time**: 4-5 hours

---

### Phase 7: Dependencies & Documentation (UPDATES)

**File to Update**: `requirements.txt`

**New Dependencies to Add**:
```python
# Geographic calculations
pyproj>=3.4.0  # Coordinate transformations
geopy>=2.3.0   # Distance and bearing calculations
shapely>=2.0.0 # Geometric operations for viewing cone

# Optional (for advanced features):
# gdal  # Viewshed analysis (terrain-aware visibility)
```

**Documentation to Create**:
1. Update `README.md` with weather features
2. Create `TEMPEST_SETUP.md` - Setup guide for Tempest integration
3. Update `CLAUDE.md` with new commands and capabilities

**Estimated Time**: 2-3 hours

---

## 📊 IMPLEMENTATION STATUS SUMMARY

| Phase | Component | Status | Time Estimate |
|-------|-----------|--------|---------------|
| 0 | Geography Calculator | ✅ DONE | - |
| 0 | Camera Configuration | ✅ DONE | - |
| 0 | Config File Updates | ✅ DONE | - |
| 1 | Tempest Monitor | ✅ DONE | - |
| 2 | Visibility Analyzer | 🚧 TODO | 2-3 hrs |
| 3 | Scheduler Integration | 🚧 TODO | 6-8 hrs |
| 4 | Storm Brilliance Score | 🚧 TODO | 8-10 hrs |
| 5 | Video Metadata Enhancement | 🚧 TODO | 2-3 hrs |
| 6 | CLI Commands & Setup | 🚧 TODO | 4-5 hrs |
| 7 | Dependencies & Docs | 🚧 TODO | 2-3 hrs |

**Total Completed**: ~12-15 hours
**Total Remaining**: ~24-32 hours
**Overall Progress**: ~33% complete

---

## 🎯 NEXT IMMEDIATE STEPS

### Priority 1: Core Functionality (Required for First Storm Capture)

1. **Create `visibility_analyzer.py`** (2-3 hrs)
   - Integrate geography calculator with Tempest
   - Filter lightning strikes by visibility
   - Test with simulated strikes

2. **Update `sunset_scheduler.py`** (6-8 hrs)
   - Initialize Tempest monitor
   - Add storm callback
   - Implement storm capture workflow
   - Handle sunset/storm conflicts

3. **Update `requirements.txt`** (15 min)
   - Add geopy, pyproj, shapely
   - Test installation on both Mac and Pi

### Priority 2: Analysis & Enhancement (Makes Better Videos)

4. **Create `storm_brilliance_score.py`** (8-10 hrs)
   - Video analysis for lightning flashes
   - Cloud darkness detection
   - Storm intensity scoring
   - Integration with Tempest data

5. **Update `youtube_uploader.py`** (2-3 hrs)
   - Storm video titles
   - Enhanced descriptions with weather stats
   - Storm-specific metadata

### Priority 3: User Experience (Quality of Life)

6. **Update `main.py`** (4-5 hrs)
   - Add weather commands
   - Add setup wizard
   - Add test commands

7. **Documentation** (2-3 hrs)
   - Update README
   - Create setup guides
   - Update CLAUDE.md

---

## 🧪 TESTING STRATEGY

### Current Testing (Done)
- ✅ Geography calculator validation
- ✅ Horizon distance calculations verified
- ✅ FOV calculations tested
- ✅ Bearing visibility logic tested

### Needed Testing
- ⏳ Tempest UDP listener (need actual Tempest station)
- ⏳ Lightning strike visibility filtering
- ⏳ Storm detection thresholds (tune with real data)
- ⏳ Capture scheduling conflicts (sunset + storm)
- ⏳ Video processing for storm footage
- ⏳ YouTube upload with storm metadata

### Testing with Actual Tempest Station
**Required to complete testing**:
1. Configure `tempest.station_id` in config.yaml
2. Set `tempest.enabled = true`
3. Run `python tempest_monitor.py` to verify UDP reception
4. Wait for actual storm to test end-to-end workflow

---

## 📝 CONFIGURATION CHECKLIST

### Before First Use:

- [ ] Measure camera azimuth with Garmin/compass ✅ (Done: 280°)
- [ ] Confirm camera elevation ✅ (Done: 1240 feet)
- [ ] Verify camera FOV specs ✅ (Done: 87° × 47°)
- [ ] Get Tempest station ID
- [ ] Set `tempest.enabled = true` in config.yaml
- [ ] Set `tempest.station_id` in config.yaml
- [ ] Test Tempest UDP reception
- [ ] Tune storm detection thresholds (after first storm)
- [ ] Adjust capture timing parameters (after first capture)

---

## 🔮 FUTURE ENHANCEMENTS (Post-MVP)

### Advanced Visibility Analysis
- Terrain-aware viewshed using USGS elevation data
- Account for trees, buildings, local obstructions
- Generate visualization maps of viewing area

### Enhanced Storm Detection
- NWS API integration for alerts and forecasts
- Radar data integration
- Predictive storm capture (start before arrival)
- Multiple Tempest station support (network of stations)

### Video Analysis Improvements
- Machine learning for cloud classification
- Automated highlight detection (best lightning strikes)
- Real-time SBS/SIS display during capture
- Adaptive capture duration based on storm intensity

### Automation & Intelligence
- Smart scheduling (know when storms are likely)
- Automatic threshold tuning based on results
- Storm path prediction and tracking
- Integration with home automation (alerts, displays)

---

## 💡 KEY INSIGHTS & DECISIONS

### Why This Approach Works

1. **Tempest + Camera = Perfect Storm Spotting**
   - Tempest's 40km lightning detection perfectly matches 69km horizon
   - Real-time UDP data enables instant response
   - Local data = no API limits or internet dependency

2. **Geographic Awareness Prevents False Triggers**
   - Lightning 40km to the east = not visible (outside FOV)
   - Lightning 40km to the west = visible and dramatic
   - Saves camera session connections and storage

3. **Multi-Factor Storm Detection = Reliable**
   - Lightning alone might be distant/high altitude
   - Lightning + pressure drop + heavy rain = storm is here
   - Confidence scoring prevents false alarms

4. **Pelham Location is Ideal**
   - 1240 ft elevation = excellent visibility
   - 280° azimuth covers SW/W/NW (prevailing weather)
   - Alabama weather = frequent thunderstorms (good content!)

### Technical Decisions Made

1. **UDP over REST API**: Real-time, no polling, no rate limits
2. **Ring buffers**: Fixed memory, automatic data management
3. **Callback pattern**: Clean separation, extensible
4. **Confidence scoring**: Multiple weak signals > one strong signal
5. **Cooldown periods**: Prevent duplicate captures of same storm

---

## 📞 QUESTIONS TO RESOLVE

1. **Tempest Station ID**: What is your station ID?
2. **API Token**: Do you have a Tempest API token? (optional, for forecasts)
3. **Capture Preferences**:
   - Should storm captures continue if sunset starts during storm?
   - Priority: separate videos or one combined video?
4. **Threshold Tuning**: After first storm, do thresholds need adjustment?
5. **Storage**: Where to store storm videos? (currently: `data/storms/`)
6. **YouTube**: Separate playlist for storms vs sunsets?

---

## 🚀 DEPLOYMENT PLAN

### Development (Mac)
1. Complete remaining phases (Priority 1 first)
2. Test with simulated data
3. Test with actual Tempest station
4. Wait for real storm to test end-to-end

### Production (Raspberry Pi)
1. Deploy tested code to Pi
2. Update Pi configuration with Tempest settings
3. Restart sunset-timelapse service
4. Monitor first storm capture
5. Review footage and tune parameters
6. Monitor for several storms
7. Refine and optimize

### Rollback Plan
- Keep sunset-only capture working
- Storm features can be disabled via config
- No breaking changes to existing functionality

---

## 📄 FILES SUMMARY

### New Files Created (2 files)
1. ✅ `geography_calculator.py` (467 lines)
2. ✅ `tempest_monitor.py` (598 lines)

### Modified Files (1 file)
1. ✅ `config.yaml` (added ~120 lines of configuration)

### Files to Create (3 files)
1. 🚧 `visibility_analyzer.py` (est. ~200 lines)
2. 🚧 `storm_brilliance_score.py` (est. ~400 lines)
3. 🚧 `TEMPEST_SETUP.md` (documentation)

### Files to Modify (4 files)
1. 🚧 `sunset_scheduler.py` (add ~200 lines)
2. 🚧 `youtube_uploader.py` (add ~100 lines)
3. 🚧 `main.py` (add ~150 lines)
4. 🚧 `requirements.txt` (add 3 packages)
5. 🚧 `README.md` (update documentation)
6. 🚧 `CLAUDE.md` (update for Claude Code)

---

## ✉️ CONTACT & SUPPORT

This plan was created through collaborative development with Claude Code.

**Questions or issues?**
- Review configuration in `config.yaml`
- Check logs in `data/logs/`
- Test components individually with test commands
- Refer to this plan document for implementation details

**Next session priorities:**
1. Create visibility_analyzer.py
2. Integrate with sunset_scheduler.py
3. Test with actual Tempest station

---

**Document Version**: 1.0
**Created**: 2025-10-14
**Last Updated**: 2025-10-14
**Status**: Active Development - 33% Complete
