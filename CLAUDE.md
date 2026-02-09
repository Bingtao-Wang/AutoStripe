# AutoStripe - Project Context

## Project Overview

AutoStripe is an automated highway lane marking system built on CARLA 0.9.15.
Full design: `docs/Project_Design.md`

## Environment

- CARLA 0.9.15, Python 3.8+, Linux
- Working dir: `/home/peter/workspaces/carla-0.9.15/CARLA_0.9.15/0MyCode/AutoStripe`
- CARLA egg path pattern: `../carla/dist/carla-*%d.%d-%s.egg`
- Map: Town05 (highway segment)
- Vehicle: `vehicle.*stl*`, spawn at (10, -210, 1.85, yaw=180)

## Current Version: V3 (Manual Painting Control + Enhanced Visualization)

V3 builds on V2's vision perception pipeline, adding:
- CityScapes-based road segmentation (replacing raw tag ID matching)
- Manual painting control with pygame (drive mode + paint toggle)
- Slope-aware visualization (blue dot markers follow road pitch)
- Nozzle-to-edge perpendicular distance visualization (green line)

### V3 Architecture

```text
manual_painting_control.py              -- V3 main entry point (pygame + manual/auto control)
main_v2.py                              -- V2 standalone entry point (auto only, no ROS)
carla_env/setup_scene_v2.py             -- Scene: semantic + depth + RGB front + overhead cameras
perception/road_segmentor.py            -- CityScapes color matching -> binary road mask
perception/edge_extractor.py            -- road mask -> left/right edge pixels
perception/depth_projector.py           -- pixel + depth -> 3D world coordinates
perception/perception_pipeline.py       -- combines segmentor + extractor + projector
planning/vision_path_planner.py         -- right edge -> driving path + nozzle path (with z)
control/marker_vehicle_v2.py            -- Pure Pursuit with dynamic path update
ros_interface/autostripe_node.py        -- main ROS node
```

### V3 Perception Pipeline (CityScapes-based)

Each frame:
1. Semantic camera -> `image.convert(cc.CityScapesPalette)` -> CityScapes colored image
2. RoadSegmentor -> match purple (128,64,128) with tolerance=10 -> binary road mask
3. Edge extractor -> per-row scan for leftmost/rightmost road pixels
4. Depth camera -> decode to meters -> pixel+depth -> world coordinates (with z)
5. Output: left/right road edges in world frame

### V3 Key Changes from V2

- Road segmentation: raw tag IDs -> CityScapes color matching (robust to non-standard tags)
- Driving path visualization: blue lines -> blue dot markers (draw_point, size=0.1)
- Slope-aware z: blue dots follow vehicle pitch angle for correct height on slopes/bridges
- Nozzle-edge distance: perpendicular to vehicle body (median lateral projection)
- Green visualization line: nozzle -> road edge intersection (perpendicular)
- Cyan visualization line: first tracking point -> road edge distance
- Distance text on all views: 3D->2D projection + cv2.putText (debug.draw_string only works in editor)
- Yellow paint line: gap handling with None markers (pause/resume no longer connects)
- Manual painting control: pygame keyboard input (SPACE=paint, TAB=mode, WASD=drive)
- Synchronous mode: fixed_delta_seconds=0.05, world.tick() each frame (eliminates spectator jitter, disabled for FPS)
- Spectator follow: V key toggles follow/free camera in CARLA editor
- Pure Pursuit lookahead: LOOKAHEAD_WPS=8 (reduced from 15 for closer target points)
- Red edge dots: right road edge visualization (red draw_point, slope z, 20m range, 2m spacing, median lateral filter)
- Short-term memory: planner keeps last valid path for 10 frames when perception fails
- Distance truncation: planner discards edge points beyond 20m (reduces depth noise)
- Visualization z offset: all debug drawings use +0.3 (lowered from +0.5 for less floating)

### V3 Key Parameters

- CityScapes road color: BGR (128, 64, 128), tolerance=10
- Road mask top cutoff: 35% (MASK_TOP_RATIO)
- Front cameras: x=2.5, z=2.8, pitch=-15, 800x600, FOV=90
- CARLA depth decode: (R + G*256 + B*65536) / (256^3-1) * 1000
- Camera intrinsics: fx = fy = 400 (for 800px, FOV=90)
- Semantic tags: Road=7, RoadLines=6, Sidewalk=8, Terrain=22
- Edge extraction: morphological close(15) + open(5), min road width 40px
- Path planner: 1m longitudinal bins, smooth_window=5, min_spacing=0.5m
- Pure Pursuit: LOOKAHEAD_WPS=8, wheelbase=2.875, Kdd=3.0, TARGET_SPEED=3.0 m/s
- Synchronous mode: fixed_delta_seconds=0.05 (20 FPS), world.tick() per frame (disabled for performance)
- Spectator: behind=10m, height=6m, pitch=-20, get_forward_vector() offset
- Visualization z offset: +0.3 above slope-projected road surface
- Red edge dots: 20m max range, 2m min spacing, median lateral ±3m outlier filter
- Path planner memory: 10 frames fallback, 20m max range truncation

### V1 (Map API based) — preserved

V1 skips LUNA-Net perception, uses CARLA Map API for path planning (Pure Pursuit),
vehicle acts as marking machine — line is the vehicle's actual trajectory offset to the right.

### V1 Architecture

```text
main_v1.py                  -- entry point, main loop (drive + paint + display)
carla_env/setup_scene.py    -- CARLA connection, vehicle/camera spawn, spectator follow, overhead display
planning/lane_planner.py    -- waypoint generation from Map API (also has edge computation, unused in V1)
control/marker_vehicle.py   -- MarkerVehicle class (Pure Pursuit controller)
```

### V1 Core Logic: Nozzle Trajectory Painting

The vehicle IS the marking machine. Each frame:
1. Get vehicle actual position + yaw
2. Compute right-side nozzle position (offset 2.0m via yaw + pi/2)
3. Draw line segment from previous nozzle pos to current nozzle pos
4. Collect trail for overhead OpenCV overlay

NOT pre-computed map edges. Line = vehicle's real trajectory + right offset.

### V1 Key Parameters

- Pure Pursuit: wheelbase=2.875, Kdd=4.0, throttle=0.3
- Nozzle offset: 2.0m to the right of vehicle
- Line color: yellow (255,255,0), thickness: 0.3m
- debug.draw_line: persistent_lines=True, life_time=1000
- Overhead camera: z=25m, pitch=-90, FOV=90, image 1800x1600
- Spectator: behind=10m, height=6m, pitch=-20, follows vehicle yaw

### V1 Solved Issues

- OpenCV white screen: camera callback runs in thread, cv2.imshow must be in main thread -> shared buffer
- debug.draw_line invisible in camera sensor -> world-to-pixel projection + cv2.line overlay
- Spectator loses vehicle on turns -> compute position based on vehicle yaw, not fixed world offset

## Project Structure

```
AutoStripe/
  main_v1.py                 # V1 entry point
  main_v2.py                 # V2 standalone entry point (auto only, no ROS)
  manual_painting_control.py # V3 main entry point (manual/auto + paint control)
  carla_env/
    __init__.py
    setup_scene.py            # V1 scene setup
    setup_scene_v2.py         # V2/V3 scene: + semantic/depth/RGB front cameras + CityScapes
  perception/
    __init__.py
    road_segmentor.py         # CityScapes color matching -> road mask
    edge_extractor.py         # Road mask -> left/right edge pixels
    depth_projector.py        # Pixel + depth -> world coordinates
    perception_pipeline.py    # Combines above three steps
  planning/
    __init__.py
    lane_planner.py           # V1 Map API planner
    vision_path_planner.py    # V2/V3 vision-based path planner (with z coords)
  control/
    __init__.py
    marker_vehicle.py         # V1 Pure Pursuit
    marker_vehicle_v2.py      # V2/V3 Pure Pursuit + dynamic path update
  ros_interface/
    __init__.py
    topic_config.py           # ROS topic name constants
    rviz_publisher.py         # RVIZ visualization publisher
    autostripe_node.py        # Main ROS node
  configs/
    rviz/
      autostripe_v2.rviz      # RVIZ layout config
  launch/
    autostripe_v2.launch      # Main launch file
    rviz_only.launch          # RVIZ-only launch
  evaluation/                 # (reserved)
  datasets/                   # (reserved)
  docs/
    Project_Design.md
  TEST/
    pure_pursuit_TOWN5.py
    pure_pursuit_windows.py
    open3d_lidar.py
```

## Code Conventions

- All modules include CARLA egg path append boilerplate at top
- Functions return dicts or tuples, not custom classes (except MarkerVehicle)
- Debug drawing uses `persistent_lines=True`, `life_time=1000`
- Cleanup via actor list passed to `cleanup(actors)`

## Run Instructions

```bash
# Terminal 1: Start CARLA
cd /home/peter/workspaces/carla-0.9.15/CARLA_0.9.15 && ./CarlaUE4.sh

# Terminal 2: Run V3 Manual Painting Control (recommended)
cd /home/peter/workspaces/carla-0.9.15/CARLA_0.9.15/0MyCode/AutoStripe
python manual_painting_control.py

# Terminal 2: Run V2 Standalone (auto only, no manual control)
python main_v2.py

# Terminal 2: Run V1 (Map API)
python main_v1.py

# Terminal 2: Run V2 with ROS + RVIZ
source /opt/ros/melodic/setup.bash
roslaunch autostripe autostripe_v2.launch
```

### V3 Keyboard Controls

| Key | Function |
|-----|----------|
| SPACE | Toggle painting ON/OFF |
| TAB | Toggle Auto/Manual drive mode |
| WASD/Arrows | Manual drive (throttle/steer/brake) |
| Q | Toggle reverse mode |
| V | Toggle spectator follow/free camera |
| X | Handbrake |
| ESC | Quit |

## Future Versions (not yet implemented)

- V3+: Replace CARLA semantic camera with LUNA-Net for real perception
- V3+: Dashed lane dividers, center lines, all-weather support
- V3+: Full evaluation pipeline: compare V3 trajectory vs Map API ground truth
- V3+: Multi-lane support, obstacle avoidance
