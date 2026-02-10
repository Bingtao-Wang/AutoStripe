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

## Current Version: V4 (VLLiNet AI Perception + Polynomial Extrapolation)

V4 replaces V3's ground-truth CityScapes segmentation with VLLiNet, a trained
deep learning model (MaxF 98.33%, IoU 96.72%), moving toward real-world
deployable perception. Key additions:
- VLLiNet AI road segmentation (replaces CityScapes GT, G key toggles AI/GT)
- Polynomial curve extrapolation for blind-spot nozzle distance
- Native 1248x384 camera resolution (matches VLLiNet training)
- CARLA-ROS Bridge integration (subscribes to Bridge topics)
- Magenta polynomial curve visualization in 3D and RVIZ

### V4 Architecture

```text
manual_painting_control_v4.py           -- V4 main entry point (AI/GT toggle + poly distance)
diag_vllinet.py                         -- Standalone VLLiNet model verification script
carla_env/setup_scene_v2.py             -- Scene: 1248x384 cameras, position (1.5, 2.4)
perception/road_segmentor_ai.py         -- VLLiNet wrapper, same segment() interface
perception/road_segmentor.py            -- GT CityScapes segmentor (preserved)
perception/perception_pipeline.py       -- AI/GT dual mode via use_ai flag
planning/vision_path_planner.py         -- + polynomial extrapolation (estimate_nozzle_edge_distance)
ros_interface/autostripe_node.py        -- V4 ROS node (subscribes to CARLA-ROS Bridge)
ros_interface/rviz_publisher.py         -- + poly curve marker publisher
```

### V4 Perception Pipeline (VLLiNet AI mode)

Each frame:
1. RGB camera -> BGRA -> RGB, ImageNet normalize -> [1, 3, 384, 1248] tensor
2. Depth camera -> CARLA decode -> min-max normalize -> [1, 3, 384, 1248] tensor
3. VLLiNet_Lite inference (mixed precision) -> sigmoid -> threshold 0.5
4. Upsample to original resolution -> apply MASK_TOP_RATIO cutoff
5. Edge extractor + depth projector (same as V3) -> world coordinates

GT mode (G key toggle): falls back to V3 CityScapes pipeline.

### V4 Key Changes from V3

- Perception: CityScapes GT -> VLLiNet AI (G key toggles AI/GT for A/B comparison)
- Camera resolution: 800x600 -> 1248x384 (native VLLiNet input, no software resize)
- Camera position: (x=2.5, z=2.8) -> (x=1.5, z=2.4) (matches VLLiNet training)
- Camera intrinsics: fx=fy=624 (for 1248px, FOV=90), cx=624, cy=192
- Polynomial extrapolation: quadratic fit to edge points, extrapolate to lon=0 for blind spot
- Magenta poly curve: 3D visualization of fitted quadratic in CARLA editor
- Poly-Edge HUD: additional distance readout from polynomial extrapolation
- ROS node: rewritten to subscribe to CARLA-ROS Bridge topics (no direct CARLA API)
- RVIZ: poly curve marker topic (/autostripe/planning/poly_curve)
- Pygame window: 1248x384 (matches camera resolution)

### V4 Key Parameters

- VLLiNet input: 1248x384, RGB ImageNet normalized, depth min-max [0,1]
- VLLiNet output: 624x192 (1/2 res), upsampled back to 1248x384
- VLLiNet depth channels: 3 (auto-detected from checkpoint)
- Checkpoint: VLLiNet_models/checkpoints_carla/best_model.pth
- Polynomial fit: quadratic (deg=2), edge points in [3m, 20m] range, min 5 points
- Polynomial sanity: 0.5m < distance < 15.0m, else fallback to V3 median method
- Front cameras: x=1.5, z=2.4, pitch=-15, 1248x384, FOV=90
- All other parameters same as V3

### V4.1 (Control Algorithm Optimization)

V4.1 adds closed-loop control to automatically converge nozzle-edge distance
to the ideal 3.0m and auto-start painting when stable. Three coordinated mechanisms:

- Adaptive steer filter: aggressive (0.50) when lateral error > 0.5m, smooth (0.15) when < 0.3m
- Dynamic driving offset: P-controller adjusts driving_offset with low-pass smoothing
- Auto-paint state machine: CONVERGING → STABILIZED (60 frames) → PAINTING

### V4.1 Key Changes from V4

- Control: fixed STEER_FILTER → adaptive via set_lateral_error() (0.15–0.50 linear interp)
- Planning: fixed driving_offset=5.0 → dynamic via set_dynamic_offset() P-controller
- Dynamic offset: Kp=0.8, low-pass OFFSET_SMOOTH=0.08, correction range [-1.0, +2.0]m
- Auto-paint: AutoPaintStateMachine class (CONVERGING/STABILIZED/PAINTING)
- Auto-paint conditions: |nozzle_dist - 3.0| < 0.3m, speed > 1.0 m/s, stable 60 frames
- SPACE key: manual override (bypasses state machine)
- HUD: AutoPaint state indicator, driving offset value, steer filter value
- Bug fix: poly_dist is vehicle-center-to-edge, subtract nozzle_arm for nozzle-to-edge

### V4.1 Key Parameters

- OFFSET_KP: 0.8 (P-controller gain)
- OFFSET_SMOOTH: 0.08 (low-pass filter rate for offset changes)
- OFFSET_MAX_CORRECTION: +2.0m (cap at 7.0m total offset)
- OFFSET_MIN_CORRECTION: -1.0m (floor at 4.0m total offset)
- STEER_FILTER_AGGRESSIVE: 0.50 (large error)
- STEER_FILTER_SMOOTH: 0.15 (small error, same as V4)
- TARGET_NOZZLE_DIST: 3.0m (ideal painting distance)
- Auto-paint tolerance: 0.3m, stability_frames: 60, min_speed: 1.0 m/s

### V4.2 (PD Control + Evaluation Pipeline + Dashed Lines)

V4.2 addresses V4.1's oscillation issues and adds quantitative evaluation,
dashed line support, and paper-grade data recording. Key additions:

- PD controller: derivative term damps oscillation, replaces P-only offset control
- Hysteresis state machine: separate enter/exit tolerances + grace frames prevent chatter
- Evaluation pipeline: Map API ground truth comparison with CSV export (E key)
- Dashed line mode: alternating paint/gap phases for lane dividers (D key)
- Per-frame CSV logger: 27-column framelog during eval recording (FrameLogger)
- Inference timing: VLLiNet forward pass timing with CUDA synchronize
- Enhanced detail CSV: 8 columns (+ GT coords, along-track dist, curvature, in_range)
- Road geometry: curvature and lane width functions in lane_planner
- Enhanced visualization: timeseries plots, curvature-deviation scatter

### V4.2 Key Changes from V4.1

- Control: P-controller → PD-controller (Kp=0.5, Kd=0.3)
- OFFSET_SMOOTH: 0.08 → 0.12 (faster response with D-term damping)
- Auto-paint: single tolerance → hysteresis (enter=0.3m, exit=0.45m)
- Auto-paint: PAINTING state has 15 grace frames before downgrade
- Dashed line: D key toggles SOLID/DASHED mode (3m paint / 3m gap)
- Evaluation: E key toggles eval recording (start/stop), runs GT comparison on stop
- Per-frame logger: FrameLogger records 27-column CSV during eval recording
- Inference timing: road_segmentor_ai.py times forward pass, pipeline forwards last_inference_ms
- Detail CSV: 3 columns → 8 columns (+ gt_nearest_x/y, along_track_dist, local_curvature, in_range)
- Road geometry: lane_planner.py adds compute_road_curvature() and get_lane_widths()
- Visualization: timeseries plots (6 subplots), curvature-deviation scatter, backwards-compatible load_detail()
- New file: evaluation/frame_logger.py (FrameLogger class, 27-col CSV)
- Modified: evaluation/trajectory_evaluator.py (8-col detail, local curvatures, along-track dist)
- Modified: evaluation/visualize_eval.py (timeseries, curvature scatter, framelog support)
- HUD: line type indicator, dash progress, updated help text

### V4.2 Key Parameters

- OFFSET_KP: 0.5 (reduced from 0.8, D-term compensates)
- OFFSET_KD: 0.3 (derivative gain, damps error rate of change)
- OFFSET_SMOOTH: 0.12 (faster low-pass with D-term damping)
- tolerance_enter: 0.3m (entering STABILIZED from CONVERGING)
- tolerance_exit: 0.45m (leaving STABILIZED/PAINTING, 1.5x wider)
- GRACE_LIMIT: 15 frames (PAINTING tolerates brief excursions)
- DASH_LENGTH: 3.0m (paint phase length)
- GAP_LENGTH: 3.0m (gap phase length)
- Evaluation coverage_threshold: 2.0m (GT point considered covered)

### V3 (Manual Painting Control + Enhanced Visualization) — preserved

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
  manual_painting_control_v4.py # V4 main entry point (AI/GT + poly distance)
  diag_vllinet.py            # V4 standalone VLLiNet model verification
  carla_env/
    __init__.py
    setup_scene.py            # V1 scene setup
    setup_scene_v2.py         # V2-V4 scene: 1248x384 cameras, position (1.5, 2.4)
  perception/
    __init__.py
    road_segmentor.py         # GT: CityScapes color matching -> road mask
    road_segmentor_ai.py      # V4: VLLiNet wrapper -> road mask
    edge_extractor.py         # Road mask -> left/right edge pixels
    depth_projector.py        # Pixel + depth -> world coordinates
    perception_pipeline.py    # AI/GT dual mode (use_ai flag)
  planning/
    __init__.py
    lane_planner.py           # V1 Map API planner
    vision_path_planner.py    # V2-V4 vision planner + polynomial extrapolation
  control/
    __init__.py
    marker_vehicle.py         # V1 Pure Pursuit
    marker_vehicle_v2.py      # V2-V4 Pure Pursuit + dynamic path update
  ros_interface/
    __init__.py
    topic_config.py           # ROS topic name constants (+ poly_curve topic)
    rviz_publisher.py         # RVIZ publisher (+ poly curve marker)
    autostripe_node.py        # V4 ROS node (CARLA-ROS Bridge subscriber)
  configs/
    rviz/
      autostripe_v2.rviz      # V2/V3 RVIZ layout
      autostripe_v4.rviz      # V4 RVIZ layout (+ poly curve)
  launch/
    autostripe_v2.launch      # V2/V3 launch file
    autostripe_v4.launch      # V4 launch (Bridge + AutoStripe + RVIZ)
    rviz_only.launch          # RVIZ-only launch
  VLLiNet_models/             # VLLiNet model code + checkpoint
    models/vllinet.py         # VLLiNet_Lite model class
    models/backbone.py        # MobileNetV3 + LiDAREncoder
    checkpoints_carla/best_model.pth  # Trained checkpoint
  evaluation/                 # V4.2 trajectory evaluation + data recording
    __init__.py
    trajectory_evaluator.py   # Map API GT comparison + 8-col detail CSV export
    frame_logger.py           # Per-frame 27-column CSV recorder
    visualize_eval.py         # Evaluation plots + timeseries + curvature scatter
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

# Terminal 2: Run V4 AI Perception (recommended)
cd /home/peter/workspaces/carla-0.9.15/CARLA_0.9.15/0MyCode/AutoStripe
python manual_painting_control_v4.py

# Terminal 2: Run V4 model diagnostic (verify VLLiNet loads)
python diag_vllinet.py

# Terminal 2: Run V3 Manual Painting Control (GT only)
python manual_painting_control.py

# Terminal 2: Run V2 Standalone (auto only, no manual control)
python main_v2.py

# Terminal 2: Run V1 (Map API)
python main_v1.py

# Terminal 2: Run V4 with ROS + RVIZ (requires CARLA-ROS Bridge)
source /opt/ros/melodic/setup.bash
roslaunch autostripe autostripe_v4.launch

# Terminal 2: Run V2/V3 with ROS + RVIZ
roslaunch autostripe autostripe_v2.launch
```

### V4 Keyboard Controls

| Key | Function |
|-----|----------|
| SPACE | Toggle painting ON/OFF |
| TAB | Toggle Auto/Manual drive mode |
| G | Toggle AI/GT perception mode |
| D | Toggle dashed/solid line mode (AUTO only) |
| E | Toggle eval recording (start/stop + framelog + GT evaluation) |
| R | Toggle video recording |
| WASD/Arrows | Manual drive (throttle/steer/brake) |
| Q | Toggle reverse mode |
| V | Toggle spectator follow/free camera |
| X | Handbrake |
| ESC | Quit |

## Future Versions (not yet implemented)

- V4+: Replace VLLiNet with real-world trained model (transfer from CARLA to real camera)
- V4+: Center lines, all-weather support
- V4+: Multi-lane support, obstacle avoidance
