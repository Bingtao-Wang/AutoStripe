#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AutoStripe Headless Experiment Runner

Runs experiments without pygame GUI. Automates:
- Perception mode selection (GT / VLLiNet / LUNA-Net)
- Auto-drive with Pure Pursuit
- Eval recording (framelog + trajectory evaluation)
- Stops after --frames N frames of eval recording

Usage:
  python experiment_runner.py --mode GT --frames 6000
  python experiment_runner.py --mode LUNA --frames 6000
  python experiment_runner.py --mode VLLiNet --frames 6000
  python experiment_runner.py --mode LUNA --weather ClearNight --frames 6000
"""

import argparse
import glob
import os
import sys
import time
import math

try:
    sys.path.append(glob.glob(
        '../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64')
    )[0])
except IndexError:
    pass

import carla
import numpy as np
import cv2

# Patch cv2 to skip GUI windows (headless mode)
cv2.namedWindow = lambda *a, **kw: None
cv2.resizeWindow = lambda *a, **kw: None

from carla_env.setup_scene_v2 import setup_scene_v2, FRONT_CAM_W, FRONT_CAM_H
from carla_env.setup_scene import update_spectator
from perception.perception_pipeline import PerceptionPipeline, PerceptionMode
from planning.vision_path_planner import VisionPathPlanner
from control.marker_vehicle_v2 import MarkerVehicleV2
from evaluation.trajectory_evaluator import TrajectoryEvaluator
from evaluation.frame_logger import FrameLogger
from evaluation.perception_metrics import compute_mask_iou, compute_edge_deviation


# --- Spawn point presets (same as main script) ---
SPAWN_POINTS = {
    1: {"x": 10,     "y": -210,   "z": 1.85, "yaw": 180,   "desc": "Highway straight (original)"},
    2: {"x": -247.1, "y": -32.3,  "z": 10.0, "yaw": 90.1,  "desc": ""},
    3: {"x": 211.0,  "y": -13.6,  "z": 0.50, "yaw": -91.2, "desc": ""},
    4: {"x": 0.0,    "y": 208.8,  "z": 9.00, "yaw": 0.0,   "desc": ""},
    5: {"x": 90.0,   "y": -190.0, "z": 0.50, "yaw": -0.2,  "desc": "Highway SP1 reverse"},
    6: {"x": 94.6,   "y": -146.1, "z": 0.30, "yaw": 189.0, "desc": "Post-turn westbound"},
    7: {"x": 60.0,   "y": 209.0,  "z": 9.00, "yaw": 0.0,   "desc": "Straight before right curve"},
    8: {"x": 210.0,  "y": 75.0,   "z": 9.00, "yaw": -90.0, "desc": "After right curve"},
}


# --- Weather presets ---
WEATHER_PRESETS = {
    'ClearDay': {
        'cloudiness': 10.0, 'precipitation': 0.0,
        'precipitation_deposits': 0.0, 'wind_intensity': 5.0,
        'sun_altitude_angle': 5.0, 'fog_density': 0.0,
        'fog_distance': 100.0, 'wetness': 0.0,
    },
    'ClearNight': {
        'cloudiness': 10.0, 'sun_altitude_angle': -30.0,
        'fog_density': 0.0,
    },
    'CloudyNoon': {
        'cloudiness': 80.0, 'sun_altitude_angle': 60.0,
    },
    'WetSunset': {
        'cloudiness': 40.0, 'precipitation': 60.0,
        'sun_altitude_angle': 5.0,
    },
    'HeavyFoggyNight': {
        'cloudiness': 80.0, 'precipitation': 0.0,
        'precipitation_deposits': 0.0, 'wind_intensity': 5.0,
        'sun_altitude_angle': -30.0, 'fog_density': 80.0,
        'fog_distance': 0.0, 'wetness': 0.0,
    },
    'HeavyRainFoggyNight': {
        'cloudiness': 90.0, 'precipitation': 80.0,
        'precipitation_deposits': 50.0, 'wind_intensity': 40.0,
        'sun_altitude_angle': -30.0, 'fog_density': 50.0,
        'fog_distance': 15.0, 'wetness': 80.0,
    },
}


def get_nozzle_position(vehicle, offset=2.0):
    """Compute nozzle position: vehicle position + right-side offset."""
    veh_tf = vehicle.get_transform()
    veh_loc = vehicle.get_location()
    yaw_rad = math.radians(veh_tf.rotation.yaw)
    dx = offset * math.cos(yaw_rad + math.pi / 2)
    dy = offset * math.sin(yaw_rad + math.pi / 2)
    return carla.Location(x=veh_loc.x + dx, y=veh_loc.y + dy, z=veh_loc.z)


def compute_point_edge_distance(ref_loc, right_world, vehicle_tf, max_lon=15.0):
    """Compute perpendicular distance from reference point to right road edge."""
    if not right_world:
        return 999.0, None

    yaw = math.radians(vehicle_tf.rotation.yaw)
    fwd_x = math.cos(yaw)
    fwd_y = math.sin(yaw)
    right_x = -fwd_y
    right_y = fwd_x

    candidates = []
    for loc in right_world:
        dx = loc.x - ref_loc.x
        dy = loc.y - ref_loc.y
        lon = dx * fwd_x + dy * fwd_y
        lat = dx * right_x + dy * right_y
        if abs(lon) < max_lon and lat > 0:
            candidates.append((abs(lon), lat))

    if not candidates:
        return 999.0, None

    candidates.sort(key=lambda c: c[0])
    top_n = min(10, len(candidates))
    nearest_lats = [c[1] for c in candidates[:top_n]]
    nearest_lats.sort()
    median_lat = nearest_lats[len(nearest_lats) // 2]

    edge_point = carla.Location(
        x=ref_loc.x + median_lat * right_x,
        y=ref_loc.y + median_lat * right_y,
        z=ref_loc.z)
    return median_lat, edge_point


class AutoPaintStateMachine:
    """Auto-paint state machine (same as main script)."""
    STATE_CONVERGING = "CONVERGING"
    STATE_STABILIZED = "STABILIZED"
    STATE_PAINTING = "PAINTING"
    GRACE_LIMIT = 15
    STABILIZED_GRACE = 10
    CURV_LO = 0.004
    CURV_HI = 0.010
    TOL_ENTER_CURVE = 0.55
    TOL_EXIT_CURVE = 0.80

    def __init__(self, target_dist=3.0, tolerance_enter=0.3,
                 tolerance_exit=0.45, stability_frames=30, min_speed=1.0):
        self.target_dist = target_dist
        self.tolerance_enter = tolerance_enter
        self.tolerance_exit = tolerance_exit
        self.stability_frames = stability_frames
        self.min_speed = min_speed
        self.state = self.STATE_CONVERGING
        self._stable_count = 0
        self._grace_count = 0

    def _adaptive_tolerances(self, poly_coeff_a):
        if poly_coeff_a is None:
            return self.tolerance_enter, self.tolerance_exit
        curv = abs(poly_coeff_a)
        if curv <= self.CURV_LO:
            return self.tolerance_enter, self.tolerance_exit
        if curv >= self.CURV_HI:
            return self.TOL_ENTER_CURVE, self.TOL_EXIT_CURVE
        t = (curv - self.CURV_LO) / (self.CURV_HI - self.CURV_LO)
        te = self.tolerance_enter + t * (self.TOL_ENTER_CURVE - self.tolerance_enter)
        tx = self.tolerance_exit + t * (self.TOL_EXIT_CURVE - self.tolerance_exit)
        return te, tx

    def update(self, nozzle_edge_dist, speed, poly_coeff_a=None):
        tol_enter, tol_exit = self._adaptive_tolerances(poly_coeff_a)
        error = abs(nozzle_edge_dist - self.target_dist)
        in_enter = error < tol_enter
        in_exit = error < tol_exit
        speed_ok = speed > self.min_speed

        if self.state == self.STATE_CONVERGING:
            if in_enter and speed_ok:
                self.state = self.STATE_STABILIZED
                self._stable_count = 0
                self._grace_count = 0
        elif self.state == self.STATE_STABILIZED:
            if not speed_ok:
                self.state = self.STATE_CONVERGING
                self._stable_count = 0
                self._grace_count = 0
            elif not in_exit:
                self._grace_count += 1
                if self._grace_count >= self.STABILIZED_GRACE:
                    self.state = self.STATE_CONVERGING
                    self._stable_count = 0
                    self._grace_count = 0
            else:
                self._grace_count = 0
                self._stable_count += 1
                if self._stable_count >= self.stability_frames:
                    self.state = self.STATE_PAINTING
                    self._grace_count = 0
        elif self.state == self.STATE_PAINTING:
            if not speed_ok:
                self.state = self.STATE_CONVERGING
                self._stable_count = 0
                self._grace_count = 0
            elif not in_exit:
                self._grace_count += 1
                if self._grace_count >= self.GRACE_LIMIT:
                    self.state = self.STATE_CONVERGING
                    self._stable_count = 0
                    self._grace_count = 0
            else:
                self._grace_count = 0

        return self.state == self.STATE_PAINTING


def set_weather(world, preset_name):
    """Apply a weather preset to the CARLA world."""
    preset = WEATHER_PRESETS.get(preset_name)
    if preset is None:
        print(f"  Unknown weather preset: {preset_name}")
        return
    weather = world.get_weather()
    for k, v in preset.items():
        setattr(weather, k, v)
    world.set_weather(weather)
    print(f"  Weather set to: {preset_name}")


def resolve_perception_mode(mode_str):
    """Convert CLI string to PerceptionMode constant."""
    m = mode_str.upper()
    if m == 'GT':
        return PerceptionMode.GT
    elif m in ('VLLINET', 'VLLiNet'):
        return PerceptionMode.VLLINET
    elif m == 'LUNA':
        return PerceptionMode.LUNA
    else:
        raise ValueError(f"Unknown perception mode: {mode_str}")


def _write_distance_comparison(run_dir, perception_mode):
    """Read framelog CSV and write a nozzle-edge distance comparison summary.

    Compares perception-based nozzle_edge_dist vs gt_nozzle_edge_dist
    for frames where both are valid (not 999.0).
    """
    import glob as _glob
    framelog_files = _glob.glob(os.path.join(run_dir, 'framelog_*.csv'))
    if not framelog_files:
        print("  No framelog found for distance comparison.")
        return

    framelog_path = sorted(framelog_files)[-1]  # latest
    percept_dists = []
    gt_dists = []
    errors = []

    with open(framelog_path, 'r') as f:
        import csv as _csv
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                nd = float(row.get('nozzle_edge_dist', 999))
                gd = float(row.get('gt_nozzle_edge_dist', 999))
                if nd < 900 and gd < 900:
                    percept_dists.append(nd)
                    gt_dists.append(gd)
                    errors.append(nd - gd)
            except (ValueError, TypeError):
                continue

    if len(errors) < 10:
        print(f"  Distance comparison: insufficient valid frames ({len(errors)})")
        return

    errors_np = np.array(errors)
    percept_np = np.array(percept_dists)
    gt_np = np.array(gt_dists)

    summary_path = os.path.join(run_dir, 'distance_comparison.csv')
    with open(summary_path, 'w', newline='') as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['perception_mode', str(perception_mode)])
        w.writerow(['valid_frames', len(errors)])
        w.writerow(['percept_mean_dist', f"{np.mean(percept_np):.4f}"])
        w.writerow(['gt_mean_dist', f"{np.mean(gt_np):.4f}"])
        w.writerow(['error_mean', f"{np.mean(errors_np):.4f}"])
        w.writerow(['error_std', f"{np.std(errors_np):.4f}"])
        w.writerow(['error_median', f"{np.median(errors_np):.4f}"])
        w.writerow(['error_abs_mean', f"{np.mean(np.abs(errors_np)):.4f}"])
        w.writerow(['error_abs_max', f"{np.max(np.abs(errors_np)):.4f}"])
        w.writerow(['error_rmse', f"{np.sqrt(np.mean(errors_np**2)):.4f}"])

    print(f"\n  [Distance Comparison] {len(errors)} valid frames")
    print(f"  Percept mean: {np.mean(percept_np):.3f}m  "
          f"GT mean: {np.mean(gt_np):.3f}m")
    print(f"  Error: mean={np.mean(errors_np):.3f}m  "
          f"std={np.std(errors_np):.3f}m  "
          f"RMSE={np.sqrt(np.mean(errors_np**2)):.3f}m")
    print(f"  Saved: {summary_path}")


def run_experiment(mode_str, weather_name, total_frames, spawn_id,
                   warmup_frames=150, target_distance=0):
    """Run a single automated experiment.

    Args:
        mode_str: 'GT', 'VLLiNet', or 'LUNA'
        weather_name: weather preset name
        total_frames: number of eval-recording frames to collect
        spawn_id: spawn point ID
        warmup_frames: frames to drive before starting eval recording
        target_distance: stop after this many meters (0=use frames)
    """
    perception_mode = resolve_perception_mode(mode_str)
    sp = SPAWN_POINTS[spawn_id]

    stop_mode = f"Distance: {target_distance:.0f}m" if target_distance > 0 else f"Frames: {total_frames}"
    print("=" * 60)
    print(f"  AutoStripe Experiment Runner")
    print(f"  Mode: {perception_mode}  Weather: {weather_name}")
    print(f"  {stop_mode}  Spawn: #{spawn_id}")
    print("=" * 60)

    actors = []
    try:
        # 1. Setup scene
        scene = setup_scene_v2(
            map_name=sp.get('map', 'Town05'),
            spawn_x=sp['x'], spawn_y=sp['y'],
            spawn_z=sp['z'], spawn_yaw=sp['yaw'])
        actors = scene['actors']
        world = scene['world']
        vehicle = scene['vehicle']

        # 2. Set weather
        set_weather(world, weather_name)

        # 3. Initialize modules
        perception = PerceptionPipeline(
            img_w=FRONT_CAM_W, img_h=FRONT_CAM_H, fov_deg=90.0,
            perception_mode=perception_mode)
        planner = VisionPathPlanner(
            line_offset=3.0, nozzle_arm=2.0, smooth_window=5)
        planner_gt = VisionPathPlanner(
            line_offset=3.0, nozzle_arm=2.0, smooth_window=5)
        controller = MarkerVehicleV2(vehicle, wheelbase=2.875, kdd=3.0)

        # 4. Auto-paint state machine
        auto_paint = AutoPaintStateMachine(
            target_dist=3.0, tolerance_enter=0.3, tolerance_exit=0.45,
            stability_frames=30, min_speed=1.0)

        # 5. Evaluator + frame logger
        evaluator = TrajectoryEvaluator(scene['map'])
        frame_logger = FrameLogger()

        # Paint trail tracking (no actual drawing in headless mode)
        paint_trail = []
        gt_paint_trail = []  # GT reference trail (same nozzle pos, GT edges)
        last_nozzle_loc = None
        painting_enabled = False

        # Poly smoothing state
        poly_dist = None
        poly_coeffs = None
        poly_dist_history = []
        POLY_SMOOTH_WINDOW = 10
        PERCEPT_INTERVAL = 3
        cached_result = None
        cached_road_mask = None
        cached_gt_road_mask = None

        # 6. Warm up sensors
        print(f"  Warming up sensors ({warmup_frames} frames)...")
        for i in range(30):
            time.sleep(0.05)

        # 7. Create run directory + start eval recording
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        map_name = sp.get('map', 'Town05')
        run_name = f"run_{perception_mode}_{weather_name}_{map_name}_{run_ts}"
        run_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'evaluation', run_name)
        os.makedirs(run_dir, exist_ok=True)
        evaluator.set_output_dir(run_dir)

        # Write experiment metadata
        with open(os.path.join(run_dir, 'experiment_info.txt'), 'w') as f:
            f.write(f"Perception Mode: {perception_mode}\n")
            f.write(f"Weather: {weather_name}\n")
            f.write(f"Map: {sp.get('map', 'Town05')}\n")
            f.write(f"Spawn: #{spawn_id} (x={sp['x']}, y={sp['y']}, z={sp['z']}, yaw={sp['yaw']})\n")
            f.write(f"Frames: {total_frames} eval ({warmup_frames} warmup)\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"  Output: {run_dir}")
        print(f"  Warmup phase: {warmup_frames} frames before eval starts")
        print("  Running...\n")

        # --- Main loop ---
        frame_count = 0
        eval_frame_count = 0
        eval_recording = False
        eval_trail_start_idx = 0
        last_time = time.time()
        fps_history = []
        _eval_cum_dist = 0.0
        _prev_x, _prev_y = 0.0, 0.0

        while True:
            frame_count += 1
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps_history.append(1.0 / dt)
                if len(fps_history) > 30:
                    fps_history.pop(0)

            # --- Read sensor data ---
            with scene['_semantic_lock']:
                sem_data = scene['_semantic_data']['image']
                cs_data = scene['_semantic_data'].get('cityscapes')
            with scene['_depth_lock']:
                depth_data = scene['_depth_data']['image']
            with scene['_rgb_front_lock']:
                rgb_front = scene['_rgb_front_data']['image']

            if sem_data is None or depth_data is None:
                time.sleep(0.05)
                continue

            # --- Perception ---
            run_percept = (frame_count % PERCEPT_INTERVAL == 1) or cached_result is None
            if run_percept:
                cam_tf = scene['semantic_cam'].get_transform()
                result = perception.process_frame(
                    sem_data, depth_data, cam_tf,
                    cityscapes_bgra=cs_data,
                    rgb_bgra=rgb_front)
                cached_result = result
                cached_road_mask = result[2]
                cached_gt_road_mask = result[7] if len(result) > 7 else None
            else:
                result = cached_result

            left_world, right_world, road_mask, left_px, right_px = result[:5]
            road_mask = cached_road_mask
            gt_right_world = result[5] if len(result) > 5 else None
            gt_right_px = result[6] if len(result) > 6 else None
            gt_road_mask = cached_gt_road_mask

            # --- Planning ---
            veh_tf = vehicle.get_transform()

            if poly_dist is not None:
                nozzle_dist_est = poly_dist - planner.nozzle_arm
                planner.set_dynamic_offset(nozzle_dist_est, poly_coeffs=poly_coeffs)

            driving_coords, _ = planner.update(right_world, veh_tf)

            # GT reference path
            driving_coords_gt = []
            if gt_right_world is not None:
                driving_coords_gt, _ = planner_gt.update(gt_right_world, veh_tf)

            # Polynomial extrapolation
            poly_dist_raw, poly_coeffs = planner.estimate_nozzle_edge_distance(
                right_world, veh_tf)
            if poly_dist_raw is not None:
                poly_dist_history.append(poly_dist_raw)
                if len(poly_dist_history) > POLY_SMOOTH_WINDOW:
                    poly_dist_history.pop(0)
                poly_dist = float(np.median(poly_dist_history))
            else:
                poly_dist = None

            # --- Control: auto-drive ---
            if poly_dist is not None:
                lateral_error = (poly_dist - planner.nozzle_arm) - 3.0
                controller.set_lateral_error(lateral_error)
            controller.update_path(driving_coords)
            controller.step()

            # --- Nozzle + distance ---
            nozzle_loc = get_nozzle_position(vehicle)
            veh_vel = vehicle.get_velocity()
            speed = math.sqrt(veh_vel.x**2 + veh_vel.y**2)

            nozzle_raised = carla.Location(
                x=nozzle_loc.x, y=nozzle_loc.y, z=nozzle_loc.z + 0.3)
            edge_dist_r, nozzle_edge_pt = compute_point_edge_distance(
                nozzle_raised, right_world, veh_tf)

            # GT nozzle-edge distance (same nozzle, GT edges)
            gt_edge_dist_r = 999.0
            if gt_right_world:
                gt_edge_dist_r, _ = compute_point_edge_distance(
                    nozzle_raised, gt_right_world, veh_tf)

            # --- Auto-paint state machine ---
            if poly_dist is not None:
                dist_for_sm = poly_dist - planner.nozzle_arm
            else:
                dist_for_sm = edge_dist_r
            _poly_a = poly_coeffs[0] if poly_coeffs is not None else None
            should_paint = auto_paint.update(dist_for_sm, speed, poly_coeff_a=_poly_a)
            painting_enabled = should_paint

            # --- Paint trail tracking (no visual drawing) ---
            if painting_enabled:
                if last_nozzle_loc is not None:
                    paint_trail.append((nozzle_loc.x, nozzle_loc.y))
                    gt_paint_trail.append((nozzle_loc.x, nozzle_loc.y))
                else:
                    if len(paint_trail) > 0:
                        paint_trail.append(None)
                        gt_paint_trail.append(None)
                    paint_trail.append((nozzle_loc.x, nozzle_loc.y))
                    gt_paint_trail.append((nozzle_loc.x, nozzle_loc.y))
                # Draw yellow line in CARLA world (visible in spectator)
                if last_nozzle_loc is not None:
                    world.debug.draw_line(
                        last_nozzle_loc, nozzle_loc,
                        thickness=0.3,
                        color=carla.Color(255, 255, 0),
                        life_time=10.0,
                        persistent_lines=True)
                last_nozzle_loc = nozzle_loc
            else:
                last_nozzle_loc = None

            # --- Spectator follow ---
            update_spectator(scene['spectator'], vehicle)

            # --- Auto-start eval recording after warmup ---
            if not eval_recording and frame_count >= warmup_frames:
                eval_recording = True
                eval_trail_start_idx = len(paint_trail)
                frame_logger = FrameLogger(output_dir=run_dir)
                frame_logger.start()
                print(f"\n  [F{frame_count}] Eval recording STARTED")
                if target_distance > 0:
                    print(f"  Collecting until {target_distance:.0f}m...\n")
                else:
                    print(f"  Collecting {total_frames} frames...\n")

            # --- Per-frame logging ---
            if eval_recording and frame_logger.active:
                eval_frame_count += 1

                # Track cumulative eval distance
                if eval_frame_count == 1:
                    _prev_x, _prev_y = veh_tf.location.x, veh_tf.location.y
                else:
                    _dx = veh_tf.location.x - _prev_x
                    _dy = veh_tf.location.y - _prev_y
                    _eval_cum_dist += math.sqrt(_dx**2 + _dy**2)
                    _prev_x, _prev_y = veh_tf.location.x, veh_tf.location.y

                _rmr = 0.0
                if road_mask is not None:
                    _rmr = float(np.count_nonzero(road_mask)) / max(1, road_mask.size)

                _pa = poly_coeffs[0] if poly_coeffs is not None else 0.0
                _pb = poly_coeffs[1] if poly_coeffs is not None else 0.0
                _pc = poly_coeffs[2] if poly_coeffs is not None else 0.0

                _lat_err = 0.0
                if poly_dist is not None:
                    _lat_err = (poly_dist - planner.nozzle_arm) - 3.0

                # Perception accuracy metrics (AI mode only)
                _mask_iou = 0.0
                _edge_mean = -1.0
                _edge_median = -1.0
                _edge_max = -1.0
                if perception.use_ai:
                    _mask_iou = compute_mask_iou(road_mask, gt_road_mask)
                    edge_dev = compute_edge_deviation(right_px, gt_right_px)
                    if edge_dev is not None:
                        _edge_mean = edge_dev['mean_px']
                        _edge_median = edge_dev['median_px']
                        _edge_max = edge_dev['max_px']

                frame_logger.log_frame({
                    'timestamp': time.time(),
                    'frame': frame_count,
                    'dt': dt,
                    'veh_x': veh_tf.location.x,
                    'veh_y': veh_tf.location.y,
                    'veh_yaw': veh_tf.rotation.yaw,
                    'speed': speed,
                    'nozzle_x': nozzle_loc.x,
                    'nozzle_y': nozzle_loc.y,
                    'nozzle_edge_dist': edge_dist_r,
                    'poly_edge_dist': poly_dist if poly_dist is not None else -1.0,
                    'driving_offset': planner.driving_offset,
                    'steer_filter': controller._effective_steer_filter,
                    'steer_cmd': 0.0,
                    'throttle_cmd': 0.0,
                    'brake_cmd': 0.0,
                    'lateral_error': _lat_err,
                    'paint_state': auto_paint.state,
                    'painting_enabled': int(painting_enabled),
                    'dash_phase': -1,
                    'perception_mode': perception.perception_mode,
                    'ai_edge_pts': len(right_world) if right_world else 0,
                    'gt_edge_pts': len(gt_right_world) if gt_right_world else 0,
                    'road_mask_ratio': _rmr,
                    'poly_coeff_a': _pa,
                    'poly_coeff_b': _pb,
                    'poly_coeff_c': _pc,
                    'inference_time_ms': perception.last_inference_ms if run_percept else -1.0,
                    'sne_time_ms': perception.last_sne_ms if run_percept else -1.0,
                    'mask_iou': _mask_iou,
                    'edge_dev_mean_px': _edge_mean,
                    'edge_dev_median_px': _edge_median,
                    'edge_dev_max_px': _edge_max,
                    'gt_nozzle_edge_dist': gt_edge_dist_r,
                })

                # --- Check stop condition ---
                if target_distance > 0:
                    if _eval_cum_dist >= target_distance:
                        print(f"\n  [F{frame_count}] Reached {_eval_cum_dist:.0f}m / {target_distance:.0f}m. Stopping...")
                        break
                elif eval_frame_count >= total_frames:
                    print(f"\n  [F{frame_count}] Reached {total_frames} eval frames. Stopping...")
                    break

            # --- Status print every 100 frames ---
            if frame_count % 100 == 0:
                fps = sum(fps_history) / max(1, len(fps_history))
                poly_d = f"{poly_dist:.1f}" if poly_dist else "N/A"
                phase = "EVAL" if eval_recording else "WARMUP"
                print(f"  [F{frame_count}] {phase} ef={eval_frame_count} "
                      f"AP:{auto_paint.state} Spd:{speed:.1f} "
                      f"Noz:{edge_dist_r:.1f}m Poly:{poly_d}m "
                      f"Off:{planner.driving_offset:.1f} FPS:{fps:.0f} "
                      f"Dist:{_eval_cum_dist:.0f}m")

        # --- End of main loop ---
        # Stop eval recording and run evaluation
        print("\n" + "=" * 60)
        print("  Stopping eval recording...")
        frame_logger.stop()

        segment = paint_trail[eval_trail_start_idx:]
        gt_segment = gt_paint_trail[eval_trail_start_idx:]

        # Eval 1: Perception-driven trail vs Map API GT
        if segment:
            print(f"\n  [Eval] Perception trail: {len(segment)} points")
            evaluator.run_evaluation(segment, vehicle.get_location())
        else:
            print("  WARNING: No paint trail collected!")

        # Eval 2: Per-frame nozzle-edge distance comparison summary
        if eval_frame_count > 0:
            _write_distance_comparison(run_dir, perception_mode)

        print(f"\n  Experiment complete!")
        print(f"  Output: {run_dir}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
        if frame_logger.active:
            frame_logger.stop()
        segment = paint_trail[eval_trail_start_idx:]
        if segment and eval_recording:
            evaluator.run_evaluation(segment, vehicle.get_location())
            _write_distance_comparison(run_dir, perception_mode)

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n  Cleaning up...")
        if 'frame_logger' in dir() and frame_logger.active:
            frame_logger.stop()
        import cv2
        cv2.destroyAllWindows()
        for actor in actors:
            if actor is not None:
                actor.destroy()
        print("  Done.")


def main():
    parser = argparse.ArgumentParser(
        description='AutoStripe Headless Experiment Runner')
    parser.add_argument('--mode', type=str, default='GT',
                        choices=['GT', 'VLLiNet', 'LUNA'],
                        help='Perception mode (default: GT)')
    parser.add_argument('--weather', type=str, default='ClearDay',
                        choices=list(WEATHER_PRESETS.keys()),
                        help='Weather preset (default: ClearDay)')
    parser.add_argument('--frames', type=int, default=6000,
                        help='Eval recording frames (default: 6000)')
    parser.add_argument('--spawn', type=int, default=2,
                        choices=list(SPAWN_POINTS.keys()),
                        help='Spawn point ID (default: 2)')
    parser.add_argument('--warmup', type=int, default=150,
                        help='Warmup frames before eval (default: 150)')
    parser.add_argument('--batch', action='store_true',
                        help='Run all remaining experiments sequentially')
    parser.add_argument('--distance', type=float, default=0,
                        help='Stop after this many meters of eval driving (0=use --frames)')
    args = parser.parse_args()

    if args.batch:
        # Run all planned experiments sequentially
        experiments = [
            ('GT',      'ClearDay'),
            ('VLLiNet', 'ClearDay'),
            ('LUNA',    'ClearDay'),
            ('LUNA',    'ClearNight'),
            ('LUNA',    'HeavyFoggyNight'),
            ('LUNA',    'HeavyRainFoggyNight'),
        ]
        for i, (mode, weather) in enumerate(experiments):
            print(f"\n{'#'*60}")
            print(f"  BATCH [{i+1}/{len(experiments)}]: {mode} + {weather}")
            print(f"{'#'*60}\n")
            try:
                run_experiment(
                    mode_str=mode,
                    weather_name=weather,
                    total_frames=args.frames,
                    spawn_id=args.spawn,
                    warmup_frames=args.warmup,
                    target_distance=args.distance,
                )
            except Exception as e:
                print(f"  BATCH ERROR: {e}")
                import traceback
                traceback.print_exc()
            time.sleep(5)  # brief pause between experiments
        print(f"\n{'#'*60}")
        print(f"  BATCH COMPLETE: {len(experiments)} experiments")
        print(f"{'#'*60}")
    else:
        run_experiment(
            mode_str=args.mode,
            weather_name=args.weather,
            total_frames=args.frames,
            spawn_id=args.spawn,
            warmup_frames=args.warmup,
            target_distance=args.distance,
        )


if __name__ == '__main__':
    main()
