#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AutoStripe V5 - Manual Painting Control with Triple Perception Mode

Based on V4, adds:
- LUNA-Net perception mode (G key cycles GT -> VLLiNet -> LUNA-Net)
- Night weather toggle (N key) for LUNA-Net low-light testing
- SNE timing tracking for LUNA-Net mode

Keyboard Controls:
  SPACE - Toggle painting ON/OFF
  TAB   - Toggle Auto/Manual drive mode
  G     - Cycle perception mode (GT -> VLLiNet -> LUNA-Net)
  N     - Toggle ClearNight weather preset
  D     - Toggle dashed/solid line mode (AUTO only)
  E     - Toggle eval recording (start/stop + framelog + GT evaluation)
  R     - Toggle video recording (front + overhead)
  WASD/Arrows - Manual drive
  Q     - Toggle reverse
  V     - Toggle spectator follow/free camera
  X     - Handbrake
  ESC   - Quit

Usage:
  1. Start CARLA: ./CarlaUE4.sh
  2. Run: python manual_painting_control_v4.py
"""

import glob
import os
import sys
import time
import math
import pygame
from pygame.locals import K_ESCAPE, K_SPACE, K_TAB, K_q, K_g, K_r, K_e, K_n
from pygame.locals import K_w, K_a, K_s, K_d, K_x, K_v
from pygame.locals import K_UP, K_DOWN, K_LEFT, K_RIGHT

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

# V4 modules
from carla_env.setup_scene_v2 import setup_scene_v2, FRONT_CAM_W, FRONT_CAM_H
from carla_env.setup_scene import update_spectator
from perception.perception_pipeline import PerceptionPipeline, PerceptionMode
from planning.vision_path_planner import VisionPathPlanner
from control.marker_vehicle_v2 import MarkerVehicleV2
from utils.video_recorder import VideoRecorder
from evaluation.trajectory_evaluator import TrajectoryEvaluator
from evaluation.frame_logger import FrameLogger
from evaluation.perception_metrics import compute_mask_iou, compute_edge_deviation


class AutoPaintStateMachine:
    """Auto-paint state machine: CONVERGING -> STABILIZED -> PAINTING.

    Automatically starts painting when nozzle-edge distance stabilizes
    near the target (3.0m). SPACE key acts as manual override.

    V4.2: Hysteresis (enter/exit tolerances) + grace frames to prevent
    frequent state transitions from momentary oscillations.
    V4.3: Curvature-adaptive tolerances — widen in curves where Pure Pursuit
    has systematic lateral error, preventing unnecessary paint breaks.
    """

    STATE_CONVERGING = "CONVERGING"
    STATE_STABILIZED = "STABILIZED"
    STATE_PAINTING = "PAINTING"

    GRACE_LIMIT = 15          # PAINTING grace frames before downgrade
    STABILIZED_GRACE = 10     # STABILIZED grace frames before downgrade

    # V4.3 curvature-adaptive tolerance parameters
    CURV_LO = 0.004           # below this |poly_a|, straight road (base tolerances)
    CURV_HI = 0.010           # above this |poly_a|, full curve (max tolerances)
    TOL_ENTER_CURVE = 0.55    # widened enter tolerance in curves
    TOL_EXIT_CURVE = 0.80     # widened exit tolerance in curves

    def __init__(self, target_dist=3.0, tolerance_enter=0.3,
                 tolerance_exit=0.45, stability_frames=60, min_speed=1.0):
        self.target_dist = target_dist
        self.tolerance_enter = tolerance_enter   # tight: entering STABILIZED
        self.tolerance_exit = tolerance_exit      # wide: leaving STABILIZED/PAINTING
        self.stability_frames = stability_frames
        self.min_speed = min_speed

        self.state = self.STATE_CONVERGING
        self._stable_count = 0
        self._grace_count = 0
        self._manual_override = False

    def _adaptive_tolerances(self, poly_coeff_a):
        """Compute curvature-adaptive enter/exit tolerances."""
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
        """Update state machine. Returns True if painting should be active."""
        tol_enter, tol_exit = self._adaptive_tolerances(poly_coeff_a)
        error = abs(nozzle_edge_dist - self.target_dist)
        in_enter = error < tol_enter
        in_exit = error < tol_exit
        speed_ok = speed > self.min_speed

        if self._manual_override:
            return True

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

    def manual_toggle(self):
        """SPACE key: toggle manual override, reset state machine."""
        self._manual_override = not self._manual_override
        if not self._manual_override:
            self.state = self.STATE_CONVERGING
            self._stable_count = 0
            self._grace_count = 0

    @property
    def progress(self):
        """Return stabilization progress 0.0-1.0 for HUD."""
        if self.state == self.STATE_STABILIZED:
            return min(1.0, self._stable_count / self.stability_frames)
        elif self.state == self.STATE_PAINTING:
            return 1.0
        return 0.0


def get_nozzle_position(vehicle, offset=2.0):
    """Compute nozzle position: vehicle position + right-side offset (V1 logic)."""
    veh_tf = vehicle.get_transform()
    veh_loc = vehicle.get_location()
    yaw_rad = math.radians(veh_tf.rotation.yaw)

    dx = offset * math.cos(yaw_rad + math.pi / 2)
    dy = offset * math.sin(yaw_rad + math.pi / 2)

    return carla.Location(
        x=veh_loc.x + dx,
        y=veh_loc.y + dy,
        z=veh_loc.z
    )


class ManualPaintingControl:
    """Manual painting controller + manual driving control."""

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.painting_enabled = False
        self.paint_trail = []
        self.last_nozzle_loc = None

        self.auto_drive = True
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        self.reverse = False

        # Dash mode state
        self.dash_mode = False
        self.DASH_LENGTH = 3.0   # meters of paint
        self.GAP_LENGTH = 3.0    # meters of gap
        self._dash_accum = 0.0   # accumulated distance in current phase
        self._dash_painting = True  # True=paint phase, False=gap phase

    def toggle_painting(self):
        self.painting_enabled = not self.painting_enabled
        # Reset dash accumulator on paint toggle
        self._dash_accum = 0.0
        self._dash_painting = True
        status = "ON" if self.painting_enabled else "OFF"
        print(f"\n{'='*50}")
        print(f"  Paint: {status}")
        print(f"{'='*50}\n")
        return self.painting_enabled

    def toggle_dash_mode(self):
        """D key: toggle dashed/solid line mode."""
        self.dash_mode = not self.dash_mode
        self._dash_accum = 0.0
        self._dash_painting = True
        mode = "DASHED" if self.dash_mode else "SOLID"
        print(f"\n{'='*50}")
        print(f"  Line: {mode} ({self.DASH_LENGTH:.0f}m/{self.GAP_LENGTH:.0f}m)")
        print(f"{'='*50}\n")

    def paint_line(self, world, nozzle_loc):
        if not self.painting_enabled:
            self.last_nozzle_loc = None
            return

        if self.dash_mode and self.last_nozzle_loc is not None:
            # Accumulate distance traveled
            seg_dist = math.sqrt(
                (nozzle_loc.x - self.last_nozzle_loc.x)**2 +
                (nozzle_loc.y - self.last_nozzle_loc.y)**2)
            self._dash_accum += seg_dist

            # Check phase transition
            threshold = self.DASH_LENGTH if self._dash_painting else self.GAP_LENGTH
            if self._dash_accum >= threshold:
                self._dash_accum -= threshold
                self._dash_painting = not self._dash_painting
                if not self._dash_painting:
                    # Entering gap: insert None marker for trail discontinuity
                    self.paint_trail.append(None)

            if self._dash_painting:
                # Paint phase: draw line segment
                world.debug.draw_line(
                    self.last_nozzle_loc, nozzle_loc,
                    thickness=0.3,
                    color=carla.Color(255, 255, 0),
                    life_time=1000.0,
                    persistent_lines=True
                )
                self.paint_trail.append((nozzle_loc.x, nozzle_loc.y))
            # Gap phase: no drawing, just track position

        elif self.last_nozzle_loc is not None:
            # Solid mode: always draw
            world.debug.draw_line(
                self.last_nozzle_loc, nozzle_loc,
                thickness=0.3,
                color=carla.Color(255, 255, 0),
                life_time=1000.0,
                persistent_lines=True
            )
            self.paint_trail.append((nozzle_loc.x, nozzle_loc.y))
        else:
            if len(self.paint_trail) > 0:
                self.paint_trail.append(None)
            self.paint_trail.append((nozzle_loc.x, nozzle_loc.y))

        self.last_nozzle_loc = nozzle_loc

    def toggle_drive_mode(self):
        self.auto_drive = not self.auto_drive
        mode = "AUTO" if self.auto_drive else "MANUAL"
        print(f"\n{'='*50}")
        print(f"  Drive: {mode}")
        print(f"{'='*50}\n")
        return self.auto_drive

    def toggle_reverse(self):
        self.reverse = not self.reverse
        status = "ON" if self.reverse else "OFF"
        print(f"\n{'='*50}")
        print(f"  Reverse: {status}")
        print(f"{'='*50}\n")

    def update_manual_control(self, keys):
        if keys[K_w] or keys[K_UP]:
            self.throttle = min(1.0, self.throttle + 0.1)
        else:
            self.throttle = 0.0

        if keys[K_s] or keys[K_DOWN]:
            self.brake = min(1.0, self.brake + 0.2)
        else:
            self.brake = 0.0

        if keys[K_a] or keys[K_LEFT]:
            if self.steer > 0:
                self.steer = 0
            else:
                self.steer = max(-0.7, self.steer - 0.05)
        elif keys[K_d] or keys[K_RIGHT]:
            if self.steer < 0:
                self.steer = 0
            else:
                self.steer = min(0.7, self.steer + 0.05)
        else:
            self.steer = 0.0

        if keys[K_x]:
            self.brake = 1.0
            self.throttle = 0.0

    def apply_manual_control(self):
        control = carla.VehicleControl()
        control.throttle = self.throttle
        control.steer = self.steer
        control.brake = self.brake
        control.reverse = self.reverse
        self.vehicle.apply_control(control)


def draw_status_overlay(img, painting_enabled, frame_count, speed, edge_dist_r,
                        drive_mode="AUTO", throttle=0.0, steer=0.0, brake=0.0,
                        perception_mode="AI", poly_dist=None, fps=0.0,
                        veh_x=0.0, veh_y=0.0, veh_yaw=0.0,
                        ap_state_str="", ap_color=(255,255,255),
                        line_mode_str="", driving_offset=0.0,
                        steer_filter=0.0, is_recording=False,
                        eval_recording=False, spectator_follow=True):
    """Draw status info on overhead image."""
    h, w = img.shape[:2]

    # Drive mode
    mode_text = f"MODE: {drive_mode}"
    mode_color = (0, 255, 255) if drive_mode == "AUTO" else (255, 0, 255)
    cv2.putText(img, mode_text, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, mode_color, 4)

    # Perception mode
    perc_text = f"PERC: {perception_mode}"
    if perception_mode == "GT":
        perc_color = (0, 255, 0)
    elif perception_mode == "LUNA":
        perc_color = (0, 200, 255)
    else:
        perc_color = (255, 0, 255)
    cv2.putText(img, perc_text, (500, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, perc_color, 4)

    # FPS
    fps_color = (0, 255, 0) if fps >= 15 else (0, 255, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(img, f"FPS: {fps:.0f}", (1000, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, fps_color, 4)

    # Vehicle coordinates (right side)
    cv2.putText(img, f"x={veh_x:.1f}  y={veh_y:.1f}  yaw={veh_yaw:.1f}",
                (w - 900, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Paint status
    status_text = "PAINT: ON" if painting_enabled else "PAINT: OFF"
    status_color = (0, 255, 0) if painting_enabled else (0, 0, 255)
    cv2.putText(img, status_text, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, status_color, 4)

    # Vehicle status
    cv2.putText(img, f"Speed: {speed:.1f} m/s", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(img, f"Nozzle-Edge: {edge_dist_r:.1f}m", (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)

    # Polynomial distance
    if poly_dist is not None:
        cv2.putText(img, f"Poly-Edge: {poly_dist:.1f}m", (20, 305),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 255), 4)
    else:
        cv2.putText(img, "Poly-Edge: N/A", (20, 305),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (150, 150, 150), 3)

    # AutoPaint state
    if ap_state_str:
        # BGR color for cv2
        ap_bgr = (ap_color[2], ap_color[1], ap_color[0])
        cv2.putText(img, f"AutoPaint: {ap_state_str}", (20, 370),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, ap_bgr, 3)

    # Line mode
    if line_mode_str:
        cv2.putText(img, f"Line: {line_mode_str}", (20, 425),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 200, 0), 3)

    # Offset + SteerFilter
    cv2.putText(img, f"Offset: {driving_offset:.1f}m  SF: {steer_filter:.2f}",
                (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (200, 200, 200), 3)

    # Recording indicators (right side)
    indicator_x = w - 400
    if is_recording:
        cv2.putText(img, "REC", (indicator_x, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
    if eval_recording:
        cv2.putText(img, "EVAL REC", (indicator_x, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 4)

    # Camera mode
    cam_str = "Cam: FOLLOW" if spectator_follow else "Cam: FREE"
    cam_color = (255, 200, 0) if spectator_follow else (0, 150, 255)
    cv2.putText(img, cam_str, (indicator_x, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, cam_color, 3)

    # Manual control status
    if drive_mode == "MANUAL":
        cv2.putText(img, f"Thr:{throttle:.2f} Str:{steer:.2f} Brk:{brake:.2f}",
                    (20, 535), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)

    # Help text
    help_y = h - 200
    cv2.putText(img, "Controls:", (20, help_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
    cv2.putText(img, "TAB=Mode SPACE=Paint G=Perc R=Rec", (20, help_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    cv2.putText(img, "D=Dash E=EvalRec N=Night V=Cam", (20, help_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    cv2.putText(img, "Q=Reverse X=Brake WASD=Drive ESC=Quit", (20, help_y + 125),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    return img


def draw_driving_path(world, driving_coords, vehicle_tf=None):
    """Draw blue driving path dots (slope-aware z)."""
    if len(driving_coords) < 1 or vehicle_tf is None:
        return

    veh_loc = vehicle_tf.location
    fwd = vehicle_tf.get_forward_vector()
    fwd_h = math.sqrt(fwd.x**2 + fwd.y**2)
    slope = fwd.z / fwd_h if fwd_h > 1e-6 else 0.0

    for i in range(0, len(driving_coords), 2):
        dx = driving_coords[i][0] - veh_loc.x
        dy = driving_coords[i][1] - veh_loc.y
        lon = (dx * fwd.x / fwd_h + dy * fwd.y / fwd_h) if fwd_h > 1e-6 else 0.0
        z = veh_loc.z + lon * slope + 0.3

        pt = carla.Location(x=driving_coords[i][0],
                           y=driving_coords[i][1], z=z)
        world.debug.draw_point(pt, size=0.1,
                              color=carla.Color(0, 0, 255),
                              life_time=0.1)


def draw_poly_curve(world, coeffs, vehicle_tf, num_points=20, max_lon=20.0):
    """Draw polynomial extrapolation curve as magenta dots in 3D."""
    if coeffs is None or vehicle_tf is None:
        return

    veh_loc = vehicle_tf.location
    yaw = math.radians(vehicle_tf.rotation.yaw)
    fwd_x = math.cos(yaw)
    fwd_y = math.sin(yaw)
    right_x = -fwd_y
    right_y = fwd_x

    fwd_vec = vehicle_tf.get_forward_vector()
    fwd_h = math.sqrt(fwd_vec.x**2 + fwd_vec.y**2)
    slope = fwd_vec.z / fwd_h if fwd_h > 1e-6 else 0.0

    a, b, c = coeffs
    for i in range(num_points + 1):
        lon = max_lon * i / num_points
        lat = a * lon**2 + b * lon + c

        wx = veh_loc.x + lon * fwd_x + lat * right_x
        wy = veh_loc.y + lon * fwd_y + lat * right_y
        wz = veh_loc.z + lon * slope + 0.3

        world.debug.draw_point(
            carla.Location(x=wx, y=wy, z=wz),
            size=0.06,
            color=carla.Color(255, 0, 255),
            life_time=0.1)


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
        z=ref_loc.z
    )

    dist = math.sqrt((edge_point.x - ref_loc.x)**2 +
                     (edge_point.y - ref_loc.y)**2)
    return dist, edge_point


def world_to_pixel(wx, wy, veh_tf, img_w=1800, img_h=1600, cam_h=25.0):
    """Project world coordinates to overhead camera pixel coordinates."""
    vx = veh_tf.location.x
    vy = veh_tf.location.y
    yaw = math.radians(veh_tf.rotation.yaw)

    dx = wx - vx
    dy = wy - vy

    local_fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
    local_right = -dx * math.sin(yaw) + dy * math.cos(yaw)

    scale_x = (img_w / 2.0) / cam_h
    scale_y = (img_h / 2.0) / cam_h

    px = int(img_w / 2.0 + local_right * scale_x)
    py = int(img_h / 2.0 - local_fwd * scale_y)
    return px, py


def world_to_front_pixel(wx, wy, wz, cam_tf,
                         img_w=FRONT_CAM_W, img_h=FRONT_CAM_H, fov=90):
    """Project world coordinates to front camera pixel coordinates."""
    f = (img_w / 2.0) / math.tan(math.radians(fov / 2.0))

    dx = wx - cam_tf.location.x
    dy = wy - cam_tf.location.y
    dz = wz - cam_tf.location.z

    yaw = math.radians(cam_tf.rotation.yaw)
    pitch = math.radians(cam_tf.rotation.pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    x1 = dx * cy + dy * sy
    y1 = -dx * sy + dy * cy
    z1 = dz

    fwd = x1 * cp + z1 * sp
    up = -x1 * sp + z1 * cp
    right = y1

    if fwd < 0.5:
        return None

    px = int(img_w / 2.0 + f * right / fwd)
    py = int(img_h / 2.0 - f * up / fwd)
    return px, py


# --- Spawn point presets (Town05 highway) ---
SPAWN_POINTS = {
    1: {"x": 10,     "y": -210,   "z": 1.85, "yaw": 180,   "desc": "Highway straight (original)"},
    2: {"x": -247.1, "y": -32.3,  "z": 10.0, "yaw": 90.1,  "desc": ""},
    3: {"x": 211.0,  "y": -13.6, "z": 0.50, "yaw": -91.2, "desc": ""},
    4: {"x": 0.0,    "y": 208.8, "z": 9.00, "yaw": 0.0,   "desc": ""},
    5: {"x": 90.0,   "y": -190.0, "z": 0.50, "yaw": -0.2,  "desc": "Highway SP1 reverse into right curve"},
    6: {"x": 94.6,   "y": -146.1, "z": 0.30, "yaw": 189.0, "desc": "Post-turn westbound (from SP4 right curve)"},
    7: {"x": 60.0,   "y": 209.0,  "z": 9.00, "yaw": 0.0,   "desc": "Straight before right curve (test zone)"},
    8: {"x": 210.0,  "y": 75.0,   "z": 9.00, "yaw": -90.0, "desc": "After right curve, straight start"},
    9: {"x": 128.0,  "y": 37.0,   "z": 20.0, "yaw": 0.0,   "desc": "Town04 highway", "map": "Town04"},
}
ACTIVE_SPAWN = 2  # <-- Change this to switch spawn point


def main():
    sp = SPAWN_POINTS[ACTIVE_SPAWN]
    print("=" * 60)
    print("  AutoStripe V5 - Triple Perception + Manual Painting Control")
    print(f"  Spawn #{ACTIVE_SPAWN}: {sp['desc']}")
    print(f"  Location: x={sp['x']}, y={sp['y']}, yaw={sp['yaw']}")
    print("=" * 60)
    print("\nControls:")
    print("  SPACE - Toggle painting ON/OFF")
    print("  TAB   - Toggle Auto/Manual drive")
    print("  G     - Cycle perception (GT/VLLiNet/LUNA)")
    print("  N     - Toggle ClearNight weather")
    print("  R     - Toggle video recording")
    print("  ESC   - Quit")
    print("\nStarting...\n")

    # Initialize pygame (keyboard + front view display)
    pygame.init()
    pg_screen = pygame.display.set_mode((FRONT_CAM_W, FRONT_CAM_H))
    pygame.display.set_caption("AutoStripe V4 - Front View + Control")
    pg_font = pygame.font.SysFont("monospace", 18)
    pg_clock = pygame.time.Clock()

    actors = []
    # V5: Three-mode perception (GT / VLLiNet / LUNA-Net)
    PERCEPTION_MODES = [PerceptionMode.GT, PerceptionMode.VLLINET, PerceptionMode.LUNA]
    perception_mode = PerceptionMode.LUNA  # V5 default
    night_weather = False  # N key toggle

    try:
        # 1. Setup scene
        scene = setup_scene_v2(
            map_name=sp.get('map', 'Town05'),
            spawn_x=sp['x'], spawn_y=sp['y'],
            spawn_z=sp['z'], spawn_yaw=sp['yaw'])
        actors = scene['actors']
        world = scene['world']
        vehicle = scene['vehicle']

        # 2. Initialize modules — start in configured perception mode
        perception = PerceptionPipeline(
            img_w=FRONT_CAM_W, img_h=FRONT_CAM_H, fov_deg=90.0,
            perception_mode=perception_mode
        )
        planner = VisionPathPlanner(
            line_offset=3.0, nozzle_arm=2.0, smooth_window=5
        )
        planner_gt = VisionPathPlanner(
            line_offset=3.0, nozzle_arm=2.0, smooth_window=5
        )
        controller = MarkerVehicleV2(vehicle, wheelbase=2.875, kdd=3.0)

        # 3. Initialize painting controller
        paint_ctrl = ManualPaintingControl(vehicle)

        # 3b. Auto-paint state machine
        auto_paint = AutoPaintStateMachine(
            target_dist=3.0, tolerance_enter=0.3, tolerance_exit=0.45,
            stability_frames=30, min_speed=1.0)

        # 3c. Trajectory evaluator (E key toggle: start/stop recording)
        evaluator = TrajectoryEvaluator(scene['map'])
        eval_recording = False
        eval_trail_start_idx = 0

        # 3d. Frame logger (records per-frame CSV during eval recording)
        frame_logger = FrameLogger()

        # 4. Warm up sensors
        print("Warming up sensors (30 frames)...")
        for _ in range(30):
            time.sleep(0.05)
        print("Sensors ready.\n")

        # 4b. Video recorder (R key to toggle)
        recorder = VideoRecorder()

        # 5. Main loop
        frame_count = 0
        spectator_follow = True
        poly_dist = None
        poly_coeffs = None
        poly_dist_history = []
        POLY_SMOOTH_WINDOW = 10
        last_time = time.time()
        fps = 0.0
        fps_history = []
        FPS_SMOOTH_WINDOW = 30
        PERCEPT_INTERVAL = 3  # run VLLiNet every N frames
        cached_result = None
        cached_road_mask = None
        cached_gt_road_mask = None

        print("=" * 60)
        print("  System ready! Press SPACE to start painting")
        print("  Press G to cycle perception: GT/VLLiNet/LUNA")
        print("  Press N to toggle ClearNight weather")
        print("=" * 60)

        while True:
            frame_count += 1
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps_history.append(1.0 / dt)
                if len(fps_history) > FPS_SMOOTH_WINDOW:
                    fps_history.pop(0)
                fps = sum(fps_history) / len(fps_history)

            # --- Read sensor data ---
            with scene['_semantic_lock']:
                sem_data = scene['_semantic_data']['image']
                cs_data = scene['_semantic_data'].get('cityscapes')
            with scene['_depth_lock']:
                depth_data = scene['_depth_data']['image']
            with scene['_frame_lock']:
                overhead_data = scene['_frame_data']['image']
            with scene['_rgb_front_lock']:
                rgb_front = scene['_rgb_front_data']['image']

            if sem_data is None or depth_data is None:
                time.sleep(0.05)
                continue

            # --- Perception: extract road edges (skip-frame) ---
            run_percept = (frame_count % PERCEPT_INTERVAL == 1) or cached_result is None
            if run_percept:
                cam_tf = scene['semantic_cam'].get_transform()
                result = perception.process_frame(
                    sem_data, depth_data, cam_tf,
                    cityscapes_bgra=cs_data,
                    rgb_bgra=rgb_front
                )
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

            # --- Planning: generate driving path (AI edges) ---
            veh_tf = vehicle.get_transform()

            # Dynamic offset: adjust driving_offset before path generation
            # poly_dist is vehicle-center-to-edge; subtract nozzle_arm to get nozzle-to-edge
            # V4.3: pass poly_coeffs for curvature feedforward (uses previous frame, 1-frame lag OK)
            if poly_dist is not None:
                nozzle_dist_est = poly_dist - planner.nozzle_arm
                planner.set_dynamic_offset(nozzle_dist_est, poly_coeffs=poly_coeffs)

            driving_coords, _ = planner.update(right_world, veh_tf)

            # --- Planning: GT reference path ---
            driving_coords_gt = []
            if gt_right_world is not None:
                driving_coords_gt, _ = planner_gt.update(gt_right_world, veh_tf)

            # --- Polynomial extrapolation ---
            poly_dist_raw, poly_coeffs = planner.estimate_nozzle_edge_distance(
                right_world, veh_tf)

            # Temporal smoothing (median filter)
            if poly_dist_raw is not None:
                poly_dist_history.append(poly_dist_raw)
                if len(poly_dist_history) > POLY_SMOOTH_WINDOW:
                    poly_dist_history.pop(0)
                poly_dist = float(np.median(poly_dist_history))
            else:
                poly_dist = None

            # --- Visualize: right edge red dots (skip in AI mode to avoid polluting camera) ---
            if not perception.use_ai:
                _draw_right_edge_dots(world, right_world, veh_tf)

            # --- Control ---
            if paint_ctrl.auto_drive:
                # Adaptive steer filter based on nozzle-edge error
                if poly_dist is not None:
                    lateral_error = (poly_dist - planner.nozzle_arm) - 3.0
                    controller.set_lateral_error(lateral_error)
                controller.update_path(driving_coords)
                controller.step()
            else:
                paint_ctrl.apply_manual_control()

            # --- Nozzle position (painting deferred until after distance computation) ---
            nozzle_loc = get_nozzle_position(vehicle)

            # --- Visualize: driving path + poly curve (skip in AI mode) ---
            if not perception.use_ai:
                draw_driving_path(world, driving_coords, veh_tf)
                draw_poly_curve(world, poly_coeffs, veh_tf)

            # --- Spectator follow ---
            if spectator_follow:
                update_spectator(scene['spectator'], vehicle)

            # --- Compute distances ---
            veh_vel = vehicle.get_velocity()
            speed = math.sqrt(veh_vel.x**2 + veh_vel.y**2)

            nozzle_edge_pt = None
            nozzle_raised = None

            # Nozzle-Edge distance (green line)
            nozzle_raised = carla.Location(
                x=nozzle_loc.x, y=nozzle_loc.y, z=nozzle_loc.z + 0.3)
            edge_dist_r, nozzle_edge_pt = compute_point_edge_distance(
                nozzle_raised, right_world, veh_tf)

            nozzle_mid = None
            if nozzle_edge_pt is not None:
                nozzle_mid = carla.Location(
                    x=(nozzle_raised.x + nozzle_edge_pt.x) / 2,
                    y=(nozzle_raised.y + nozzle_edge_pt.y) / 2,
                    z=nozzle_raised.z + 0.3)
                if not perception.use_ai:
                    world.debug.draw_line(
                        nozzle_raised, nozzle_edge_pt,
                        thickness=0.08,
                        color=carla.Color(0, 255, 0),
                        life_time=0.1)
                    world.debug.draw_string(
                        nozzle_mid, f"{edge_dist_r:.1f}m",
                        color=carla.Color(0, 255, 0),
                        life_time=0.1)

            # Poly-Edge: cyan line from first TP -> poly-extrapolated edge (like V3)
            poly_edge_pt = None
            tp_loc = None
            if poly_coeffs is not None and len(driving_coords) > 0:
                veh_tf_now = veh_tf
                yaw = math.radians(veh_tf_now.rotation.yaw)
                fwd_x = math.cos(yaw)
                fwd_y = math.sin(yaw)
                right_x = -fwd_y
                right_y = fwd_x

                # First tracking point
                tp = driving_coords[0]
                tp_wx = tp[0] if not hasattr(tp, 'x') else tp.x
                tp_wy = tp[1] if not hasattr(tp, 'y') else tp.y
                dx_tp = tp_wx - veh_tf_now.location.x
                dy_tp = tp_wy - veh_tf_now.location.y
                lon_tp = dx_tp * fwd_x + dy_tp * fwd_y

                # Slope-aware z for TP
                fwd_vec = veh_tf_now.get_forward_vector()
                fwd_h = math.sqrt(fwd_vec.x**2 + fwd_vec.y**2)
                slope = fwd_vec.z / fwd_h if fwd_h > 1e-6 else 0.0
                tp_z = veh_tf_now.location.z + lon_tp * slope + 0.3
                tp_loc = carla.Location(x=tp_wx, y=tp_wy, z=tp_z)

                # Evaluate polynomial at TP's longitudinal position
                a, b, c = poly_coeffs
                lat_at_tp = a * lon_tp**2 + b * lon_tp + c
                poly_edge_pt = carla.Location(
                    x=tp_wx + lat_at_tp * right_x,
                    y=tp_wy + lat_at_tp * right_y,
                    z=tp_z)

            # --- Auto-paint state machine ---
            # Convert poly_dist (vehicle-center-to-edge) to nozzle-to-edge
            if poly_dist is not None:
                dist_for_sm = poly_dist - planner.nozzle_arm
            else:
                dist_for_sm = edge_dist_r
            _poly_a = poly_coeffs[0] if poly_coeffs is not None else None
            should_paint = auto_paint.update(dist_for_sm, speed, poly_coeff_a=_poly_a)
            if paint_ctrl.auto_drive and not auto_paint._manual_override:
                paint_ctrl.painting_enabled = should_paint

            # --- Painting (after distance computation) ---
            paint_ctrl.paint_line(world, nozzle_loc)

            # --- Per-frame logging (only when eval_recording) ---
            if eval_recording and frame_logger.active:
                # Compute road_mask_ratio
                _rmr = 0.0
                if road_mask is not None:
                    _rmr = float(np.count_nonzero(road_mask)) / max(1, road_mask.size)

                # Poly coefficients
                _pa = poly_coeffs[0] if poly_coeffs is not None else 0.0
                _pb = poly_coeffs[1] if poly_coeffs is not None else 0.0
                _pc = poly_coeffs[2] if poly_coeffs is not None else 0.0

                # Lateral error for logging
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
                    'steer_cmd': paint_ctrl.steer,
                    'throttle_cmd': paint_ctrl.throttle,
                    'brake_cmd': paint_ctrl.brake,
                    'lateral_error': _lat_err,
                    'paint_state': auto_paint.state,
                    'painting_enabled': int(paint_ctrl.painting_enabled),
                    'dash_phase': int(paint_ctrl._dash_painting) if paint_ctrl.dash_mode else -1,
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
                })

            # --- Render overhead view ---
            overhead_img = _render_overhead(overhead_data, paint_ctrl, veh_tf, world,
                             edge_dist_r, nozzle_mid,
                             speed, frame_count, poly_dist, perception.use_ai,
                             right_world=right_world,
                             driving_coords=driving_coords,
                             poly_coeffs=poly_coeffs,
                             nozzle_raised=nozzle_raised,
                             nozzle_edge_pt=nozzle_edge_pt,
                             driving_coords_gt=driving_coords_gt,
                             poly_edge_pt=poly_edge_pt,
                             tp_loc=tp_loc, fps=fps,
                             auto_paint=auto_paint,
                             planner=planner,
                             controller=controller,
                             is_recording=recorder.is_recording,
                             eval_recording=eval_recording,
                             spectator_follow=spectator_follow,
                             perception_mode_str=perception.perception_mode)

            # --- Record overhead frame ---
            recorder.write_overhead(overhead_img)

            # --- Depth camera visualization (diagnostic) ---
            if depth_data is not None:
                from perception.depth_projector import decode_depth_image
                depth_m = decode_depth_image(depth_data)
                depth_vis = np.clip(depth_m / 50.0, 0, 1)  # normalize 0-50m
                depth_vis = (depth_vis * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
                cv2.imshow("Depth Camera", depth_color)
                cv2.waitKey(1)

            # --- Render front view in pygame ---
            _render_front_view(pg_screen, rgb_front, road_mask, scene,
                               nozzle_mid, edge_dist_r,
                               right_world, driving_coords, poly_coeffs,
                               veh_tf, perception.use_ai,
                               nozzle_raised, nozzle_edge_pt,
                               right_px=right_px, poly_dist=poly_dist,
                               driving_coords_gt=driving_coords_gt,
                               poly_edge_pt=poly_edge_pt,
                               tp_loc=tp_loc)

            # --- Event handling ---
            should_exit = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE:
                        should_exit = True
                    elif event.key == K_SPACE:
                        auto_paint.manual_toggle()
                        paint_ctrl.toggle_painting()
                    elif event.key == K_TAB:
                        paint_ctrl.toggle_drive_mode()
                    elif event.key == K_q:
                        paint_ctrl.toggle_reverse()
                    elif event.key == K_v:
                        spectator_follow = not spectator_follow
                        mode = "FOLLOW" if spectator_follow else "FREE"
                        print(f"\n  Camera: {mode}\n")
                    elif event.key == K_r:
                        recorder.toggle(
                            front_size=(FRONT_CAM_W, FRONT_CAM_H),
                            overhead_size=(1800, 1600))
                    elif event.key == K_g:
                        # V5: cycle through GT -> VLLiNet -> LUNA-Net
                        idx = PERCEPTION_MODES.index(perception_mode)
                        perception_mode = PERCEPTION_MODES[(idx + 1) % len(PERCEPTION_MODES)]
                        perception = PerceptionPipeline(
                            img_w=FRONT_CAM_W, img_h=FRONT_CAM_H,
                            fov_deg=90.0, perception_mode=perception_mode
                        )
                        cached_result = None
                        print(f"\n{'='*50}")
                        print(f"  Perception: {perception_mode}")
                        print(f"{'='*50}\n")
                    elif event.key == K_n:
                        # V5: toggle ClearNight weather
                        night_weather = not night_weather
                        weather = world.get_weather()
                        if night_weather:
                            weather.sun_altitude_angle = -30.0
                            weather.cloudiness = 10.0
                            weather.fog_density = 0.0
                        else:
                            weather.sun_altitude_angle = 5.0
                            weather.cloudiness = 10.0
                            weather.precipitation = 0.0
                            weather.precipitation_deposits = 0.0
                            weather.wind_intensity = 5.0
                            weather.fog_density = 0.0
                            weather.fog_distance = 100.0
                            weather.wetness = 0.0
                        world.set_weather(weather)
                        w_str = "ClearNight" if night_weather else "ClearDay"
                        print(f"\n{'='*50}")
                        print(f"  Weather: {w_str}")
                        print(f"{'='*50}\n")
                    elif event.key == K_e:
                        if evaluator is None:
                            print("  Evaluator not available.")
                        elif not eval_recording:
                            # 开始记录 — 创建 run 子目录
                            eval_recording = True
                            eval_trail_start_idx = len(paint_ctrl.paint_trail)
                            run_ts = time.strftime("%Y%m%d_%H%M%S")
                            run_dir = os.path.join(
                                os.path.dirname(os.path.abspath(__file__)),
                                'evaluation', f'run_{run_ts}')
                            os.makedirs(run_dir, exist_ok=True)
                            frame_logger = FrameLogger(output_dir=run_dir)
                            frame_logger.start()
                            evaluator.set_output_dir(run_dir)
                            print(f"\n{'='*50}")
                            print(f"  Eval: RECORDING started (idx={eval_trail_start_idx})")
                            print(f"  Press E again to stop and evaluate")
                            print(f"{'='*50}\n")
                        else:
                            # 停止记录，取这段 paint trail 评估
                            eval_recording = False
                            frame_logger.stop()
                            segment = paint_ctrl.paint_trail[eval_trail_start_idx:]
                            print(f"\n{'='*50}")
                            print(f"  Eval: RECORDING stopped")
                            print(f"{'='*50}")
                            evaluator.run_evaluation(
                                segment, vehicle.get_location())
                    elif event.key == K_d and paint_ctrl.auto_drive:
                        # D key: toggle dash mode (only in AUTO drive)
                        paint_ctrl.toggle_dash_mode()

            if should_exit:
                print("\nExiting...")
                break

            # Continuous key detection for manual driving
            keys = pygame.key.get_pressed()
            if not paint_ctrl.auto_drive:
                paint_ctrl.update_manual_control(keys)

            # --- Pygame HUD overlay ---
            perc_mode = perception.perception_mode
            ap_state = auto_paint.state
            if auto_paint._manual_override:
                ap_state_str = "MANUAL"
                ap_color = (255, 255, 0)
            elif ap_state == AutoPaintStateMachine.STATE_PAINTING:
                ap_state_str = "PAINTING"
                ap_color = (0, 255, 0)
            elif ap_state == AutoPaintStateMachine.STATE_STABILIZED:
                pct = int(auto_paint.progress * 100)
                ap_state_str = f"STABLE {pct}%"
                ap_color = (0, 200, 255)
            else:
                ap_state_str = "CONVERGING"
                ap_color = (255, 100, 100)
            lines = [
                ("Drive: AUTO" if paint_ctrl.auto_drive else "Drive: MANUAL",
                 (0, 255, 0) if paint_ctrl.auto_drive else (255, 255, 0)),
                (f"Perc: {perc_mode}",
                 (0, 255, 0) if perc_mode == "GT" else (0, 200, 255) if perc_mode == "LUNA" else (255, 0, 255)),
                (f"FPS: {fps:.0f}",
                 (0, 255, 0) if fps >= 15 else (0, 255, 255) if fps >= 10 else (255, 0, 0)),
                ("Paint: ON" if paint_ctrl.painting_enabled else "Paint: OFF",
                 (0, 255, 0) if paint_ctrl.painting_enabled else (150, 150, 150)),
                (f"AutoPaint: {ap_state_str}", ap_color),
                (f"Line: {'DASH' if paint_ctrl.dash_mode else 'SOLID'}"
                 + (f" ({paint_ctrl._dash_accum:.1f}m)" if paint_ctrl.dash_mode else ""),
                 (0, 200, 255) if paint_ctrl.dash_mode else (200, 200, 200)),
                (f"Speed: {speed:.1f} m/s", (255, 255, 255)),
                (f"Nozzle-Edge: {edge_dist_r:.1f}m", (0, 255, 0)),
                (f"Poly-Edge: {poly_dist:.1f}m" if poly_dist else "Poly-Edge: N/A",
                 (255, 0, 255)),
                (f"Offset: {planner.driving_offset:.1f}m  SteerF: {controller._effective_steer_filter:.2f}",
                 (200, 200, 200)),
                (f"Thr:{paint_ctrl.throttle:.1f} Str:{paint_ctrl.steer:.2f} Brk:{paint_ctrl.brake:.1f}",
                 (255, 255, 255)),
                ("REC" if recorder.is_recording else "",
                 (255, 0, 0)),
                ("EVAL REC" if eval_recording else "",
                 (255, 165, 0)),
                ("Cam: FOLLOW" if spectator_follow else "Cam: FREE",
                 (0, 200, 255) if spectator_follow else (255, 150, 0)),
                ("TAB=Mode SPACE=Paint G=Perc N=Night R=Rec", (200, 200, 200)),
                ("D=Dash E=EvalRec V=Cam WASD=Drive ESC=Quit", (200, 200, 200)),
            ]
            for i, (text, color) in enumerate(lines):
                if text:
                    surf = pg_font.render(text, True, color)
                    pg_screen.blit(surf, (10, 8 + i * 22))

            # --- Record front frame (capture pygame surface with HUD) ---
            if recorder.is_recording:
                front_arr = pygame.surfarray.array3d(pg_screen)
                front_bgr = cv2.cvtColor(
                    front_arr.swapaxes(0, 1), cv2.COLOR_RGB2BGR)
                recorder.write_front(front_bgr)

            pygame.display.flip()
            pg_clock.tick(30)

            # Periodic status print
            if frame_count % 50 == 0:
                status = "ON" if paint_ctrl.painting_enabled else "OFF"
                poly_d = f"{poly_dist:.1f}" if poly_dist else "N/A"
                gt_pts = len(driving_coords_gt) if driving_coords_gt else 0
                print(f"[F{frame_count}] Paint:{status} AP:{ap_state_str} "
                      f"Perc:{perc_mode} Spd:{speed:.1f} "
                      f"Noz:{edge_dist_r:.1f}m Poly:{poly_d}m "
                      f"Off:{planner.driving_offset:.1f} "
                      f"SF:{controller._effective_steer_filter:.2f} "
                      f"AI:{len(driving_coords)}pts GT:{gt_pts}pts")

    except KeyboardInterrupt:
        print("\nInterrupted...")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nCleaning up...")
        if 'frame_logger' in dir():
            frame_logger.stop()
        recorder.release()
        pygame.quit()
        cv2.destroyAllWindows()
        for actor in actors:
            if actor is not None:
                actor.destroy()
        print("Done.")


def _draw_right_edge_dots(world, right_world, veh_tf):
    """Draw red dots on right road edge (filtered, slope-aware)."""
    if not right_world or len(right_world) < 3:
        return

    veh_loc = veh_tf.location
    fwd = veh_tf.get_forward_vector()
    fwd_h = math.sqrt(fwd.x**2 + fwd.y**2)
    slope = fwd.z / fwd_h if fwd_h > 1e-6 else 0.0
    right_x = -fwd.y / fwd_h if fwd_h > 1e-6 else 0.0
    right_y = fwd.x / fwd_h if fwd_h > 1e-6 else 0.0

    # Median lateral filter
    lats = []
    for pt in right_world:
        dx = pt.x - veh_loc.x
        dy = pt.y - veh_loc.y
        lats.append(dx * right_x + dy * right_y)
    lats_sorted = sorted(lats)
    median_lat = lats_sorted[len(lats_sorted) // 2]

    last_drawn = None
    for pt in right_world:
        dx = pt.x - veh_loc.x
        dy = pt.y - veh_loc.y
        lon = (dx * fwd.x / fwd_h + dy * fwd.y / fwd_h) if fwd_h > 1e-6 else 0.0
        lat = dx * right_x + dy * right_y
        if lon < 0.5 or lon > 20.0:
            continue
        if abs(lat - median_lat) > 3.0:
            continue
        if last_drawn is not None:
            d = math.sqrt((pt.x - last_drawn.x)**2 + (pt.y - last_drawn.y)**2)
            if d < 2.0:
                continue
        z = veh_loc.z + lon * slope + 0.3
        world.debug.draw_point(
            carla.Location(x=pt.x, y=pt.y, z=z),
            size=0.08,
            color=carla.Color(255, 0, 0),
            life_time=0.1)
        last_drawn = pt


def _render_overhead(overhead_data, paint_ctrl, veh_tf, world,
                     edge_dist_r, nozzle_mid,
                     speed, frame_count, poly_dist, use_ai_mode,
                     perception_mode_str=None,
                     right_world=None, driving_coords=None,
                     poly_coeffs=None, nozzle_raised=None,
                     nozzle_edge_pt=None, driving_coords_gt=None,
                     poly_edge_pt=None, tp_loc=None, fps=0.0,
                     auto_paint=None, planner=None, controller=None,
                     is_recording=False, eval_recording=False,
                     spectator_follow=True):
    """Render overhead view with overlays."""
    if overhead_data is None:
        return

    img = overhead_data.copy()

    # Yellow paint trail overlay
    trail = paint_ctrl.paint_trail
    if len(trail) >= 2:
        for i in range(1, len(trail)):
            if trail[i - 1] is None or trail[i] is None:
                continue
            x0, y0 = trail[i - 1]
            x1, y1 = trail[i]
            px0, py0 = world_to_pixel(x0, y0, veh_tf)
            px1, py1 = world_to_pixel(x1, y1, veh_tf)
            cv2.line(img, (px0, py0), (px1, py1), (0, 255, 255), 3)

    # --- 2D overlays: edge dots, path dots, poly curve, distance lines ---
    if veh_tf is not None:
        oh, ow = img.shape[:2]

        # Red dots: right road edge
        if right_world:
            for pt in right_world:
                wx = pt.x if hasattr(pt, 'x') else pt[0]
                wy = pt.y if hasattr(pt, 'y') else pt[1]
                px, py = world_to_pixel(wx, wy, veh_tf, ow, oh)
                if 0 <= px < ow and 0 <= py < oh:
                    cv2.circle(img, (px, py), 4, (0, 0, 255), -1)

        # Blue dots: AI driving path
        if driving_coords:
            for i in range(0, len(driving_coords), 2):
                pt = driving_coords[i]
                dx = pt[0] if not hasattr(pt, 'x') else pt.x
                dy = pt[1] if not hasattr(pt, 'y') else pt.y
                px, py = world_to_pixel(dx, dy, veh_tf, ow, oh)
                if 0 <= px < ow and 0 <= py < oh:
                    cv2.circle(img, (px, py), 5, (255, 0, 0), -1)

        # Purple dots: GT reference path
        if driving_coords_gt:
            for i in range(0, len(driving_coords_gt), 2):
                pt = driving_coords_gt[i]
                dx = pt[0] if not hasattr(pt, 'x') else pt.x
                dy = pt[1] if not hasattr(pt, 'y') else pt.y
                px, py = world_to_pixel(dx, dy, veh_tf, ow, oh)
                if 0 <= px < ow and 0 <= py < oh:
                    cv2.circle(img, (px, py), 5, (128, 0, 128), -1)

        # Magenta curve: polynomial extrapolation
        if poly_coeffs is not None and len(poly_coeffs) == 3:
            fwd = veh_tf.get_forward_vector()
            fwd_h = math.sqrt(fwd.x**2 + fwd.y**2)
            if fwd_h > 1e-6:
                right_x = -fwd.y / fwd_h
                right_y = fwd.x / fwd_h
                poly_pts = []
                for lon in np.linspace(0, 20, 40):
                    lat = poly_coeffs[0]*lon**2 + poly_coeffs[1]*lon + poly_coeffs[2]
                    wx = veh_tf.location.x + (fwd.x/fwd_h)*lon + right_x*lat
                    wy = veh_tf.location.y + (fwd.y/fwd_h)*lon + right_y*lat
                    px, py = world_to_pixel(wx, wy, veh_tf, ow, oh)
                    if 0 <= px < ow and 0 <= py < oh:
                        poly_pts.append((px, py))
                for i in range(len(poly_pts) - 1):
                    cv2.line(img, poly_pts[i], poly_pts[i+1], (255, 0, 255), 2)

        # Green line: nozzle -> road edge
        if nozzle_raised is not None and nozzle_edge_pt is not None:
            p1 = world_to_pixel(nozzle_raised.x, nozzle_raised.y, veh_tf, ow, oh)
            p2 = world_to_pixel(nozzle_edge_pt.x, nozzle_edge_pt.y, veh_tf, ow, oh)
            cv2.line(img, p1, p2, (0, 255, 0), 2)

        # Cyan line: first TP -> poly-extrapolated edge (like V3)
        if tp_loc is not None and poly_edge_pt is not None:
            p1 = world_to_pixel(tp_loc.x, tp_loc.y, veh_tf, ow, oh)
            p2 = world_to_pixel(poly_edge_pt.x, poly_edge_pt.y, veh_tf, ow, oh)
            cv2.line(img, p1, p2, (255, 255, 0), 2)
            # Poly distance text at midpoint
            mx = (p1[0] + p2[0]) // 2
            my = (p1[1] + p2[1]) // 2
            if poly_dist is not None:
                cv2.putText(img, f"{poly_dist:.1f}m", (mx, my - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Distance text overlays
    if nozzle_mid is not None:
        npx, npy = world_to_pixel(nozzle_mid.x, nozzle_mid.y, veh_tf)
        cv2.putText(img, f"{edge_dist_r:.1f}m", (npx, npy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    drive_mode = "AUTO" if paint_ctrl.auto_drive else "MANUAL"
    perc_mode = perception_mode_str if perception_mode_str else ("AI" if use_ai_mode else "GT")

    # AutoPaint state string + color (RGB for draw_status_overlay)
    ap_state_str = ""
    ap_color = (255, 255, 255)
    if auto_paint is not None:
        if auto_paint._manual_override:
            ap_state_str = "MANUAL"
            ap_color = (255, 255, 0)
        elif auto_paint.state == AutoPaintStateMachine.STATE_PAINTING:
            ap_state_str = "PAINTING"
            ap_color = (0, 255, 0)
        elif auto_paint.state == AutoPaintStateMachine.STATE_STABILIZED:
            pct = int(auto_paint.progress * 100)
            ap_state_str = f"STABLE {pct}%"
            ap_color = (0, 200, 255)
        else:
            ap_state_str = "CONVERGING"
            ap_color = (255, 100, 100)

    # Line mode string
    line_mode_str = ""
    if paint_ctrl.dash_mode:
        line_mode_str = f"DASH ({paint_ctrl._dash_accum:.1f}m)"
    else:
        line_mode_str = "SOLID"

    # Offset + steer filter
    d_offset = planner.driving_offset if planner else 5.0
    sf = controller._effective_steer_filter if controller else 0.15

    img = draw_status_overlay(
        img, paint_ctrl.painting_enabled,
        frame_count, speed, edge_dist_r,
        drive_mode, paint_ctrl.throttle, paint_ctrl.steer, paint_ctrl.brake,
        perc_mode, poly_dist, fps,
        veh_x=veh_tf.location.x, veh_y=veh_tf.location.y,
        veh_yaw=veh_tf.rotation.yaw,
        ap_state_str=ap_state_str, ap_color=ap_color,
        line_mode_str=line_mode_str, driving_offset=d_offset,
        steer_filter=sf, is_recording=is_recording,
        eval_recording=eval_recording, spectator_follow=spectator_follow
    )
    cv2.imshow("Overhead View", img)
    cv2.waitKey(1)
    return img


def _render_front_view(pg_screen, rgb_front, road_mask, scene,
                       nozzle_mid, edge_dist_r,
                       right_world=None, driving_coords=None,
                       poly_coeffs=None, veh_tf=None, use_ai=False,
                       nozzle_raised=None, nozzle_edge_pt=None,
                       right_px=None, poly_dist=None,
                       driving_coords_gt=None, poly_edge_pt=None,
                       tp_loc=None):
    """Render front camera view with mask overlay in pygame window."""
    if rgb_front is None:
        return

    front_display = rgb_front[:, :, :3].copy()

    # Green road mask overlay
    if road_mask is not None:
        mask_overlay = np.zeros_like(front_display)
        mask_overlay[road_mask > 0] = [0, 255, 0]
        front_display = cv2.addWeighted(
            front_display, 0.7, mask_overlay, 0.3, 0)

    # Distance text on front view
    rgb_cam_tf = scene['rgb_front_cam'].get_transform()
    if nozzle_mid is not None:
        fp = world_to_front_pixel(
            nozzle_mid.x, nozzle_mid.y, nozzle_mid.z, rgb_cam_tf)
        if fp is not None:
            fpx, fpy = fp
            if 0 <= fpx < FRONT_CAM_W and 0 <= fpy < FRONT_CAM_H:
                cv2.putText(front_display, f"{edge_dist_r:.1f}m",
                            (fpx, fpy), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 3)

    # --- 2D overlay: edge dots, path dots, poly curve (AI mode) ---
    rgb_cam_tf = scene['rgb_front_cam'].get_transform()
    if use_ai and veh_tf is not None:
        # Red dots: right road edge (direct 2D pixels, no 3D round-trip)
        if right_px:
            for u, v in right_px:
                if 0 <= u < FRONT_CAM_W and 0 <= v < FRONT_CAM_H:
                    cv2.circle(front_display, (int(u), int(v)), 3, (0, 0, 255), -1)

        # Blue dots: AI driving path
        if driving_coords:
            for i in range(0, len(driving_coords), 2):
                pt = driving_coords[i]
                px = pt.x if hasattr(pt, 'x') else pt[0]
                py = pt.y if hasattr(pt, 'y') else pt[1]
                z = veh_tf.location.z + 0.3
                fp = world_to_front_pixel(px, py, z, rgb_cam_tf)
                if fp and 0 <= fp[0] < FRONT_CAM_W and 0 <= fp[1] < FRONT_CAM_H:
                    cv2.circle(front_display, fp, 4, (255, 0, 0), -1)

        # Purple dots: GT reference path
        if driving_coords_gt:
            for i in range(0, len(driving_coords_gt), 2):
                pt = driving_coords_gt[i]
                px = pt.x if hasattr(pt, 'x') else pt[0]
                py = pt.y if hasattr(pt, 'y') else pt[1]
                z = veh_tf.location.z + 0.3
                fp = world_to_front_pixel(px, py, z, rgb_cam_tf)
                if fp and 0 <= fp[0] < FRONT_CAM_W and 0 <= fp[1] < FRONT_CAM_H:
                    cv2.circle(front_display, fp, 4, (128, 0, 128), -1)

        # Magenta curve: polynomial extrapolation
        if poly_coeffs is not None and len(poly_coeffs) == 3:
            fwd = veh_tf.get_forward_vector()
            right = carla.Vector3D(-fwd.y, fwd.x, 0)
            fwd_h = math.sqrt(fwd.x**2 + fwd.y**2)
            if fwd_h > 1e-6:
                poly_pts = []
                for lon in np.linspace(0, 20, 40):
                    lat = poly_coeffs[0]*lon**2 + poly_coeffs[1]*lon + poly_coeffs[2]
                    wx = veh_tf.location.x + (fwd.x/fwd_h)*lon + right.x*lat
                    wy = veh_tf.location.y + (fwd.y/fwd_h)*lon + right.y*lat
                    wz = veh_tf.location.z + 0.3
                    fp = world_to_front_pixel(wx, wy, wz, rgb_cam_tf)
                    if fp and 0 <= fp[0] < FRONT_CAM_W and 0 <= fp[1] < FRONT_CAM_H:
                        poly_pts.append(fp)
                for i in range(len(poly_pts) - 1):
                    cv2.line(front_display, poly_pts[i], poly_pts[i+1],
                             (255, 0, 255), 2)

    # --- 2D distance line segments (AI mode) ---
    if use_ai and veh_tf is not None:
        rgb_cam_tf = scene['rgb_front_cam'].get_transform()

        # Green line: nozzle -> road edge
        if nozzle_raised is not None and nozzle_edge_pt is not None:
            fp_a = world_to_front_pixel(
                nozzle_raised.x, nozzle_raised.y, nozzle_raised.z, rgb_cam_tf)
            fp_b = world_to_front_pixel(
                nozzle_edge_pt.x, nozzle_edge_pt.y, nozzle_edge_pt.z, rgb_cam_tf)
            if fp_a and fp_b:
                if (0 <= fp_a[0] < FRONT_CAM_W and 0 <= fp_a[1] < FRONT_CAM_H and
                        0 <= fp_b[0] < FRONT_CAM_W and 0 <= fp_b[1] < FRONT_CAM_H):
                    cv2.line(front_display, fp_a, fp_b, (0, 255, 0), 2)

        # Cyan line: first TP -> poly-extrapolated edge (like V3)
        if tp_loc is not None and poly_edge_pt is not None:
            fp_a = world_to_front_pixel(
                tp_loc.x, tp_loc.y, tp_loc.z, rgb_cam_tf)
            fp_b = world_to_front_pixel(
                poly_edge_pt.x, poly_edge_pt.y, poly_edge_pt.z, rgb_cam_tf)
            if fp_a and fp_b:
                if (0 <= fp_a[0] < FRONT_CAM_W and 0 <= fp_a[1] < FRONT_CAM_H and
                        0 <= fp_b[0] < FRONT_CAM_W and 0 <= fp_b[1] < FRONT_CAM_H):
                    cv2.line(front_display, fp_a, fp_b, (255, 255, 0), 2)
                    # Poly distance text at midpoint
                    mx = (fp_a[0] + fp_b[0]) // 2
                    my = (fp_a[1] + fp_b[1]) // 2
                    if poly_dist is not None:
                        cv2.putText(front_display, f"{poly_dist:.1f}m",
                                    (mx, my - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 255, 0), 2)

    # BGR -> RGB for pygame
    front_rgb = cv2.cvtColor(front_display, cv2.COLOR_BGR2RGB)
    pg_surface = pygame.surfarray.make_surface(front_rgb.swapaxes(0, 1))
    pg_screen.blit(pg_surface, (0, 0))


if __name__ == '__main__':
    main()
