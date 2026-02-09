#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AutoStripe V4 - Manual Painting Control with VLLiNet AI Perception

Based on V3 manual_painting_control.py, adds:
- VLLiNet AI road segmentation (G key toggles AI/GT mode)
- Polynomial extrapolation for nozzle-edge distance (blind spot)
- Camera resolution 1248x384 to match VLLiNet training
- Magenta polynomial curve visualization

Keyboard Controls:
  SPACE - Toggle painting ON/OFF
  TAB   - Toggle Auto/Manual drive mode
  G     - Toggle AI/GT perception mode
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
import threading
import pygame
from pygame.locals import K_ESCAPE, K_SPACE, K_TAB, K_q, K_g
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
from perception.perception_pipeline import PerceptionPipeline
from planning.vision_path_planner import VisionPathPlanner
from control.marker_vehicle_v2 import MarkerVehicleV2


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

    def toggle_painting(self):
        self.painting_enabled = not self.painting_enabled
        status = "ON" if self.painting_enabled else "OFF"
        print(f"\n{'='*50}")
        print(f"  Paint: {status}")
        print(f"{'='*50}\n")
        return self.painting_enabled

    def paint_line(self, world, nozzle_loc):
        if not self.painting_enabled:
            self.last_nozzle_loc = None
            return

        if self.last_nozzle_loc is not None:
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
                        tp_edge_dist=999.0, perception_mode="AI",
                        poly_dist=None):
    """Draw status info on overhead image."""
    h, w = img.shape[:2]

    # Drive mode
    mode_text = f"MODE: {drive_mode}"
    mode_color = (0, 255, 255) if drive_mode == "AUTO" else (255, 0, 255)
    cv2.putText(img, mode_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, mode_color, 3)

    # Perception mode
    perc_text = f"PERC: {perception_mode}"
    perc_color = (255, 0, 255) if perception_mode == "AI" else (0, 255, 0)
    cv2.putText(img, perc_text, (350, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, perc_color, 3)

    # Paint status
    status_text = "PAINT: ON" if painting_enabled else "PAINT: OFF"
    status_color = (0, 255, 0) if painting_enabled else (0, 0, 255)
    cv2.putText(img, status_text, (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)

    # Vehicle status
    cv2.putText(img, f"Speed: {speed:.1f} m/s", (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Nozzle-Edge: {edge_dist_r:.1f}m", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    tp_dist_text = f"TP-Edge: {tp_edge_dist:.1f}m" if tp_edge_dist < 900 else "TP-Edge: N/A"
    cv2.putText(img, tp_dist_text, (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    # Polynomial distance
    if poly_dist is not None:
        cv2.putText(img, f"Poly-Edge: {poly_dist:.1f}m", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
    else:
        cv2.putText(img, "Poly-Edge: N/A", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)

    # Manual control status
    if drive_mode == "MANUAL":
        cv2.putText(img, f"Throttle: {throttle:.2f}", (20, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, f"Steer: {steer:.2f}", (20, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, f"Brake: {brake:.2f}", (20, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Help text
    help_y = h - 150
    cv2.putText(img, "Controls:", (20, help_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(img, "TAB=Mode SPACE=Paint G=AI/GT", (20, help_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, "WASD=Drive Q=Reverse X=Brake", (20, help_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, "V=Camera ESC=Quit", (20, help_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

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


def main():
    print("=" * 60)
    print("  AutoStripe V4 - AI Perception + Manual Painting Control")
    print("=" * 60)
    print("\nControls:")
    print("  SPACE - Toggle painting ON/OFF")
    print("  TAB   - Toggle Auto/Manual drive")
    print("  G     - Toggle AI/GT perception")
    print("  ESC   - Quit")
    print("\nStarting...\n")

    # Initialize pygame (keyboard + front view display)
    pygame.init()
    pg_screen = pygame.display.set_mode((FRONT_CAM_W, FRONT_CAM_H))
    pygame.display.set_caption("AutoStripe V4 - Front View + Control")
    pg_font = pygame.font.SysFont("monospace", 18)
    pg_clock = pygame.time.Clock()

    actors = []
    # Perception mode: True=AI (VLLiNet), False=GT (CityScapes)
    use_ai_mode = True

    try:
        # 1. Setup scene
        scene = setup_scene_v2()
        actors = scene['actors']
        world = scene['world']
        vehicle = scene['vehicle']

        # 2. Initialize modules — start in AI mode
        perception = PerceptionPipeline(
            img_w=FRONT_CAM_W, img_h=FRONT_CAM_H, fov_deg=90.0,
            use_ai=use_ai_mode
        )
        planner = VisionPathPlanner(
            line_offset=3.0, nozzle_arm=2.0, smooth_window=5
        )
        controller = MarkerVehicleV2(vehicle, wheelbase=2.875, kdd=3.0)

        # 3. Initialize painting controller
        paint_ctrl = ManualPaintingControl(vehicle)

        # 4. Warm up sensors
        print("Warming up sensors (30 frames)...")
        for _ in range(30):
            time.sleep(0.05)
        print("Sensors ready.\n")

        # 5. Main loop
        frame_count = 0
        spectator_follow = True
        poly_dist = None
        poly_coeffs = None

        print("=" * 60)
        print("  System ready! Press SPACE to start painting")
        print("  Press G to toggle AI/GT perception mode")
        print("=" * 60)

        while True:
            frame_count += 1

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

            # --- Perception: extract road edges ---
            cam_tf = scene['semantic_cam'].get_transform()
            left_world, right_world, road_mask, left_px, right_px = \
                perception.process_frame(
                    sem_data, depth_data, cam_tf,
                    cityscapes_bgra=cs_data,
                    rgb_bgra=rgb_front
                )

            # --- Planning: generate driving path ---
            veh_tf = vehicle.get_transform()
            driving_coords, _ = planner.update(right_world, veh_tf)

            # --- Polynomial extrapolation ---
            poly_dist, poly_coeffs = planner.estimate_nozzle_edge_distance(
                right_world, veh_tf)

            # --- Visualize: right edge red dots ---
            _draw_right_edge_dots(world, right_world, veh_tf)

            # --- Control ---
            if paint_ctrl.auto_drive:
                controller.update_path(driving_coords)
                controller.step()
            else:
                paint_ctrl.apply_manual_control()

            # --- Painting ---
            nozzle_loc = get_nozzle_position(vehicle)
            paint_ctrl.paint_line(world, nozzle_loc)

            # --- Visualize: driving path + poly curve ---
            draw_driving_path(world, driving_coords, vehicle.get_transform())
            draw_poly_curve(world, poly_coeffs, vehicle.get_transform())

            # --- Spectator follow ---
            if spectator_follow:
                update_spectator(scene['spectator'], vehicle)

            # --- Compute distances ---
            veh_vel = vehicle.get_velocity()
            speed = math.sqrt(veh_vel.x**2 + veh_vel.y**2)

            # TP-Edge distance (cyan line)
            tp_edge_dist = 999.0
            tp_mid = None
            if len(driving_coords) > 0:
                veh_tf_now = vehicle.get_transform()
                fwd = veh_tf_now.get_forward_vector()
                fwd_h = math.sqrt(fwd.x**2 + fwd.y**2)
                slope = fwd.z / fwd_h if fwd_h > 1e-6 else 0.0

                tp = driving_coords[0]
                dx_tp = tp[0] - veh_tf_now.location.x
                dy_tp = tp[1] - veh_tf_now.location.y
                lon_tp = (dx_tp * fwd.x / fwd_h + dy_tp * fwd.y / fwd_h) \
                    if fwd_h > 1e-6 else 0.0
                tp_z = veh_tf_now.location.z + lon_tp * slope + 0.3

                tp_loc = carla.Location(x=tp[0], y=tp[1], z=tp_z)
                tp_edge_dist, tp_edge_pt = compute_point_edge_distance(
                    tp_loc, right_world, veh_tf_now)

                if tp_edge_pt is not None:
                    tp_edge_pt.z = tp_z
                    world.debug.draw_line(
                        tp_loc, tp_edge_pt,
                        thickness=0.08,
                        color=carla.Color(0, 255, 255),
                        life_time=0.1)
                    tp_mid = carla.Location(
                        x=(tp_loc.x + tp_edge_pt.x) / 2,
                        y=(tp_loc.y + tp_edge_pt.y) / 2,
                        z=tp_z + 0.3)
                    world.debug.draw_string(
                        tp_mid, f"{tp_edge_dist:.1f}m",
                        color=carla.Color(0, 255, 255),
                        life_time=0.1)

            # Nozzle-Edge distance (green line)
            nozzle_raised = carla.Location(
                x=nozzle_loc.x, y=nozzle_loc.y, z=nozzle_loc.z + 0.3)
            edge_dist_r, nozzle_edge_pt = compute_point_edge_distance(
                nozzle_raised, right_world, vehicle.get_transform())

            nozzle_mid = None
            if nozzle_edge_pt is not None:
                world.debug.draw_line(
                    nozzle_raised, nozzle_edge_pt,
                    thickness=0.08,
                    color=carla.Color(0, 255, 0),
                    life_time=0.1)
                nozzle_mid = carla.Location(
                    x=(nozzle_raised.x + nozzle_edge_pt.x) / 2,
                    y=(nozzle_raised.y + nozzle_edge_pt.y) / 2,
                    z=nozzle_raised.z + 0.3)
                world.debug.draw_string(
                    nozzle_mid, f"{edge_dist_r:.1f}m",
                    color=carla.Color(0, 255, 0),
                    life_time=0.1)

            # --- Render overhead view ---
            _render_overhead(overhead_data, paint_ctrl, veh_tf, world,
                             edge_dist_r, tp_edge_dist, nozzle_mid, tp_mid,
                             speed, frame_count, poly_dist, use_ai_mode)

            # --- Render front view in pygame ---
            _render_front_view(pg_screen, rgb_front, road_mask, scene,
                               nozzle_mid, tp_mid, edge_dist_r, tp_edge_dist)

            # --- Event handling ---
            should_exit = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE:
                        should_exit = True
                    elif event.key == K_SPACE:
                        paint_ctrl.toggle_painting()
                    elif event.key == K_TAB:
                        paint_ctrl.toggle_drive_mode()
                    elif event.key == K_q:
                        paint_ctrl.toggle_reverse()
                    elif event.key == K_v:
                        spectator_follow = not spectator_follow
                        mode = "FOLLOW" if spectator_follow else "FREE"
                        print(f"\n  Camera: {mode}\n")
                    elif event.key == K_g:
                        use_ai_mode = not use_ai_mode
                        perception = PerceptionPipeline(
                            img_w=FRONT_CAM_W, img_h=FRONT_CAM_H,
                            fov_deg=90.0, use_ai=use_ai_mode
                        )
                        mode_str = "AI (VLLiNet)" if use_ai_mode else "GT (CityScapes)"
                        print(f"\n{'='*50}")
                        print(f"  Perception: {mode_str}")
                        print(f"{'='*50}\n")

            if should_exit:
                print("\nExiting...")
                break

            # Continuous key detection for manual driving
            keys = pygame.key.get_pressed()
            if not paint_ctrl.auto_drive:
                paint_ctrl.update_manual_control(keys)

            cv2.waitKey(1)

            # --- Pygame HUD overlay ---
            perc_mode = "AI" if use_ai_mode else "GT"
            lines = [
                ("Drive: AUTO" if paint_ctrl.auto_drive else "Drive: MANUAL",
                 (0, 255, 0) if paint_ctrl.auto_drive else (255, 255, 0)),
                (f"Perc: {perc_mode}",
                 (255, 0, 255) if use_ai_mode else (0, 255, 0)),
                ("Paint: ON" if paint_ctrl.painting_enabled else "Paint: OFF",
                 (0, 255, 0) if paint_ctrl.painting_enabled else (150, 150, 150)),
                ("Reverse: ON" if paint_ctrl.reverse else "",
                 (255, 100, 100)),
                (f"Speed: {speed:.1f} m/s", (255, 255, 255)),
                (f"Nozzle-Edge: {edge_dist_r:.1f}m", (0, 255, 0)),
                (f"Poly-Edge: {poly_dist:.1f}m" if poly_dist else "Poly-Edge: N/A",
                 (255, 0, 255)),
                (f"TP-Edge: {tp_edge_dist:.1f}m" if tp_edge_dist < 900
                 else "TP-Edge: N/A", (0, 255, 255)),
                (f"Thr:{paint_ctrl.throttle:.1f} Str:{paint_ctrl.steer:.2f} Brk:{paint_ctrl.brake:.1f}",
                 (255, 255, 255)),
                ("", (0, 0, 0)),
                ("Cam: FOLLOW" if spectator_follow else "Cam: FREE",
                 (0, 200, 255) if spectator_follow else (255, 150, 0)),
                ("TAB=Mode SPACE=Paint G=AI/GT Q=Rev", (200, 200, 200)),
                ("V=Cam WASD=Drive X=Brake ESC=Quit", (200, 200, 200)),
            ]
            for i, (text, color) in enumerate(lines):
                if text:
                    surf = pg_font.render(text, True, color)
                    pg_screen.blit(surf, (10, 8 + i * 22))
            pygame.display.flip()
            pg_clock.tick(30)

            # Periodic status print
            if frame_count % 50 == 0:
                status = "ON" if paint_ctrl.painting_enabled else "OFF"
                tp_d = f"{tp_edge_dist:.1f}" if tp_edge_dist < 900 else "N/A"
                poly_d = f"{poly_dist:.1f}" if poly_dist else "N/A"
                print(f"[F{frame_count}] Paint:{status} Perc:{perc_mode} "
                      f"Spd:{speed:.1f} Noz:{edge_dist_r:.1f}m "
                      f"Poly:{poly_d}m TP:{tp_d}m "
                      f"Path:{len(driving_coords)}pts")

    except KeyboardInterrupt:
        print("\nInterrupted...")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nCleaning up...")
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
                     edge_dist_r, tp_edge_dist, nozzle_mid, tp_mid,
                     speed, frame_count, poly_dist, use_ai_mode):
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

    # Distance text overlays
    if nozzle_mid is not None:
        npx, npy = world_to_pixel(nozzle_mid.x, nozzle_mid.y, veh_tf)
        cv2.putText(img, f"{edge_dist_r:.1f}m", (npx, npy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    if tp_mid is not None:
        tpx, tpy = world_to_pixel(tp_mid.x, tp_mid.y, veh_tf)
        cv2.putText(img, f"{tp_edge_dist:.1f}m", (tpx, tpy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    drive_mode = "AUTO" if paint_ctrl.auto_drive else "MANUAL"
    perc_mode = "AI" if use_ai_mode else "GT"
    img = draw_status_overlay(
        img, paint_ctrl.painting_enabled,
        frame_count, speed, edge_dist_r,
        drive_mode, paint_ctrl.throttle, paint_ctrl.steer, paint_ctrl.brake,
        tp_edge_dist, perc_mode, poly_dist
    )
    cv2.imshow("Overhead View", img)


def _render_front_view(pg_screen, rgb_front, road_mask, scene,
                       nozzle_mid, tp_mid, edge_dist_r, tp_edge_dist):
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

    if tp_mid is not None:
        fp2 = world_to_front_pixel(
            tp_mid.x, tp_mid.y, tp_mid.z, rgb_cam_tf)
        if fp2 is not None:
            fpx2, fpy2 = fp2
            if 0 <= fpx2 < FRONT_CAM_W and 0 <= fpy2 < FRONT_CAM_H:
                cv2.putText(front_display, f"{tp_edge_dist:.1f}m",
                            (fpx2, fpy2), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 255), 3)

    # BGR -> RGB for pygame
    front_rgb = cv2.cvtColor(front_display, cv2.COLOR_BGR2RGB)
    pg_surface = pygame.surfarray.make_surface(front_rgb.swapaxes(0, 1))
    pg_screen.blit(pg_surface, (0, 0))


if __name__ == '__main__':
    main()
