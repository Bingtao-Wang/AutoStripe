"""AutoStripe V2 — vision-based road edge marking (standalone, no ROS).

Replaces Map API with perception pipeline:
  semantic segmentation -> edge extraction -> depth projection -> path planning

Usage:
    1. Start CARLA server:  ./CarlaUE4.sh
    2. Run:  python main_v2.py
    3. Ctrl+C to exit cleanly.
"""

import glob
import os
import sys
import math
import time

import cv2
import numpy as np

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

from carla_env.setup_scene_v2 import setup_scene_v2
from carla_env.setup_scene import update_spectator
from perception.perception_pipeline import PerceptionPipeline
from planning.vision_path_planner import VisionPathPlanner
from control.marker_vehicle_v2 import MarkerVehicleV2

# --- Drawing constants (same as V1) ---
EDGE_LINE_COLOR = carla.Color(255, 255, 0)
EDGE_LINE_THICKNESS = 0.3
NOZZLE_OFFSET = 2.0

# --- Perception warm-up frames ---
WARMUP_FRAMES = 30


def get_front_camera_transform(vehicle):
    """Compute the world transform of the front camera from vehicle transform."""
    from carla_env.setup_scene_v2 import FRONT_CAM_X, FRONT_CAM_Z, FRONT_CAM_PITCH
    veh_tf = vehicle.get_transform()
    yaw_rad = math.radians(veh_tf.rotation.yaw)

    cam_world_x = veh_tf.location.x + FRONT_CAM_X * math.cos(yaw_rad)
    cam_world_y = veh_tf.location.y + FRONT_CAM_X * math.sin(yaw_rad)
    cam_world_z = veh_tf.location.z + FRONT_CAM_Z

    return carla.Transform(
        carla.Location(x=cam_world_x, y=cam_world_y, z=cam_world_z),
        carla.Rotation(pitch=veh_tf.rotation.pitch + FRONT_CAM_PITCH,
                       yaw=veh_tf.rotation.yaw,
                       roll=veh_tf.rotation.roll),
    )


def get_nozzle_position(veh_location, yaw_deg, offset=NOZZLE_OFFSET):
    """Compute the paint nozzle position on the right side of the vehicle."""
    yaw_rad = math.radians(yaw_deg)
    dx = offset * math.cos(yaw_rad + math.pi / 2)
    dy = offset * math.sin(yaw_rad + math.pi / 2)
    return carla.Location(
        x=veh_location.x + dx,
        y=veh_location.y + dy,
        z=veh_location.z,
    )


def compute_edge_distance_from_perception(vehicle, right_world, left_world):
    """Compute distance from vehicle to nearest right/left edge point."""
    veh_loc = vehicle.get_location()
    right_d = None
    left_d = None

    if right_world:
        dists = [math.sqrt((loc.x - veh_loc.x)**2 + (loc.y - veh_loc.y)**2)
                 for loc in right_world]
        right_d = min(dists) if dists else None

    if left_world:
        dists = [math.sqrt((loc.x - veh_loc.x)**2 + (loc.y - veh_loc.y)**2)
                 for loc in left_world]
        left_d = min(dists) if dists else None

    return left_d, right_d


def draw_driving_path(world, driving_coords, vehicle_tf=None,
                      color=carla.Color(0, 120, 255),
                      thickness=0.15, life_time=0.2):
    """Draw the planned driving path in CARLA world (blue lines).
    Uses vehicle pitch to estimate z on slopes.
    """
    if len(driving_coords) < 2 or vehicle_tf is None:
        return

    veh_loc = vehicle_tf.location
    fwd = vehicle_tf.get_forward_vector()
    fwd_h = math.sqrt(fwd.x**2 + fwd.y**2)
    slope = fwd.z / fwd_h if fwd_h > 1e-6 else 0.0

    # Draw every 3rd segment to reduce overhead
    for i in range(0, len(driving_coords) - 1, 3):
        x1, y1 = driving_coords[i][0], driving_coords[i][1]
        dx1 = x1 - veh_loc.x
        dy1 = y1 - veh_loc.y
        lon1 = (dx1 * fwd.x + dy1 * fwd.y) / fwd_h if fwd_h > 1e-6 else 0.0
        z1 = veh_loc.z + lon1 * slope + 0.5

        j = min(i + 3, len(driving_coords) - 1)
        x2, y2 = driving_coords[j][0], driving_coords[j][1]
        dx2 = x2 - veh_loc.x
        dy2 = y2 - veh_loc.y
        lon2 = (dx2 * fwd.x + dy2 * fwd.y) / fwd_h if fwd_h > 1e-6 else 0.0
        z2 = veh_loc.z + lon2 * slope + 0.5

        world.debug.draw_line(
            carla.Location(x=x1, y=y1, z=z1),
            carla.Location(x=x2, y=y2, z=z2),
            thickness=thickness,
            color=color,
            life_time=life_time,
        )


def display_front_perception(scene, road_mask, left_px, right_px):
    """Show front camera with edge detection overlay.

    Crops the bottom portion of the image where the vehicle's own
    hood/body is visible (not useful for perception).
    """
    with scene["_rgb_front_lock"]:
        rgb = scene["_rgb_front_data"]["image"]
    if rgb is None:
        return

    frame = rgb[:, :, :3].copy()  # drop alpha

    # Draw edge pixels
    for u, v in left_px:
        cv2.circle(frame, (u, v), 3, (0, 255, 0), -1)  # green = left
    for u, v in right_px:
        cv2.circle(frame, (u, v), 3, (0, 0, 255), -1)  # red = right

    cv2.imshow("Front Perception", frame)
    cv2.waitKey(1)
    return frame


def display_overhead_v2(scene, nozzle_trail, edge_distances):
    """Show overhead view with nozzle trail (reuses V1 projection logic)."""
    from carla_env.setup_scene import _world_to_pixel

    with scene["_frame_lock"]:
        frame = scene["_frame_data"]["image"]
    if frame is None:
        return

    frame = frame.copy()

    if nozzle_trail and len(nozzle_trail) >= 2:
        veh_tf = scene["vehicle"].get_transform()
        pts = []
        for loc in nozzle_trail:
            px, py = _world_to_pixel(loc.x, loc.y, veh_tf)
            if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 3)

    # HUD text
    y0 = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "V2 — Vision-based", (20, y0), font, 1.0, (255, 255, 0), 2)

    if edge_distances is not None:
        left_d, right_d = edge_distances
        lt = f"Left edge: {left_d:.2f} m" if left_d else "Left edge: --"
        rt = f"Right edge: {right_d:.2f} m" if right_d else "Right edge: --"
        cv2.putText(frame, lt, (20, y0 + 40), font, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, rt, (20, y0 + 80), font, 1.0, (0, 200, 255), 2)

    cv2.imshow("Overhead View", frame)
    cv2.waitKey(1)
    return frame


# Diagnostic snapshot interval
SNAPSHOT_INTERVAL = 200


def main():
    print("AutoStripe V2 — starting (vision-based)...")
    scene = setup_scene_v2()
    world = scene["world"]
    vehicle = scene["vehicle"]
    spectator = scene["spectator"]

    # --- Initialize modules ---
    perception = PerceptionPipeline(
        scene["front_cam_w"], scene["front_cam_h"], scene["front_cam_fov"])
    planner = VisionPathPlanner(nozzle_arm=NOZZLE_OFFSET)
    controller = MarkerVehicleV2(vehicle)

    # --- Warm up: let sensors fill buffers ---
    print(f"Warming up sensors ({WARMUP_FRAMES} frames)...")
    for _ in range(WARMUP_FRAMES):
        world.tick()
        time.sleep(0.05)
    print("Sensors ready.")

    # --- Main loop ---
    prev_nozzle = None
    nozzle_trail = []
    frame_count = 0

    # Cache for edge pixels (for display)
    last_left_px = []
    last_right_px = []
    last_road_mask = None
    last_right_world = []
    last_left_world = []

    try:
        while True:
            frame_count += 1

            # 1. Read sensor data
            with scene["_semantic_lock"]:
                sem_img = scene["_semantic_data"]["image"]
                cs_img = scene["_semantic_data"].get("cityscapes")
            with scene["_depth_lock"]:
                depth_img = scene["_depth_data"]["image"]

            # 2. Perception (every frame if data available)
            has_sensor = sem_img is not None and depth_img is not None
            if frame_count <= 5:
                print(f"[F{frame_count}] sensor data: sem={sem_img is not None}, "
                      f"depth={depth_img is not None}")

            if has_sensor:
                # Debug: inspect raw semantic image
                if frame_count <= 3:
                    tags = sem_img[:, :, 2]  # R channel
                    h = tags.shape[0]
                    bot = tags[h//2:, :]  # bottom half only
                    print(f"[F{frame_count}] BOTTOM HALF tag distribution:")
                    for t in np.unique(bot):
                        cnt = np.count_nonzero(bot == t)
                        pct = cnt / bot.size * 100
                        print(f"  tag {t:3d}: {cnt:7d} px ({pct:5.1f}%)")
                    # Save semantic R-channel as grayscale for inspection
                    cv2.imwrite(f"/tmp/sem_frame{frame_count}.png",
                                tags * 10)  # scale up for visibility

                cam_tf = get_front_camera_transform(vehicle)
                left_world, right_world, road_mask, left_px, right_px = \
                    perception.process_frame(sem_img, depth_img, cam_tf,
                                             cityscapes_bgra=cs_img)
                last_road_mask = road_mask
                last_left_px = left_px
                last_right_px = right_px
                last_right_world = right_world
                last_left_world = left_world

                if frame_count <= 3:
                    road_pct = np.count_nonzero(road_mask) / road_mask.size * 100
                    print(f"[F{frame_count}] road_mask: {road_pct:.1f}% road pixels")

                if frame_count <= 10 or frame_count % 50 == 0:
                    print(f"[F{frame_count}] edges: L={len(left_world)}, "
                          f"R={len(right_world)}, px: L={len(left_px)}, "
                          f"R={len(right_px)}")

                # 3. Planning (right-edge-only strategy)
                driving_coords, nozzle_locs = planner.update(
                    right_world, vehicle.get_transform())

                if frame_count <= 10 or frame_count % 50 == 0:
                    print(f"[F{frame_count}] path: drive={len(driving_coords)}, "
                          f"nozzle={len(nozzle_locs)}")
                    vl = vehicle.get_location()
                    if driving_coords:
                        p0 = driving_coords[0]
                        pn = driving_coords[-1]
                        print(f"  veh=({vl.x:.1f},{vl.y:.1f}) "
                              f"path[0]=({p0[0]:.1f},{p0[1]:.1f}) "
                              f"path[-1]=({pn[0]:.1f},{pn[1]:.1f})")

                # 4. Control: update path
                controller.update_path(driving_coords)

            # 5. Step controller
            controller.step()

            # 6. Spectator follow
            update_spectator(spectator, vehicle)

            # 7. Nozzle painting (same as V1)
            veh_tf = vehicle.get_transform()
            veh_loc = vehicle.get_location()
            yaw = veh_tf.rotation.yaw

            cur_nozzle = get_nozzle_position(veh_loc, yaw)
            nozzle_trail.append(cur_nozzle)

            if prev_nozzle is not None:
                world.debug.draw_line(
                    prev_nozzle, cur_nozzle,
                    thickness=EDGE_LINE_THICKNESS,
                    color=EDGE_LINE_COLOR,
                    life_time=1000,
                    persistent_lines=True,
                )
            prev_nozzle = cur_nozzle

            # 7b. Draw planned driving path in CARLA world (blue)
            if has_sensor:
                draw_driving_path(world, controller.waypoint_coords,
                                  vehicle_tf=vehicle.get_transform())

            # 8. Display — use perception-based edge distance
            edge_dists = compute_edge_distance_from_perception(
                vehicle, last_right_world, last_left_world)
            overhead_frame = display_overhead_v2(scene, nozzle_trail, edge_dists)
            front_frame = display_front_perception(scene, last_road_mask,
                                     last_left_px, last_right_px)

            # 9. Save diagnostic snapshots periodically
            if frame_count % SNAPSHOT_INTERVAL == 0:
                if overhead_frame is not None:
                    cv2.imwrite("/tmp/v2_overhead.png", overhead_frame)
                if front_frame is not None:
                    cv2.imwrite("/tmp/v2_front.png", front_frame)

            if frame_count % 100 == 0:
                n_path = len(controller.waypoint_coords)
                print(f"Frame {frame_count}: path={n_path} pts, "
                      f"trail={len(nozzle_trail)} pts")

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        from carla_env.setup_scene import cleanup
        cleanup(scene["actors"])
        print("Done.")


if __name__ == "__main__":
    main()
