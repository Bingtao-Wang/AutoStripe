#!/usr/bin/env python
"""Spawn Point Finder - Manual drive + real-time position display.

Drive around Town05 and find good spawn points for AutoStripe.
Press P to print current position as a ready-to-paste spawn dict entry.

Controls:
  WASD/Arrows - Drive (throttle/steer/brake)
  Q           - Toggle reverse
  X           - Handbrake
  P           - Print current position (copy-paste ready)
  V           - Toggle spectator follow/free camera
  ESC         - Quit
"""
import glob, os, sys, math, time
import pygame
from pygame.locals import (K_ESCAPE, K_w, K_a, K_s, K_d, K_x, K_q, K_p, K_v,
                           K_UP, K_DOWN, K_LEFT, K_RIGHT)

try:
    sys.path.append(glob.glob(
        '../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major, sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np
import cv2


# --- Config ---
SPAWN_X, SPAWN_Y, SPAWN_Z, SPAWN_YAW = 155.0, -5.7, 0.60, -90.1
WINDOW_W, WINDOW_H = 960, 540
NOZZLE_OFFSET = 2.0  # meters to the right of vehicle center


def get_nozzle_position(vehicle):
    """Compute nozzle position: vehicle center + right-side offset."""
    tf = vehicle.get_transform()
    loc = vehicle.get_location()
    yaw_rad = math.radians(tf.rotation.yaw)
    dx = NOZZLE_OFFSET * math.cos(yaw_rad + math.pi / 2)
    dy = NOZZLE_OFFSET * math.sin(yaw_rad + math.pi / 2)
    return carla.Location(x=loc.x + dx, y=loc.y + dy, z=loc.z)


def get_nozzle_edge_distance(carla_map, nozzle_loc, vehicle_tf):
    """Compute distance from nozzle to right road edge via Map API.

    Uses waypoint lane center + lane_width, projected along vehicle's right axis.
    """
    wp = carla_map.get_waypoint(nozzle_loc)
    if wp is None:
        return None

    # Waypoint = lane center
    wp_loc = wp.transform.location
    half_width = wp.lane_width / 2.0

    # Use vehicle's right direction
    veh_yaw = math.radians(vehicle_tf.rotation.yaw)
    right_x = -math.sin(veh_yaw)
    right_y = math.cos(veh_yaw)

    # Nozzle lateral offset from lane center (positive = right of center)
    dx = nozzle_loc.x - wp_loc.x
    dy = nozzle_loc.y - wp_loc.y
    lat_offset = dx * right_x + dy * right_y

    # Right edge = lane center + half_width in right direction
    # Distance from nozzle to right edge
    dist = half_width - lat_offset
    return dist


def main():
    print("=" * 50)
    print("  Spawn Point Finder")
    print(f"  Start: x={SPAWN_X}, y={SPAWN_Y}, yaw={SPAWN_YAW}")
    print("=" * 50)
    print("\nControls:")
    print("  WASD/Arrows - Drive")
    print("  Q - Reverse | X - Brake")
    print("  P - Print position (copy-paste ready)")
    print("  V - Camera follow/free | ESC - Quit\n")

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Spawn Point Finder")
    font_big = pygame.font.SysFont("monospace", 28, bold=True)
    font_med = pygame.font.SysFont("monospace", 22)
    font_sm = pygame.font.SysFont("monospace", 18)
    clock = pygame.time.Clock()

    actors = []
    try:
        # Connect
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()

        # Spawn vehicle
        vehicle_bp = bp_lib.filter('vehicle.*stl*')[0]
        spawn_tf = carla.Transform(
            carla.Location(x=SPAWN_X, y=SPAWN_Y, z=SPAWN_Z),
            carla.Rotation(yaw=SPAWN_YAW))
        vehicle = world.spawn_actor(vehicle_bp, spawn_tf)
        actors.append(vehicle)
        print(f"Vehicle spawned: {vehicle.type_id}")

        # Attach overhead camera
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(WINDOW_W))
        cam_bp.set_attribute('image_size_y', str(WINDOW_H))
        cam_bp.set_attribute('fov', '90')
        cam_tf = carla.Transform(
            carla.Location(z=25),
            carla.Rotation(pitch=-90))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        actors.append(camera)

        # Camera data buffer
        cam_image = {'data': None}
        def cam_callback(image):
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
            cam_image['data'] = arr

        camera.listen(cam_callback)

        # Spectator
        spectator = world.get_spectator()
        spectator_follow = True

        # Drive state
        throttle = 0.0
        steer = 0.0
        brake = 0.0
        reverse = False
        print_count = 0
        carla_map = world.get_map()

        print("\nReady! Drive with WASD, press P to save position.\n")

        while True:
            # --- Event handling ---
            should_exit = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE:
                        should_exit = True
                    elif event.key == K_q:
                        reverse = not reverse
                        print(f"  Reverse: {'ON' if reverse else 'OFF'}")
                    elif event.key == K_v:
                        spectator_follow = not spectator_follow
                        print(f"  Camera: {'FOLLOW' if spectator_follow else 'FREE'}")
                    elif event.key == K_p:
                        tf = vehicle.get_transform()
                        loc = tf.location
                        yaw = tf.rotation.yaw
                        nz = get_nozzle_position(vehicle)
                        ed = get_nozzle_edge_distance(carla_map, nz, tf)
                        ed_str = f"{ed:.1f}m" if ed is not None else "N/A"
                        print_count += 1
                        print(f"\n  === Spawn Point #{print_count} (Nozzle-Edge: {ed_str}) ===")
                        print(f'  {{"x": {loc.x:.1f}, "y": {loc.y:.1f}, '
                              f'"z": {loc.z:.2f}, "yaw": {yaw:.1f}, '
                              f'"desc": ""}},')
                        print()

            if should_exit:
                break

            # --- Continuous key input ---
            keys = pygame.key.get_pressed()

            if keys[K_w] or keys[K_UP]:
                throttle = min(1.0, throttle + 0.1)
            else:
                throttle = 0.0

            if keys[K_s] or keys[K_DOWN]:
                brake = min(1.0, brake + 0.2)
            else:
                brake = 0.0

            if keys[K_a] or keys[K_LEFT]:
                steer = max(-0.7, steer - 0.05) if steer <= 0 else 0.0
            elif keys[K_d] or keys[K_RIGHT]:
                steer = min(0.7, steer + 0.05) if steer >= 0 else 0.0
            else:
                steer = 0.0

            if keys[K_x]:
                brake = 1.0
                throttle = 0.0

            # Apply control
            ctrl = carla.VehicleControl()
            ctrl.throttle = throttle
            ctrl.steer = steer
            ctrl.brake = brake
            ctrl.reverse = reverse
            vehicle.apply_control(ctrl)

            # --- Spectator follow ---
            if spectator_follow:
                veh_tf = vehicle.get_transform()
                fwd = veh_tf.get_forward_vector()
                spec_loc = carla.Location(
                    x=veh_tf.location.x - 10 * fwd.x,
                    y=veh_tf.location.y - 10 * fwd.y,
                    z=veh_tf.location.z + 6)
                spec_rot = carla.Rotation(pitch=-20, yaw=veh_tf.rotation.yaw)
                spectator.set_transform(carla.Transform(spec_loc, spec_rot))

            # --- Get vehicle state ---
            tf = vehicle.get_transform()
            loc = tf.location
            yaw = tf.rotation.yaw
            vel = vehicle.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2)

            # --- Nozzle-to-edge distance ---
            nozzle_loc = get_nozzle_position(vehicle)
            edge_dist = get_nozzle_edge_distance(carla_map, nozzle_loc, tf)

            # Draw green line: nozzle -> right edge in 3D
            if edge_dist is not None:
                wp = carla_map.get_waypoint(nozzle_loc)
                wp_yaw = math.radians(wp.transform.rotation.yaw)
                right_x = -math.sin(wp_yaw)
                right_y = math.cos(wp_yaw)
                edge_loc = carla.Location(
                    x=nozzle_loc.x + edge_dist * right_x,
                    y=nozzle_loc.y + edge_dist * right_y,
                    z=nozzle_loc.z + 0.3)
                nozzle_raised = carla.Location(
                    x=nozzle_loc.x, y=nozzle_loc.y, z=nozzle_loc.z + 0.3)
                world.debug.draw_line(
                    nozzle_raised, edge_loc,
                    thickness=0.08, color=carla.Color(0, 255, 0),
                    life_time=0.1)
                world.debug.draw_point(
                    nozzle_raised, size=0.1,
                    color=carla.Color(255, 255, 0), life_time=0.1)

            # --- Render ---
            screen.fill((30, 30, 30))

            # Camera image
            if cam_image['data'] is not None:
                img_rgb = cam_image['data']
                surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
                screen.blit(surf, (0, 0))

            # HUD overlay
            edge_text = f"{edge_dist:.1f}m" if edge_dist is not None else "N/A"
            edge_color = (0, 255, 0) if edge_dist is not None else (150, 150, 150)
            y_off = 10
            lines = [
                (f"x={loc.x:8.1f}  y={loc.y:8.1f}  z={loc.z:5.2f}", (0, 255, 0), font_big),
                (f"yaw={yaw:7.1f}   speed={speed:.1f} m/s", (0, 255, 0), font_big),
                ("", None, font_sm),
                (f"Nozzle-Edge: {edge_text}", edge_color, font_big),
                (f"road_id={_get_road_info(world, loc)}", (255, 255, 255), font_med),
                ("", None, font_sm),
                (f"Thr:{throttle:.1f} Str:{steer:.2f} Brk:{brake:.1f}"
                 + (" REV" if reverse else ""),
                 (255, 255, 0), font_med),
                ("", None, font_sm),
                ("P=Save position  V=Camera  ESC=Quit", (180, 180, 180), font_sm),
                ("WASD=Drive  Q=Reverse  X=Brake", (180, 180, 180), font_sm),
            ]
            for text, color, f in lines:
                if text:
                    surf = f.render(text, True, color)
                    screen.blit(surf, (10, y_off))
                y_off += f.get_height() + 2

            pygame.display.flip()
            clock.tick(30)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        pygame.quit()
        for a in actors:
            if a is not None:
                a.destroy()
        print("Done.")


def _get_road_info(world, loc):
    """Get road/lane info for current location."""
    try:
        wp = world.get_map().get_waypoint(loc)
        return f"{wp.road_id}  lane={wp.lane_id}"
    except Exception:
        return "N/A"


if __name__ == '__main__':
    main()
