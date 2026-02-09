"""AutoStripe V1 — automatic road edge marking using CARLA Map API.

Usage:
    1. Start CARLA server:  ./CarlaUE4.sh
    2. Run:  python main_v1.py
    3. Ctrl+C to exit cleanly.
"""

import glob
import os
import sys
import math
import time

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

from carla_env.setup_scene import (setup_scene, update_spectator, display_overhead,
                                   compute_road_edge_distances, cleanup)
from planning.lane_planner import generate_center_waypoints
from control.marker_vehicle import MarkerVehicle

# --- Drawing constants ---
EDGE_LINE_COLOR = carla.Color(255, 255, 0)     # yellow (distinct from road white)
EDGE_LINE_THICKNESS = 0.3                       # visible from overhead
NOZZLE_OFFSET = 2.0                             # nozzle distance to the right of vehicle (m)


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


def main():
    print("AutoStripe V1 — starting...")
    scene = setup_scene()
    world = scene["world"]
    vehicle = scene["vehicle"]
    spectator = scene["spectator"]
    carla_map = scene["map"]

    # --- Planning: generate waypoints for Pure Pursuit path only ---
    start_loc = vehicle.get_location()
    wp_objects, wp_coords = generate_center_waypoints(
        carla_map, start_loc, num_waypoints=200, spacing=1.0,
    )
    print(f"Planned {len(wp_coords)} waypoints.")

    # --- Control: create Pure Pursuit controller ---
    controller = MarkerVehicle(vehicle, wp_coords, wp_objects)

    # --- Main loop ---
    # The vehicle IS the marking machine.  Each frame we compute the
    # paint-nozzle position (right side of vehicle) and draw a line
    # segment from the previous nozzle position to the current one.
    prev_nozzle = None
    nozzle_trail = []

    try:
        while True:
            controller.step()
            update_spectator(spectator, vehicle)

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

            edge_dists = compute_road_edge_distances(scene)
            display_overhead(scene, trail=nozzle_trail, edge_distances=edge_dists)

            if controller.is_route_complete():
                print("Route complete — keeping alive for inspection. "
                      "Press Ctrl+C to exit.")
                while True:
                    update_spectator(spectator, vehicle)
                    edge_dists = compute_road_edge_distances(scene)
                    display_overhead(scene, trail=nozzle_trail,
                                     edge_distances=edge_dists)
                    time.sleep(0.1)

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup(scene["actors"])
        print("Done.")


if __name__ == "__main__":
    main()
