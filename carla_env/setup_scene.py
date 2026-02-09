import glob
import os
import sys
import math
import threading
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


def setup_scene(map_name="Town05", spawn_x=10, spawn_y=-210,
                spawn_z=1.85, spawn_yaw=180):
    """Connect to CARLA, load map, spawn vehicle and overhead camera.

    Returns a dict with: client, world, map, vehicle, spectator,
    overhead_camera, and actors (list for cleanup).
    """
    client = carla.Client('localhost', 2000)
    client.set_timeout(200)

    world = client.load_world(map_name)
    weather = carla.WeatherParameters.ClearNoon
    weather.sun_altitude_angle = 45.0
    world.set_weather(weather)

    bp_lib = world.get_blueprint_library()

    # Spawn vehicle
    vehicle_bp = bp_lib.filter('vehicle.*stl*')[0]
    spawn_tf = carla.Transform(
        carla.Location(x=spawn_x, y=spawn_y, z=spawn_z),
        carla.Rotation(yaw=spawn_yaw),
    )
    vehicle = world.spawn_actor(vehicle_bp, spawn_tf)

    spectator = world.get_spectator()

    # Overhead RGB camera attached to vehicle
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", "1800")
    camera_bp.set_attribute("image_size_y", "1600")
    camera_bp.set_attribute("fov", "90")

    cam_tf = carla.Transform(
        carla.Location(x=0, y=0, z=25),
        carla.Rotation(pitch=-90, yaw=0, roll=0),
    )
    overhead_camera = world.spawn_actor(camera_bp, cam_tf, attach_to=vehicle)

    # Shared buffer for overhead camera (callback runs in a separate thread)
    frame_lock = threading.Lock()
    frame_data = {"image": None}

    def _process_img(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        with frame_lock:
            frame_data["image"] = array

    overhead_camera.listen(_process_img)

    # OpenCV window (created in main thread)
    cv2.namedWindow("Overhead View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Overhead View", 800, 600)

    # Semantic LiDAR for road-edge distance measurement
    lidar_bp = bp_lib.find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('range', '30')
    lidar_bp.set_attribute('points_per_second', '200000')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_bp.set_attribute('upper_fov', '5')
    lidar_bp.set_attribute('lower_fov', '-25')

    lidar_tf = carla.Transform(carla.Location(x=0, y=0, z=1.8))
    lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)

    lidar_lock = threading.Lock()
    lidar_data = {"points": None}

    def _process_lidar(point_cloud):
        raw = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32),
        ]))
        with lidar_lock:
            lidar_data["points"] = raw.copy()

    lidar.listen(_process_lidar)

    actors = [vehicle, overhead_camera, lidar]

    return {
        "client": client,
        "world": world,
        "map": world.get_map(),
        "vehicle": vehicle,
        "spectator": spectator,
        "overhead_camera": overhead_camera,
        "actors": actors,
        "_frame_lock": frame_lock,
        "_frame_data": frame_data,
        "_lidar_lock": lidar_lock,
        "_lidar_data": lidar_data,
    }


def update_spectator(spectator, vehicle, behind=10, height=6, pitch=-20):
    """Move spectator to follow behind the vehicle.

    In synchronous mode with fixed delta, direct set is smooth enough.
    Uses get_forward_vector() for heading-based offset.
    """
    veh_tf = vehicle.get_transform()
    fwd = veh_tf.get_forward_vector()

    sp_loc = veh_tf.location + carla.Location(
        x=-behind * fwd.x,
        y=-behind * fwd.y,
        z=height
    )

    sp_tf = carla.Transform(
        sp_loc,
        carla.Rotation(yaw=veh_tf.rotation.yaw, pitch=pitch),
    )
    spectator.set_transform(sp_tf)


def compute_road_edge_distances(scene):
    """Compute left/right road-edge distances from semantic LiDAR.

    Returns (left_dist, right_dist) in meters, or (None, None) if no data.
    Semantic tags: Road=7, Sidewalk=8, Terrain=22, Vegetation=9.
    """
    with scene["_lidar_lock"]:
        pts = scene["_lidar_data"]["points"]
    if pts is None or len(pts) == 0:
        return None, None

    # Non-road tags that indicate road boundary
    tags = pts['ObjTag']
    edge_mask = (tags == 8) | (tags == 22) | (tags == 9)  # Sidewalk/Terrain/Vegetation
    edge_pts = pts[edge_mask]
    if len(edge_pts) == 0:
        return None, None

    # LiDAR coords are sensor-local: x=forward, y=right, z=up
    # Filter points near ground level (z < 0 means below sensor at 1.8m)
    ground_mask = edge_pts['z'] < -0.5
    edge_pts = edge_pts[ground_mask]
    if len(edge_pts) == 0:
        return None, None

    # Lateral distance = |y|, split by sign
    y_vals = edge_pts['y']
    x_vals = edge_pts['x']

    # Only consider points roughly beside the vehicle (|x| < 8m)
    beside_mask = np.abs(x_vals) < 8.0

    right_mask = beside_mask & (y_vals > 0.5)
    left_mask = beside_mask & (y_vals < -0.5)

    right_dist = None
    left_dist = None

    if np.any(right_mask):
        right_dist = float(np.min(np.abs(y_vals[right_mask])))
    if np.any(left_mask):
        left_dist = float(np.min(np.abs(y_vals[left_mask])))

    return left_dist, right_dist


def _world_to_pixel(wx, wy, veh_tf, img_w=1800, img_h=1600, cam_h=25.0):
    """Project a world XY point onto the overhead camera image.

    Camera: attached to vehicle at z=cam_h, pitch=-90 (straight down), FOV=90.
    """
    vx = veh_tf.location.x
    vy = veh_tf.location.y
    yaw = math.radians(veh_tf.rotation.yaw)

    dx = wx - vx
    dy = wy - vy

    # Rotate to vehicle-local frame (forward / right)
    local_fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
    local_right = -dx * math.sin(yaw) + dy * math.cos(yaw)

    # FOV=90 => at height 25 m the visible half-extent = 25 m
    scale_x = (img_w / 2.0) / cam_h
    scale_y = (img_h / 2.0) / cam_h

    px = int(img_w / 2.0 + local_right * scale_x)
    py = int(img_h / 2.0 - local_fwd * scale_y)
    return px, py


def display_overhead(scene, trail=None, color=(0, 255, 255), thickness=3,
                     edge_distances=None):
    """Show overhead frame with optional painted trail overlay and edge distances."""
    with scene["_frame_lock"]:
        frame = scene["_frame_data"]["image"]
    if frame is None:
        return

    frame = frame.copy()

    if trail and len(trail) >= 2:
        veh_tf = scene["vehicle"].get_transform()
        pts = []
        for loc in trail:
            px, py = _world_to_pixel(loc.x, loc.y, veh_tf)
            if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, thickness)

    # Draw road-edge distance info
    if edge_distances is not None:
        left_d, right_d = edge_distances
        y0 = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        if left_d is not None:
            txt = f"Left edge: {left_d:.2f} m"
            cv2.putText(frame, txt, (20, y0), font, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Left edge: --", (20, y0), font, 1.0, (0, 255, 0), 2)
        if right_d is not None:
            txt = f"Right edge: {right_d:.2f} m"
            cv2.putText(frame, txt, (20, y0 + 40), font, 1.0, (0, 200, 255), 2)
        else:
            cv2.putText(frame, "Right edge: --", (20, y0 + 40), font, 1.0, (0, 200, 255), 2)

    cv2.imshow("Overhead View", frame)
    cv2.waitKey(1)


def cleanup(actors):
    """Destroy all actors safely."""
    for actor in actors:
        if actor is not None and actor.is_alive:
            actor.destroy()
    cv2.destroyAllWindows()
