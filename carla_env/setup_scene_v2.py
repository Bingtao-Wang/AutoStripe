"""AutoStripe V2 scene setup — adds semantic segmentation + depth + RGB front cameras."""

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
from carla import ColorConverter as cc


# ---------- Front camera parameters (shared by semantic / depth / RGB) ----------
FRONT_CAM_X = 2.5
FRONT_CAM_Z = 2.8
FRONT_CAM_PITCH = -15
FRONT_CAM_W = 800
FRONT_CAM_H = 600
FRONT_CAM_FOV = 90


def _make_shared_buffer():
    """Create a thread-safe shared image buffer."""
    lock = threading.Lock()
    data = {"image": None}
    return lock, data


def _camera_callback(lock, data):
    """Return a listener callback that stores raw BGRA into shared buffer."""
    def _cb(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        with lock:
            data["image"] = array
    return _cb


def setup_scene_v2(map_name="Town05", spawn_x=10, spawn_y=-210,
                   spawn_z=1.85, spawn_yaw=180):
    """Connect to CARLA, spawn vehicle and all V2 sensors.

    Returns a dict extending V1 scene with additional sensor data buffers.
    """
    client = carla.Client('localhost', 2000)
    client.set_timeout(200)

    world = client.load_world(map_name)
    weather = carla.WeatherParameters.ClearNoon
    weather.sun_altitude_angle = 45.0
    world.set_weather(weather)

    bp_lib = world.get_blueprint_library()

    # --- Spawn vehicle ---
    vehicle_bp = bp_lib.filter('vehicle.*stl*')[0]
    spawn_tf = carla.Transform(
        carla.Location(x=spawn_x, y=spawn_y, z=spawn_z),
        carla.Rotation(yaw=spawn_yaw),
    )
    vehicle = world.spawn_actor(vehicle_bp, spawn_tf)

    spectator = world.get_spectator()
    actors = [vehicle]

    # --- Front camera transform (shared by semantic / depth / RGB) ---
    front_cam_tf = carla.Transform(
        carla.Location(x=FRONT_CAM_X, y=0, z=FRONT_CAM_Z),
        carla.Rotation(pitch=FRONT_CAM_PITCH),
    )

    # --- Semantic segmentation camera ---
    sem_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    sem_bp.set_attribute("image_size_x", str(FRONT_CAM_W))
    sem_bp.set_attribute("image_size_y", str(FRONT_CAM_H))
    sem_bp.set_attribute("fov", str(FRONT_CAM_FOV))

    sem_cam = world.spawn_actor(sem_bp, front_cam_tf, attach_to=vehicle)
    sem_lock, sem_data = _make_shared_buffer()

    def _semantic_cb(image):
        # Store raw BGRA first (R channel = tag ID, for diag scripts)
        raw = np.frombuffer(image.raw_data, dtype=np.uint8).copy()
        raw = np.reshape(raw, (image.height, image.width, 4))
        # Convert to CityScapes palette (modifies image in-place)
        image.convert(cc.CityScapesPalette)
        cs = np.frombuffer(image.raw_data, dtype=np.uint8)
        cs = np.reshape(cs, (image.height, image.width, 4))
        with sem_lock:
            sem_data["image"] = raw          # raw tags (backward compat)
            sem_data["cityscapes"] = cs.copy()  # CityScapes colored

    sem_cam.listen(_semantic_cb)
    actors.append(sem_cam)

    # --- Depth camera (same position as semantic) ---
    depth_bp = bp_lib.find('sensor.camera.depth')
    depth_bp.set_attribute("image_size_x", str(FRONT_CAM_W))
    depth_bp.set_attribute("image_size_y", str(FRONT_CAM_H))
    depth_bp.set_attribute("fov", str(FRONT_CAM_FOV))

    depth_cam = world.spawn_actor(depth_bp, front_cam_tf, attach_to=vehicle)
    depth_lock, depth_data = _make_shared_buffer()
    depth_cam.listen(_camera_callback(depth_lock, depth_data))
    actors.append(depth_cam)

    # --- RGB front camera (for RVIZ / overlay display) ---
    rgb_front_bp = bp_lib.find('sensor.camera.rgb')
    rgb_front_bp.set_attribute("image_size_x", str(FRONT_CAM_W))
    rgb_front_bp.set_attribute("image_size_y", str(FRONT_CAM_H))
    rgb_front_bp.set_attribute("fov", str(FRONT_CAM_FOV))

    rgb_front_cam = world.spawn_actor(rgb_front_bp, front_cam_tf, attach_to=vehicle)
    rgb_front_lock, rgb_front_data = _make_shared_buffer()
    rgb_front_cam.listen(_camera_callback(rgb_front_lock, rgb_front_data))
    actors.append(rgb_front_cam)

    # --- Overhead RGB camera (same as V1) ---
    overhead_bp = bp_lib.find('sensor.camera.rgb')
    overhead_bp.set_attribute("image_size_x", "1800")
    overhead_bp.set_attribute("image_size_y", "1600")
    overhead_bp.set_attribute("fov", "90")

    overhead_tf = carla.Transform(
        carla.Location(x=0, y=0, z=25),
        carla.Rotation(pitch=-90, yaw=0, roll=0),
    )
    overhead_cam = world.spawn_actor(overhead_bp, overhead_tf, attach_to=vehicle)
    overhead_lock, overhead_data = _make_shared_buffer()

    def _overhead_cb(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # drop alpha for display
        with overhead_lock:
            overhead_data["image"] = array

    overhead_cam.listen(_overhead_cb)
    actors.append(overhead_cam)

    # --- Semantic LiDAR (same as V1) ---
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
    actors.append(lidar)

    # --- OpenCV windows ---
    cv2.namedWindow("Overhead View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Overhead View", 800, 600)

    return {
        "client": client,
        "world": world,
        "map": world.get_map(),
        "vehicle": vehicle,
        "spectator": spectator,
        "overhead_camera": overhead_cam,
        "semantic_cam": sem_cam,
        "depth_cam": depth_cam,
        "rgb_front_cam": rgb_front_cam,
        "lidar": lidar,
        "actors": actors,
        # V1-compatible buffers
        "_frame_lock": overhead_lock,
        "_frame_data": overhead_data,
        "_lidar_lock": lidar_lock,
        "_lidar_data": lidar_data,
        # V2 new sensor buffers
        "_semantic_lock": sem_lock,
        "_semantic_data": sem_data,
        "_depth_lock": depth_lock,
        "_depth_data": depth_data,
        "_rgb_front_lock": rgb_front_lock,
        "_rgb_front_data": rgb_front_data,
        # Camera params (needed by depth projector)
        "front_cam_w": FRONT_CAM_W,
        "front_cam_h": FRONT_CAM_H,
        "front_cam_fov": FRONT_CAM_FOV,
    }
