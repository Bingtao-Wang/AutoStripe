#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CARLA scene setup for ORB-SLAM3 testing."""

import sys
import glob
import os

# Add CARLA egg
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np

# Sensor parameters (from V7)
STEREO_WIDTH = 752
STEREO_HEIGHT = 480
STEREO_FOV = 90
STEREO_BASELINE = 0.6
CAMERA_X = 2.5
CAMERA_Z = 3.5
CAMERA_PITCH = -15

IMU_FREQ = 200
IMU_X = 0
IMU_Y = 0
IMU_Z = 1.5


def setup_carla_scene(map_name="Town05", spawn_x=-50, spawn_y=100, spawn_z=1.85, spawn_yaw=180):
    """Setup CARLA scene with vehicle and sensors.

    Returns:
        dict: Scene containing client, world, vehicle, sensors, and actor list
    """
    actors = []

    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Load map
    world = client.load_world(map_name)

    # Get blueprint library
    blueprint_library = world.get_blueprint_library()

    # Spawn vehicle
    vehicle_bp = blueprint_library.filter('vehicle.*stl*')[0]
    spawn_point = carla.Transform(
        carla.Location(x=spawn_x, y=spawn_y, z=spawn_z),
        carla.Rotation(yaw=spawn_yaw)
    )
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    actors.append(vehicle)

    # Set autopilot with smooth constant speed
    vehicle.set_autopilot(True)
    traffic_manager = client.get_trafficmanager()

    # Set target speed to 2 m/s (better for curves)
    traffic_manager.vehicle_percentage_speed_difference(vehicle, 80.0)

    # Optimize for smooth driving
    traffic_manager.auto_lane_change(vehicle, False)  # No lane changes
    traffic_manager.ignore_lights_percentage(vehicle, 100)  # Ignore traffic lights
    traffic_manager.ignore_vehicles_percentage(vehicle, 100)  # Ignore other vehicles
    traffic_manager.distance_to_leading_vehicle(vehicle, 0)  # No distance keeping

    # Setup stereo cameras
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(STEREO_WIDTH))
    camera_bp.set_attribute('image_size_y', str(STEREO_HEIGHT))
    camera_bp.set_attribute('fov', str(STEREO_FOV))

    # Left camera
    left_transform = carla.Transform(
        carla.Location(x=CAMERA_X, y=-STEREO_BASELINE/2, z=CAMERA_Z),
        carla.Rotation(pitch=CAMERA_PITCH)
    )
    camera_left = world.spawn_actor(camera_bp, left_transform, attach_to=vehicle)
    actors.append(camera_left)

    # Right camera
    right_transform = carla.Transform(
        carla.Location(x=CAMERA_X, y=STEREO_BASELINE/2, z=CAMERA_Z),
        carla.Rotation(pitch=CAMERA_PITCH)
    )
    camera_right = world.spawn_actor(camera_bp, right_transform, attach_to=vehicle)
    actors.append(camera_right)

    # Setup IMU at 200Hz (sensor_tick = 1/200 = 0.005s)
    # Without sensor_tick, IMU fires at sim framerate (~20-30Hz),
    # causing ORB-SLAM3 "Empty IMU measurements vector" errors
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', str(1.0 / IMU_FREQ))
    imu_transform = carla.Transform(carla.Location(x=IMU_X, y=IMU_Y, z=IMU_Z))
    imu = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
    actors.append(imu)

    return {
        'client': client,
        'world': world,
        'vehicle': vehicle,
        'camera_left': camera_left,
        'camera_right': camera_right,
        'imu': imu,
        'actors': actors
    }


def cleanup(actors):
    """Cleanup CARLA actors."""
    for actor in actors:
        if actor is not None:
            actor.destroy()
