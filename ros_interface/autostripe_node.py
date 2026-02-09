#!/usr/bin/env python3
"""AutoStripe V2 main ROS node.

Combines perception + planning + control, publishes all data for RVIZ.
Connects directly to CARLA (not through CARLA-ROS bridge sensors)
to maintain the same architecture as standalone mode.
"""

import glob
import os
import sys
import math
import time
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

try:
    import rospy
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from carla_env.setup_scene_v2 import setup_scene_v2
from carla_env.setup_scene import update_spectator, compute_road_edge_distances, cleanup
from perception.perception_pipeline import PerceptionPipeline
from planning.vision_path_planner import VisionPathPlanner
from control.marker_vehicle_v2 import MarkerVehicleV2
from ros_interface.rviz_publisher import RvizPublisher

EDGE_LINE_COLOR = carla.Color(255, 255, 0)
EDGE_LINE_THICKNESS = 0.3
NOZZLE_OFFSET = 2.0
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
    yaw_rad = math.radians(yaw_deg)
    dx = offset * math.cos(yaw_rad + math.pi / 2)
    dy = offset * math.sin(yaw_rad + math.pi / 2)
    return carla.Location(
        x=veh_location.x + dx,
        y=veh_location.y + dy,
        z=veh_location.z,
    )


class AutoStripeNode:
    """V2 main ROS node: perception + planning + control + RVIZ publishing."""

    def __init__(self):
        rospy.init_node('autostripe_v2', anonymous=False)
        rospy.loginfo("AutoStripe V2 ROS node starting...")

        # Setup CARLA scene
        self.scene = setup_scene_v2()
        self.world = self.scene["world"]
        self.vehicle = self.scene["vehicle"]
        self.spectator = self.scene["spectator"]

        # Initialize modules
        self.perception = PerceptionPipeline(
            self.scene["front_cam_w"],
            self.scene["front_cam_h"],
            self.scene["front_cam_fov"])
        self.planner = VisionPathPlanner(nozzle_offset=NOZZLE_OFFSET)
        self.controller = MarkerVehicleV2(self.vehicle)
        self.rviz_pub = RvizPublisher()

        # State
        self.prev_nozzle = None
        self.nozzle_trail = []
        self.frame_count = 0

    def warmup(self):
        """Let sensors fill buffers before starting main loop."""
        rospy.loginfo("Warming up sensors...")
        for _ in range(WARMUP_FRAMES):
            if rospy.is_shutdown():
                return
            self.world.tick()
            time.sleep(0.05)
        rospy.loginfo("Sensors ready.")

    def spin(self):
        """Main loop: perception -> planning -> control -> paint -> publish."""
        rate = rospy.Rate(30)  # 30 Hz

        left_world = []
        right_world = []

        while not rospy.is_shutdown():
            self.frame_count += 1

            # 1. Read sensor data
            with self.scene["_semantic_lock"]:
                sem_img = self.scene["_semantic_data"]["image"]
                cs_img = self.scene["_semantic_data"].get("cityscapes")
            with self.scene["_depth_lock"]:
                depth_img = self.scene["_depth_data"]["image"]

            # 2. Perception
            road_mask = None
            if sem_img is not None and depth_img is not None:
                cam_tf = get_front_camera_transform(self.vehicle)
                left_world, right_world, road_mask, _, _ = \
                    self.perception.process_frame(sem_img, depth_img, cam_tf,
                                                  cityscapes_bgra=cs_img)

                # 3. Planning
                driving_coords, nozzle_locs = self.planner.update(
                    left_world, right_world, self.vehicle.get_transform())
                self.controller.update_path(driving_coords)

            # 4. Step controller
            self.controller.step()
            update_spectator(self.spectator, self.vehicle)

            # 5. Nozzle painting
            veh_tf = self.vehicle.get_transform()
            veh_loc = self.vehicle.get_location()
            cur_nozzle = get_nozzle_position(veh_loc, veh_tf.rotation.yaw)
            self.nozzle_trail.append(cur_nozzle)

            if self.prev_nozzle is not None:
                self.world.debug.draw_line(
                    self.prev_nozzle, cur_nozzle,
                    thickness=EDGE_LINE_THICKNESS,
                    color=EDGE_LINE_COLOR,
                    life_time=1000,
                    persistent_lines=True,
                )
            self.prev_nozzle = cur_nozzle

            # 6. Publish to RVIZ
            if road_mask is not None:
                self.rviz_pub.publish_road_mask(road_mask)
            if left_world or right_world:
                self.rviz_pub.publish_edge_markers(left_world, right_world)

            if self.controller.has_path():
                self.rviz_pub.publish_driving_path(
                    self.controller.waypoint_coords)
            if self.planner.nozzle_locations:
                self.rviz_pub.publish_nozzle_path(
                    self.planner.nozzle_locations)

            if self.nozzle_trail:
                self.rviz_pub.publish_paint_trail(self.nozzle_trail)

            self.rviz_pub.publish_vehicle_marker(veh_tf)

            # Status text
            edge_dists = compute_road_edge_distances(self.scene)
            left_d, right_d = edge_dists if edge_dists else (None, None)
            n_path = len(self.controller.waypoint_coords)
            status = (f"V2 Vision | Path: {n_path} pts | "
                      f"Trail: {len(self.nozzle_trail)} pts\n"
                      f"L: {left_d:.1f}m" if left_d else "L: --")
            self.rviz_pub.publish_status_text(status, veh_tf)

            if self.frame_count % 100 == 0:
                rospy.loginfo(f"Frame {self.frame_count}: path={n_path}, "
                              f"trail={len(self.nozzle_trail)}")

            rate.sleep()

    def shutdown(self):
        """Clean up CARLA actors."""
        rospy.loginfo("Shutting down AutoStripe node...")
        cleanup(self.scene["actors"])
        rospy.loginfo("Done.")


def main():
    if not ROS_AVAILABLE:
        print("ERROR: ROS not available. Use main_v2.py for standalone mode.")
        return

    node = AutoStripeNode()
    rospy.on_shutdown(node.shutdown)
    node.warmup()
    node.spin()


if __name__ == "__main__":
    main()
