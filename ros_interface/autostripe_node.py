#!/usr/bin/env python3
"""AutoStripe V4 ROS node — subscribes to CARLA-ROS Bridge topics.

V4 changes from V2:
- Subscribes to CARLA-ROS Bridge sensor topics (no direct CARLA API)
- Supports AI (VLLiNet) and GT (CityScapes) perception modes
- Polynomial extrapolation for nozzle-edge distance
- Publishes poly curve markers to RVIZ

Requires:
  - CARLA-ROS Bridge running (carla_ros_bridge)
  - Sensors spawned via carla_spawn_objects
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
    from sensor_msgs.msg import Image
    from nav_msgs.msg import Odometry
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from carla_env.setup_scene_v2 import FRONT_CAM_W, FRONT_CAM_H, FRONT_CAM_FOV
from carla_env.setup_scene_v2 import FRONT_CAM_X, FRONT_CAM_Z, FRONT_CAM_PITCH
from perception.perception_pipeline import PerceptionPipeline
from planning.vision_path_planner import VisionPathPlanner
from control.marker_vehicle_v2 import MarkerVehicleV2
from ros_interface.rviz_publisher import RvizPublisher
from ros_interface.topic_config import (
    TOPIC_SEMANTIC_IMAGE, TOPIC_DEPTH_IMAGE,
    TOPIC_RGB_FRONT_IMAGE, TOPIC_ODOMETRY,
)

NOZZLE_OFFSET = 2.0


def _get_nozzle_position(x, y, z, yaw_deg, offset=NOZZLE_OFFSET):
    """Compute nozzle position from vehicle pose."""
    yaw_rad = math.radians(yaw_deg)
    dx = offset * math.cos(yaw_rad + math.pi / 2)
    dy = offset * math.sin(yaw_rad + math.pi / 2)
    return carla.Location(x=x + dx, y=y + dy, z=z)


def _odom_to_transform(odom_msg):
    """Convert Odometry message to carla.Transform (approximate)."""
    pos = odom_msg.pose.pose.position
    ori = odom_msg.pose.pose.orientation
    # Quaternion to yaw (simplified, assumes near-level vehicle)
    siny = 2.0 * (ori.w * ori.z + ori.x * ori.y)
    cosy = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
    yaw = math.degrees(math.atan2(siny, cosy))
    # CARLA-ROS Bridge uses: ROS x=forward, y=left, z=up
    # CARLA uses: x=forward, y=right, z=up
    # Bridge flips y sign in odometry
    return carla.Transform(
        carla.Location(x=pos.x, y=-pos.y, z=pos.z),
        carla.Rotation(yaw=-yaw),
    )


class AutoStripeNode:
    """V4 ROS node: subscribes to CARLA-ROS Bridge topics, runs perception
    + planning + control, publishes results to RVIZ."""

    def __init__(self):
        rospy.init_node('autostripe_v4', anonymous=False)
        rospy.loginfo("AutoStripe V4 ROS node starting...")

        # ROS params
        self.use_ai = rospy.get_param('~use_ai', True)
        checkpoint = rospy.get_param('~checkpoint', None)

        # cv_bridge for image conversion
        self.bridge = CvBridge()

        # Sensor buffers (thread-safe)
        self._lock = threading.Lock()
        self._sem_img = None
        self._depth_img = None
        self._rgb_img = None
        self._odom = None

        # Initialize perception + planning modules
        self.perception = PerceptionPipeline(
            img_w=FRONT_CAM_W, img_h=FRONT_CAM_H,
            fov_deg=FRONT_CAM_FOV, use_ai=self.use_ai,
            checkpoint_path=checkpoint)
        self.planner = VisionPathPlanner(
            line_offset=3.0, nozzle_arm=2.0, smooth_window=5)
        self.rviz_pub = RvizPublisher()

        # State
        self.prev_nozzle = None
        self.nozzle_trail = []
        self.frame_count = 0

        # Subscribe to CARLA-ROS Bridge topics
        rospy.Subscriber(TOPIC_SEMANTIC_IMAGE, Image,
                         self._cb_semantic, queue_size=1)
        rospy.Subscriber(TOPIC_DEPTH_IMAGE, Image,
                         self._cb_depth, queue_size=1)
        rospy.Subscriber(TOPIC_RGB_FRONT_IMAGE, Image,
                         self._cb_rgb, queue_size=1)
        rospy.Subscriber(TOPIC_ODOMETRY, Odometry,
                         self._cb_odom, queue_size=1)

        mode_str = "AI (VLLiNet)" if self.use_ai else "GT (CityScapes)"
        rospy.loginfo(f"Perception mode: {mode_str}")
        rospy.loginfo("Subscribed to Bridge topics. Waiting for data...")

    # --- ROS callbacks ---

    def _cb_semantic(self, msg):
        """Semantic segmentation image callback."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgra8')
            with self._lock:
                self._sem_img = img
        except Exception as e:
            rospy.logwarn_throttle(5, f"Semantic CB error: {e}")

    def _cb_depth(self, msg):
        """Depth image callback."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgra8')
            with self._lock:
                self._depth_img = img
        except Exception as e:
            rospy.logwarn_throttle(5, f"Depth CB error: {e}")

    def _cb_rgb(self, msg):
        """RGB front camera callback."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgra8')
            with self._lock:
                self._rgb_img = img
        except Exception as e:
            rospy.logwarn_throttle(5, f"RGB CB error: {e}")

    def _cb_odom(self, msg):
        """Odometry callback."""
        with self._lock:
            self._odom = msg

    # --- Camera transform from odometry ---

    def _cam_transform_from_odom(self, veh_tf):
        """Compute front camera world transform from vehicle transform."""
        yaw_rad = math.radians(veh_tf.rotation.yaw)
        cam_x = veh_tf.location.x + FRONT_CAM_X * math.cos(yaw_rad)
        cam_y = veh_tf.location.y + FRONT_CAM_X * math.sin(yaw_rad)
        cam_z = veh_tf.location.z + FRONT_CAM_Z
        return carla.Transform(
            carla.Location(x=cam_x, y=cam_y, z=cam_z),
            carla.Rotation(
                pitch=veh_tf.rotation.pitch + FRONT_CAM_PITCH,
                yaw=veh_tf.rotation.yaw,
                roll=veh_tf.rotation.roll),
        )

    # --- Main loop ---

    def spin(self):
        """Main processing loop at 30 Hz."""
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            self.frame_count += 1

            # Snapshot sensor data
            with self._lock:
                sem_img = self._sem_img
                depth_img = self._depth_img
                rgb_img = self._rgb_img
                odom = self._odom

            if sem_img is None or depth_img is None or odom is None:
                rate.sleep()
                continue

            # Vehicle transform from odometry
            veh_tf = _odom_to_transform(odom)
            cam_tf = self._cam_transform_from_odom(veh_tf)

            # Perception
            left_world, right_world, road_mask, _, _ = \
                self.perception.process_frame(
                    sem_img, depth_img, cam_tf,
                    cityscapes_bgra=sem_img,
                    rgb_bgra=rgb_img)

            # Planning
            driving_coords, nozzle_locs = \
                self.planner.update(right_world, veh_tf)

            # Polynomial extrapolation
            poly_dist, poly_coeffs = \
                self.planner.estimate_nozzle_edge_distance(
                    right_world, veh_tf)

            # Nozzle position
            veh_loc = veh_tf.location
            cur_nozzle = _get_nozzle_position(
                veh_loc.x, veh_loc.y, veh_loc.z,
                veh_tf.rotation.yaw)
            self.nozzle_trail.append(cur_nozzle)
            self.prev_nozzle = cur_nozzle

            # Publish to RVIZ
            self._publish_rviz(
                road_mask, left_world, right_world,
                driving_coords, veh_tf, poly_dist, poly_coeffs)

            if self.frame_count % 100 == 0:
                n_path = len(driving_coords)
                pd = f"{poly_dist:.1f}" if poly_dist else "N/A"
                rospy.loginfo(
                    f"Frame {self.frame_count}: "
                    f"path={n_path}, poly={pd}m, "
                    f"trail={len(self.nozzle_trail)}")

            rate.sleep()

    def _publish_rviz(self, road_mask, left_world, right_world,
                      driving_coords, veh_tf, poly_dist, poly_coeffs):
        """Publish all visualization data to RVIZ."""
        if road_mask is not None:
            self.rviz_pub.publish_road_mask(road_mask)

        if left_world or right_world:
            self.rviz_pub.publish_edge_markers(left_world, right_world)

        if driving_coords:
            self.rviz_pub.publish_driving_path(driving_coords)

        if self.planner.nozzle_locations:
            self.rviz_pub.publish_nozzle_path(self.planner.nozzle_locations)

        if self.nozzle_trail:
            self.rviz_pub.publish_paint_trail(self.nozzle_trail)

        self.rviz_pub.publish_vehicle_marker(veh_tf)

        if poly_coeffs is not None:
            self.rviz_pub.publish_poly_curve(poly_coeffs, veh_tf)

        n_path = len(driving_coords)
        pd = f"{poly_dist:.1f}m" if poly_dist else "N/A"
        mode = "AI" if self.use_ai else "GT"
        status = (f"V4 {mode} | Path: {n_path} pts | "
                  f"Poly: {pd} | Trail: {len(self.nozzle_trail)}")
        self.rviz_pub.publish_status_text(status, veh_tf)


def main():
    if not ROS_AVAILABLE:
        print("ERROR: ROS not available. Use manual_painting_control_v4.py "
              "for standalone mode.")
        return

    node = AutoStripeNode()
    node.spin()


if __name__ == "__main__":
    main()
