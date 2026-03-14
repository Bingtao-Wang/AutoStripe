#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main program for ORB-SLAM3 testing with CARLA autopilot."""

import sys
import glob
import os
import argparse
import time
from datetime import datetime

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
import cv2

try:
    import rospy
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: rospy not available, SLAM interface disabled")

from carla_setup import setup_carla_scene, cleanup
from visualization import SLAMVisualizer
from slam_interface import ORBSlam3Wrapper
from evaluator import SLAMEvaluator


class ORBSLAM3Tester:
    """ORB-SLAM3 testing with CARLA autopilot."""

    def __init__(self, map_name, spawn_x, spawn_y, spawn_z, spawn_yaw):
        self.scene = None
        self.slam = None
        self.visualizer = None
        self.evaluator = None
        self.recording = False

        # Sensor data buffers
        self.stereo_left = None
        self.stereo_right = None
        self.latest_imu = None
        self.left_timestamp = None
        self.right_timestamp = None

        # Setup scene
        print("Setting up CARLA scene: {}".format(map_name))
        self.scene = setup_carla_scene(map_name, spawn_x, spawn_y, spawn_z, spawn_yaw)

        # Initialize visualizer
        self.visualizer = SLAMVisualizer()
        self.visualizer.enabled = True  # Enable visualization by default

        # Initialize ROS node FIRST
        if ROS_AVAILABLE:
            try:
                rospy.init_node('orb_slam3_tester', anonymous=True)
                print("ROS node initialized")
            except rospy.exceptions.ROSException:
                print("ROS node already initialized")

        # Initialize SLAM interface BEFORE sensor callbacks
        print("Initializing ORB-SLAM3 interface...")
        self.slam = ORBSlam3Wrapper()

        # Setup sensor callbacks AFTER SLAM is ready
        self.scene['camera_left'].listen(lambda img: self._on_camera_left(img))
        self.scene['camera_right'].listen(lambda img: self._on_camera_right(img))
        self.scene['imu'].listen(lambda data: self._on_imu(data))

        print("Setup complete. Waiting for sensors...")
        time.sleep(2)

    def _on_camera_left(self, image):
        """Left camera callback."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.stereo_left = array[:, :, :3]
        self.left_timestamp = image.timestamp  # Store timestamp

    def _on_camera_right(self, image):
        """Right camera callback."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.stereo_right = array[:, :, :3]
        self.right_timestamp = image.timestamp  # Store timestamp

    def _on_imu(self, imu_data):
        """IMU callback - publish immediately at 200Hz."""
        self.latest_imu = imu_data
        if self.slam and imu_data is not None:
            # Use IMU's own timestamp (more reliable than world snapshot)
            timestamp = imu_data.timestamp
            self.slam.publish_imu_only(imu_data, timestamp)

    def toggle_recording(self):
        """Toggle evaluation recording."""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"experiments/run_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            self.evaluator = SLAMEvaluator(output_dir)
            self.recording = True
            print("Recording started: {}".format(output_dir))
        else:
            if self.evaluator:
                self.evaluator.finalize()
            self.recording = False
            print("Recording stopped")

    def run(self):
        """Main loop."""
        print("\nControls:")
        print("  S - Toggle visualization")
        print("  E - Toggle evaluation recording")
        print("  ESC - Quit")
        print("\nWaiting for ORB-SLAM3 initialization...")

        try:
            while True:
                # Check keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('s') or key == ord('S'):
                    self.visualizer.toggle()
                    print("Visualization: {}".format('ON' if self.visualizer.enabled else 'OFF'))
                elif key == ord('e') or key == ord('E'):
                    self.toggle_recording()

                # Get vehicle pose (ground truth)
                vehicle_transform = self.scene['vehicle'].get_transform()
                gt_pose = (
                    vehicle_transform.location.x,
                    vehicle_transform.location.y,
                    vehicle_transform.location.z,
                    vehicle_transform.rotation.roll,
                    vehicle_transform.rotation.pitch,
                    vehicle_transform.rotation.yaw
                )

                # Publish stereo images to SLAM (use camera's own timestamp for sync)
                if self.stereo_left is not None and self.stereo_right is not None and self.left_timestamp is not None:
                    self.slam.publish_stereo_only(self.stereo_left, self.stereo_right, self.left_timestamp)

                # Get ORB-SLAM3 pose
                orb_pose = self.slam.get_pose()

                # Record data
                if self.recording and self.evaluator and orb_pose is not None:
                    self.evaluator.add_pose(gt_pose, orb_pose)

                # Update visualization
                ate = self.evaluator.compute_ate() if self.evaluator else None
                self.visualizer.update(
                    self.stereo_left if self.stereo_left is not None else np.zeros((480, 752, 3), dtype=np.uint8),
                    self.stereo_right if self.stereo_right is not None else np.zeros((480, 752, 3), dtype=np.uint8),
                    ate
                )

                time.sleep(0.05)

        finally:
            print("\nCleaning up...")
            if self.recording and self.evaluator:
                self.evaluator.finalize()
            cleanup(self.scene['actors'])
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='ORB-SLAM3 Testing with CARLA')
    parser.add_argument('--map', default='Town05', help='CARLA map name')
    parser.add_argument('--spawn-x', type=float, default=10, help='Spawn X coordinate')
    parser.add_argument('--spawn-y', type=float, default=-210, help='Spawn Y coordinate')
    parser.add_argument('--spawn-z', type=float, default=1.85, help='Spawn Z coordinate')
    parser.add_argument('--spawn-yaw', type=float, default=180, help='Spawn yaw angle')
    args = parser.parse_args()

    tester = ORBSLAM3Tester(args.map, args.spawn_x, args.spawn_y, args.spawn_z, args.spawn_yaw)
    tester.run()


if __name__ == '__main__':
    main()

