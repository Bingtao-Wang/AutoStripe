"""ORB-SLAM3 ROS wrapper for stereo-inertial SLAM."""
import sys
import glob
import os

try:
    carla_egg = glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major, sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
    sys.path.append(carla_egg)
except IndexError:
    pass

import carla
import numpy as np

try:
    import rospy
    from sensor_msgs.msg import Image, Imu, CameraInfo
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Int32, Header
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("[WARN] ROS not available. ORB-SLAM3 disabled.")

class ORBSlam3Wrapper:
    def __init__(self):
        if not ROS_AVAILABLE:
            self.enabled = False
            return

        self.enabled = True
        self.last_pose = None
        self.tracking_state = 0

        # Publishers
        self.pub_left = rospy.Publisher('/autostripe/slam/stereo/left/image_raw', Image, queue_size=1)
        self.pub_right = rospy.Publisher('/autostripe/slam/stereo/right/image_raw', Image, queue_size=1)
        self.pub_imu = rospy.Publisher('/autostripe/slam/imu', Imu, queue_size=10)
        self.pub_left_info = rospy.Publisher('/autostripe/slam/stereo/left/camera_info', CameraInfo, queue_size=1)
        self.pub_right_info = rospy.Publisher('/autostripe/slam/stereo/right/camera_info', CameraInfo, queue_size=1)

        # Subscribers
        rospy.Subscriber('/orb_slam3_stereo_inertial/pose', PoseStamped, self._pose_callback)
        rospy.Subscriber('/orb_slam3_stereo_inertial/tracking_state', Int32, self._tracking_callback)

        # Camera info (752x480, FOV=90, baseline=0.6m)
        self.camera_info = self._create_camera_info()

    def _numpy_to_image_msg(self, cv_image, encoding="bgr8"):
        """Convert numpy array to ROS Image message without cv_bridge."""
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = encoding
        img_msg.is_bigendian = 0
        img_msg.step = cv_image.shape[1] * cv_image.shape[2]
        img_msg.data = cv_image.tobytes()
        return img_msg

    def publish_frame(self, stereo_left, stereo_right, imu_data, timestamp):
        if not self.enabled:
            return

        try:
            # Publish stereo images
            if stereo_left is not None:
                img_msg = self._numpy_to_image_msg(stereo_left, encoding="bgr8")
                img_msg.header.stamp = rospy.Time.from_sec(timestamp)
                img_msg.header.frame_id = "stereo_left"
                self.pub_left.publish(img_msg)

                info_msg = self.camera_info
                info_msg.header = img_msg.header
                self.pub_left_info.publish(info_msg)

            if stereo_right is not None:
                img_msg = self._numpy_to_image_msg(stereo_right, encoding="bgr8")
                img_msg.header.stamp = rospy.Time.from_sec(timestamp)
                img_msg.header.frame_id = "stereo_right"
                self.pub_right.publish(img_msg)

                info_msg = self.camera_info
                info_msg.header = img_msg.header
                self.pub_right_info.publish(info_msg)

            # Publish IMU
            if imu_data is not None:
                imu_msg = Imu()
                imu_msg.header.stamp = rospy.Time.from_sec(timestamp)
                imu_msg.header.frame_id = "imu"
                imu_msg.linear_acceleration.x = imu_data.accelerometer.x
                imu_msg.linear_acceleration.y = imu_data.accelerometer.y
                imu_msg.linear_acceleration.z = imu_data.accelerometer.z
                imu_msg.angular_velocity.x = imu_data.gyroscope.x
                imu_msg.angular_velocity.y = imu_data.gyroscope.y
                imu_msg.angular_velocity.z = imu_data.gyroscope.z
                self.pub_imu.publish(imu_msg)
        except Exception as e:
            print(f"[ORB-SLAM3] Publish error: {e}")

    def publish_imu_only(self, imu_data, timestamp):
        """Publish IMU data only (for high-frequency IMU)."""
        if not self.enabled or imu_data is None:
            return
        try:
            # Get raw IMU data in vehicle coordinate system
            acc_x_v, acc_y_v, acc_z_v = imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z
            gyro_x_v, gyro_y_v, gyro_z_v = imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z

            # Check for NaN or infinity
            import math
            if any(math.isnan(v) or math.isinf(v) for v in [acc_x_v, acc_y_v, acc_z_v, gyro_x_v, gyro_y_v, gyro_z_v]):
                print("[SLAM] Skipping invalid IMU data (NaN/Inf)")
                return

            # Transform from CARLA vehicle frame to camera frame
            # Vehicle: X=forward, Y=right, Z=up (left-handed)
            # Camera: X=right, Y=down, Z=forward (right-handed, OpenCV convention)
            acc_x = acc_y_v      # right
            acc_y = -acc_z_v     # down
            acc_z = acc_x_v      # forward

            gyro_x = gyro_y_v    # right
            gyro_y = -gyro_z_v   # down
            gyro_z = gyro_x_v    # forward

            imu_msg = Imu()
            imu_msg.header.stamp = rospy.Time.from_sec(timestamp)
            imu_msg.header.frame_id = "imu"
            imu_msg.linear_acceleration.x = acc_x
            imu_msg.linear_acceleration.y = acc_y
            imu_msg.linear_acceleration.z = acc_z
            imu_msg.angular_velocity.x = gyro_x
            imu_msg.angular_velocity.y = gyro_y
            imu_msg.angular_velocity.z = gyro_z

            # Add covariance (required by ORB-SLAM3)
            imu_msg.linear_acceleration_covariance[0] = 0.001
            imu_msg.linear_acceleration_covariance[4] = 0.001
            imu_msg.linear_acceleration_covariance[8] = 0.001
            imu_msg.angular_velocity_covariance[0] = 0.001
            imu_msg.angular_velocity_covariance[4] = 0.001
            imu_msg.angular_velocity_covariance[8] = 0.001

            self.pub_imu.publish(imu_msg)
        except Exception as e:
            print("[ORB-SLAM3] IMU publish error: {}".format(e))

    def publish_stereo_only(self, stereo_left, stereo_right, timestamp):
        """Publish stereo images only (IMU published separately)."""
        if not self.enabled:
            return
        try:
            if stereo_left is not None:
                img_msg = self._numpy_to_image_msg(stereo_left, encoding="bgr8")
                img_msg.header.stamp = rospy.Time.from_sec(timestamp)
                img_msg.header.frame_id = "stereo_left"
                self.pub_left.publish(img_msg)

                info_msg = self.camera_info
                info_msg.header = img_msg.header
                self.pub_left_info.publish(info_msg)

            if stereo_right is not None:
                img_msg = self._numpy_to_image_msg(stereo_right, encoding="bgr8")
                img_msg.header.stamp = rospy.Time.from_sec(timestamp)
                img_msg.header.frame_id = "stereo_right"
                self.pub_right.publish(img_msg)

                info_msg = self.camera_info
                info_msg.header = img_msg.header
                self.pub_right_info.publish(info_msg)
        except Exception as e:
            print(f"[ORB-SLAM3] Stereo publish error: {e}")

    def get_pose(self):
        if not self.enabled or self.tracking_state != 2:  # 2 = OK
            return None
        return self.last_pose

    def _pose_callback(self, msg):
        pos = msg.pose.position
        ori = msg.pose.orientation

        # Convert quaternion to Euler (simplified)
        import math
        yaw = math.atan2(2*(ori.w*ori.z + ori.x*ori.y), 1 - 2*(ori.y**2 + ori.z**2))
        pitch = math.asin(2*(ori.w*ori.y - ori.z*ori.x))
        roll = math.atan2(2*(ori.w*ori.x + ori.y*ori.z), 1 - 2*(ori.x**2 + ori.y**2))

        self.last_pose = carla.Transform(
            carla.Location(x=pos.x, y=pos.y, z=pos.z),
            carla.Rotation(pitch=math.degrees(pitch), yaw=math.degrees(yaw), roll=math.degrees(roll))
        )

    def _tracking_callback(self, msg):
        self.tracking_state = msg.data

    def _create_camera_info(self):
        info = CameraInfo()
        info.width = 752
        info.height = 480
        info.distortion_model = "plumb_bob"
        info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.K = [376.0, 0.0, 376.0,
                  0.0, 376.0, 240.0,
                  0.0, 0.0, 1.0]
        info.R = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        info.P = [376.0, 0.0, 376.0, 0.0,
                  0.0, 376.0, 240.0, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        info.binning_x = 0
        info.binning_y = 0
        return info
