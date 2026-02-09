#!/usr/bin/env python3
"""Publish AutoStripe data as ROS messages for RVIZ visualization.

Uses ros_compatibility from CARLA-ROS bridge for Python 3 + ROS Melodic support.
"""

import numpy as np

try:
    import rospy
    from std_msgs.msg import Header, ColorRGBA
    from sensor_msgs.msg import Image, PointCloud2, PointField
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped, Point, Vector3
    from visualization_msgs.msg import Marker, MarkerArray
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from ros_interface.topic_config import (
    TOPIC_ROAD_MASK, TOPIC_EDGE_OVERLAY, TOPIC_EDGE_MARKERS,
    TOPIC_DRIVING_PATH, TOPIC_NOZZLE_PATH,
    TOPIC_PAINT_TRAIL, TOPIC_VEHICLE_MARKER, TOPIC_STATUS_TEXT,
    FRAME_MAP,
)


def _make_header(frame_id=FRAME_MAP):
    """Create a stamped Header with current time."""
    h = Header()
    h.stamp = rospy.Time.now()
    h.frame_id = frame_id
    return h


def _image_to_msg(image_bgr, encoding="bgr8"):
    """Convert OpenCV BGR image to sensor_msgs/Image."""
    msg = Image()
    msg.header = _make_header()
    msg.height, msg.width = image_bgr.shape[:2]
    msg.encoding = encoding
    if encoding == "mono8":
        msg.step = msg.width
    else:
        msg.step = msg.width * 3
    msg.data = image_bgr.tobytes()
    return msg


class RvizPublisher:
    """Publishes all AutoStripe visualization data to ROS topics."""

    def __init__(self):
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS not available — cannot create RvizPublisher")

        self.pub_road_mask = rospy.Publisher(
            TOPIC_ROAD_MASK, Image, queue_size=1)
        self.pub_edge_overlay = rospy.Publisher(
            TOPIC_EDGE_OVERLAY, Image, queue_size=1)
        self.pub_edge_markers = rospy.Publisher(
            TOPIC_EDGE_MARKERS, MarkerArray, queue_size=1)
        self.pub_driving_path = rospy.Publisher(
            TOPIC_DRIVING_PATH, Path, queue_size=1)
        self.pub_nozzle_path = rospy.Publisher(
            TOPIC_NOZZLE_PATH, Path, queue_size=1)
        self.pub_paint_trail = rospy.Publisher(
            TOPIC_PAINT_TRAIL, Marker, queue_size=1)
        self.pub_vehicle_marker = rospy.Publisher(
            TOPIC_VEHICLE_MARKER, Marker, queue_size=1)
        self.pub_status_text = rospy.Publisher(
            TOPIC_STATUS_TEXT, Marker, queue_size=1)

    def publish_road_mask(self, road_mask):
        """Publish binary road mask as mono8 image."""
        self.pub_road_mask.publish(_image_to_msg(road_mask, encoding="mono8"))

    def publish_edge_overlay(self, overlay_bgr):
        """Publish front camera with edge overlay as BGR image."""
        self.pub_edge_overlay.publish(_image_to_msg(overlay_bgr))

    def publish_edge_markers(self, left_world, right_world):
        """Publish left/right road edge points as MarkerArray (spheres)."""
        ma = MarkerArray()
        marker_id = 0

        # Left edges — green spheres
        for loc in left_world:
            m = Marker()
            m.header = _make_header()
            m.ns = "left_edge"
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = Point(x=loc.x, y=-loc.y, z=loc.z)
            m.scale = Vector3(x=0.3, y=0.3, z=0.3)
            m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            m.lifetime = rospy.Duration(0.5)
            ma.markers.append(m)
            marker_id += 1

        # Right edges — red spheres
        for loc in right_world:
            m = Marker()
            m.header = _make_header()
            m.ns = "right_edge"
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = Point(x=loc.x, y=-loc.y, z=loc.z)
            m.scale = Vector3(x=0.3, y=0.3, z=0.3)
            m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
            m.lifetime = rospy.Duration(0.5)
            ma.markers.append(m)
            marker_id += 1

        self.pub_edge_markers.publish(ma)

    def _make_path_msg(self, coords):
        """Convert list of (x,y) to nav_msgs/Path."""
        path = Path()
        path.header = _make_header()
        for x, y in coords:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position = Point(x=x, y=-y, z=0.1)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        return path

    def publish_driving_path(self, driving_coords):
        """Publish driving path as nav_msgs/Path (blue in RVIZ)."""
        self.pub_driving_path.publish(self._make_path_msg(driving_coords))

    def publish_nozzle_path(self, nozzle_locations):
        """Publish nozzle path as nav_msgs/Path (yellow in RVIZ)."""
        coords = [(loc.x, loc.y) for loc in nozzle_locations]
        self.pub_nozzle_path.publish(self._make_path_msg(coords))

    def publish_paint_trail(self, nozzle_trail):
        """Publish paint trail as LINE_STRIP marker (yellow thick line)."""
        m = Marker()
        m.header = _make_header()
        m.ns = "paint_trail"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.3  # line width
        m.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        m.lifetime = rospy.Duration(0)  # persistent

        for loc in nozzle_trail:
            m.points.append(Point(x=loc.x, y=-loc.y, z=loc.z))

        self.pub_paint_trail.publish(m)

    def publish_vehicle_marker(self, vehicle_transform):
        """Publish vehicle position as a CUBE marker."""
        loc = vehicle_transform.location
        rot = vehicle_transform.rotation

        m = Marker()
        m.header = _make_header()
        m.ns = "vehicle"
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position = Point(x=loc.x, y=-loc.y, z=loc.z + 1.0)
        m.scale = Vector3(x=4.5, y=2.0, z=1.5)
        m.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.7)
        m.lifetime = rospy.Duration(0.2)
        self.pub_vehicle_marker.publish(m)

    def publish_status_text(self, text, vehicle_transform):
        """Publish floating status text above the vehicle."""
        loc = vehicle_transform.location

        m = Marker()
        m.header = _make_header()
        m.ns = "status"
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position = Point(x=loc.x, y=-loc.y, z=loc.z + 5.0)
        m.scale.z = 1.0
        m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        m.text = text
        m.lifetime = rospy.Duration(0.5)
        self.pub_status_text.publish(m)
