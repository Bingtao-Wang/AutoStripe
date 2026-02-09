"""ROS topic name constants for AutoStripe V2."""

# --- Subscribed topics (from CARLA-ROS bridge) ---
TOPIC_SEMANTIC_IMAGE = "/carla/ego_vehicle/semantic_segmentation/image"
TOPIC_DEPTH_IMAGE = "/carla/ego_vehicle/depth/image"
TOPIC_RGB_FRONT_IMAGE = "/carla/ego_vehicle/rgb_front/image"
TOPIC_ODOMETRY = "/carla/ego_vehicle/odometry"

# --- Published topics (AutoStripe) ---
# Perception
TOPIC_ROAD_MASK = "/autostripe/perception/road_mask"
TOPIC_EDGE_OVERLAY = "/autostripe/perception/edge_overlay"
TOPIC_EDGE_MARKERS = "/autostripe/perception/edge_markers"

# Planning
TOPIC_DRIVING_PATH = "/autostripe/planning/driving_path"
TOPIC_NOZZLE_PATH = "/autostripe/planning/nozzle_path"
TOPIC_POLY_CURVE = "/autostripe/planning/poly_curve"

# Control
TOPIC_PAINT_TRAIL = "/autostripe/control/paint_trail"
TOPIC_VEHICLE_MARKER = "/autostripe/control/vehicle_marker"
TOPIC_STATUS_TEXT = "/autostripe/control/status_text"

# LiDAR
TOPIC_LIDAR_POINTS = "/autostripe/lidar/points"

# --- Frame IDs ---
FRAME_MAP = "map"
FRAME_VEHICLE = "ego_vehicle"
