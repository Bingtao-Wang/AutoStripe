"""
LUNA-Net CARLA Deployment Configuration
"""

# ============ Model ============
MODEL_WEIGHTS = "weights/best_net_LUNA_ClearNight.pth"
SWIN_MODEL = "swin_tiny_patch4_window7_224"
NUM_CLASSES = 2

# LUNA modules (must match training config)
USE_LLEM = True
USE_ROBUST_SNE = False      # Not used during training
USE_IAF = True
USE_NAA_DECODER = True
USE_EDGE_HEAD = True

# ============ Input ============
INPUT_WIDTH = 1248
INPUT_HEIGHT = 384

# ============ CARLA ============
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

# Camera sensor config
CAMERA_WIDTH = 1600
CAMERA_HEIGHT = 900
CAMERA_FOV = 90.0

# Camera mount position (relative to vehicle)
CAMERA_X = 1.5     # forward
CAMERA_Y = 0.0     # lateral
CAMERA_Z = 2.4     # height

# ============ Visualization ============
ROAD_COLOR = (255, 0, 255)      # magenta
OVERLAY_ALPHA = 0.4
WINDOW_NAME = "LUNA-Net Drivable Area"
