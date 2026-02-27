"""Project pixel coordinates + depth to 3D world coordinates.

Coordinate chain:
  pixel (u,v) + depth -> camera coords -> UE4 coords -> world coords

CARLA depth encoding: depth_m = (R + G*256 + B*65536) / (256^3 - 1) * 1000
"""

import math
import numpy as np

try:
    import glob, os, sys
    sys.path.append(glob.glob(
        '../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64')
    )[0])
except IndexError:
    pass

import carla


def decode_depth_image(depth_bgra):
    """Decode CARLA depth camera BGRA image to meters.

    Args:
        depth_bgra: np.ndarray (H, W, 4) uint8 — raw BGRA from depth camera.

    Returns:
        np.ndarray (H, W) float32 — depth in meters.
    """
    r = depth_bgra[:, :, 2].astype(np.float64)
    g = depth_bgra[:, :, 1].astype(np.float64)
    b = depth_bgra[:, :, 0].astype(np.float64)
    depth_m = (r + g * 256.0 + b * 65536.0) / (256.0 ** 3 - 1) * 1000.0
    return depth_m.astype(np.float32)


class DepthProjector:
    """Project pixel coordinates + depth to 3D world coordinates."""

    # Max depth to consider valid (meters)
    MAX_DEPTH = 25.0

    def __init__(self, img_w, img_h, fov_deg):
        fov_rad = math.radians(fov_deg)
        self.fx = img_w / (2.0 * math.tan(fov_rad / 2.0))
        self.fy = self.fx  # square pixels
        self.cx = img_w / 2.0
        self.cy = img_h / 2.0

    def pixel_to_camera(self, u, v, depth_m):
        """Pixel + depth -> camera-frame 3D point.

        Camera frame: x=right, y=down, z=forward (OpenCV convention).
        """
        z = depth_m
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z])

    def camera_to_world(self, cam_pt, camera_transform):
        """Camera-frame point -> CARLA world coordinates.

        Steps:
          1. Camera coords (x=right, y=down, z=forward)
             -> UE4 local (x=forward, y=right, z=up)
          2. UE4 local -> world via rotation + translation
        """
        # Camera -> UE4 local axis swap
        ue_x = cam_pt[2]   # forward = camera z
        ue_y = cam_pt[0]   # right   = camera x
        ue_z = -cam_pt[1]  # up      = -camera y

        # Rotation from camera_transform (degrees -> radians)
        rot = camera_transform.rotation
        yaw = math.radians(rot.yaw)
        pitch = math.radians(rot.pitch)
        roll = math.radians(rot.roll)

        # Rotation matrix (yaw-pitch-roll, UE4 convention)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        # Combined rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr],
        ])

        local_pt = np.array([ue_x, ue_y, ue_z])
        world_pt = R @ local_pt

        loc = camera_transform.location
        wx = world_pt[0] + loc.x
        wy = world_pt[1] + loc.y
        wz = world_pt[2] + loc.z

        return carla.Location(x=float(wx), y=float(wy), z=float(wz))

    def project_pixels(self, pixels, depth_image, camera_transform):
        """Project a list of (u,v) pixels to world coordinates using depth.

        Args:
            pixels: list of (u, v) tuples
            depth_image: np.ndarray (H, W) float32 — depth in meters
            camera_transform: carla.Transform of the camera

        Returns:
            list of carla.Location in world frame
        """
        h, w = depth_image.shape
        world_points = []

        for u, v in pixels:
            if not (0 <= u < w and 0 <= v < h):
                continue
            d = float(depth_image[v, u])
            if d <= 0.1 or d > self.MAX_DEPTH:
                continue
            cam_pt = self.pixel_to_camera(u, v, d)
            world_loc = self.camera_to_world(cam_pt, camera_transform)
            world_points.append(world_loc)

        return world_points
