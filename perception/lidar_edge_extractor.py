"""Extract road edges from CARLA semantic LiDAR point cloud.

Replaces depth-camera-based projection for more stable 3D edge points.
Semantic LiDAR gives direct world-coordinate points with per-point ObjTag,
eliminating the depth decode -> pixel projection -> world transform chain.

CARLA semantic tags: Road=7, RoadLine=6, Sidewalk=8, Terrain=22
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


def extract_right_edge_lidar(lidar_points, lidar_transform,
                              min_lon=3.0, max_lon=25.0, bin_size=1.0):
    """Extract right road edge from semantic LiDAR points.

    Algorithm:
      1. Filter points with ObjTag == 7 (Road)
      2. In sensor-local frame: keep points ahead (x > min_lon)
      3. Bin by longitudinal distance (x), find rightmost road point per bin
      4. Transform to world coordinates via lidar_transform

    Args:
        lidar_points: structured numpy array with fields
                      (x, y, z, CosAngle, ObjIdx, ObjTag)
        lidar_transform: carla.Transform of the LiDAR sensor (world frame)
        min_lon: minimum forward distance (m)
        max_lon: maximum forward distance (m)
        bin_size: longitudinal bin width (m)

    Returns:
        list of carla.Location — right edge points in world coordinates
    """
    if lidar_points is None or len(lidar_points) == 0:
        return []

    # Filter road points (ObjTag == 7)
    road_mask = lidar_points['ObjTag'] == 7
    road_pts = lidar_points[road_mask]

    if len(road_pts) == 0:
        return []

    # Sensor-local coords (UE4: x=forward, y=right, z=up)
    lx = road_pts['x'].astype(np.float64)
    ly = road_pts['y'].astype(np.float64)
    lz = road_pts['z'].astype(np.float64)

    # Keep points ahead and within range, right side only (y > 0)
    valid = (lx > min_lon) & (lx < max_lon) & (ly > 0)
    lx = lx[valid]
    ly = ly[valid]
    lz = lz[valid]

    if len(lx) == 0:
        return []

    # Bin by longitudinal distance, find rightmost road point per bin
    edge_local = []
    bin_starts = np.arange(min_lon, max_lon, bin_size)
    for b in bin_starts:
        in_bin = (lx >= b) & (lx < b + bin_size)
        if not np.any(in_bin):
            continue
        bin_ly = ly[in_bin]
        bin_lx = lx[in_bin]
        bin_lz = lz[in_bin]
        idx = np.argmax(bin_ly)
        edge_local.append((float(bin_lx[idx]),
                           float(bin_ly[idx]),
                           float(bin_lz[idx])))

    if not edge_local:
        return []

    # Transform sensor-local -> world coordinates
    world_points = []
    for px, py, pz in edge_local:
        local_loc = carla.Location(x=px, y=py, z=pz)
        world_loc = lidar_transform.transform(local_loc)
        world_points.append(world_loc)

    return world_points
