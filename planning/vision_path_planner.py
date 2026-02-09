"""V2 path planner: generate driving path + nozzle path from right road edge.

Strategy: track the right road edge only, offset left to get driving path
and nozzle (paint) path. This matches real marking machine workflow:
follow the right curb, paint at a fixed offset from it.
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


class VisionPathPlanner:
    """From right road edge, generate driving path and nozzle path.

    Geometry:
        right_edge -----> road boundary
        |<- line_offset ->| nozzle paints here
        |<--- driving_offset --->| vehicle center here
        driving_offset = line_offset + nozzle_arm
    """

    def __init__(self, nozzle_arm=2.0, line_offset=3.0,
                 smooth_window=5, min_point_spacing=0.5,
                 max_buffer_size=500, max_range=20.0, memory_frames=10):
        """
        Args:
            nozzle_arm:   distance from vehicle center to nozzle (right side), meters.
            line_offset:  distance from right edge where the line should be painted, meters.
            smooth_window: moving average window for edge smoothing.
            max_range:    discard edge points beyond this forward distance (meters).
            memory_frames: frames to keep last valid path when perception fails.
        """
        self.nozzle_arm = nozzle_arm
        self.line_offset = line_offset
        self.driving_offset = line_offset + nozzle_arm
        self.smooth_window = smooth_window
        self.min_point_spacing = min_point_spacing
        self.max_buffer_size = max_buffer_size
        self.max_range = max_range
        self.memory_frames = memory_frames

        # Accumulated path buffers (kept in sync)
        self.driving_coords = []   # list of (x, y)
        self.nozzle_locations = [] # list of carla.Location

        # Short-term memory: fallback when perception fails
        self._last_valid_driving = []
        self._last_valid_nozzle = []
        self._memory_counter = 0

    def update(self, right_edges, vehicle_transform):
        """Process one frame of right road edge and generate driving path.

        Includes short-term memory: if perception fails, returns last valid
        path for up to memory_frames frames before giving up.

        Args:
            right_edges: list of carla.Location — right road edge points
            vehicle_transform: carla.Transform of the vehicle

        Returns:
            (driving_coords, nozzle_locations) — current frame path
        """
        if len(right_edges) < 2:
            return self._use_memory()

        veh_loc = vehicle_transform.location
        veh_yaw = math.radians(vehicle_transform.rotation.yaw)
        fwd_x = math.cos(veh_yaw)
        fwd_y = math.sin(veh_yaw)

        # Sort right edge by longitudinal distance, keep only ahead + within max_range
        right_sorted = self._sort_by_longitudinal(
            right_edges, veh_loc, fwd_x, fwd_y)
        if len(right_sorted) < 2:
            return self._use_memory()

        # Resample into 1m longitudinal bins
        edge_xy = self._resample_edge(right_sorted, veh_loc, fwd_x, fwd_y)
        if len(edge_xy) < 2:
            return self._use_memory()

        # Smooth the edge
        edge_xy = self._smooth_points(edge_xy)

        # Offset left into road for driving path and nozzle path
        driving_pts = self._offset_left(edge_xy, self.driving_offset)
        nozzle_pts = self._offset_left_loc(edge_xy, self.line_offset)

        # Replace buffers with current frame result
        n = min(len(driving_pts), len(nozzle_pts))
        self.driving_coords = driving_pts[:n]
        self.nozzle_locations = nozzle_pts[:n]

        # Save as last valid path and reset memory counter
        self._last_valid_driving = list(self.driving_coords)
        self._last_valid_nozzle = list(self.nozzle_locations)
        self._memory_counter = self.memory_frames

        return self.driving_coords, self.nozzle_locations

    def _use_memory(self):
        """Fallback: return last valid path if within memory window."""
        if self._memory_counter > 0:
            self._memory_counter -= 1
            self.driving_coords = self._last_valid_driving
            self.nozzle_locations = self._last_valid_nozzle
        else:
            self.driving_coords = []
            self.nozzle_locations = []
        return self.driving_coords, self.nozzle_locations

    def _sort_by_longitudinal(self, edges, veh_loc, fwd_x, fwd_y):
        """Sort edge points by longitudinal distance from vehicle."""
        def lon_dist(loc):
            dx = loc.x - veh_loc.x
            dy = loc.y - veh_loc.y
            return dx * fwd_x + dy * fwd_y

        # Only keep points ahead of vehicle and within max_range
        ahead = [(loc, lon_dist(loc)) for loc in edges
                 if 0.5 < lon_dist(loc) < self.max_range]
        ahead.sort(key=lambda x: x[1])
        return [loc for loc, _ in ahead]

    def _resample_edge(self, sorted_locs, veh_loc, fwd_x, fwd_y):
        """Resample edge points into 1m longitudinal bins, return (x,y,z) list."""
        def lon_dist(loc):
            dx = loc.x - veh_loc.x
            dy = loc.y - veh_loc.y
            return dx * fwd_x + dy * fwd_y

        max_lon = lon_dist(sorted_locs[-1])
        result = []

        for d in np.arange(1.0, max_lon, 1.0):
            best = None
            best_diff = float('inf')
            for loc in sorted_locs:
                diff = abs(lon_dist(loc) - d)
                if diff < best_diff and diff < 2.0:
                    best_diff = diff
                    best = loc
            if best is not None:
                result.append((best.x, best.y, best.z))  # 保留 z 坐标

        return result

    def _smooth_points(self, points):
        """Apply moving average smoothing to path points (x,y,z)."""
        if len(points) < self.smooth_window:
            return points

        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        zs = np.array([p[2] for p in points])

        kernel = np.ones(self.smooth_window) / self.smooth_window
        # Pad to avoid shrinking
        pad = self.smooth_window // 2
        xs_pad = np.pad(xs, pad, mode='edge')
        ys_pad = np.pad(ys, pad, mode='edge')
        zs_pad = np.pad(zs, pad, mode='edge')

        xs_smooth = np.convolve(xs_pad, kernel, mode='valid')
        ys_smooth = np.convolve(ys_pad, kernel, mode='valid')
        zs_smooth = np.convolve(zs_pad, kernel, mode='valid')

        return list(zip(xs_smooth.tolist(), ys_smooth.tolist(), zs_smooth.tolist()))

    def _offset_left(self, edge_xyz, offset):
        """Offset edge points to the LEFT (into road), return (x,y,z) tuples.

        CARLA uses a left-handed coordinate system (y increases to the right).
        Left perpendicular in CARLA is (dy, -dx), not (-dy, dx).
        """
        result = []
        n = len(edge_xyz)
        for i in range(n):
            dx, dy = self._local_direction(edge_xyz, i)
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-6:
                continue
            # Left perpendicular in CARLA's left-handed coords
            lx = dy / length
            ly = -dx / length
            nx = edge_xyz[i][0] + lx * offset
            ny = edge_xyz[i][1] + ly * offset
            nz = edge_xyz[i][2]  # 保留原始 z 坐标
            result.append((nx, ny, nz))
        return result

    def _offset_left_loc(self, edge_xyz, offset):
        """Offset edge points to the LEFT, return carla.Location list with actual z."""
        result = []
        n = len(edge_xyz)
        for i in range(n):
            dx, dy = self._local_direction(edge_xyz, i)
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-6:
                continue
            # Left perpendicular in CARLA's left-handed coords
            lx = dy / length
            ly = -dx / length
            nx = edge_xyz[i][0] + lx * offset
            ny = edge_xyz[i][1] + ly * offset
            nz = edge_xyz[i][2]  # 使用实际的 z 坐标（天桥路面高度）
            result.append(carla.Location(x=nx, y=ny, z=nz))
        return result

    def _local_direction(self, points, i):
        """Compute local forward direction at index i."""
        n = len(points)
        if i == 0:
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
        elif i == n - 1:
            dx = points[-1][0] - points[-2][0]
            dy = points[-1][1] - points[-2][1]
        else:
            dx = points[i+1][0] - points[i-1][0]
            dy = points[i+1][1] - points[i-1][1]
        return dx, dy

    def estimate_nozzle_edge_distance(self, right_edges, vehicle_transform):
        """Estimate lateral distance from nozzle to road edge via polynomial extrapolation.

        Fits a quadratic curve to detected edge points in vehicle-local frame,
        then extrapolates to lon=0 (vehicle position) to get the edge's lateral
        distance at the nozzle.

        Args:
            right_edges: list of carla.Location — right road edge points
            vehicle_transform: carla.Transform of the vehicle

        Returns:
            (distance_meters, poly_coefficients) or (None, None) if insufficient data.
        """
        if not right_edges or len(right_edges) < 5:
            return None, None

        veh_loc = vehicle_transform.location
        veh_yaw = math.radians(vehicle_transform.rotation.yaw)
        fwd_x = math.cos(veh_yaw)
        fwd_y = math.sin(veh_yaw)
        right_x = -fwd_y
        right_y = fwd_x

        # Transform edge points to vehicle-local (lon, lat)
        lon_arr = []
        lat_arr = []
        for loc in right_edges:
            dx = loc.x - veh_loc.x
            dy = loc.y - veh_loc.y
            lon = dx * fwd_x + dy * fwd_y
            lat = dx * right_x + dy * right_y
            if 3.0 < lon < 20.0 and lat > 0:
                lon_arr.append(lon)
                lat_arr.append(lat)

        if len(lon_arr) < 5:
            return None, None

        lon_np = np.array(lon_arr)
        lat_np = np.array(lat_arr)

        # Fit quadratic: lat = a*lon^2 + b*lon + c
        try:
            coeffs = np.polyfit(lon_np, lat_np, deg=2)
        except (np.linalg.LinAlgError, ValueError):
            return None, None

        # Extrapolated nozzle distance = polynomial value at lon=0 = c
        nozzle_dist = float(coeffs[2])

        # Sanity check: distance should be positive and reasonable
        if nozzle_dist < 0.5 or nozzle_dist > 15.0:
            return None, None

        return nozzle_dist, coeffs

    def _extend_buffer(self, driving_pts, nozzle_pts, veh_loc, fwd_x, fwd_y):
        """Append only new points that extend beyond the buffer's furthest point.

        Computes the longitudinal distance of each point from the vehicle.
        Only appends points whose longitudinal position exceeds the current
        buffer's furthest point by at least min_point_spacing.
        """
        n = min(len(driving_pts), len(nozzle_pts))
        if n == 0:
            return

        def lon(cx, cy):
            return (cx - veh_loc.x) * fwd_x + (cy - veh_loc.y) * fwd_y

        # Find the furthest longitudinal position in existing buffer
        max_lon_existing = -float('inf')
        if self.driving_coords:
            for pt in self.driving_coords:
                d = lon(pt[0], pt[1])
                if d > max_lon_existing:
                    max_lon_existing = d

        # Append new points that are beyond the existing buffer end
        for i in range(n):
            cx, cy = driving_pts[i][0], driving_pts[i][1]
            d = lon(cx, cy)
            if d > max_lon_existing + self.min_point_spacing:
                self.driving_coords.append(driving_pts[i])
                self.nozzle_locations.append(nozzle_pts[i])
                max_lon_existing = d

        # Cap buffer size (trim from front)
        if len(self.driving_coords) > self.max_buffer_size:
            trim = len(self.driving_coords) - self.max_buffer_size
            self.driving_coords = self.driving_coords[trim:]
            self.nozzle_locations = self.nozzle_locations[trim:]

    def _append_new_points(self, center_points, nozzle_points):
        """Append new points as pairs, keeping driving/nozzle buffers in sync."""
        n = min(len(center_points), len(nozzle_points))
        for i in range(n):
            cx, cy = center_points[i]
            if self.driving_coords:
                lx, ly = self.driving_coords[-1]
                dist = math.sqrt((cx - lx)**2 + (cy - ly)**2)
                if dist < self.min_point_spacing:
                    continue
            self.driving_coords.append((cx, cy))
            self.nozzle_locations.append(nozzle_points[i])

        # Cap buffer size (trim from front, both in sync)
        if len(self.driving_coords) > self.max_buffer_size:
            trim = len(self.driving_coords) - self.max_buffer_size
            self.driving_coords = self.driving_coords[trim:]
            self.nozzle_locations = self.nozzle_locations[trim:]

    def _prune_behind_vehicle(self, veh_loc, fwd_x, fwd_y, keep_behind=10.0):
        """Remove points far behind the vehicle (both buffers in sync)."""
        if not self.driving_coords:
            return

        cutoff = 0
        for i, pt in enumerate(self.driving_coords):
            dx = pt[0] - veh_loc.x
            dy = pt[1] - veh_loc.y
            lon = dx * fwd_x + dy * fwd_y
            if lon > -keep_behind:
                cutoff = i
                break

        if cutoff > 0:
            self.driving_coords = self.driving_coords[cutoff:]
            self.nozzle_locations = self.nozzle_locations[cutoff:]
