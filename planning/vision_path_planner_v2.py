"""V2 Nozzle-Centric Path Planner.

Key change from V1: instead of controlling driving_offset (vehicle-to-edge)
with a PD controller, we compute:
  1. nozzle_path = edge offset by line_offset (3.0m)
  2. driving_path = nozzle_path offset by nozzle_arm (2.0m)

This ensures the vehicle path is geometrically derived FROM the desired
nozzle position, automatically compensating for curvature effects.
The PD controller on driving_offset is removed entirely.
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


class VisionPathPlannerV2:
    """Nozzle-centric path planner.

    Geometry (fixed, no dynamic offset):
        right_edge -----> road boundary
        |<- line_offset (3.0m) ->| nozzle target path
        |<- + nozzle_arm (2.0m) ->| vehicle driving path

    The driving path is derived FROM the nozzle path, not independently
    from the edge. This automatically compensates for curvature effects.
    """

    def __init__(self, nozzle_arm=2.0, line_offset=3.1,
                 smooth_window=5, max_range=20.0, memory_frames=10,
                 curv_ff_gain=55.0):
        self.nozzle_arm = nozzle_arm
        self.line_offset = line_offset          # C: 3.0 → 3.1 baseline correction
        self.driving_offset = line_offset + nozzle_arm  # for logging only
        self.smooth_window = smooth_window
        self.max_range = max_range
        self.memory_frames = memory_frames
        self.min_point_spacing = 0.5
        self.max_buffer_size = 500

        # A: Curvature feedforward gain (from linear fit: deficit ≈ K * |a2|)
        self.K_CURV_FF = curv_ff_gain

        # Target nozzle distance (for state machine / logging)
        self.TARGET_NOZZLE_DIST = 3.0

        # Polynomial EMA (same as V1, for curvature estimation)
        self.POLY_EMA_ALPHA = 0.4
        self._prev_poly_coeffs = None
        self._last_curvature = 0.0              # smoothed curvature for logging

        # Path buffers
        self.driving_coords = []
        self.nozzle_locations = []

        # Short-term memory
        self._last_valid_driving = []
        self._last_valid_nozzle = []
        self._memory_counter = 0

        # Path temporal smoothing (EMA blend with previous frame)
        self.PATH_EMA_ALPHA = 0.15  # lower = smoother (15% new, 85% old)

    # ------------------------------------------------------------------
    # Core: two-stage offset (nozzle-centric)
    # ------------------------------------------------------------------
    def update(self, right_edges, vehicle_transform):
        """Generate driving path via nozzle-centric two-stage offset.

        Stage 1: nozzle_path = edge offset LEFT by line_offset (3.0m)
        Stage 2: driving_path = nozzle_path offset LEFT by nozzle_arm (2.0m)

        Returns:
            (driving_coords, nozzle_locations)
        """
        if len(right_edges) < 2:
            return self._use_memory()

        veh_loc = vehicle_transform.location
        veh_yaw = math.radians(vehicle_transform.rotation.yaw)
        fwd_x = math.cos(veh_yaw)
        fwd_y = math.sin(veh_yaw)

        # Sort + resample + smooth edge
        right_sorted = self._sort_by_longitudinal(
            right_edges, veh_loc, fwd_x, fwd_y)
        if len(right_sorted) < 2:
            return self._use_memory()

        edge_xy = self._resample_edge(right_sorted, veh_loc, fwd_x, fwd_y)
        if len(edge_xy) < 2:
            return self._use_memory()
        edge_xy = self._smooth_points(edge_xy)

        # Spatial outlier rejection: remove points far from polynomial fit
        edge_xy = self._reject_outliers(edge_xy, veh_loc, fwd_x, fwd_y)
        if len(edge_xy) < 2:
            return self._use_memory()

        # Stage 1: nozzle path (line_offset from edge)
        nozzle_xy = self._offset_left(edge_xy, self.line_offset)
        if len(nozzle_xy) < 2:
            return self._use_memory()

        # A: Estimate curvature from edge, compute compensated arm
        curv = self._estimate_curvature(edge_xy, veh_loc, fwd_x, fwd_y)
        self._last_curvature = curv
        compensated_arm = self.nozzle_arm + self.K_CURV_FF * curv

        # Stage 2: driving path (compensated arm from nozzle path)
        driving_pts = self._offset_left(nozzle_xy, compensated_arm)
        self.driving_offset = self.line_offset + compensated_arm  # update for logging

        # Convert nozzle to carla.Location
        nozzle_pts = [carla.Location(x=p[0], y=p[1], z=p[2])
                      for p in nozzle_xy]

        n = min(len(driving_pts), len(nozzle_pts))
        new_driving = driving_pts[:n]
        new_nozzle = nozzle_pts[:n]

        # Temporal EMA: blend new path with previous to suppress frame jitter
        if self._last_valid_driving and len(self._last_valid_driving) >= 2:
            a = self.PATH_EMA_ALPHA
            m = min(n, len(self._last_valid_driving))
            for i in range(m):
                ox, oy, oz = self._last_valid_driving[i]
                nx, ny, nz = new_driving[i]
                new_driving[i] = (a * nx + (1 - a) * ox,
                                  a * ny + (1 - a) * oy,
                                  a * nz + (1 - a) * oz)

        self.driving_coords = new_driving
        self.nozzle_locations = new_nozzle

        self._last_valid_driving = list(self.driving_coords)
        self._last_valid_nozzle = list(self.nozzle_locations)
        self._memory_counter = self.memory_frames

        return self.driving_coords, self.nozzle_locations

    def _estimate_curvature(self, edge_xy, veh_loc, fwd_x, fwd_y):
        """Estimate path curvature from resampled edge points.

        Fits a quadratic in vehicle-local coords, returns |a2| (smoothed).
        """
        if len(edge_xy) < 5:
            return self._last_curvature

        right_x, right_y = -fwd_y, fwd_x
        lon_arr, lat_arr = [], []
        for pt in edge_xy:
            dx = pt[0] - veh_loc.x
            dy = pt[1] - veh_loc.y
            lon = dx * fwd_x + dy * fwd_y
            lat = dx * right_x + dy * right_y
            if 2.0 < lon < self.max_range:
                lon_arr.append(lon)
                lat_arr.append(lat)

        if len(lon_arr) < 5:
            return self._last_curvature

        try:
            coeffs = np.polyfit(np.array(lon_arr), np.array(lat_arr), deg=2)
        except (np.linalg.LinAlgError, ValueError):
            return self._last_curvature

        curv = abs(coeffs[0])
        # V4: slower EMA (0.3 → 0.10) to reduce curve-entry overshoot
        alpha = 0.10
        curv = alpha * curv + (1.0 - alpha) * self._last_curvature
        # V4: rate limiter — max Δcurvature per frame = 0.0005
        max_delta = 0.0005
        curv = max(self._last_curvature - max_delta,
                   min(curv, self._last_curvature + max_delta))
        return curv

    def _use_memory(self):
        if self._memory_counter > 0:
            self._memory_counter -= 1
            self.driving_coords = self._last_valid_driving
            self.nozzle_locations = self._last_valid_nozzle
        else:
            self.driving_coords = []
            self.nozzle_locations = []
        return self.driving_coords, self.nozzle_locations

    # ------------------------------------------------------------------
    # Edge processing helpers (same as V1)
    # ------------------------------------------------------------------
    def _sort_by_longitudinal(self, edges, veh_loc, fwd_x, fwd_y):
        def lon_dist(loc):
            dx = loc.x - veh_loc.x
            dy = loc.y - veh_loc.y
            return dx * fwd_x + dy * fwd_y
        ahead = [(loc, lon_dist(loc)) for loc in edges
                 if 0.5 < lon_dist(loc) < self.max_range]
        ahead.sort(key=lambda x: x[1])
        return [loc for loc, _ in ahead]

    def _resample_edge(self, sorted_locs, veh_loc, fwd_x, fwd_y):
        def lon_dist(loc):
            dx = loc.x - veh_loc.x
            dy = loc.y - veh_loc.y
            return dx * fwd_x + dy * fwd_y
        max_lon = lon_dist(sorted_locs[-1])
        result = []
        for d in np.arange(3.0, max_lon, 1.0):
            best = None
            best_diff = float('inf')
            for loc in sorted_locs:
                diff = abs(lon_dist(loc) - d)
                if diff < best_diff and diff < 2.0:
                    best_diff = diff
                    best = loc
            if best is not None:
                result.append((best.x, best.y, best.z))
        return result

    def _smooth_points(self, points):
        if len(points) < self.smooth_window:
            return points
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        zs = np.array([p[2] for p in points])
        kernel = np.ones(self.smooth_window) / self.smooth_window
        pad = self.smooth_window // 2
        xs_s = np.convolve(np.pad(xs, pad, mode='edge'), kernel, mode='valid')
        ys_s = np.convolve(np.pad(ys, pad, mode='edge'), kernel, mode='valid')
        zs_s = np.convolve(np.pad(zs, pad, mode='edge'), kernel, mode='valid')
        return list(zip(xs_s.tolist(), ys_s.tolist(), zs_s.tolist()))

    def _reject_outliers(self, edge_xy, veh_loc, fwd_x, fwd_y,
                         max_dev=0.5):
        """Reject edge points that deviate too far from a polynomial fit.

        Fits a quadratic in vehicle-local (lon, lat) coords, then removes
        points whose lateral residual exceeds max_dev meters.
        """
        if len(edge_xy) < 5:
            return edge_xy

        right_x, right_y = -fwd_y, fwd_x
        lons, lats, pts = [], [], []
        for pt in edge_xy:
            dx = pt[0] - veh_loc.x
            dy = pt[1] - veh_loc.y
            lon = dx * fwd_x + dy * fwd_y
            lat = dx * right_x + dy * right_y
            lons.append(lon)
            lats.append(lat)
            pts.append(pt)

        try:
            coeffs = np.polyfit(np.array(lons), np.array(lats), deg=2)
        except (np.linalg.LinAlgError, ValueError):
            return edge_xy

        kept = []
        for i, (lon, lat, pt) in enumerate(zip(lons, lats, pts)):
            fitted = coeffs[0] * lon**2 + coeffs[1] * lon + coeffs[2]
            if abs(lat - fitted) <= max_dev:
                kept.append(pt)

        return kept if len(kept) >= 2 else edge_xy

    def _local_direction(self, points, i):
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

    def _offset_left(self, edge_xyz, offset):
        """Offset points LEFT (into road). CARLA left-handed coords."""
        result = []
        for i in range(len(edge_xyz)):
            dx, dy = self._local_direction(edge_xyz, i)
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-6:
                continue
            lx = dy / length
            ly = -dx / length
            nx = edge_xyz[i][0] + lx * offset
            ny = edge_xyz[i][1] + ly * offset
            nz = edge_xyz[i][2]
            result.append((nx, ny, nz))
        return result

    # ------------------------------------------------------------------
    # Polynomial edge distance estimation (for logging / state machine)
    # ------------------------------------------------------------------
    def estimate_nozzle_edge_distance(self, right_edges, vehicle_transform):
        """Estimate nozzle-to-edge distance via polynomial extrapolation.

        Same as V1 — used for logging and state machine, NOT for control.
        """
        if not right_edges or len(right_edges) < 5:
            return None, None

        veh_loc = vehicle_transform.location
        veh_yaw = math.radians(vehicle_transform.rotation.yaw)
        fwd_x = math.cos(veh_yaw)
        fwd_y = math.sin(veh_yaw)
        right_x = -fwd_y
        right_y = fwd_x

        lon_arr, lat_arr = [], []
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

        try:
            coeffs = np.polyfit(np.array(lon_arr),
                                np.array(lat_arr), deg=2)
        except (np.linalg.LinAlgError, ValueError):
            return None, None

        if self._prev_poly_coeffs is not None:
            a = self.POLY_EMA_ALPHA
            coeffs = a * coeffs + (1.0 - a) * self._prev_poly_coeffs
        self._prev_poly_coeffs = coeffs.copy()

        nozzle_dist = float(coeffs[2])
        if nozzle_dist < 0.5 or nozzle_dist > 15.0:
            return None, None

        return nozzle_dist, coeffs
