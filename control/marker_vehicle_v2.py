"""V2 Pure Pursuit controller with dynamic path update support.

Reuses V1 steering math, monotonic search, and low-pass filter.
Adds update_path() for receiving new waypoints from the vision planner.
"""

import glob
import os
import sys
import math

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


class MarkerVehicleV2:
    """Pure Pursuit controller that accepts dynamic path updates."""

    LOOKAHEAD_WPS = 8
    SEARCH_WINDOW = 30
    STEER_FILTER = 0.15
    STEER_FILTER_AGGRESSIVE = 0.50
    STEER_FILTER_SMOOTH = 0.15
    MIN_PATH_POINTS = 5
    TARGET_SPEED = 3.0    # m/s target cruising speed
    MIN_THROTTLE = 0.2
    MAX_THROTTLE = 0.5

    def __init__(self, vehicle, wheelbase=2.875, kdd=3.0):
        self.vehicle = vehicle
        self.L = wheelbase
        self.Kdd = kdd

        self.control = carla.VehicleControl()
        self.control.throttle = self.MIN_THROTTLE
        self.vehicle.apply_control(self.control)

        self.waypoint_coords = []
        self._alpha_prev = 0.0
        self._nearest_index = 0
        self._steer_prev = 0.0
        self._effective_steer_filter = self.STEER_FILTER_SMOOTH

    def set_lateral_error(self, error_m):
        """Adapt steer filter based on lateral error magnitude.

        |error| > 0.5m -> aggressive (0.50)
        |error| < 0.3m -> smooth (0.15)
        Linear interpolation between 0.3m and 0.5m.
        """
        err = abs(error_m)
        if err >= 0.5:
            self._effective_steer_filter = self.STEER_FILTER_AGGRESSIVE
        elif err <= 0.3:
            self._effective_steer_filter = self.STEER_FILTER_SMOOTH
        else:
            t = (err - 0.3) / (0.5 - 0.3)
            self._effective_steer_filter = (
                self.STEER_FILTER_SMOOTH
                + t * (self.STEER_FILTER_AGGRESSIVE - self.STEER_FILTER_SMOOTH)
            )

    def update_path(self, new_coords):
        """Receive new path points from the vision planner.

        Args:
            new_coords: list of (x, y) tuples — full driving path buffer.
        """
        self.waypoint_coords = new_coords
        if not new_coords:
            self._nearest_index = 0
            return

        # Re-locate nearest index by searching the entire new buffer.
        # The planner prunes/appends points each frame, so the old index
        # is invalid — find the closest point to the vehicle now.
        veh_loc = self.vehicle.get_location()
        best_dist = float('inf')
        best_i = 0
        for i, coord in enumerate(new_coords):
            cx, cy = coord[0], coord[1]  # Handle (x,y,z) tuples, only use x,y
            d = (veh_loc.x - cx) ** 2 + (veh_loc.y - cy) ** 2
            if d < best_dist:
                best_dist = d
                best_i = i
        self._nearest_index = best_i

    def _get_target_index(self, veh_location):
        """Monotonic forward search for nearest waypoint."""
        n = len(self.waypoint_coords)
        if n == 0:
            return 0

        start = self._nearest_index
        end = min(start + self.SEARCH_WINDOW, n)

        best_dist = float('inf')
        best_i = start
        for i in range(start, end):
            dx = veh_location.x - self.waypoint_coords[i][0]
            dy = veh_location.y - self.waypoint_coords[i][1]
            d = dx * dx + dy * dy
            if d < best_dist:
                best_dist = d
                best_i = i

        self._nearest_index = best_i

        target = best_i + self.LOOKAHEAD_WPS
        if target >= n:
            target = n - 1
        return target

    def step(self):
        """One Pure Pursuit iteration. Returns nearest index or -1 if no path."""
        if len(self.waypoint_coords) < self.MIN_PATH_POINTS:
            # Not enough path points yet — hold still
            self.control.steer = 0.0
            self.control.throttle = 0.1
            self.vehicle.apply_control(self.control)
            return -1

        veh_tf = self.vehicle.get_transform()
        veh_loc = self.vehicle.get_location()
        veh_vel = self.vehicle.get_velocity()

        vf = np.sqrt(veh_vel.x ** 2 + veh_vel.y ** 2)
        vf = np.clip(vf, 0.1, 2.5)

        idx = self._get_target_index(veh_loc)
        tx, ty = self.waypoint_coords[idx][0], self.waypoint_coords[idx][1]  # Handle (x,y,z) tuples
        # Use actual distance to target point, with a minimum floor.
        # Kdd*vf becomes tiny at low speed, causing over-steering.
        ld_actual = math.sqrt((tx - veh_loc.x)**2 + (ty - veh_loc.y)**2)
        ld = max(ld_actual, 5.0)

        yaw = math.radians(veh_tf.rotation.yaw)
        alpha = math.atan2(ty - veh_loc.y, tx - veh_loc.x) - yaw
        if math.isnan(alpha):
            alpha = self._alpha_prev
        else:
            self._alpha_prev = alpha

        delta = math.atan2(2 * self.L * np.sin(alpha), ld)
        steer_raw = float(np.clip(delta, -1.0, 1.0))

        steer = (self._effective_steer_filter * steer_raw
                 + (1.0 - self._effective_steer_filter) * self._steer_prev)
        self._steer_prev = steer

        self.control.steer = steer

        # Speed maintenance: boost throttle when speed drops
        vf_raw = np.sqrt(veh_vel.x ** 2 + veh_vel.y ** 2)
        if vf_raw < self.TARGET_SPEED * 0.5:
            throttle = self.MAX_THROTTLE
        elif vf_raw < self.TARGET_SPEED:
            throttle = self.MIN_THROTTLE + (self.MAX_THROTTLE - self.MIN_THROTTLE) * (
                1.0 - vf_raw / self.TARGET_SPEED)
        else:
            throttle = self.MIN_THROTTLE
        self.control.throttle = throttle
        self.vehicle.apply_control(self.control)
        return self._nearest_index

    def has_path(self):
        return len(self.waypoint_coords) >= self.MIN_PATH_POINTS
