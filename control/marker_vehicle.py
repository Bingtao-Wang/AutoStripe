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


class MarkerVehicle:
    """Pure Pursuit controller with monotonic search and steering filter."""

    LOOKAHEAD_WPS = 4          # waypoints ahead of nearest for target
    SEARCH_WINDOW = 30         # how far ahead to search for nearest
    STEER_FILTER = 0.3         # EMA factor (0=ignore new, 1=no filter)

    def __init__(self, vehicle, waypoint_coords, waypoint_objects,
                 wheelbase=2.875, kdd=4.0, throttle=0.3):
        self.vehicle = vehicle
        self.waypoint_coords = waypoint_coords
        self.waypoint_objects = waypoint_objects
        self.L = wheelbase
        self.Kdd = kdd

        self.control = carla.VehicleControl()
        self.control.throttle = throttle
        self.vehicle.apply_control(self.control)

        self._alpha_prev = 0.0
        self._nearest_index = 0
        self._steer_prev = 0.0

    # ------------------------------------------------------------------
    # Helpers (same logic as reference script)
    # ------------------------------------------------------------------

    def _get_target_index(self, veh_location):
        """Monotonic forward search: find nearest in a local window, never go back."""
        n = len(self.waypoint_coords)
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

        # Update nearest (raw, no lookahead added)
        self._nearest_index = best_i

        # Target = nearest + lookahead
        target = best_i + self.LOOKAHEAD_WPS
        if target >= n:
            target = n - 1
        return target

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self):
        """One Pure Pursuit iteration with monotonic search + steering filter."""
        veh_tf = self.vehicle.get_transform()
        veh_loc = self.vehicle.get_location()
        veh_vel = self.vehicle.get_velocity()

        vf = np.sqrt(veh_vel.x ** 2 + veh_vel.y ** 2)
        vf = np.clip(vf, 0.1, 2.5)

        idx = self._get_target_index(veh_loc)

        tx, ty = self.waypoint_coords[idx]
        ld = self.Kdd * vf

        yaw = math.radians(veh_tf.rotation.yaw)
        alpha = math.atan2(ty - veh_loc.y, tx - veh_loc.x) - yaw
        if math.isnan(alpha):
            alpha = self._alpha_prev
        else:
            self._alpha_prev = alpha

        delta = math.atan2(2 * self.L * np.sin(alpha), ld)
        steer_raw = float(np.clip(delta, -1.0, 1.0))

        # Low-pass filter on steering to remove high-frequency jitter
        steer = (self.STEER_FILTER * steer_raw
                 + (1.0 - self.STEER_FILTER) * self._steer_prev)
        self._steer_prev = steer

        self.control.steer = steer
        self.vehicle.apply_control(self.control)
        return self._nearest_index

    def is_route_complete(self):
        return self._nearest_index >= len(self.waypoint_coords) - 5
