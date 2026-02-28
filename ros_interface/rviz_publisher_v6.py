#!/usr/bin/env python3
"""V6 RViz publisher: 3-panel image display + 2D map view.

Lightweight class with image publishers + interactive 2D map renderer.
No CARLA-ROS Bridge dependency — publishes directly from the V6 main loop.
"""

import math
import cv2
import numpy as np

try:
    import rospy
    from std_msgs.msg import Header, ColorRGBA
    from sensor_msgs.msg import Image
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point, Vector3
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from ros_interface.topic_config import (
    TOPIC_V6_FRONT_OVERLAY,
    TOPIC_V6_PERCEPTION_DETAIL,
    TOPIC_V6_OVERHEAD,
    TOPIC_V6_MAP_ROADS,
    TOPIC_V6_DASHBOARD,
    FRAME_MAP,
)

# Overhead downscale target
OVERHEAD_W, OVERHEAD_H = 900, 800


def _make_header(frame_id=FRAME_MAP):
    h = Header()
    h.stamp = rospy.Time.now()
    h.frame_id = frame_id
    return h


def _bgr_to_img_msg(bgr):
    """Convert BGR numpy array to sensor_msgs/Image."""
    msg = Image()
    msg.header = _make_header()
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = "bgr8"
    msg.step = msg.width * 3
    msg.data = bgr.tobytes()
    return msg


class RvizPublisherV6:
    """Publishes 3 image panels + map roads to ROS for RViz display."""

    def __init__(self):
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS not available")
        self.pub_front = rospy.Publisher(
            TOPIC_V6_FRONT_OVERLAY, Image, queue_size=1)
        self.pub_detail = rospy.Publisher(
            TOPIC_V6_PERCEPTION_DETAIL, Image, queue_size=1)
        self.pub_overhead = rospy.Publisher(
            TOPIC_V6_OVERHEAD, Image, queue_size=1)
        self.pub_map = rospy.Publisher(
            TOPIC_V6_MAP_ROADS, Image, queue_size=1)
        self.pub_dashboard = rospy.Publisher(
            TOPIC_V6_DASHBOARD, Image, queue_size=1)

    def publish_front_overlay(self, bgr):
        """Panel 1: RGB front camera + edge overlay (1248x384)."""
        if bgr is not None:
            self.pub_front.publish(_bgr_to_img_msg(bgr))

    def publish_perception_detail(self, bgr):
        """Panel 2: SNE normal map (LUNA) or depth heatmap (VLLiNet/GT)."""
        if bgr is not None:
            self.pub_detail.publish(_bgr_to_img_msg(bgr))

    def publish_overhead(self, bgr):
        """Panel 3: Overhead bird's eye view (downscaled to 900x800)."""
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        if w != OVERHEAD_W or h != OVERHEAD_H:
            bgr = cv2.resize(bgr, (OVERHEAD_W, OVERHEAD_H),
                             interpolation=cv2.INTER_AREA)
        self.pub_overhead.publish(_bgr_to_img_msg(bgr))

    def publish_all(self, front_bgr, detail_bgr, overhead_bgr,
                    map_bgr=None, dashboard_bgr=None):
        """Publish all panels + composed dashboard."""
        self.publish_front_overlay(front_bgr)
        self.publish_perception_detail(detail_bgr)
        self.publish_overhead(overhead_bgr)
        if map_bgr is not None:
            self.pub_map.publish(_bgr_to_img_msg(map_bgr))
        if dashboard_bgr is not None:
            self.pub_dashboard.publish(_bgr_to_img_msg(dashboard_bgr))

def _nozzle_dist_color(d, vcenter=3.0, vmin=2.6, vmax=3.4):
    """Map nozzle-edge distance to BGR color (paper's diverging colormap).

    Blue(too close) -> Green(on target) -> Red(too far).
    Center 30% of colormap is pure green.
    Uses TwoSlopeNorm: [vmin,vcenter] -> [0,0.5], [vcenter,vmax] -> [0.5,1].
    """
    # TwoSlopeNorm
    if d <= vmin:
        t = 0.0
    elif d < vcenter:
        t = 0.5 * (d - vmin) / (vcenter - vmin)
    elif d < vmax:
        t = 0.5 + 0.5 * (d - vcenter) / (vmax - vcenter)
    else:
        t = 1.0

    # 6-stop colormap (BGR), bright colors for dark map background
    stops = [
        (0.0,    (255, 160,  60)),   # bright blue   #3ca0ff
        (0.175,  (255, 220, 140)),   # sky blue      #8cdcff
        (0.35,   (  0, 255,   0)),   # bright green  #00ff00
        (0.65,   (  0, 255,   0)),   # bright green  #00ff00
        (0.825,  ( 80, 160, 255)),   # bright orange #ffa050
        (1.0,    ( 50,  50, 255)),   # bright red    #ff3232
    ]

    # Find segment and interpolate
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            s = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return (int(c0[0] + s * (c1[0] - c0[0])),
                    int(c0[1] + s * (c1[1] - c0[1])),
                    int(c0[2] + s * (c1[2] - c0[2])))
    return stops[-1][1]


class MapView:
    """2D top-down map renderer with keyboard zoom/pan/rotate.

    Keys (handled via handle_key):
        ]  = zoom in       [  = zoom out
        UP/DOWN/LEFT/RIGHT (shift held) = pan
        ,  = rotate left   .  = rotate right
        M  = toggle auto-follow vehicle
    """

    MAP_W, MAP_H = 900, 800  # output image size

    def __init__(self, carla_map, spacing=2.0):
        self.zoom = 1.0          # pixels per meter
        self.cx = 0.0            # view center X (CARLA world)
        self.cy = 0.0            # view center Y (CARLA world)
        self.angle = 0.0         # rotation degrees
        self.follow = False      # start centered on map, not following vehicle

        # Sample road geometry once
        self._build_road_data(carla_map, spacing)
        # cx, cy already set to map center by _build_road_data

    def _build_road_data(self, carla_map, spacing):
        """Sample waypoints and store road polygons as numpy arrays."""
        waypoints = carla_map.generate_waypoints(spacing)

        # Group by (road_id, lane_id)
        lanes = {}
        for wp in waypoints:
            key = (wp.road_id, wp.lane_id)
            if key not in lanes:
                lanes[key] = []
            lanes[key].append(wp)

        self.road_polys = []  # list of (left_xy, right_xy) numpy arrays
        for wps in lanes.values():
            wps.sort(key=lambda w: w.s)
            left = []
            right = []
            for wp in wps:
                tf = wp.transform
                rv = tf.get_right_vector()
                hw = wp.lane_width / 2.0
                loc = tf.location
                left.append([loc.x - rv.x * hw, loc.y - rv.y * hw])
                right.append([loc.x + rv.x * hw, loc.y + rv.y * hw])
            self.road_polys.append((
                np.array(left, dtype=np.float32),
                np.array(right, dtype=np.float32),
            ))

        # Compute world bounding box
        all_pts = np.concatenate(
            [l for l, r in self.road_polys] +
            [r for l, r in self.road_polys], axis=0)
        self.world_min = all_pts.min(axis=0)
        self.world_max = all_pts.max(axis=0)
        world_center = (self.world_min + self.world_max) / 2.0
        self.cx, self.cy = float(world_center[0]), float(world_center[1])

    def _auto_fit(self):
        """Set zoom so the full map fits in the image."""
        span = self.world_max - self.world_min  # [dx, dy]
        zx = (self.MAP_W - 40) / max(span[0], 1.0)
        zy = (self.MAP_H - 40) / max(span[1], 1.0)
        self.zoom = min(zx, zy)

    def _world_to_px(self, pts):
        """Convert Nx2 world coords to Nx2 pixel coords (int32)."""
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        dx = pts[:, 0] - self.cx
        dy = pts[:, 1] - self.cy
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        px = (rx * self.zoom + self.MAP_W / 2).astype(np.int32)
        py = (ry * self.zoom + self.MAP_H / 2).astype(np.int32)
        return np.stack([px, py], axis=1)

    def handle_key(self, key):
        """Process pygame key for map controls. Returns True if handled."""
        import pygame
        PAN = 20.0 / max(self.zoom, 0.1)  # pan step in world meters
        if key == pygame.K_RIGHTBRACKET:
            self.zoom *= 1.3
            return True
        elif key == pygame.K_LEFTBRACKET:
            self.zoom /= 1.3
            return True
        elif key == pygame.K_COMMA:
            self.angle -= 5.0
            return True
        elif key == pygame.K_PERIOD:
            self.angle += 5.0
            return True
        elif key == pygame.K_BACKSLASH:
            self._auto_fit()
            self.angle = 0.0
            return True
        elif key == pygame.K_m:
            self.follow = not self.follow
            return True
        # Shift + arrows for pan
        mods = pygame.key.get_mods()
        if mods & pygame.KMOD_SHIFT:
            if key == pygame.K_UP:
                self.cy -= PAN; self.follow = False; return True
            elif key == pygame.K_DOWN:
                self.cy += PAN; self.follow = False; return True
            elif key == pygame.K_LEFT:
                self.cx -= PAN; self.follow = False; return True
            elif key == pygame.K_RIGHT:
                self.cx += PAN; self.follow = False; return True
        return False

    def handle_cv_key(self, key):
        """Process OpenCV waitKey code for map controls. Returns True if handled."""
        if key == -1 or key == 255:
            return False
        k = key & 0xFF
        PAN = 20.0 / max(self.zoom, 0.1)
        if k == ord(']'):
            self.zoom *= 1.3; return True
        elif k == ord('['):
            self.zoom /= 1.3; return True
        elif k == ord(','):
            self.angle -= 5.0; return True
        elif k == ord('.'):
            self.angle += 5.0; return True
        elif k == ord('\\'):
            self._auto_fit(); self.angle = 0.0; return True
        elif k == ord('m'):
            self.follow = not self.follow; return True
        # Arrow keys (special codes vary by platform)
        elif key == 82 or key == 65362:  # Up
            self.cy -= PAN; self.follow = False; return True
        elif key == 84 or key == 65364:  # Down
            self.cy += PAN; self.follow = False; return True
        elif key == 81 or key == 65361:  # Left
            self.cx -= PAN; self.follow = False; return True
        elif key == 83 or key == 65363:  # Right
            self.cx += PAN; self.follow = False; return True
        return False

    def render(self, vehicle_xy=None, vehicle_yaw=None,
               paint_trail=None, trail_dists=None, driving_coords=None):
        """Render 2D map image (900x800 BGR).

        Args:
            vehicle_xy: (x, y) in CARLA world coords
            vehicle_yaw: degrees
            paint_trail: list of (x, y) or None entries
            trail_dists: list of float or None (nozzle-edge distance per point)
            driving_coords: list of (x, y) tuples
        """
        if self.follow and vehicle_xy is not None:
            self.cx, self.cy = vehicle_xy

        img = np.zeros((self.MAP_H, self.MAP_W, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)  # dark background

        # Draw road surfaces
        for left_xy, right_xy in self.road_polys:
            lp = self._world_to_px(left_xy)
            rp = self._world_to_px(right_xy)
            # Build polygon: left forward + right reversed
            poly = np.concatenate([lp, rp[::-1]], axis=0)
            cv2.fillPoly(img, [poly], (60, 60, 60))

        # Draw road edges
        for left_xy, right_xy in self.road_polys:
            lp = self._world_to_px(left_xy)
            rp = self._world_to_px(right_xy)
            cv2.polylines(img, [lp], False, (100, 100, 100), 1, cv2.LINE_AA)
            cv2.polylines(img, [rp], False, (100, 100, 100), 1, cv2.LINE_AA)

        # Draw paint trail (gradient: green=close, red=far from edge)
        if paint_trail:
            has_dists = (trail_dists is not None
                         and len(trail_dists) == len(paint_trail))
            # Build segments between consecutive non-None points
            segments = []  # list of (xy_list, dist_list) per continuous segment
            cur_pts = []
            cur_dists = []
            for i, p in enumerate(paint_trail):
                if p is None:
                    if len(cur_pts) >= 2:
                        segments.append((cur_pts, cur_dists))
                    cur_pts = []
                    cur_dists = []
                else:
                    cur_pts.append((p[0], p[1]))
                    d = trail_dists[i] if has_dists else None
                    cur_dists.append(d)
            if len(cur_pts) >= 2:
                segments.append((cur_pts, cur_dists))

            for seg_pts, seg_dists in segments:
                arr = np.array(seg_pts, dtype=np.float32)
                px = self._world_to_px(arr)
                if not has_dists or all(d is None for d in seg_dists):
                    # Fallback: yellow if no distance data
                    cv2.polylines(img, [px], False, (0, 255, 255), 2,
                                  cv2.LINE_AA)
                else:
                    # Per-segment diverging colormap (paper style)
                    for j in range(len(px) - 1):
                        d = seg_dists[j + 1] if seg_dists[j + 1] is not None \
                            else seg_dists[j]
                        if d is None:
                            color = (80, 152, 26)  # green fallback
                        else:
                            color = _nozzle_dist_color(d)
                        pt1 = (int(px[j][0]), int(px[j][1]))
                        pt2 = (int(px[j + 1][0]), int(px[j + 1][1]))
                        cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

        # Draw driving path (blue)
        if driving_coords and len(driving_coords) >= 2:
            arr = np.array(driving_coords, dtype=np.float32)
            px = self._world_to_px(arr)
            cv2.polylines(img, [px], False, (255, 170, 85), 1, cv2.LINE_AA)

        # Draw vehicle (green triangle)
        if vehicle_xy is not None:
            vp = self._world_to_px(
                np.array([vehicle_xy], dtype=np.float32))[0]
            yaw_rad = math.radians(vehicle_yaw or 0) + math.radians(self.angle)
            sz = max(8, int(self.zoom * 3))
            tip = (int(vp[0] + sz * math.cos(yaw_rad)),
                   int(vp[1] + sz * math.sin(yaw_rad)))
            bl = (int(vp[0] - sz * 0.5 * math.cos(yaw_rad) - sz * 0.4 * math.sin(yaw_rad)),
                  int(vp[1] - sz * 0.5 * math.sin(yaw_rad) + sz * 0.4 * math.cos(yaw_rad)))
            br = (int(vp[0] - sz * 0.5 * math.cos(yaw_rad) + sz * 0.4 * math.sin(yaw_rad)),
                  int(vp[1] - sz * 0.5 * math.sin(yaw_rad) - sz * 0.4 * math.cos(yaw_rad)))
            tri = np.array([tip, bl, br], dtype=np.int32)
            cv2.fillPoly(img, [tri], (0, 255, 0))

        # HUD: controls hint + zoom level
        follow_str = "FOLLOW" if self.follow else "FREE"
        hud = f"Zoom:{self.zoom:.1f}  Rot:{self.angle:.0f}  [{follow_str}]"
        cv2.putText(img, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1, cv2.LINE_AA)
        keys_hint = "[ ] zoom  Shift+Arrow pan  , . rot  \\ reset  M follow"
        cv2.putText(img, keys_hint, (10, self.MAP_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120),
                    1, cv2.LINE_AA)

        return img
