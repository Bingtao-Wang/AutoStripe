import glob
import os
import sys

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


def generate_center_waypoints(carla_map, start_location, num_waypoints=200,
                              spacing=1.0):
    """Generate a sequence of center-lane waypoints starting from a location.

    Returns:
        waypoint_objects: list of carla.Waypoint
        waypoint_coords:  list of (x, y) tuples
    """
    wp = carla_map.get_waypoint(start_location, project_to_road=True,
                                lane_type=carla.LaneType.Driving)

    waypoint_objects = []
    waypoint_coords = []

    for _ in range(num_waypoints):
        wp_next = wp.next(spacing)
        if not wp_next:
            break
        # Prefer second option at forks (matches reference script behaviour)
        wp = wp_next[1] if len(wp_next) > 1 else wp_next[0]
        waypoint_objects.append(wp)
        waypoint_coords.append((wp.transform.location.x,
                                wp.transform.location.y))

    return waypoint_objects, waypoint_coords


def _find_outermost_lane(waypoint, direction):
    """Traverse laterally to find the outermost Driving lane.

    Args:
        waypoint: starting carla.Waypoint
        direction: "left" or "right"

    Returns:
        The outermost Driving-type waypoint on that side, or the
        original waypoint if no lateral neighbour exists.
    """
    current = waypoint
    original_road_id = waypoint.road_id
    original_lane_sign = (waypoint.lane_id > 0)

    while True:
        if direction == "left":
            neighbour = current.get_left_lane()
        else:
            neighbour = current.get_right_lane()

        if neighbour is None:
            break
        if neighbour.lane_type != carla.LaneType.Driving:
            break
        if neighbour.road_id != original_road_id:
            break
        # Sign flip means we crossed the center divider
        if (neighbour.lane_id > 0) != original_lane_sign:
            break

        current = neighbour

    return current


def _compute_edge_position(waypoint, side, z_offset=0.1):
    """Compute the road edge location for one side of a lane.

    Args:
        waypoint: carla.Waypoint at the outermost lane
        side: "left" or "right" — which edge of this lane to return
        z_offset: height above road to avoid z-fighting

    Returns:
        carla.Location at the lane edge
    """
    tf = waypoint.transform
    right_vec = tf.get_right_vector()
    half_w = waypoint.lane_width / 2.0

    loc = tf.location
    if side == "right":
        edge = carla.Location(
            x=loc.x + right_vec.x * half_w,
            y=loc.y + right_vec.y * half_w,
            z=loc.z + z_offset,
        )
    else:
        edge = carla.Location(
            x=loc.x - right_vec.x * half_w,
            y=loc.y - right_vec.y * half_w,
            z=loc.z + z_offset,
        )
    return edge


def compute_road_edges(waypoint_objects):
    """Compute left and right road-edge positions for a waypoint sequence.

    For each center waypoint the function walks laterally to the outermost
    Driving lane on each side, then offsets by half the lane width to reach
    the actual road boundary.

    Returns:
        left_edges:  list of carla.Location
        right_edges: list of carla.Location
    """
    left_edges = []
    right_edges = []

    for wp in waypoint_objects:
        leftmost = _find_outermost_lane(wp, "left")
        rightmost = _find_outermost_lane(wp, "right")

        left_edges.append(_compute_edge_position(leftmost, "left"))
        right_edges.append(_compute_edge_position(rightmost, "right"))

    return left_edges, right_edges
