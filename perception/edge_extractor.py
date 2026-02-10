"""Extract left and right road edges from semantic + depth images.

Strategy: scan outward from image center per row, find the first pixel
whose semantic tag indicates a road boundary (Sidewalk, Fence, Terrain,
etc.).  Poles (tag 5) are skipped — they stand ON the road, not at the
edge.  Depth is used only to reject the vehicle hood (very close) and
sky (very far).
"""

import numpy as np

# --- Semantic tag sets -----------------------------------------------
# Tags that definitively mark the road boundary (curb / edge).
ROAD_EDGE_TAGS = {8, 22, 2, 9, 11, 27}
#  8=Sidewalk  22=Terrain  2=Fence  9=Vegetation
#  11=Wall     27=GuardRail

# Tags that sit ON the road surface — skip them when scanning.
ROAD_SURFACE_TAGS = {0, 1, 6}
#  0=Unlabeled (vehicle hood / far road)  1=Road  6=RoadLines

# Tags to ignore during scanning (objects on road, not edges).
SKIP_TAGS = {5, 10, 24}
#  5=Pole (road bollards)  10=Vehicle  24=???

# --- Scan parameters -------------------------------------------------
ROW_START_RATIO = 0.3      # only scan lower 70% of image
SMOOTH_WINDOW = 7          # median filter window for edge smoothing
MIN_DEPTH = 1.5            # skip vehicle hood (very close pixels)
MAX_DEPTH = 30.0           # skip far background (mountains/trees)
MIN_CONFIRM = 2            # consecutive edge-tag pixels to confirm
MIN_ROAD_RUN = 20          # min consecutive road pixels to confirm road surface
MAX_LEFT_SCAN = 200        # max pixels to scan left from center (prevent crossing lanes)


def extract_road_edges_semantic(semantic_bgra, depth_m=None,
                                projector=None, camera_transform=None):
    """Extract road edges by scanning for semantic boundary tags.

    For each row, scans outward from image center.  The road edge is the
    first pixel with a ROAD_EDGE_TAG and valid depth.  Poles and vehicles
    are skipped.

    Args:
        semantic_bgra: (H, W, 4) CARLA semantic BGRA image.
        depth_m:       (H, W) float32 depth in meters (optional but
                       recommended for hood/sky filtering).
        projector:     unused (kept for API compatibility).
        camera_transform: unused (kept for API compatibility).

    Returns:
        left_pixels, right_pixels: lists of (u, v) tuples.
    """
    tags = semantic_bgra[:, :, 2]  # R channel = semantic tag
    h, w = tags.shape

    has_depth = depth_m is not None
    row_start = int(h * ROW_START_RATIO)
    cx = w // 2
    right_raw = []
    left_raw = []

    for v in range(row_start, h):
        # Skip rows where center depth is too small (vehicle hood)
        if has_depth and depth_m[v, cx] < MIN_DEPTH:
            continue

        row_tags = tags[v, :]
        row_depth = depth_m[v, :] if has_depth else None

        # --- Right edge: scan from center rightward ---
        ru = _find_edge_tag(row_tags, row_depth, range(cx, w))
        if ru is not None:
            right_raw.append((ru, v))

        # --- Left edge: scan from center leftward (limited range) ---
        left_limit = max(0, cx - MAX_LEFT_SCAN)
        lu = _find_edge_tag(row_tags, row_depth, range(cx, left_limit - 1, -1))
        if lu is not None:
            left_raw.append((lu, v))

    return _smooth_edge(left_raw), _smooth_edge(right_raw)


def _find_edge_tag(row_tags, row_depth, u_range):
    """Scan along u_range for the first road-edge tag after solid road.

    Requires MIN_ROAD_RUN consecutive road-surface pixels before accepting
    an edge tag.  Poles/vehicles (SKIP_TAGS) are transparent — they don't
    break the road run or the edge confirmation streak.
    """
    road_run = 0
    found_road = False
    confirm = 0
    edge_u = None

    for u in u_range:
        t = int(row_tags[u])

        # Depth filter: skip hood and sky
        if row_depth is not None:
            d = float(row_depth[u])
            if d < MIN_DEPTH or d > MAX_DEPTH:
                road_run = 0
                confirm = 0
                edge_u = None
                continue

        if t in ROAD_SURFACE_TAGS:
            road_run += 1
            if road_run >= MIN_ROAD_RUN:
                found_road = True
            confirm = 0
            edge_u = None
        elif t in ROAD_EDGE_TAGS:
            road_run = 0
            if found_road:
                if confirm == 0:
                    edge_u = u
                confirm += 1
                if confirm >= MIN_CONFIRM:
                    return edge_u
        elif t in SKIP_TAGS:
            # Pole / vehicle on road — transparent, don't break anything
            pass
        else:
            road_run = 0
            confirm = 0
            edge_u = None

    return None


def _smooth_edge(pixels):
    """Apply median filter to the u-coordinates of edge pixels."""
    if len(pixels) < SMOOTH_WINDOW:
        return pixels

    us = np.array([p[0] for p in pixels], dtype=np.float32)
    vs = [p[1] for p in pixels]

    pad = SMOOTH_WINDOW // 2
    us_padded = np.pad(us, pad, mode='reflect')
    smoothed = np.array([
        np.median(us_padded[i:i + SMOOTH_WINDOW])
        for i in range(len(us))
    ], dtype=np.int32)

    return [(int(u), v) for u, v in zip(smoothed, vs)]


def extract_road_edges_mask(road_mask, depth_m=None):
    """Extract road edges from a binary mask (e.g. VLLiNet output).

    Per-row scan from center outward: the edge is where road (mask>0)
    transitions to non-road (mask==0).

    Args:
        road_mask: (H, W) uint8, 255=road, 0=other.
        depth_m:   (H, W) float32 depth in meters (optional).

    Returns:
        left_pixels, right_pixels: lists of (u, v) tuples.
    """
    h, w = road_mask.shape[:2]
    has_depth = depth_m is not None
    row_start = int(h * ROW_START_RATIO)
    cx = w // 2
    right_raw = []
    left_raw = []

    for v in range(row_start, h):
        if has_depth and depth_m[v, cx] < MIN_DEPTH:
            continue

        row = road_mask[v, :]

        # Right edge: scan rightward from center, find last road pixel
        ru = _find_mask_edge(row, depth_m[v, :] if has_depth else None,
                             range(cx, w))
        if ru is not None:
            right_raw.append((ru, v))

        # Left edge: scan leftward from center
        left_limit = max(0, cx - MAX_LEFT_SCAN)
        lu = _find_mask_edge(row, depth_m[v, :] if has_depth else None,
                             range(cx, left_limit - 1, -1))
        if lu is not None:
            left_raw.append((lu, v))

    return _smooth_edge(left_raw), _smooth_edge(right_raw)


def _find_mask_edge(row_mask, row_depth, u_range):
    """Find the pixel where road mask transitions to non-road.

    Scans along u_range. Requires MIN_ROAD_RUN consecutive road pixels,
    then returns the first non-road pixel (the edge).
    """
    road_run = 0
    found_road = False

    for u in u_range:
        if row_depth is not None:
            d = float(row_depth[u])
            if d < MIN_DEPTH or d > MAX_DEPTH:
                road_run = 0
                continue

        if row_mask[u] > 0:
            road_run += 1
            if road_run >= MIN_ROAD_RUN:
                found_road = True
        else:
            if found_road:
                return u
            road_run = 0

    return None
