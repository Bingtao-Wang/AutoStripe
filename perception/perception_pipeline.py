"""Complete perception pipeline: segmentation -> edge extraction -> depth projection.

Combines RoadSegmentor, extract_road_edges, and DepthProjector into a single
per-frame call that outputs world-coordinate road edges.
"""

import numpy as np

from perception.road_segmentor import RoadSegmentor
from perception.edge_extractor import extract_road_edges_semantic
from perception.depth_projector import DepthProjector, decode_depth_image


class PerceptionPipeline:
    """Per-frame perception: semantic image + depth -> world-coordinate road edges."""

    def __init__(self, img_w, img_h, fov_deg):
        self.segmentor = RoadSegmentor()
        self.projector = DepthProjector(img_w, img_h, fov_deg)

    def process_frame(self, semantic_bgra, depth_bgra, camera_transform,
                       cityscapes_bgra=None):
        """Run full perception pipeline on one frame.

        Args:
            semantic_bgra: np.ndarray (H,W,4) RAW semantic BGRA (R=tag ID).
            depth_bgra:    np.ndarray (H,W,4) from depth camera.
            camera_transform: carla.Transform of the front camera (world frame).
            cityscapes_bgra: np.ndarray (H,W,4) CityScapes-colored BGRA.
                             If None, falls back to raw semantic_bgra.

        Returns:
            left_world, right_world, road_mask, left_px, right_px
        """
        seg_input = cityscapes_bgra if cityscapes_bgra is not None else semantic_bgra
        road_mask = self.segmentor.segment(seg_input)
        depth_m = decode_depth_image(depth_bgra)

        left_px, right_px = extract_road_edges_semantic(
            semantic_bgra, depth_m, self.projector, camera_transform)

        left_world = self.projector.project_pixels(
            left_px, depth_m, camera_transform)
        right_world = self.projector.project_pixels(
            right_px, depth_m, camera_transform)

        return left_world, right_world, road_mask, left_px, right_px
