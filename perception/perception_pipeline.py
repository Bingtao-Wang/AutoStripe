"""Complete perception pipeline: segmentation -> edge extraction -> depth projection.

Combines RoadSegmentor, extract_road_edges, and DepthProjector into a single
per-frame call that outputs world-coordinate road edges.

V4: Supports both GT mode (CityScapes) and AI mode (VLLiNet) via use_ai flag.
"""

import numpy as np

from perception.road_segmentor import RoadSegmentor
from perception.edge_extractor import extract_road_edges_semantic, extract_road_edges_mask
from perception.depth_projector import DepthProjector, decode_depth_image


class PerceptionPipeline:
    """Per-frame perception: semantic image + depth -> world-coordinate road edges.

    Args:
        img_w, img_h, fov_deg: Camera parameters for depth projection.
        use_ai: If True, use VLLiNet AI segmentor instead of GT CityScapes.
        checkpoint_path: Path to VLLiNet checkpoint (only used if use_ai=True).
    """

    def __init__(self, img_w, img_h, fov_deg,
                 use_ai=False, checkpoint_path=None):
        self.use_ai = use_ai

        if use_ai:
            from perception.road_segmentor_ai import RoadSegmentorAI
            self.segmentor = RoadSegmentorAI(
                checkpoint_path=checkpoint_path)
            self.gt_segmentor = RoadSegmentor()
        else:
            self.segmentor = RoadSegmentor()
            self.gt_segmentor = None

        self.last_inference_ms = 0.0
        self.projector = DepthProjector(img_w, img_h, fov_deg)

    def process_frame(self, semantic_bgra, depth_bgra, camera_transform,
                       cityscapes_bgra=None, rgb_bgra=None):
        """Run full perception pipeline on one frame.

        Args:
            semantic_bgra: np.ndarray (H,W,4) RAW semantic BGRA (R=tag ID).
            depth_bgra:    np.ndarray (H,W,4) from depth camera.
            camera_transform: carla.Transform of the front camera (world frame).
            cityscapes_bgra: np.ndarray (H,W,4) CityScapes-colored BGRA.
                             If None, falls back to raw semantic_bgra.
            rgb_bgra: np.ndarray (H,W,4) from RGB camera (needed for AI mode).

        Returns:
            8-tuple: (left_world, right_world, road_mask, left_px, right_px,
                      gt_right_world, gt_right_px, gt_road_mask)
            gt_road_mask is the GT CityScapes road mask (AI mode only, else None).
        """
        if self.use_ai:
            road_mask = self.segmentor.segment(rgb_bgra, depth_bgra)
            self.last_inference_ms = self.segmentor.last_inference_ms
        else:
            seg_input = cityscapes_bgra if cityscapes_bgra is not None else semantic_bgra
            road_mask = self.segmentor.segment(seg_input)
            self.last_inference_ms = 0.0

        depth_m = decode_depth_image(depth_bgra)

        if self.use_ai:
            # AI edges: from VLLiNet road mask
            left_px, right_px = extract_road_edges_mask(road_mask, depth_m)
            left_world = self.projector.project_pixels(
                left_px, depth_m, camera_transform)
            right_world = self.projector.project_pixels(
                right_px, depth_m, camera_transform)

            # GT edges: from semantic tags (reference)
            _, gt_right_px = extract_road_edges_semantic(
                semantic_bgra, depth_m)
            gt_right_world = self.projector.project_pixels(
                gt_right_px, depth_m, camera_transform)

            # GT road mask for IoU comparison
            seg_input = cityscapes_bgra if cityscapes_bgra is not None else semantic_bgra
            gt_road_mask = self.gt_segmentor.segment(seg_input)

            return (left_world, right_world, road_mask, left_px, right_px,
                    gt_right_world, gt_right_px, gt_road_mask)
        else:
            left_px, right_px = extract_road_edges_semantic(
                semantic_bgra, depth_m)
            left_world = self.projector.project_pixels(
                left_px, depth_m, camera_transform)
            right_world = self.projector.project_pixels(
                right_px, depth_m, camera_transform)

            return (left_world, right_world, road_mask, left_px, right_px,
                    None, None, None)
