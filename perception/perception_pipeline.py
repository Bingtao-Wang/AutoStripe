"""Complete perception pipeline: segmentation -> edge extraction -> depth projection.

Combines RoadSegmentor, extract_road_edges, and DepthProjector into a single
per-frame call that outputs world-coordinate road edges.

V4: Supports both GT mode (CityScapes) and AI mode (VLLiNet) via use_ai flag.
V5: Three-mode perception: GT / VLLiNet / LUNA-Net via perception_mode.
"""

import numpy as np

from perception.road_segmentor import RoadSegmentor
from perception.edge_extractor import extract_road_edges_semantic, extract_road_edges_mask
from perception.depth_projector import DepthProjector, decode_depth_image


class PerceptionMode:
    """Perception mode constants."""
    GT = "GT"
    VLLINET = "VLLiNet"
    LUNA = "LUNA"


class PerceptionPipeline:
    """Per-frame perception: semantic image + depth -> world-coordinate road edges.

    Args:
        img_w, img_h, fov_deg: Camera parameters for depth projection.
        use_ai: (V4 compat) If True, use VLLiNet. Ignored if perception_mode is set.
        perception_mode: One of PerceptionMode.GT / VLLINET / LUNA.
        checkpoint_path: Path to model checkpoint (VLLiNet or LUNA-Net).
    """

    def __init__(self, img_w, img_h, fov_deg,
                 use_ai=False, checkpoint_path=None,
                 perception_mode=None):
        # Resolve mode: explicit perception_mode takes priority over use_ai
        if perception_mode is not None:
            self._mode = perception_mode
        elif use_ai:
            self._mode = PerceptionMode.VLLINET
        else:
            self._mode = PerceptionMode.GT

        if self._mode == PerceptionMode.VLLINET:
            from perception.road_segmentor_ai import RoadSegmentorAI
            self.segmentor = RoadSegmentorAI(
                checkpoint_path=checkpoint_path)
            self.gt_segmentor = RoadSegmentor()
        elif self._mode == PerceptionMode.LUNA:
            from perception.road_segmentor_luna import RoadSegmentorLuna
            self.segmentor = RoadSegmentorLuna(
                checkpoint_path=checkpoint_path)
            self.gt_segmentor = RoadSegmentor()
        else:
            self.segmentor = RoadSegmentor()
            self.gt_segmentor = None

        self.last_inference_ms = 0.0
        self.last_sne_ms = 0.0
        self.projector = DepthProjector(img_w, img_h, fov_deg)

    @property
    def perception_mode(self):
        return self._mode

    @property
    def use_ai(self):
        """Backward-compatible property: True for any AI mode."""
        return self._mode in (PerceptionMode.VLLINET, PerceptionMode.LUNA)

    @property
    def last_normal(self):
        """V6: SNE surface normal from LUNA mode, else None."""
        if self._mode == PerceptionMode.LUNA:
            return getattr(self.segmentor, 'last_normal', None)
        return None

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
            # Forward SNE timing for LUNA mode
            if self._mode == PerceptionMode.LUNA:
                self.last_sne_ms = self.segmentor.last_sne_ms
            else:
                self.last_sne_ms = 0.0
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
