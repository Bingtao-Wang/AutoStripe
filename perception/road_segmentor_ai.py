"""AI road segmentation using VLLiNet model.

Wraps VLLiNet_Lite with the same segment() interface as RoadSegmentor,
enabling drop-in replacement in the perception pipeline.

Requires both RGB and depth images (unlike GT segmentor which only needs semantic).
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Add VLLiNet model path
VLLINET_DIR = os.path.join(os.path.dirname(__file__), '..', 'VLLiNet_models')
sys.path.insert(0, VLLINET_DIR)
from models.vllinet import VLLiNet_Lite
from models.backbone import LiDAREncoder

# Top portion of image to mask out (same as GT segmentor)
MASK_TOP_RATIO = 0.35


class RoadSegmentorAI:
    """VLLiNet-based road segmentation.

    Unlike RoadSegmentor (GT), this requires both RGB and depth images.
    The segment() interface accepts rgb_bgra and depth_bgra parameters.
    """

    def __init__(self, checkpoint_path=None, device='cuda',
                 model_h=384, model_w=1248):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                VLLINET_DIR, 'checkpoints_carla', 'best_model.pth')

        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.model_h = model_h
        self.model_w = model_w

        # ImageNet normalization for RGB
        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Load checkpoint and detect depth channels
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path):
        """Load VLLiNet_Lite from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']

        # Detect depth encoder input channels
        depth_key = 'lidar_encoder.stage1.0.weight'
        if depth_key in state_dict:
            self.depth_channels = state_dict[depth_key].shape[1]
        else:
            self.depth_channels = 3

        # Build model with correct depth channels
        self.model = VLLiNet_Lite(
            pretrained=False, use_deep_supervision=True)
        if self.depth_channels != 3:
            self.model.lidar_encoder = LiDAREncoder(
                in_channels=self.depth_channels)

        # Fix key naming: checkpoint uses 'fusion.' but code uses 'fusion_module.'
        fixed_sd = {}
        for k, v in state_dict.items():
            new_k = k.replace('fusion.fusion_modules',
                              'fusion_module.fusion_modules')
            fixed_sd[new_k] = v
        self.model.load_state_dict(fixed_sd)
        self.model = self.model.to(self.device).eval()

        epoch = checkpoint.get('epoch', '?')
        maxf = checkpoint.get('val_maxf', 0.0)
        print(f"[RoadSegmentorAI] Loaded VLLiNet: epoch={epoch}, "
              f"MaxF={maxf:.4f}, depth_ch={self.depth_channels}, "
              f"device={self.device}")

    def segment(self, rgb_bgra, depth_bgra=None):
        """Segment road from RGB + depth images.

        Args:
            rgb_bgra: np.ndarray (H, W, 4) BGRA from CARLA RGB camera.
            depth_bgra: np.ndarray (H, W, 4) BGRA from CARLA depth camera.
                        Required for AI mode.

        Returns:
            np.ndarray (H, W) uint8 — 255 for road, 0 otherwise.
        """
        if depth_bgra is None:
            h, w = rgb_bgra.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        orig_h, orig_w = rgb_bgra.shape[:2]

        # Preprocess RGB
        rgb_tensor = self._preprocess_rgb(rgb_bgra)

        # Preprocess depth
        depth_tensor = self._preprocess_depth(depth_bgra)

        # Inference
        mask = self._infer(rgb_tensor, depth_tensor, orig_h, orig_w)

        # Apply top cutoff (same as GT segmentor)
        cutoff = int(orig_h * MASK_TOP_RATIO)
        mask[:cutoff, :] = 0

        return mask

    def _preprocess_rgb(self, bgra):
        """BGRA -> normalized RGB tensor [1, 3, H, W]."""
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        rgb = cv2.resize(rgb, (self.model_w, self.model_h),
                         interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - self.rgb_mean) / self.rgb_std
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _preprocess_depth(self, bgra):
        """CARLA depth BGRA -> normalized depth tensor [1, C, H, W]."""
        r = bgra[:, :, 2].astype(np.float32)
        g = bgra[:, :, 1].astype(np.float32)
        b = bgra[:, :, 0].astype(np.float32)
        depth_m = (r + g * 256.0 + b * 65536.0) / (256.0**3 - 1) * 1000.0

        depth_m = cv2.resize(depth_m, (self.model_w, self.model_h),
                             interpolation=cv2.INTER_LINEAR)

        # Min-max normalize to [0, 1]
        d_min, d_max = depth_m.min(), depth_m.max()
        if d_max > d_min:
            depth_norm = (depth_m - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_m)

        # Stack to required channels
        depth_stack = np.stack(
            [depth_norm] * self.depth_channels, axis=0)  # [C, H, W]
        tensor = torch.from_numpy(depth_stack).unsqueeze(0)  # [1, C, H, W]
        return tensor.to(self.device)

    @torch.no_grad()
    def _infer(self, rgb_tensor, depth_tensor, orig_h, orig_w):
        """Run model and return uint8 mask at original resolution."""
        if self.device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                output = self.model(rgb_tensor, depth_tensor, return_aux=False)
        else:
            output = self.model(rgb_tensor, depth_tensor, return_aux=False)

        pred = torch.sigmoid(output)
        pred = F.interpolate(
            pred, size=(orig_h, orig_w),
            mode='bilinear', align_corners=False)
        mask = (pred > 0.5).squeeze().cpu().numpy().astype(np.uint8) * 255
        return mask
