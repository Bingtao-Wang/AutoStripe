"""AI road segmentation using LUNA-Net model.

Wraps LUNA-Net (Swin Transformer backbone) with the same segment() interface
as RoadSegmentorAI, enabling drop-in replacement in the perception pipeline.

Key differences from VLLiNet:
- Input: RGB [0,1] (no ImageNet norm) + Surface Normal via SNE (not depth)
- Output: 2-class logits -> argmax (not sigmoid > 0.5)
- Output resolution: native 1248x384 (no upsample needed from model)
- Extra step: SNE computes surface normals from depth on CPU

Requires both RGB and depth images (depth is used to compute surface normals).
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Add LUNA-Net_carla directory to path
# Clear cached 'models' package (may point to VLLiNet_models/models from prior import)
for _mod in list(sys.modules.keys()):
    if _mod == 'models' or _mod.startswith('models.'):
        del sys.modules[_mod]
LUNA_DIR = os.path.join(os.path.dirname(__file__), '..', 'LUNA-Net_carla')
sys.path.insert(0, LUNA_DIR)

from models.sne_model import SNE
from models_luna.luna_net import LUNANet

# Top portion of image to mask out (same as GT and VLLiNet segmentors)
MASK_TOP_RATIO = 0.35


class RoadSegmentorLuna:
    """LUNA-Net-based road segmentation.

    Unlike VLLiNet, this uses Surface Normal Estimation (SNE) instead of
    raw depth as the second modality. RGB is normalized to [0,1] without
    ImageNet mean/std subtraction.

    The segment() interface is identical to RoadSegmentorAI.
    """

    def __init__(self, checkpoint_path=None, device='cuda',
                 model_h=384, model_w=1248):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                LUNA_DIR, 'weights', 'best_net_LUNA_ClearNight.pth')

        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.model_h = model_h
        self.model_w = model_w
        self.last_inference_ms = 0.0
        self.last_sne_ms = 0.0

        # Camera intrinsics (3x4 matrix for SNE)
        # fx=fy=624 for 1248px width, FOV=90
        fx = fy = model_w / 2.0  # 624.0
        cx = model_w / 2.0       # 624.0
        cy = model_h / 2.0       # 192.0
        self.cam_param = np.array([
            [fx,  0, cx, 0],
            [ 0, fy, cy, 0],
            [ 0,  0,  1, 0],
        ], dtype=np.float32)

        # SNE model (CPU, same as training pipeline)
        self.sne = SNE()

        # Load LUNA-Net
        self._load_model(checkpoint_path)

    @staticmethod
    def _remap_checkpoint(state_dict):
        """Remap checkpoint keys to match timm 0.6.x Swin layer naming.

        Checkpoint uses layers_0/layers_1 (underscore, downsample at stage start).
        timm uses layers.0/layers.1 (dot, downsample at stage end).
        """
        import re
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k
            m = re.match(r'(.*\.swin\.)layers_(\d+)\.downsample\.(.*)', k)
            if m:
                prefix, stage, suffix = m.group(1), int(m.group(2)), m.group(3)
                new_k = f'{prefix}layers.{stage - 1}.downsample.{suffix}'
            else:
                m = re.match(r'(.*\.swin\.)layers_(\d+)\.(.*)', k)
                if m:
                    prefix, stage, suffix = m.group(1), int(m.group(2)), m.group(3)
                    new_k = f'{prefix}layers.{stage}.{suffix}'
            new_sd[new_k] = v
        return new_sd

    def _load_model(self, checkpoint_path):
        """Load LUNA-Net from checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        state_dict = self._remap_checkpoint(state_dict)

        self.model = LUNANet(
            swin_model='swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=2,
            use_llem=True,
            use_robust_sne=False,
            use_iaf=True,
            use_naa_decoder=True,
            use_edge_head=True,
        )

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        real_missing = [k for k in missing
                        if 'attn_mask' not in k and 'relative_position_index' not in k]
        if real_missing:
            print(f"[RoadSegmentorLuna] WARNING: {len(real_missing)} missing keys")
        if unexpected:
            print(f"[RoadSegmentorLuna] WARNING: {len(unexpected)} unexpected keys")
        self.model = self.model.to(self.device).eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[RoadSegmentorLuna] Loaded LUNA-Net: "
              f"params={n_params:,}, device={self.device}")

    def segment(self, rgb_bgra, depth_bgra=None):
        """Segment road from RGB + depth images.

        Args:
            rgb_bgra: np.ndarray (H, W, 4) BGRA from CARLA RGB camera.
            depth_bgra: np.ndarray (H, W, 4) BGRA from CARLA depth camera.
                        Required (used to compute surface normals via SNE).

        Returns:
            np.ndarray (H, W) uint8 — 255 for road, 0 otherwise.
        """
        if depth_bgra is None:
            h, w = rgb_bgra.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        orig_h, orig_w = rgb_bgra.shape[:2]

        # 1. Decode depth: CARLA BGRA -> meters
        depth_m = self._decode_depth(depth_bgra)

        # 2. Resize depth to model resolution for SNE
        depth_resized = cv2.resize(
            depth_m, (self.model_w, self.model_h),
            interpolation=cv2.INTER_LINEAR)

        # 3. Compute surface normals via SNE (timed)
        normal = self._compute_sne(depth_resized)

        # 4. Preprocess RGB: BGRA -> RGB, [0,1] (no ImageNet norm)
        rgb_tensor = self._preprocess_rgb(rgb_bgra)

        # 5. Preprocess normal: numpy -> tensor
        normal_tensor = self._preprocess_normal(normal)

        # 6. Inference (timed)
        mask = self._infer(rgb_tensor, normal_tensor, orig_h, orig_w)

        # 7. Apply top cutoff
        cutoff = int(orig_h * MASK_TOP_RATIO)
        mask[:cutoff, :] = 0

        return mask

    def _decode_depth(self, bgra):
        """CARLA depth BGRA -> depth in meters."""
        r = bgra[:, :, 2].astype(np.float32)
        g = bgra[:, :, 1].astype(np.float32)
        b = bgra[:, :, 0].astype(np.float32)
        depth_m = (r + g * 256.0 + b * 65536.0) / (256.0**3 - 1) * 1000.0
        return depth_m

    def _compute_sne(self, depth_resized):
        """Compute surface normal from depth using SNE (CPU).

        Args:
            depth_resized: (model_h, model_w) float32 depth in meters.

        Returns:
            normal: (3, model_h, model_w) float32 surface normal.
        """
        t0 = time.time()
        depth_t = torch.from_numpy(depth_resized).float()
        cam_t = torch.from_numpy(self.cam_param).float()
        with torch.no_grad():
            normal = self.sne(depth_t, cam_t)  # (3, H, W)
        self.last_sne_ms = (time.time() - t0) * 1000.0
        return normal.numpy()

    def _preprocess_rgb(self, bgra):
        """BGRA -> RGB tensor [1, 3, H, W] in [0, 1]."""
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        rgb = cv2.resize(rgb, (self.model_w, self.model_h),
                         interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _preprocess_normal(self, normal):
        """Surface normal (3, H, W) numpy -> tensor [1, 3, H, W]."""
        tensor = torch.from_numpy(normal).unsqueeze(0).float()
        return tensor.to(self.device)

    @torch.no_grad()
    def _infer(self, rgb_tensor, normal_tensor, orig_h, orig_w):
        """Run LUNA-Net and return uint8 mask at original resolution."""
        t0 = time.time()
        if self.device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                output, _aux = self.model(
                    rgb_tensor, normal_tensor, is_normal=True)
        else:
            output, _aux = self.model(
                rgb_tensor, normal_tensor, is_normal=True)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.last_inference_ms = (time.time() - t0) * 1000.0

        # Crop padding: model outputs at padded size (448x1344),
        # valid region is top-left (model_h x model_w) = (384x1248)
        output = output[:, :, :self.model_h, :self.model_w]

        # 2-class logits -> argmax -> 0/1
        pred = torch.argmax(output, dim=1)  # (1, model_h, model_w)

        # Resize to original resolution if needed
        if pred.shape[1] != orig_h or pred.shape[2] != orig_w:
            pred = F.interpolate(
                pred.unsqueeze(1).float(),
                size=(orig_h, orig_w),
                mode='nearest').squeeze(1)

        mask = pred.squeeze().cpu().numpy().astype(np.uint8) * 255
        return mask
