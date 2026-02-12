"""
Swin Transformer Backbone for SNE-RoadSegV2

Manually extracts multi-scale features from Swin-T, compatible with
timm 0.6.x which doesn't support features_only + custom img_size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _swin_compatible_size(h, w, patch_size=4, window_size=7, num_stages=4):
    """Smallest (H, W) >= (h, w) compatible with Swin window partitioning."""
    divisor = patch_size * window_size * (2 ** (num_stages - 1))  # 224
    new_h = ((h + divisor - 1) // divisor) * divisor
    new_w = ((w + divisor - 1) // divisor) * divisor
    return (new_h, new_w)


class SwinTransformerBackbone(nn.Module):
    """Swin-T backbone with manual multi-scale feature extraction.

    Avoids timm's features_only wrapper which is broken with custom
    img_size in timm 0.6.x.  Instead, creates a standard Swin model
    and hooks into each stage to collect intermediate features.

    Output channels (Swin-T): [96, 192, 384, 768]
    """

    # Swin-T channel dims per stage
    SWIN_T_DIMS = [96, 192, 384, 768]

    def __init__(self, model_name='swin_tiny_patch4_window7_224',
                 pretrained=True, out_indices=(0, 1, 2, 3),
                 pretrained_path=None):
        super().__init__()
        self.out_indices = out_indices
        self._compat_size = _swin_compatible_size(384, 1248)  # (448, 1344)

        # Resolve local pretrained weights
        if pretrained_path is None and pretrained:
            import os
            local_weights = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'data', 'swin_tiny_patch4_window7_224.safetensors')
            if os.path.exists(local_weights):
                pretrained_path = local_weights

        # Create base Swin model (NOT features_only)
        self.swin = timm.create_model(
            model_name,
            pretrained=(pretrained and pretrained_path is None),
            num_classes=0,  # remove classification head
            img_size=self._compat_size,
        )

        # Load local weights if provided
        if pretrained_path is not None:
            print(f"Loading pretrained weights: {pretrained_path}")
            if pretrained_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(pretrained_path)
            else:
                state_dict = torch.load(pretrained_path, map_location='cpu')
            missing, unexpected = self.swin.load_state_dict(
                state_dict, strict=False)
            print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        self.feature_info = self.SWIN_T_DIMS
        print(f"Swin Backbone: {model_name}, img_size={self._compat_size}")
        print(f"Feature channels: {self.feature_info}")

    def forward(self, x):
        """Extract multi-scale features from Swin stages.

        Args:
            x: (B, 3, H, W) input tensor.

        Returns:
            List of (B, C, H_i, W_i) feature maps per stage.
        """
        _, _, h, w = x.shape
        th, tw = self._compat_size
        pad_h, pad_w = th - h, tw - w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Patch embedding
        x = self.swin.patch_embed(x)
        if hasattr(self.swin, 'absolute_pos_embed') and self.swin.absolute_pos_embed is not None:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.pos_drop(x)

        features = []
        for i, layer in enumerate(self.swin.layers):
            # Run blocks ONLY (not downsample) so we capture pre-downsample features
            for blk in layer.blocks:
                x = blk(x)

            if i in self.out_indices:
                # x is (B, H*W, C) — reshape to (B, C, H, W)
                B, L, C = x.shape
                scale = 2 ** (i + 2)  # stage0: /4, stage1: /8, ...
                fh, fw = th // scale, tw // scale
                feat = x.reshape(B, fh, fw, C).permute(0, 3, 1, 2).contiguous()
                features.append(feat)

            # Now apply downsample (if present) to prepare for next stage
            if layer.downsample is not None:
                x = layer.downsample(x)

        return features

    def get_feature_dims(self):
        return self.feature_info


class DualSwinBackbone(nn.Module):
    """Dual-stream Swin Transformer for RGB + Surface Normal."""

    def __init__(self, model_name='swin_tiny_patch4_window7_224',
                 pretrained=True, pretrained_path=None):
        super().__init__()
        self.encoder_rgb = SwinTransformerBackbone(
            model_name, pretrained, (0, 1, 2, 3), pretrained_path)
        self.encoder_normal = SwinTransformerBackbone(
            model_name, pretrained, (0, 1, 2, 3), pretrained_path)
        self.feature_dims = self.encoder_rgb.get_feature_dims()
        print(f"Dual Swin Backbone: {self.feature_dims}")

    def forward(self, rgb, normal):
        return self.encoder_rgb(rgb), self.encoder_normal(normal)

    def get_feature_dims(self):
        return self.feature_dims
