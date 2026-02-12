"""
HF2B Fusion Components: HAM, HFCD, AWFR

Hierarchical Feature Fusion Block components used by IAF (Illumination-Adaptive Fusion).

Layer names and shapes match the trained checkpoint (best_net_LUNA_ClearNight.pth):
- HAM: fc (1x1 Conv2d channel attention) + spatial_conv (7x7 spatial attention)
- HFCD: contrast_conv (1x1, 2C->C) + refine_conv (3x3, C->C)
- AWFR: affinity_conv (1x1, 2C->C) + calibrate_conv (3x3, C->C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HAM(nn.Module):
    """Hierarchical Attention Module — channel then spatial attention.

    Checkpoint shapes (e.g. C=96, mid=6):
      ham.fc.0.weight: (6, 96, 1, 1)   — Conv2d squeeze
      ham.fc.2.weight: (96, 6, 1, 1)   — Conv2d excite
      ham.spatial_conv.0.weight: (1, 2, 7, 7) — spatial attention
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        # Channel attention via 1x1 conv (NOT Linear — checkpoint uses Conv2d)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),   # fc.0
            nn.ReLU(inplace=True),                         # fc.1
            nn.Conv2d(mid, in_channels, 1, bias=False),   # fc.2
        )
        # Spatial attention: avg+max concat -> 7x7 conv -> sigmoid
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),    # spatial_conv.0
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Channel attention (global pool -> 1x1 conv on (B,C,1,1))
        avg = x.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
        mx = x.amax(dim=(2, 3), keepdim=True)    # (B, C, 1, 1)
        ca = torch.sigmoid(self.fc(avg) + self.fc(mx))  # (B, C, 1, 1)
        x = x * ca
        # Spatial attention
        avg_s = x.mean(dim=1, keepdim=True)
        mx_s = x.amax(dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial_conv(torch.cat([avg_s, mx_s], dim=1)))
        return x * sa


class HFCD(nn.Module):
    """Heterogeneous Feature Contrast Descriptor.

    Checkpoint shapes (e.g. C=96):
      hfcd.contrast_conv.0.weight: (96, 192, 1, 1) — 1x1, input=2C (concat)
      hfcd.contrast_conv.1: BatchNorm2d(96)
      hfcd.refine_conv.0.weight: (96, 96, 3, 3)    — 3x3
      hfcd.refine_conv.1: BatchNorm2d(96)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.contrast_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, feat_a, feat_b):
        cat = torch.cat([feat_a, feat_b], dim=1)
        contrast = F.relu(self.contrast_conv(cat), inplace=True)
        return F.relu(self.refine_conv(contrast), inplace=True)


class AWFR(nn.Module):
    """Affinity-Weighted Feature Refiner.

    Checkpoint shapes (e.g. C=96):
      awfr.affinity_conv.0.weight: (96, 192, 1, 1) — 1x1, input=2C
      awfr.affinity_conv.1: BatchNorm2d(96)
      awfr.calibrate_conv.0.weight: (96, 96, 3, 3) — 3x3
      awfr.calibrate_conv.1: BatchNorm2d(96)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.affinity_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.calibrate_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, fused, feat_a, feat_b):
        aff = torch.sigmoid(self.affinity_conv(torch.cat([feat_a, feat_b], dim=1)))
        return self.calibrate_conv(fused * aff + fused)
