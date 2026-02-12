"""
Loss functions for SNE-RoadSegV2.

- FocalLoss: Class-imbalanced CE with focusing parameter
- SemanticTransitionAwareLoss (STA): Boundary-aware loss
- DepthInconsistencyAwareLoss (DIA): Depth-consistency loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced segmentation."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return focal.mean()
        return focal


class SemanticTransitionAwareLoss(nn.Module):
    """Boundary-aware loss that up-weights pixels near class transitions."""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        pad = kernel_size // 2
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=pad)

    def forward(self, pred, target):
        target_f = target.float().unsqueeze(1)
        dilated = self.pool(target_f)
        eroded = -self.pool(-target_f)
        boundary = (dilated - eroded).squeeze(1).clamp(0, 1)

        ce = F.cross_entropy(pred, target, reduction='none')
        weight = 1.0 + boundary * 4.0
        return (ce * weight).mean()


class DepthInconsistencyAwareLoss(nn.Module):
    """Penalises predictions that disagree with depth discontinuities."""

    def __init__(self, scale_factor=0.1):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, pred, target, depth, intrinsics):
        # Depth gradient magnitude
        dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
        dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        grad_x = F.pad(dx, (0, 1, 0, 0))
        grad_y = F.pad(dy, (0, 0, 0, 1))
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6).squeeze(1)

        ce = F.cross_entropy(pred, target, reduction='none')
        weight = 1.0 + self.scale_factor * grad_mag
        return (ce * weight).mean()
