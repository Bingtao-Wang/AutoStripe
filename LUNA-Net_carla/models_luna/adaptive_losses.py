"""
ALW: Adaptive Loss Weighting

Extends Fallibility-aware Loss with:
1. Brightness-adaptive loss weights
2. Edge-preserving loss for boundary refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import base loss components
import sys
sys.path.append('..')
from modelsV2.losses import SemanticTransitionAwareLoss, DepthInconsistencyAwareLoss, FocalLoss


class EdgePreservingLoss(nn.Module):
    """
    Edge-Preserving Loss for boundary refinement.

    Uses Sobel operator to extract GT edges and computes BCE loss
    between predicted edges and GT edges.

    Args:
        edge_threshold: Threshold for edge detection (default: 0.5)
    """

    def __init__(self, edge_threshold=0.5):
        super().__init__()
        self.edge_threshold = edge_threshold

        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def compute_edge_gt(self, target):
        """
        Compute edge ground truth from segmentation mask.

        Args:
            target: Segmentation mask (B, H, W) with class labels

        Returns:
            edge_gt: Edge map (B, 1, H, W) in [0, 1]
        """
        # Convert to float and add channel dimension
        target_float = target.float().unsqueeze(1)  # (B, 1, H, W)

        # Compute gradients
        grad_x = F.conv2d(target_float, self.sobel_x, padding=1)
        grad_y = F.conv2d(target_float, self.sobel_y, padding=1)

        # Edge magnitude
        edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalize and threshold
        edge_gt = (edge_mag > self.edge_threshold).float()

        return edge_gt

    def forward(self, edge_pred, target):
        """
        Args:
            edge_pred: Predicted edge map (B, 1, H, W) in [0, 1]
            target: Segmentation mask (B, H, W)

        Returns:
            Edge loss value
        """
        # Compute GT edges
        edge_gt = self.compute_edge_gt(target)

        # Resize prediction if needed
        if edge_pred.shape[2:] != edge_gt.shape[2:]:
            edge_pred = F.interpolate(
                edge_pred, size=edge_gt.shape[2:],
                mode='bilinear', align_corners=True
            )

        # BCE loss with edge weighting
        # Give more weight to edge pixels (they are sparse)
        pos_weight = (edge_gt == 0).sum() / (edge_gt == 1).sum().clamp(min=1)
        pos_weight = pos_weight.clamp(max=10)  # Limit weight

        loss = F.binary_cross_entropy(
            edge_pred, edge_gt,
            weight=edge_gt * pos_weight + (1 - edge_gt)
        )

        return loss


class BrightnessToLossWeights(nn.Module):
    """
    Maps brightness to loss component weights.

    In dark scenes:
    - Increase DIA weight (depth is more reliable)
    - Increase Edge weight (boundaries are harder to see)

    In bright scenes:
    - Use default weights
    """

    def __init__(self, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # [STA_scale, DIA_scale, Edge_scale]
            nn.Softplus()  # Ensure positive weights
        )

        # Initialize to output ~1.0 for all weights
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.ones_(m.bias)

    def forward(self, brightness):
        """
        Args:
            brightness: (B, 1) brightness score in [0, 1]

        Returns:
            scales: (3,) loss weight scales [STA, DIA, Edge]
        """
        # Get per-sample scales
        scales = self.mlp(brightness)  # (B, 3)
        # Average across batch for stable training
        scales = scales.mean(dim=0)  # (3,)
        return scales


class AdaptiveFallibilityLoss(nn.Module):
    """
    Adaptive Fallibility-aware Loss (ALW)

    Extends the base Fallibility-aware Loss with:
    1. Brightness-adaptive weight scaling
    2. Edge-preserving loss component

    Total Loss:
    L = L_CE + λ_STA * s_STA * L_STA + λ_DIA * s_DIA * L_DIA + λ_EDGE * s_EDGE * L_EDGE

    where s_* are brightness-dependent scales.

    Args:
        lambda_sta: Base weight for STA loss (default: 0.3)
        lambda_dia: Base weight for DIA loss (default: 0.1)
        lambda_edge: Base weight for Edge loss (default: 0.1)
        use_focal: Whether to use Focal Loss for CE (default: False)
        use_adaptive_weights: Whether to use brightness-adaptive weights (default: True)
        dia_scale_factor: Scale factor for DIA loss (default: 0.1)
    """

    def __init__(self, lambda_sta=0.3, lambda_dia=0.1, lambda_edge=0.1,
                 use_focal=False, focal_alpha=0.25, focal_gamma=2.0,
                 use_adaptive_weights=True, dia_scale_factor=0.1):
        super().__init__()

        self.lambda_sta = lambda_sta
        self.lambda_dia = lambda_dia
        self.lambda_edge = lambda_edge
        self.use_adaptive_weights = use_adaptive_weights

        # Base CE loss
        if use_focal:
            self.ce_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        # Component losses
        self.sta_loss = SemanticTransitionAwareLoss()
        self.dia_loss = DepthInconsistencyAwareLoss(scale_factor=dia_scale_factor)
        self.edge_loss = EdgePreservingLoss()

        # Adaptive weight network
        if use_adaptive_weights:
            self.brightness_to_weights = BrightnessToLossWeights(hidden_dim=16)

    def forward(self, pred, target, depth=None, intrinsics=None,
                brightness=None, edge_pred=None):
        """
        Args:
            pred: Predicted logits (B, 2, H, W)
            target: Ground truth (B, H, W)
            depth: Depth map (B, 1, H, W), optional
            intrinsics: Camera intrinsics (B, 3, 3), optional
            brightness: Brightness score (B, 1), optional
            edge_pred: Predicted edges (B, 1, H, W), optional

        Returns:
            total_loss: Combined loss value
        """
        # 1. Base CE Loss
        loss_ce = self.ce_loss(pred, target)

        # 2. STA Loss (Boundary awareness)
        loss_sta = self.sta_loss(pred, target)

        # 3. DIA Loss (Depth consistency)
        if depth is not None and intrinsics is not None:
            loss_dia = self.dia_loss(pred, target, depth, intrinsics)
        else:
            loss_dia = torch.tensor(0.0, device=pred.device)

        # 4. Edge Loss
        if edge_pred is not None:
            loss_edge = self.edge_loss(edge_pred, target)
        else:
            loss_edge = torch.tensor(0.0, device=pred.device)

        # 5. Compute adaptive scales
        if self.use_adaptive_weights and brightness is not None:
            scales = self.brightness_to_weights(brightness)
            s_sta, s_dia, s_edge = scales[0], scales[1], scales[2]
        else:
            s_sta, s_dia, s_edge = 1.0, 1.0, 1.0

        # 6. Combine losses
        total_loss = loss_ce
        total_loss = total_loss + self.lambda_sta * s_sta * loss_sta
        total_loss = total_loss + self.lambda_dia * s_dia * loss_dia
        total_loss = total_loss + self.lambda_edge * s_edge * loss_edge

        return total_loss

    def get_loss_components(self, pred, target, depth=None, intrinsics=None,
                           brightness=None, edge_pred=None):
        """Get individual loss components for logging."""
        with torch.no_grad():
            loss_ce = self.ce_loss(pred, target)
            loss_sta = self.sta_loss(pred, target)

            if depth is not None and intrinsics is not None:
                loss_dia = self.dia_loss(pred, target, depth, intrinsics)
            else:
                loss_dia = torch.tensor(0.0, device=pred.device)

            if edge_pred is not None:
                loss_edge = self.edge_loss(edge_pred, target)
            else:
                loss_edge = torch.tensor(0.0, device=pred.device)

            # Get scales
            if self.use_adaptive_weights and brightness is not None:
                scales = self.brightness_to_weights(brightness)
                s_sta, s_dia, s_edge = scales[0].item(), scales[1].item(), scales[2].item()
            else:
                s_sta, s_dia, s_edge = 1.0, 1.0, 1.0

        return {
            'CE': loss_ce.item(),
            'STA': loss_sta.item(),
            'DIA': loss_dia.item(),
            'Edge': loss_edge.item(),
            'scale_STA': s_sta,
            'scale_DIA': s_dia,
            'scale_Edge': s_edge,
            'Total': (
                loss_ce +
                self.lambda_sta * s_sta * loss_sta +
                self.lambda_dia * s_dia * loss_dia +
                self.lambda_edge * s_edge * loss_edge
            ).item()
        }


class MultiScaleEdgeLoss(nn.Module):
    """
    Multi-scale edge loss for deep supervision.

    Computes edge loss at multiple decoder scales.
    """

    def __init__(self):
        super().__init__()
        self.edge_loss = EdgePreservingLoss()

    def forward(self, edge_preds, target):
        """
        Args:
            edge_preds: Dict of edge predictions at multiple scales
                       {'edge_0': (B,1,H,W), 'edge_1': ..., 'edge_2': ...}
            target: Ground truth segmentation (B, H, W)

        Returns:
            Combined multi-scale edge loss
        """
        total_loss = 0
        weights = {'edge_0': 1.0, 'edge_1': 0.5, 'edge_2': 0.25}

        for name, edge_pred in edge_preds.items():
            if name in weights:
                loss = self.edge_loss(edge_pred, target)
                total_loss = total_loss + weights[name] * loss

        return total_loss


if __name__ == "__main__":
    print("Testing Adaptive Fallibility-aware Loss...")

    # Create loss function
    criterion = AdaptiveFallibilityLoss(
        lambda_sta=0.3,
        lambda_dia=0.1,
        lambda_edge=0.1,
        use_adaptive_weights=True
    )

    # Test inputs
    B, C, H, W = 2, 2, 384, 1248
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pred = torch.randn(B, C, H, W).to(device)
    target = torch.randint(0, 2, (B, H, W)).to(device)
    depth = torch.rand(B, 1, H, W).to(device) * 80.0
    edge_pred = torch.rand(B, 1, H, W).to(device)

    # Camera intrinsics
    intrinsics = torch.tensor([
        [721.5, 0.0, 609.5],
        [0.0, 721.5, 172.8],
        [0.0, 0.0, 1.0]
    ]).unsqueeze(0).repeat(B, 1, 1).to(device)

    # Test with different brightness levels
    print("\nTesting with different brightness levels:")
    for brightness_val in [0.1, 0.5, 0.9]:
        brightness = torch.full((B, 1), brightness_val).to(device)

        loss = criterion(pred, target, depth, intrinsics, brightness, edge_pred)
        components = criterion.get_loss_components(
            pred, target, depth, intrinsics, brightness, edge_pred
        )

        print(f"\n  Brightness={brightness_val:.1f}:")
        print(f"    Total Loss: {loss.item():.4f}")
        print(f"    CE: {components['CE']:.4f}")
        print(f"    STA: {components['STA']:.4f} (scale={components['scale_STA']:.2f})")
        print(f"    DIA: {components['DIA']:.4f} (scale={components['scale_DIA']:.2f})")
        print(f"    Edge: {components['Edge']:.4f} (scale={components['scale_Edge']:.2f})")

    # Count parameters
    total_params = sum(p.numel() for p in criterion.parameters())
    print(f"\nTotal parameters: {total_params}")
