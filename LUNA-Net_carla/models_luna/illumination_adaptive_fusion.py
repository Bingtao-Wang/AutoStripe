"""
IAF: Illumination-Adaptive Fusion

Dynamically adjusts RGB/Normal feature weights based on image brightness.
In dark scenes, increases Normal weight (depth is light-invariant).
In bright scenes, maintains RGB weight (RGB features are reliable).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import existing fusion components
import sys
sys.path.append('..')
from modelsV2.fusion_modules import HAM, HFCD, AWFR


class BrightnessToWeightMLP(nn.Module):
    """
    Maps brightness score to modality fusion weights.

    Input: brightness (B, 1) in [0, 1]
    Output: weights (B, 2) for [RGB, Normal], summing to 1
    """

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)
        )
        # Initialize to output equal weights
        self._init_weights()

    def _init_weights(self):
        """Initialize to produce balanced weights initially."""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, brightness):
        """
        Args:
            brightness: (B, 1) brightness score in [0, 1]

        Returns:
            weights: (B, 2) fusion weights [w_rgb, w_normal]
        """
        logits = self.mlp(brightness)
        weights = F.softmax(logits, dim=-1)
        return weights


class IlluminationAdaptiveFusion(nn.Module):
    """
    Illumination-Adaptive Fusion (IAF)

    Replaces fixed 1:1 fusion in HF2B with brightness-adaptive weighting.

    Key improvement over HF2B:
    - Original: fused = rgb_att + normal_att + contrast (fixed 1:1)
    - IAF: fused = w_rgb * rgb_att + w_normal * normal_att + contrast (adaptive)

    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio for HAM (default: 16)
        use_contrast: Whether to use HFCD contrast descriptor (default: True)
        use_refiner: Whether to use AWFR refinement (default: True)
    """

    def __init__(self, in_channels, reduction=16, use_contrast=True, use_refiner=True):
        super().__init__()
        self.in_channels = in_channels
        self.use_contrast = use_contrast
        self.use_refiner = use_refiner

        # Hierarchical Attention Modules (from HF2B)
        self.ham_rgb = HAM(in_channels, reduction)
        self.ham_normal = HAM(in_channels, reduction)

        # Brightness to weight mapping
        self.brightness_mlp = BrightnessToWeightMLP(hidden_dim=32)

        # Optional: Heterogeneous Feature Contrast Descriptor
        if use_contrast:
            self.hfcd = HFCD(in_channels)

        # Optional: Affinity-Weighted Feature Refiner
        if use_refiner:
            self.awfr = AWFR(in_channels)

        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, rgb_feat, normal_feat, brightness):
        """
        Args:
            rgb_feat: RGB features (B, C, H, W)
            normal_feat: Normal features (B, C, H, W)
            brightness: Brightness score (B, 1) in [0, 1]

        Returns:
            fused: Fused features (B, C, H, W)
            weights: Fusion weights (B, 2) for logging/visualization
        """
        B, C, H, W = rgb_feat.shape

        # Step 1: Apply hierarchical attention
        rgb_att = self.ham_rgb(rgb_feat)
        normal_att = self.ham_normal(normal_feat)

        # Step 2: Compute adaptive weights based on brightness
        # brightness: (B, 1) -> weights: (B, 2)
        weights = self.brightness_mlp(brightness)
        w_rgb = weights[:, 0:1].view(B, 1, 1, 1)      # (B, 1, 1, 1)
        w_normal = weights[:, 1:2].view(B, 1, 1, 1)   # (B, 1, 1, 1)

        # Step 3: Weighted attention features
        weighted_rgb = rgb_att * w_rgb
        weighted_normal = normal_att * w_normal

        # Step 4: Compute contrast descriptor (optional)
        if self.use_contrast:
            contrast = self.hfcd(weighted_rgb, weighted_normal)
            fused = weighted_rgb + weighted_normal + contrast
        else:
            fused = weighted_rgb + weighted_normal

        # Step 5: Affinity-weighted refinement (optional)
        if self.use_refiner:
            refined = self.awfr(fused, weighted_rgb, weighted_normal)
        else:
            refined = fused

        # Step 6: Residual connection
        output = refined + self.residual_weight * (rgb_feat + normal_feat)

        return output, weights


class MultiScaleIAF(nn.Module):
    """
    Multi-scale Illumination-Adaptive Fusion.

    Applies IAF at multiple encoder scales with shared brightness input.

    Args:
        channel_list: List of channel sizes for each scale
                     e.g., [96, 192, 384, 768] for Swin-T
    """

    def __init__(self, channel_list, reduction=16):
        super().__init__()
        self.num_scales = len(channel_list)

        # Create IAF module for each scale
        self.iaf_modules = nn.ModuleList([
            IlluminationAdaptiveFusion(ch, reduction=reduction)
            for ch in channel_list
        ])

    def forward(self, rgb_features, normal_features, brightness):
        """
        Args:
            rgb_features: List of RGB features at each scale
            normal_features: List of Normal features at each scale
            brightness: Brightness score (B, 1)

        Returns:
            fused_features: List of fused features at each scale
            all_weights: List of fusion weights at each scale
        """
        fused_features = []
        all_weights = []

        for i, iaf in enumerate(self.iaf_modules):
            fused, weights = iaf(rgb_features[i], normal_features[i], brightness)
            fused_features.append(fused)
            all_weights.append(weights)

        return fused_features, all_weights


class ConfidenceAwareFusion(nn.Module):
    """
    Extended IAF that also considers depth confidence.

    Combines brightness-based weights with depth confidence for
    more robust fusion in challenging scenarios.

    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio for HAM
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels

        # Base IAF
        self.iaf = IlluminationAdaptiveFusion(in_channels, reduction)

        # Confidence integration
        self.confidence_conv = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, normal_feat, brightness, confidence=None):
        """
        Args:
            rgb_feat: RGB features (B, C, H, W)
            normal_feat: Normal features (B, C, H, W)
            brightness: Brightness score (B, 1)
            confidence: Depth confidence map (B, 1, H, W), optional

        Returns:
            fused: Fused features (B, C, H, W)
            weights: Fusion weights (B, 2)
        """
        # Get IAF fusion result
        fused, weights = self.iaf(rgb_feat, normal_feat, brightness)

        # Apply confidence weighting if available
        if confidence is not None:
            # Resize confidence to match feature size
            if confidence.shape[2:] != fused.shape[2:]:
                confidence = F.interpolate(
                    confidence, size=fused.shape[2:],
                    mode='bilinear', align_corners=True
                )

            # Convert confidence to channel-wise weights
            conf_weights = self.confidence_conv(confidence)

            # Modulate normal contribution based on confidence
            # High confidence -> trust normal more
            # Low confidence -> rely more on RGB
            w_normal = weights[:, 1:2].view(-1, 1, 1, 1)
            normal_contribution = normal_feat * conf_weights * w_normal

            # Recompute fusion with confidence
            w_rgb = weights[:, 0:1].view(-1, 1, 1, 1)
            fused = rgb_feat * w_rgb + normal_contribution + fused * 0.5

        return fused, weights


if __name__ == "__main__":
    print("Testing Illumination-Adaptive Fusion Module...")

    # Test single-scale IAF
    in_channels = 96
    model = IlluminationAdaptiveFusion(in_channels=in_channels, reduction=16)
    model.eval()

    # Test inputs
    B, H, W = 2, 96, 312
    rgb_feat = torch.randn(B, in_channels, H, W)
    normal_feat = torch.randn(B, in_channels, H, W)

    # Test with different brightness levels
    print("\nTesting with different brightness levels:")
    for brightness_val in [0.1, 0.5, 0.9]:
        brightness = torch.full((B, 1), brightness_val)
        with torch.no_grad():
            fused, weights = model(rgb_feat, normal_feat, brightness)
        print(f"  Brightness={brightness_val:.1f}: RGB weight={weights[0, 0]:.3f}, Normal weight={weights[0, 1]:.3f}")

    print(f"\nOutput shape: {fused.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e3:.2f}K")

    # Test multi-scale IAF
    print("\nTesting Multi-Scale IAF...")
    channel_list = [96, 192, 384, 768]
    ms_model = MultiScaleIAF(channel_list)
    ms_model.eval()

    rgb_features = [
        torch.randn(B, 96, 96, 312),
        torch.randn(B, 192, 48, 156),
        torch.randn(B, 384, 24, 78),
        torch.randn(B, 768, 12, 39)
    ]
    normal_features = [
        torch.randn(B, 96, 96, 312),
        torch.randn(B, 192, 48, 156),
        torch.randn(B, 384, 24, 78),
        torch.randn(B, 768, 12, 39)
    ]
    brightness = torch.full((B, 1), 0.3)

    with torch.no_grad():
        fused_features, all_weights = ms_model(rgb_features, normal_features, brightness)

    for i, (f, w) in enumerate(zip(fused_features, all_weights)):
        print(f"  Scale {i}: shape={f.shape}, RGB_w={w[0, 0]:.3f}, Normal_w={w[0, 1]:.3f}")

    ms_params = sum(p.numel() for p in ms_model.parameters())
    print(f"Multi-scale total parameters: {ms_params / 1e3:.2f}K")
