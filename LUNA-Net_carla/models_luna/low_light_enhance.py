"""
LLEM: Low-Light Enhancement Module

Based on Zero-DCE++ architecture, uses learnable curve estimation
to enhance low-light images without paired training data.

Reference: Zero-DCE++ (arXiv:2103.00860)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution for efficiency."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class LowLightEnhanceModule(nn.Module):
    """
    Low-Light Enhancement Module (LLEM)

    Uses iterative curve estimation to enhance dark images:
    x_enhanced = x + alpha * x * (1 - x)

    Args:
        num_iterations: Number of enhancement iterations (default: 4)
        num_filters: Number of intermediate filters (default: 32)

    Input:
        RGB image (B, 3, H, W) normalized to [0, 1]

    Output:
        enhanced: Enhanced RGB image (B, 3, H, W)
        brightness: Global brightness score (B, 1) in [0, 1]
    """

    def __init__(self, num_iterations=4, num_filters=32):
        super().__init__()
        self.num_iterations = num_iterations

        # Lightweight curve estimator using depthwise separable convolutions
        self.curve_estimator = nn.Sequential(
            DepthwiseSeparableConv(3, num_filters, 3, 1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(num_filters, num_filters, 3, 1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(num_filters, num_filters, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, 3 * num_iterations, kernel_size=1, bias=True)
        )

        # Global brightness estimator
        self.brightness_pool = nn.AdaptiveAvgPool2d(1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Input RGB image (B, 3, H, W), values in [0, 1]

        Returns:
            enhanced: Enhanced image (B, 3, H, W)
            brightness: Brightness score (B, 1)
        """
        # Estimate enhancement curves
        curves = self.curve_estimator(x)  # (B, 3*num_iter, H, W)

        # Split curves for each iteration
        curve_list = curves.split(3, dim=1)  # List of (B, 3, H, W)

        # Iterative enhancement
        enhanced = x
        for alpha in curve_list:
            # Apply tanh to constrain alpha to [-1, 1]
            alpha = torch.tanh(alpha)
            # Enhancement formula: x = x + alpha * x * (1 - x)
            enhanced = enhanced + alpha * enhanced * (1 - enhanced)

        # Clamp to valid range
        enhanced = torch.clamp(enhanced, 0, 1)

        # Compute global brightness score
        # Use mean of enhanced image as brightness indicator
        brightness = self.brightness_pool(enhanced.mean(dim=1, keepdim=True))
        brightness = brightness.view(x.size(0), 1)  # (B, 1)

        return enhanced, brightness


class LLEMLoss(nn.Module):
    """
    Loss function for training LLEM module.

    Combines:
    1. Spatial consistency loss
    2. Exposure control loss
    3. Color constancy loss
    4. Illumination smoothness loss
    """

    def __init__(self, spa_weight=1.0, exp_weight=10.0, col_weight=5.0, smooth_weight=200.0):
        super().__init__()
        self.spa_weight = spa_weight
        self.exp_weight = exp_weight
        self.col_weight = col_weight
        self.smooth_weight = smooth_weight

        # Spatial consistency kernel
        self.spa_kernel = nn.Parameter(
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False
        )

    def spatial_consistency_loss(self, enhanced, original):
        """Preserve spatial structure."""
        # Compute gradients
        enhanced_grad = F.conv2d(enhanced.mean(dim=1, keepdim=True), self.spa_kernel, padding=1)
        original_grad = F.conv2d(original.mean(dim=1, keepdim=True), self.spa_kernel, padding=1)
        return F.l1_loss(enhanced_grad, original_grad)

    def exposure_control_loss(self, enhanced, target_exposure=0.6):
        """Control overall exposure level."""
        # Pool to 16x16 patches
        pooled = F.adaptive_avg_pool2d(enhanced, (16, 16))
        return F.mse_loss(pooled, torch.ones_like(pooled) * target_exposure)

    def color_constancy_loss(self, enhanced):
        """Preserve color balance."""
        mean_rgb = enhanced.mean(dim=(2, 3))  # (B, 3)
        r, g, b = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        loss = (r - g).pow(2) + (r - b).pow(2) + (g - b).pow(2)
        return loss.mean()

    def illumination_smoothness_loss(self, curves):
        """Encourage smooth enhancement curves."""
        # Compute gradients in x and y directions
        grad_x = torch.abs(curves[:, :, :, :-1] - curves[:, :, :, 1:])
        grad_y = torch.abs(curves[:, :, :-1, :] - curves[:, :, 1:, :])
        return grad_x.mean() + grad_y.mean()

    def forward(self, enhanced, original, curves=None):
        """
        Args:
            enhanced: Enhanced image (B, 3, H, W)
            original: Original image (B, 3, H, W)
            curves: Enhancement curves (B, 3*num_iter, H, W), optional
        """
        loss = 0

        loss += self.spa_weight * self.spatial_consistency_loss(enhanced, original)
        loss += self.exp_weight * self.exposure_control_loss(enhanced)
        loss += self.col_weight * self.color_constancy_loss(enhanced)

        if curves is not None:
            loss += self.smooth_weight * self.illumination_smoothness_loss(curves)

        return loss


if __name__ == "__main__":
    print("Testing Low-Light Enhancement Module...")

    # Create model
    model = LowLightEnhanceModule(num_iterations=4, num_filters=32)
    model.eval()

    # Test input (simulated dark image)
    x = torch.rand(2, 3, 384, 1248) * 0.3  # Dark image

    # Forward pass
    with torch.no_grad():
        enhanced, brightness = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Input mean brightness: {x.mean():.3f}")
    print(f"Enhanced shape: {enhanced.shape}")
    print(f"Enhanced mean brightness: {enhanced.mean():.3f}")
    print(f"Brightness score: {brightness.squeeze()}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e3:.2f}K")
