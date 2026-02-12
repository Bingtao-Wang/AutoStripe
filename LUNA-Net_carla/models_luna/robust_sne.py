"""
R-SNE: Robust Surface Normal Estimation

Enhanced SNE module with depth denoising and confidence estimation
for handling noisy depth data in nighttime scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthConfidenceNet(nn.Module):
    """
    Lightweight network to estimate depth confidence.

    Outputs a confidence map indicating reliability of depth values.
    """

    def __init__(self, hidden_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, depth):
        """
        Args:
            depth: Depth map (B, 1, H, W)
        Returns:
            confidence: Confidence map (B, 1, H, W) in [0, 1]
        """
        return self.net(depth)


class RobustSNE(nn.Module):
    """
    Robust Surface Normal Estimation (R-SNE)

    Enhances the original SNE with:
    1. Outlier rejection based on local statistics
    2. Bilateral filtering for edge-preserving smoothing
    3. Learnable confidence estimation

    Args:
        use_confidence_net: Whether to use learnable confidence (default: True)
        outlier_threshold: Z-score threshold for outlier rejection (default: 3.0)
        bilateral_sigma_space: Spatial sigma for bilateral filter (default: 5.0)
        bilateral_sigma_range: Range sigma for bilateral filter (default: 0.1)
    """

    def __init__(self, use_confidence_net=True, outlier_threshold=3.0,
                 bilateral_sigma_space=5.0, bilateral_sigma_range=0.1):
        super().__init__()
        self.use_confidence_net = use_confidence_net
        self.outlier_threshold = outlier_threshold
        self.bilateral_sigma_space = bilateral_sigma_space
        self.bilateral_sigma_range = bilateral_sigma_range

        if use_confidence_net:
            self.confidence_net = DepthConfidenceNet(hidden_channels=16)

        # Sobel kernels for gradient computation
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 8.0)
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 8.0)

        # Gaussian kernel for bilateral filter (spatial component)
        self._create_gaussian_kernel(kernel_size=5)

    def _create_gaussian_kernel(self, kernel_size=5):
        """Create Gaussian kernel for spatial weighting."""
        sigma = self.bilateral_sigma_space
        x = torch.arange(kernel_size) - kernel_size // 2
        gauss_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d / gauss_2d.sum()
        self.register_buffer('gaussian_kernel', gauss_2d.view(1, 1, kernel_size, kernel_size))
        self.kernel_size = kernel_size

    def reject_outliers(self, depth, threshold=3.0):
        """
        Reject depth outliers based on local statistics.

        Args:
            depth: Depth map (B, 1, H, W)
            threshold: Z-score threshold

        Returns:
            cleaned_depth: Depth with outliers replaced by local median
        """
        B, C, H, W = depth.shape

        # Compute local mean and std using average pooling
        kernel_size = 5
        padding = kernel_size // 2

        # Create valid mask (non-zero depth)
        valid_mask = (depth > 0).float()

        # Local mean
        local_sum = F.avg_pool2d(depth * valid_mask, kernel_size, stride=1, padding=padding)
        local_count = F.avg_pool2d(valid_mask, kernel_size, stride=1, padding=padding).clamp(min=1e-6)
        local_mean = local_sum / local_count

        # Local variance
        local_sq_sum = F.avg_pool2d((depth ** 2) * valid_mask, kernel_size, stride=1, padding=padding)
        local_var = (local_sq_sum / local_count) - local_mean ** 2
        local_std = torch.sqrt(local_var.clamp(min=1e-6))

        # Z-score
        z_score = torch.abs(depth - local_mean) / local_std.clamp(min=1e-6)

        # Outlier mask
        outlier_mask = (z_score > threshold) & (valid_mask > 0)

        # Replace outliers with local mean
        cleaned_depth = torch.where(outlier_mask, local_mean, depth)

        return cleaned_depth

    def bilateral_filter(self, depth):
        """
        Apply bilateral filtering for edge-preserving smoothing.

        Args:
            depth: Depth map (B, 1, H, W)

        Returns:
            filtered_depth: Smoothed depth map
        """
        B, C, H, W = depth.shape
        k = self.kernel_size
        pad = k // 2

        # Pad input
        depth_padded = F.pad(depth, (pad, pad, pad, pad), mode='reflect')

        # Unfold to get local patches
        patches = depth_padded.unfold(2, k, 1).unfold(3, k, 1)  # (B, 1, H, W, k, k)
        patches = patches.contiguous().view(B, 1, H, W, k * k)

        # Center values
        center = depth.unsqueeze(-1)  # (B, 1, H, W, 1)

        # Range weights (intensity similarity)
        range_diff = (patches - center).abs()
        range_weights = torch.exp(-range_diff / (2 * self.bilateral_sigma_range ** 2))

        # Spatial weights (from Gaussian kernel)
        spatial_weights = self.gaussian_kernel.view(1, 1, 1, 1, k * k)

        # Combined weights
        weights = range_weights * spatial_weights
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # Weighted average
        filtered = (patches * weights).sum(dim=-1)

        return filtered

    def compute_sne(self, depth, cam_param):
        """
        Compute surface normals using SNE algorithm.

        Args:
            depth: Depth map (H, W) or (B, 1, H, W)
            cam_param: Camera intrinsic matrix (3, 3) or (B, 3, 3)

        Returns:
            normal: Surface normal map (3, H, W) or (B, 3, H, W)
        """
        # Handle batch dimension
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
            squeeze_output = True
        elif depth.dim() == 3:
            depth = depth.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        B, _, H, W = depth.shape
        device = depth.device

        # Handle camera parameters
        if cam_param.dim() == 2:
            cam_param = cam_param.unsqueeze(0).expand(B, -1, -1)

        # Extract camera intrinsics
        fx = cam_param[:, 0, 0].view(B, 1, 1, 1)
        fy = cam_param[:, 1, 1].view(B, 1, 1, 1)
        cx = cam_param[:, 0, 2].view(B, 1, 1, 1)
        cy = cam_param[:, 1, 2].view(B, 1, 1, 1)

        # Create coordinate grids
        v, u = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        u = u.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        v = v.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)

        # Back-project to 3D
        Z = depth
        X = Z * (u - cx) / fx
        Y = Z * (v - cy) / fy

        # Compute gradients
        dZdx = F.conv2d(Z, self.sobel_x, padding=1)
        dZdy = F.conv2d(Z, self.sobel_y, padding=1)
        dXdx = F.conv2d(X, self.sobel_x, padding=1)
        dXdy = F.conv2d(X, self.sobel_y, padding=1)
        dYdx = F.conv2d(Y, self.sobel_x, padding=1)
        dYdy = F.conv2d(Y, self.sobel_y, padding=1)

        # Compute normal via cross product of tangent vectors
        # T1 = (dXdx, dYdx, dZdx), T2 = (dXdy, dYdy, dZdy)
        # N = T1 x T2
        nx = dYdx * dZdy - dZdx * dYdy
        ny = dZdx * dXdy - dXdx * dZdy
        nz = dXdx * dYdy - dYdx * dXdy

        # Normalize
        norm = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2).clamp(min=1e-6)
        nx = nx / norm
        ny = ny / norm
        nz = nz / norm

        # Handle invalid regions (zero depth)
        invalid_mask = (depth <= 0)
        nx = torch.where(invalid_mask, torch.zeros_like(nx), nx)
        ny = torch.where(invalid_mask, torch.zeros_like(ny), ny)
        nz = torch.where(invalid_mask, -torch.ones_like(nz), nz)

        # Ensure normals point towards camera (nz should be negative for ground)
        # Flip normals where nz > 0
        flip_mask = (nz > 0)
        nx = torch.where(flip_mask, -nx, nx)
        ny = torch.where(flip_mask, -ny, ny)
        nz = torch.where(flip_mask, -nz, nz)

        # Concatenate to (B, 3, H, W)
        normal = torch.cat([nx, ny, nz], dim=1)

        if squeeze_output:
            normal = normal.squeeze(0)

        return normal

    def forward(self, depth, cam_param):
        """
        Compute robust surface normals with confidence.

        Args:
            depth: Depth map (H, W) or (B, 1, H, W)
            cam_param: Camera intrinsic matrix (3, 3) or (B, 3, 3)

        Returns:
            normal: Surface normal map (3, H, W) or (B, 3, H, W)
            confidence: Confidence map (1, H, W) or (B, 1, H, W)
        """
        # Ensure batch dimension
        original_dim = depth.dim()
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(0)

        # Step 1: Outlier rejection
        depth_clean = self.reject_outliers(depth, self.outlier_threshold)

        # Step 2: Bilateral filtering
        depth_filtered = self.bilateral_filter(depth_clean)

        # Step 3: Confidence estimation
        if self.use_confidence_net:
            confidence = self.confidence_net(depth_filtered)
        else:
            # Simple confidence based on depth validity and gradient magnitude
            valid_mask = (depth_filtered > 0).float()
            grad_x = F.conv2d(depth_filtered, self.sobel_x, padding=1)
            grad_y = F.conv2d(depth_filtered, self.sobel_y, padding=1)
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            # Lower confidence for high gradients (potential noise)
            confidence = valid_mask * torch.exp(-grad_mag * 10)

        # Step 4: Compute surface normals
        normal = self.compute_sne(depth_filtered.squeeze(1), cam_param)

        # Ensure normal has batch dimension
        if normal.dim() == 3:
            normal = normal.unsqueeze(0)

        # Apply confidence weighting to normals
        normal_weighted = normal * confidence

        # Restore original dimensions
        if original_dim == 2:
            normal_weighted = normal_weighted.squeeze(0)
            confidence = confidence.squeeze(0)
        elif original_dim == 3:
            normal_weighted = normal_weighted.squeeze(0)
            confidence = confidence.squeeze(0)

        return normal_weighted, confidence


if __name__ == "__main__":
    print("Testing Robust SNE Module...")

    # Create model
    model = RobustSNE(use_confidence_net=True)
    model.eval()

    # Test input
    H, W = 384, 1248
    depth = torch.rand(2, 1, H, W) * 80.0  # Random depth 0-80m
    depth[depth < 10] = 0  # Add some invalid regions

    # Add noise
    depth = depth + torch.randn_like(depth) * 0.5
    depth = depth.clamp(min=0)

    # Camera intrinsics (KITTI-like)
    cam_param = torch.tensor([
        [721.5, 0.0, 609.5],
        [0.0, 721.5, 172.8],
        [0.0, 0.0, 1.0]
    ]).unsqueeze(0).expand(2, -1, -1)

    # Forward pass
    with torch.no_grad():
        normal, confidence = model(depth, cam_param)

    print(f"Depth shape: {depth.shape}")
    print(f"Normal shape: {normal.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print(f"Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e3:.2f}K")
