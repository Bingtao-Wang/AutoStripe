"""
Lightweight Decoder with Depthwise Separable Convolutions for SNE-RoadSegV2

This module implements a simplified decoder that:
1. Uses Depthwise Separable Convolutions (DSConv) for efficiency
2. Prunes redundant dense skip connections from Nested U-Net
3. Keeps only essential cross-scale skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution (DSConv)

    Splits standard convolution into:
    1. Depthwise convolution (spatial filtering per channel)
    2. Pointwise convolution (1x1 conv for channel mixing)

    This reduces parameters and computation significantly.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Key: groups = in_channels
            bias=bias
        )

        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DSConvBlock(nn.Module):
    """
    Depthwise Separable Convolution Block with BatchNorm, ReLU, and Dropout.

    This replaces the standard conv_block_nested in V1.
    Added Dropout for regularization to prevent overfitting.
    """

    def __init__(self, in_channels, mid_channels, out_channels, dropout_rate=0.1):
        super(DSConvBlock, self).__init__()

        self.conv1 = DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.conv2 = DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        return x


class UpsampleBlock(nn.Module):
    """
    Upsampling block using bilinear interpolation + 1x1 conv.

    This is more efficient than transposed convolution.
    """

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()

        self.scale_factor = scale_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x


class SimplifiedDecoder(nn.Module):
    """
    Simplified Decoder with DSConv and pruned skip connections.

    Architecture changes from V1:
    1. Replace standard conv with DSConv
    2. Remove dense connections within same level
    3. Keep only cross-scale skip connections (encoder -> decoder)
    4. Reduce number of decoder stages

    Feature flow:
        Encoder outputs: [x0, x1, x2, x3] (from Swin backbone)

        Level 3: x3 -> d3
        Level 2: [x2, Up(d3)] -> d2
        Level 1: [x1, Up(d2)] -> d1
        Level 0: [x0, Up(d1)] -> d0

        Output: d0 -> final segmentation
    """

    def __init__(self, encoder_channels, decoder_channels=None, num_classes=2, dropout_rate=0.1):
        """
        Args:
            encoder_channels: List of encoder output channels [C0, C1, C2, C3]
                             For Swin-T: [96, 192, 384, 768]
            decoder_channels: List of decoder channels (default: [64, 128, 256, 512])
            num_classes: Number of output classes (default: 2 for road/background)
            dropout_rate: Dropout rate for regularization (default: 0.1)
        """
        super(SimplifiedDecoder, self).__init__()

        if decoder_channels is None:
            decoder_channels = [64, 128, 256, 512]

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

        # Decoder blocks (bottom-up)
        # Level 3: Process deepest features
        self.dec3 = DSConvBlock(
            in_channels=encoder_channels[3],
            mid_channels=decoder_channels[3],
            out_channels=decoder_channels[3],
            dropout_rate=dropout_rate
        )
        self.up3 = UpsampleBlock(decoder_channels[3], decoder_channels[2])

        # Level 2: Fuse with encoder level 2
        self.dec2 = DSConvBlock(
            in_channels=encoder_channels[2] + decoder_channels[2],
            mid_channels=decoder_channels[2],
            out_channels=decoder_channels[2],
            dropout_rate=dropout_rate
        )
        self.up2 = UpsampleBlock(decoder_channels[2], decoder_channels[1])

        # Level 1: Fuse with encoder level 1
        self.dec1 = DSConvBlock(
            in_channels=encoder_channels[1] + decoder_channels[1],
            mid_channels=decoder_channels[1],
            out_channels=decoder_channels[1],
            dropout_rate=dropout_rate
        )
        self.up1 = UpsampleBlock(decoder_channels[1], decoder_channels[0])

        # Level 0: Fuse with encoder level 0
        self.dec0 = DSConvBlock(
            in_channels=encoder_channels[0] + decoder_channels[0],
            mid_channels=decoder_channels[0],
            out_channels=decoder_channels[0],
            dropout_rate=dropout_rate
        )

        # Final segmentation head
        self.final = nn.Sequential(
            nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # Upsample to original size
        )

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List of encoder outputs [x0, x1, x2, x3]
                             x0: 1/4 resolution
                             x1: 1/8 resolution
                             x2: 1/16 resolution
                             x3: 1/32 resolution

        Returns:
            Segmentation output (B, num_classes, H, W)
        """
        x0, x1, x2, x3 = encoder_features

        # Level 3: Process deepest features
        d3 = self.dec3(x3)
        d3_up = self.up3(d3)

        # Level 2: Fuse with encoder level 2
        d2 = torch.cat([x2, d3_up], dim=1)
        d2 = self.dec2(d2)
        d2_up = self.up2(d2)

        # Level 1: Fuse with encoder level 1
        d1 = torch.cat([x1, d2_up], dim=1)
        d1 = self.dec1(d1)
        d1_up = self.up1(d1)

        # Level 0: Fuse with encoder level 0
        d0 = torch.cat([x0, d1_up], dim=1)
        d0 = self.dec0(d0)

        # Final segmentation
        output = self.final(d0)

        return output


class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections.

    This can be optionally added to refine skip connections
    by suppressing irrelevant features.
    """

    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of channels in gating signal (from decoder)
            F_l: Number of channels in skip connection (from encoder)
            F_int: Number of intermediate channels
        """
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (B, F_g, H, W)
            x: Skip connection from encoder (B, F_l, H, W)

        Returns:
            Attention-weighted skip connection (B, F_l, H, W)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionDecoder(SimplifiedDecoder):
    """
    Decoder with attention gates on skip connections.

    This is an enhanced version of SimplifiedDecoder.
    """

    def __init__(self, encoder_channels, decoder_channels=None, num_classes=2, dropout_rate=0.1):
        super(AttentionDecoder, self).__init__(encoder_channels, decoder_channels, num_classes, dropout_rate)

        if decoder_channels is None:
            decoder_channels = [64, 128, 256, 512]

        # Attention gates
        self.att2 = AttentionGate(F_g=decoder_channels[2], F_l=encoder_channels[2], F_int=decoder_channels[2] // 2)
        self.att1 = AttentionGate(F_g=decoder_channels[1], F_l=encoder_channels[1], F_int=decoder_channels[1] // 2)
        self.att0 = AttentionGate(F_g=decoder_channels[0], F_l=encoder_channels[0], F_int=decoder_channels[0] // 2)

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List of encoder outputs [x0, x1, x2, x3]

        Returns:
            Segmentation output (B, num_classes, H, W)
        """
        x0, x1, x2, x3 = encoder_features

        # Level 3: Process deepest features
        d3 = self.dec3(x3)
        d3_up = self.up3(d3)

        # Level 2: Attention-weighted skip connection
        x2_att = self.att2(g=d3_up, x=x2)
        d2 = torch.cat([x2_att, d3_up], dim=1)
        d2 = self.dec2(d2)
        d2_up = self.up2(d2)

        # Level 1: Attention-weighted skip connection
        x1_att = self.att1(g=d2_up, x=x1)
        d1 = torch.cat([x1_att, d2_up], dim=1)
        d1 = self.dec1(d1)
        d1_up = self.up1(d1)

        # Level 0: Attention-weighted skip connection
        x0_att = self.att0(g=d1_up, x=x0)
        d0 = torch.cat([x0_att, d1_up], dim=1)
        d0 = self.dec0(d0)

        # Final segmentation
        output = self.final(d0)

        return output


if __name__ == "__main__":
    # Test decoder
    print("Testing Simplified Decoder with DSConv...")

    # Swin-T encoder channels
    encoder_channels = [96, 192, 384, 768]
    decoder_channels = [64, 128, 256, 512]

    # Create model
    model = SimplifiedDecoder(
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        num_classes=2
    )
    model.eval()

    # Test input (simulated encoder outputs)
    x0 = torch.randn(2, 96, 96, 312)    # 1/4 resolution
    x1 = torch.randn(2, 192, 48, 156)   # 1/8 resolution
    x2 = torch.randn(2, 384, 24, 78)    # 1/16 resolution
    x3 = torch.randn(2, 768, 12, 39)    # 1/32 resolution

    encoder_features = [x0, x1, x2, x3]

    # Forward pass
    with torch.no_grad():
        output = model(encoder_features)

    print(f"Output shape: {output.shape}")
    print(f"Expected: (2, 2, 384, 1248)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # Test attention decoder
    print("\nTesting Attention Decoder...")
    att_model = AttentionDecoder(
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        num_classes=2
    )
    att_model.eval()

    with torch.no_grad():
        att_output = att_model(encoder_features)

    print(f"Output shape: {att_output.shape}")
    att_params = sum(p.numel() for p in att_model.parameters())
    print(f"Total parameters: {att_params / 1e6:.2f}M")
