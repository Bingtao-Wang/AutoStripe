"""
NAA: Night-Aware Attention Decoder

Enhanced decoder with brightness-modulated attention gates
and edge prediction head for nighttime scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import base decoder components
import sys
sys.path.append('..')
from modelsV2.decoder import DepthwiseSeparableConv, DSConvBlock, UpsampleBlock


class NightAwareAttentionGate(nn.Module):
    """
    Brightness-modulated Attention Gate.

    In dark scenes, sharpens attention to be more selective.
    In bright scenes, uses standard attention.

    Args:
        F_g: Channels in gating signal (from decoder)
        F_l: Channels in skip connection (from encoder)
        F_int: Intermediate channels
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()

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

    def forward(self, g, x, brightness=None):
        """
        Args:
            g: Gating signal from decoder (B, F_g, H, W)
            x: Skip connection from encoder (B, F_l, H, W)
            brightness: Brightness score (B, 1), optional

        Returns:
            Attention-weighted skip connection (B, F_l, H, W)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # (B, 1, H, W)

        # Brightness modulation: sharpen attention in dark scenes
        if brightness is not None:
            # brightness: (B, 1) -> (B, 1, 1, 1)
            brightness = brightness.view(-1, 1, 1, 1)
            # Sharpness factor: 1.0 (bright) to 1.5 (dark)
            sharpness = 1.0 + (1.0 - brightness) * 0.5
            # Apply power function to sharpen attention
            psi = torch.pow(psi.clamp(min=1e-6), sharpness)

        return x * psi


class EdgePredictionHead(nn.Module):
    """
    Auxiliary head for edge prediction.

    Predicts road boundaries to help with edge-preserving loss.

    Args:
        in_channels: Input feature channels
        hidden_channels: Hidden layer channels (default: 32)
    """

    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()

        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W)

        Returns:
            edge_pred: Edge prediction (B, 1, H, W) in [0, 1]
        """
        return self.edge_conv(x)


class NightAwareDecoder(nn.Module):
    """
    Night-Aware Attention Decoder (NAA)

    Enhanced decoder with:
    1. Brightness-modulated attention gates
    2. Edge prediction head for boundary refinement
    3. DSConv blocks for efficiency

    Args:
        encoder_channels: List of encoder output channels [C0, C1, C2, C3]
        decoder_channels: List of decoder channels (default: [64, 128, 256, 512])
        num_classes: Number of output classes (default: 2)
        dropout_rate: Dropout rate (default: 0.1)
        use_edge_head: Whether to use edge prediction head (default: True)
    """

    def __init__(self, encoder_channels, decoder_channels=None, num_classes=2,
                 dropout_rate=0.1, use_edge_head=True):
        super().__init__()

        if decoder_channels is None:
            decoder_channels = [64, 128, 256, 512]

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.use_edge_head = use_edge_head

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
        self.att2 = NightAwareAttentionGate(
            F_g=decoder_channels[2],
            F_l=encoder_channels[2],
            F_int=decoder_channels[2] // 2
        )
        self.dec2 = DSConvBlock(
            in_channels=encoder_channels[2] + decoder_channels[2],
            mid_channels=decoder_channels[2],
            out_channels=decoder_channels[2],
            dropout_rate=dropout_rate
        )
        self.up2 = UpsampleBlock(decoder_channels[2], decoder_channels[1])

        # Level 1: Fuse with encoder level 1
        self.att1 = NightAwareAttentionGate(
            F_g=decoder_channels[1],
            F_l=encoder_channels[1],
            F_int=decoder_channels[1] // 2
        )
        self.dec1 = DSConvBlock(
            in_channels=encoder_channels[1] + decoder_channels[1],
            mid_channels=decoder_channels[1],
            out_channels=decoder_channels[1],
            dropout_rate=dropout_rate
        )
        self.up1 = UpsampleBlock(decoder_channels[1], decoder_channels[0])

        # Level 0: Fuse with encoder level 0
        self.att0 = NightAwareAttentionGate(
            F_g=decoder_channels[0],
            F_l=encoder_channels[0],
            F_int=decoder_channels[0] // 2
        )
        self.dec0 = DSConvBlock(
            in_channels=encoder_channels[0] + decoder_channels[0],
            mid_channels=decoder_channels[0],
            out_channels=decoder_channels[0],
            dropout_rate=dropout_rate
        )

        # Final segmentation head
        self.final = nn.Sequential(
            nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )

        # Edge prediction head (optional)
        if use_edge_head:
            self.edge_head = EdgePredictionHead(
                in_channels=decoder_channels[0],
                hidden_channels=32
            )

    def forward(self, encoder_features, brightness=None):
        """
        Args:
            encoder_features: List of encoder outputs [x0, x1, x2, x3]
                             x0: 1/4 resolution
                             x1: 1/8 resolution
                             x2: 1/16 resolution
                             x3: 1/32 resolution
            brightness: Brightness score (B, 1), optional

        Returns:
            output: Segmentation output (B, num_classes, H, W)
            edge_pred: Edge prediction (B, 1, H, W) if use_edge_head else None
        """
        x0, x1, x2, x3 = encoder_features

        # Level 3: Process deepest features
        d3 = self.dec3(x3)
        d3_up = self.up3(d3)

        # Level 2: Attention-weighted skip connection
        x2_att = self.att2(g=d3_up, x=x2, brightness=brightness)
        d2 = torch.cat([x2_att, d3_up], dim=1)
        d2 = self.dec2(d2)
        d2_up = self.up2(d2)

        # Level 1: Attention-weighted skip connection
        x1_att = self.att1(g=d2_up, x=x1, brightness=brightness)
        d1 = torch.cat([x1_att, d2_up], dim=1)
        d1 = self.dec1(d1)
        d1_up = self.up1(d1)

        # Level 0: Attention-weighted skip connection
        x0_att = self.att0(g=d1_up, x=x0, brightness=brightness)
        d0 = torch.cat([x0_att, d1_up], dim=1)
        d0 = self.dec0(d0)

        # Final segmentation
        output = self.final(d0)

        # Edge prediction (optional)
        edge_pred = None
        if self.use_edge_head:
            edge_pred = self.edge_head(d0)
            # Upsample edge prediction to match output size
            edge_pred = F.interpolate(
                edge_pred, scale_factor=4,
                mode='bilinear', align_corners=True
            )

        return output, edge_pred


class MultiScaleEdgeDecoder(NightAwareDecoder):
    """
    Extended NAA decoder with multi-scale edge supervision.

    Predicts edges at multiple scales for better boundary learning.
    """

    def __init__(self, encoder_channels, decoder_channels=None, num_classes=2,
                 dropout_rate=0.1):
        super().__init__(
            encoder_channels, decoder_channels, num_classes,
            dropout_rate, use_edge_head=True
        )

        if decoder_channels is None:
            decoder_channels = [64, 128, 256, 512]

        # Multi-scale edge heads
        self.edge_head_1 = EdgePredictionHead(decoder_channels[1], 32)
        self.edge_head_2 = EdgePredictionHead(decoder_channels[2], 32)

    def forward(self, encoder_features, brightness=None):
        """
        Returns:
            output: Segmentation output
            edge_preds: Dict of edge predictions at multiple scales
        """
        x0, x1, x2, x3 = encoder_features

        # Level 3
        d3 = self.dec3(x3)
        d3_up = self.up3(d3)

        # Level 2
        x2_att = self.att2(g=d3_up, x=x2, brightness=brightness)
        d2 = torch.cat([x2_att, d3_up], dim=1)
        d2 = self.dec2(d2)
        edge_2 = self.edge_head_2(d2)
        d2_up = self.up2(d2)

        # Level 1
        x1_att = self.att1(g=d2_up, x=x1, brightness=brightness)
        d1 = torch.cat([x1_att, d2_up], dim=1)
        d1 = self.dec1(d1)
        edge_1 = self.edge_head_1(d1)
        d1_up = self.up1(d1)

        # Level 0
        x0_att = self.att0(g=d1_up, x=x0, brightness=brightness)
        d0 = torch.cat([x0_att, d1_up], dim=1)
        d0 = self.dec0(d0)
        edge_0 = self.edge_head(d0)

        # Final output
        output = self.final(d0)

        # Collect edge predictions
        edge_preds = {
            'edge_0': F.interpolate(edge_0, scale_factor=4, mode='bilinear', align_corners=True),
            'edge_1': F.interpolate(edge_1, scale_factor=8, mode='bilinear', align_corners=True),
            'edge_2': F.interpolate(edge_2, scale_factor=16, mode='bilinear', align_corners=True)
        }

        return output, edge_preds


if __name__ == "__main__":
    print("Testing Night-Aware Attention Decoder...")

    # Swin-T encoder channels
    encoder_channels = [96, 192, 384, 768]
    decoder_channels = [64, 128, 256, 512]

    # Create model
    model = NightAwareDecoder(
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        num_classes=2,
        use_edge_head=True
    )
    model.eval()

    # Test input (simulated encoder outputs)
    B = 2
    x0 = torch.randn(B, 96, 96, 312)    # 1/4 resolution
    x1 = torch.randn(B, 192, 48, 156)   # 1/8 resolution
    x2 = torch.randn(B, 384, 24, 78)    # 1/16 resolution
    x3 = torch.randn(B, 768, 12, 39)    # 1/32 resolution

    encoder_features = [x0, x1, x2, x3]

    # Test with different brightness levels
    print("\nTesting with different brightness levels:")
    for brightness_val in [0.1, 0.5, 0.9]:
        brightness = torch.full((B, 1), brightness_val)
        with torch.no_grad():
            output, edge_pred = model(encoder_features, brightness)
        print(f"  Brightness={brightness_val:.1f}: output={output.shape}, edge={edge_pred.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")

    # Test multi-scale edge decoder
    print("\nTesting Multi-Scale Edge Decoder...")
    ms_model = MultiScaleEdgeDecoder(
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        num_classes=2
    )
    ms_model.eval()

    brightness = torch.full((B, 1), 0.3)
    with torch.no_grad():
        output, edge_preds = ms_model(encoder_features, brightness)

    print(f"Output shape: {output.shape}")
    for name, edge in edge_preds.items():
        print(f"  {name}: {edge.shape}")

    ms_params = sum(p.numel() for p in ms_model.parameters())
    print(f"Multi-scale total parameters: {ms_params / 1e6:.2f}M")
