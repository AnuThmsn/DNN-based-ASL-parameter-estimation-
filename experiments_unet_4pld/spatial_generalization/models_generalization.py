"""
Architectural Candidates for ASL Spatial Generalization:
1. ASLUNet (Original Baseline)
2. ASLResidualUNet (Residual blocks to preserve high-frequency features)
3. ASLHybridUNet (Dual-branch: Spatial U-Net + Point-wise 1x1 MLP with learned gating)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Standard DoubleConv: Conv => BN => ReLU => Conv => BN => ReLU"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResidualBlock(nn.Module):
    """Residual Double Conv: Conv => BN => ReLU => Conv => BN + Shortcut => ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_residual=False):
        super().__init__()
        block = ResidualBlock(in_channels, out_channels) if use_residual else DoubleConv(in_channels, out_channels)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            block
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, use_residual=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResidualBlock(in_channels, out_channels) if use_residual else DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ASLUNet(nn.Module):
    """Original 4-PLD U-Net baseline."""
    def __init__(self, in_channels=4, out_channels=2, features=[32, 64, 128, 256]):
        super().__init__()
        f0, f1, f2, f3 = features
        f4 = f3 * 2
        self.inc = DoubleConv(in_channels, f0)
        self.down1 = Down(f0, f1)
        self.down2 = Down(f1, f2)
        self.down3 = Down(f2, f3)
        self.down4 = Down(f3, f4)
        self.up1 = Up(f4, f3)
        self.up2 = Up(f3, f2)
        self.up3 = Up(f2, f1)
        self.up4 = Up(f1, f0)
        self.outc = nn.Conv2d(f0, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

class ASLResidualUNet(nn.Module):
    """Candidate 1: Residual U-Net preserving high-frequency boundary information."""
    def __init__(self, in_channels=4, out_channels=2, features=[32, 64, 128, 256]):
        super().__init__()
        f0, f1, f2, f3 = features
        f4 = f3 * 2
        self.inc = ResidualBlock(in_channels, f0)
        self.down1 = Down(f0, f1, use_residual=True)
        self.down2 = Down(f1, f2, use_residual=True)
        self.down3 = Down(f2, f3, use_residual=True)
        self.down4 = Down(f3, f4, use_residual=True)
        self.up1 = Up(f4, f3, use_residual=True)
        self.up2 = Up(f3, f2, use_residual=True)
        self.up3 = Up(f2, f1, use_residual=True)
        self.up4 = Up(f1, f0, use_residual=True)
        self.outc = nn.Conv2d(f0, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

class ASLHybridUNet(nn.Module):
    """
    Candidate 2: Dual-Branch Spatial U-Net + Point-wise 1x1 MLP.
    Combines spatial contextual denoising with voxel-independent local signal fitting
    via learned spatial-adaptive gating.
    """
    def __init__(self, in_channels=4, out_channels=2, features=[32, 64, 128, 256]):
        super().__init__()
        # 1. Spatial branch (U-Net)
        self.spatial_net = ASLResidualUNet(in_channels=in_channels, out_channels=out_channels, features=features)
        
        # 2. Point-wise branch (1x1 Conv MLP - exactly 0D voxel-wise)
        self.pointwise_mlp = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

        # 3. Adaptive gating network (predicts spatial weighting per voxel)
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * 2, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid() # alpha in [0, 1] per channel
        )

    def forward(self, x):
        spatial_pred = self.spatial_net(x)
        pointwise_pred = self.pointwise_mlp(x)
        
        # Gate conditioned on input and both predictions
        fusion_input = torch.cat([x, spatial_pred, pointwise_pred], dim=1)
        gate = self.gate_net(fusion_input)
        
        # Blended output: spatial denoising where gate -> 1, pointwise fidelity where gate -> 0
        out = gate * spatial_pred + (1.0 - gate) * pointwise_pred
        return out

