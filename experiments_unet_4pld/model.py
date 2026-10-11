import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""
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

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ASLUNet(nn.Module):
    """
    Standard 2D Convolutional U-Net for ASL Parameter Estimation.
    Input: [B, 4, H, W] - 4 PLD perfusion channels
    Output: [B, 2, H, W] - continuous regression maps for CBF (channel 0) and ATT (channel 1)
    """
    def __init__(self, in_channels=4, out_channels=2, features=[32, 64, 128, 256], bilinear=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        f0, f1, f2, f3 = features
        f4 = f3 * 2 # bottleneck channels: e.g. 512

        self.inc = DoubleConv(in_channels, f0)
        self.down1 = Down(f0, f1)
        self.down2 = Down(f1, f2)
        self.down3 = Down(f2, f3)
        self.down4 = Down(f3, f4)

        self.up1 = Up(f4, f3, bilinear)
        self.up2 = Up(f3, f2, bilinear)
        self.up3 = Up(f2, f1, bilinear)
        self.up4 = Up(f1, f0, bilinear)
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
        logits = self.outc(x)
        return logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = ASLUNet(in_channels=4, out_channels=2, features=[32, 64, 128, 256])
    n_params = count_parameters(model)
    print(f"ASLUNet initialized. Trainable parameters: {n_params:,}")
    
    # Test forward pass
    dummy = torch.randn(2, 4, 64, 64)
    out = model(dummy)
    print(f"Input shape: {dummy.shape} -> Output shape: {out.shape}")
    assert out.shape == (2, 2, 64, 64), f"Unexpected shape {out.shape}"
    print("ASLUNet model tests passed successfully!")

