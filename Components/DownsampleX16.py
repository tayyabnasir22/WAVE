import torch.nn as nn

class DownsampleX16(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.PixelUnshuffle(16),
            nn.Conv2d(in_channels*16*16, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)