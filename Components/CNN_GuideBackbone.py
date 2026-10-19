import torch.nn as nn
from Components.ConvOps import ConvOps
from Components.ResBlock import ResBlock


class CNN_GuideBackbone(nn.Module):
    def __init__(self, num_feats: int = 64, blocks_per_stage: int = 2, kernel_size: int = 3):
        super().__init__()
        self.widths = [num_feats, num_feats * 2, num_feats * 3, num_feats * 4]  # fine -> coarse

        self.stem = nn.Sequential(
            ConvOps.DefaultConv(3, num_feats, kernel_size),
            nn.SiLU(),
        )

        stages = []
        in_ch = num_feats
        for w in self.widths:
            layers = [
                nn.Conv2d(in_ch, w, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
            ]
            layers += [
                ResBlock(ConvOps.DefaultConv, w, kernel_size, act=nn.SiLU())
                for _ in range(blocks_per_stage)
            ]
            stages.append(nn.Sequential(*layers))
            in_ch = w
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        feats = []
        h = self.stem(x)
        for stage in self.stages:
            h = stage(h)
            feats.append(h)
        return feats 