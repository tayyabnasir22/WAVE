from Components.ConvOps import ConvOps
from Components.ResBlock import ResBlock
import torch.nn as nn

class RGBEncoder(nn.Module):
    def __init__(self, num_feats: int, kernel_size: int):
        super(RGBEncoder, self).__init__()

        self.blocks = ResBlock(ConvOps.DefaultConv, num_feats, kernel_size, act=nn.SiLU(),)

    def forward(self, x):
        return self.blocks(x)