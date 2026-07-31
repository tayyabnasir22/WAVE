from Components.RCAB import RCAB
import torch.nn as nn
from Components.ConvOps import ConvOps

class DepthEncoder(nn.Module):
    def __init__(
        self, num_feats: int, blocks: int, kernel_size: int):

        super(DepthEncoder, self).__init__()
        
        modules = [
            RCAB(
                ConvOps.DefaultConv,
                num_feats,
                kernel_size,
                16,
                act=nn.SiLU(),
            )
            for _ in range(blocks)
        ]

        modules.append(ConvOps.DefaultConv(num_feats, num_feats, kernel_size))

        self.att_blocks = nn.Sequential(*modules)   

    def forward(self, x):
        return self.att_blocks(x) + x