import torch.nn as nn
from Components.ConvOps import ConvOps

# DBP styled upsampling
class DBPUpsampling(nn.Module):
    def __init__(self, inter_channels, nr, scale):
        super(DBPUpsampling, self).__init__()
        
        self.up_1 = nn.Sequential(
            ConvOps.ProjectionConv(inter_channels, nr, scale, up=True),
            nn.SiLU(),
        )
        
        self.down_1 = nn.Sequential(
            ConvOps.ProjectionConv(nr, inter_channels, scale, up=False),
            nn.SiLU(),
        )
        
        self.up_2 = nn.Sequential(
            ConvOps.ProjectionConv(inter_channels, nr, scale, up=True),
        )

    def forward(self, x):
        h_0 = self.up_1(x)
        
        l_0 = self.down_1(h_0)
        
        e = l_0.sub(x)
        
        h_1 = self.up_2(e)

        return h_0.add(h_1)