from Components.UNetConvBlock import UNetConvBlock
import torch.nn as nn
import torch
from Components.ConvOps import ConvOps

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d = 1, gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.SiLU()

        ConvOps.XavierInitWeights([self.conv1, self.conv2, self.conv3], 0.1)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3