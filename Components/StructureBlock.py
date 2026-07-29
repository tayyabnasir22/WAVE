from Components.ConvOps import ConvOps
import torch.nn as nn
import torch
from Components.DenseBlock import DenseBlock
from Components.DepthEncoder import DepthEncoder
from Components.InvBlock import InvBlock
from Components.RGBEncoder import RGBEncoder
import torch.nn.functional as F

class StructureBlock(nn.Module):
    def __init__(self, dino_dim: int, hidden_dim: int, kernel_size: int = 3):

        super(StructureBlock, self).__init__()
        self.dino_projector = nn.Sequential(
            nn.Conv2d(dino_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        self.depth_enc = DepthEncoder(hidden_dim, 6, kernel_size)
        self.rgb_encoder = RGBEncoder(hidden_dim, kernel_size)

        self.irn_fusion = nn.Sequential(
            InvBlock(DenseBlock, hidden_dim + hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim + hidden_dim, hidden_dim, 1, 1, 0)
        )

        self.init_rgb_projection = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, token, ll, depth_feat):
        token = F.interpolate(
                    token,
                    size=(depth_feat.shape[-2], depth_feat.shape[-1]),
                    mode="bilinear",
                    align_corners=False
                )        
        s_guide = self.dino_projector(token)

        rgb0 = self.init_rgb_projection(ll)
        encoded_rgb = self.rgb_encoder(rgb0)
        
        depth_semantics = ConvOps.compute_linear_attention(depth_feat, s_guide, s_guide)

        encoded_depth = self.depth_enc(depth_feat) + self.alpha * depth_semantics

        return self.irn_fusion(torch.cat([encoded_depth, encoded_rgb], dim=1))
