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

        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim + hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        self.init_rgb_projection = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def compute_linear_attention(self, q, k, v):
        B, C, H, W = q.shape
        N = H * W

        N_k = k.shape[-1] * k.shape[-2] 

        # Flatten and transpose: [B, N, C]
        q = q.view(B, C, N).transpose(1, 2)
        k = k.view(B, C, N_k).transpose(1, 2)
        v = v.view(B, C, N_k).transpose(1, 2)

        # Apply ELU + 1 as the feature map (standard for Linear Attention)
        # This keeps the values positive to avoid division by zero in the kernel
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Compute the "Context" first: [B, C, C]
        # This is the "Weighted Sum of Values" per channel feature
        context = torch.bmm(k.transpose(1, 2), v) 

        # Compute the denominator (Normalization factor)
        k_sum = k.sum(dim=1, keepdim=True) # [B, 1, C]
        denom = torch.bmm(q, k_sum.transpose(1, 2)) # [B, N, C] @ [B, C, 1] =>  [B, N, 1]

        # Compute Attended Values: [B, N, C]
        out = torch.bmm(q, context)
        out = out / (denom + 1e-6) # Normalize

        return out.transpose(1, 2).view(B, C, H, W)

    def forward(self, token, ll, depth_feat):
        # 1. Encode and interpolate dino token
        token = F.interpolate(
                    token,
                    size=(depth_feat.shape[-2], depth_feat.shape[-1]),
                    mode="bilinear",
                    align_corners=False
                )        
        s_guide = self.dino_projector(token)

        # 2. Interpolate and encode RGB features form ll band
        rgb0 = self.init_rgb_projection(ll)
        encoded_rgb = self.rgb_encoder(rgb0)
        
        # 3. Get Relevant semantics from the encoded token using the depth as query
        depth_semantics = self.compute_linear_attention(depth_feat, s_guide, s_guide)

        # 4. Encode the depth feature map and add residual semantics
        encoded_depth = self.depth_enc(depth_feat) + self.alpha * depth_semantics

        # 5. Fusion of encoded LL with the output depth, returning the relevant semantics to be used by other bands
        return self.fuse(torch.cat([encoded_depth, encoded_rgb], dim=1))
