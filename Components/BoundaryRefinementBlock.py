import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class BoundaryRefinementBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_dim: int, 
    ):
        super().__init__()

        self.rgb_proj = nn.Sequential(
            nn.Conv2d(channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.gate_net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),

            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.out_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

        self.merger = nn.Conv2d(hidden_dim+1, hidden_dim, kernel_size=1)

        self.merger_final = nn.Conv2d(hidden_dim+1, hidden_dim, kernel_size=1)

    def semantic_boundary_map(self, feat):
        # 1. Normalize feature vectors
        feat_map = F.normalize(feat, dim=1)

        # 2. Cosine similarity with neighboring pixels
        cos_x = (feat_map[:, :, :, 1:] * feat_map[:, :, :, :-1]).sum(dim=1, keepdim=True)
        cos_y = (feat_map[:, :, 1:, :] * feat_map[:, :, :-1, :]).sum(dim=1, keepdim=True)

        # 3. Convert similarity to boundary strength
        dx = 1.0 - cos_x
        dy = 1.0 - cos_y

        # 4. Pad back to original size
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))

        # 5. Combine horizontal + vertical boundary responses
        boundary = torch.sqrt(dx**2 + dy**2)

        # 6. Smooth + normalize
        boundary = TF.gaussian_blur(boundary, kernel_size=5)

        boundary = boundary / (boundary.mean(dim=(2,3), keepdim=True) + 1e-3)

        return boundary

    def forward(self, rgb_feat, dino_feat, depth):

        rgb = self.rgb_proj(rgb_feat)
        boundary = self.semantic_boundary_map(dino_feat)

        boundary = F.interpolate(
            boundary,
            size=rgb_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        depth = F.interpolate(
            depth,
            size=rgb_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        gate_input = self.merger(torch.cat(
            [rgb, boundary],
            dim=1
        ))

        gate = self.gate_net(gate_input)
        fused = gate * rgb

        fused = self.out_fusion(self.merger_final(torch.cat([fused, depth], dim=1)))

        return fused, boundary