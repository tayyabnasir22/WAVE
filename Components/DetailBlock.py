from Components.ConvOps import ConvOps
import torch
import torch.nn as nn
import torch.nn.functional as F

class DetailBlock(nn.Module):
    def __init__(self, dino_dim: int, hidden_dim: int):
        super(DetailBlock, self).__init__()
        
        self.dino_projector_s = nn.Sequential(
            nn.Conv2d(dino_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        self.dino_projector_t = nn.Sequential(
            nn.Conv2d(dino_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        self.band_projector_lh_hl = nn.Sequential(
                    nn.Conv2d(2 * 3, hidden_dim, 1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                )
                
        self.band_projector_hh = nn.Sequential(
            nn.Conv2d(3*3, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        # Edge branch
        self.structure_extractor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

        # Texture branch
        self.texture_extractor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2),
            nn.SiLU(),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),   # fill gaps
            nn.SiLU(),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, dilation=4, padding=4),
            nn.SiLU(),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

    def forward(self, dino_tokens, lh, hl, hh):
        dino_tokens = F.interpolate(
                    dino_tokens,
                    size=(lh.shape[-2], lh.shape[-1]),
                    mode="bilinear",
                    align_corners=False
                )
        s_guide = self.dino_projector_s(dino_tokens)
        t_guide = self.dino_projector_t(dino_tokens)
        
        struct_v = self.band_projector_lh_hl(torch.cat([lh, hl], dim=1))
        structure = self.structure_extractor(struct_v)

        struct_semantics = ConvOps.compute_linear_attention(s_guide, structure, structure)

        gated_structure = self.alpha * struct_semantics + (1 - self.alpha) * structure

        texture_v = self.band_projector_hh(torch.cat([lh, hl, hh], dim=1))

        texture = self.texture_extractor(texture_v)

        texture_semantics = ConvOps.compute_linear_attention(t_guide, texture, texture)

        gated_texture = self.beta * texture_semantics + (1 - self.beta) * texture

        return self.refine(gated_structure + gated_texture)