import torch.nn as nn

class SToRA(nn.Module):
    def __init__(self, embed_dim=384, rank=16, alpha=1.0):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        
        # LoRA like matrices
        self.down = nn.Linear(embed_dim, rank, bias=False)
        self.up = nn.Linear(rank, embed_dim, bias=False)
        nn.init.zeros_(self.up.weight) 
        
        # scaling
        self.scaling = alpha / rank
        
    def forward(self, x, patch_h, patch_w):
        delta = self.up(self.down(x)) * self.scaling

        # Add the learned representaion as a residual to the original token
        x = x + delta
        
        # reshape to spatial
        x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
        
        return x