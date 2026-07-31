import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch

class ConvOps:
    @staticmethod
    def DefaultConv(in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), bias=bias)

    @staticmethod
    def ProjectionConv(in_channels, out_channels, scale, up=True):
        kernel_size, stride, padding = {
            2: (6, 2, 2),
            4: (8, 4, 2),
            8: (12, 8, 2),
            16: (20, 16, 2)
        }[scale]
        if up:
            conv_f = nn.ConvTranspose2d
        else:
            conv_f = nn.Conv2d

        return conv_f(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
    
    @staticmethod
    def XavierInitWeights(net_l, scale=1):
        if not isinstance(net_l, list):
            net_l = [net_l]
        for net in net_l:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    init.xavier_normal_(m.weight)
                    m.weight.data *= scale  # for residual block
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias.data, 0.0)

    @staticmethod
    def compute_linear_attention(q, k, v):
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