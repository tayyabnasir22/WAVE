import torch
import torch.nn as nn
import torch.nn.functional as F

# INR-ASSR: Empirical Analysis
def sobel_filters(img):
    """Compute Sobel gradients for an image tensor (N,C,H,W)."""
    C = img.shape[1]
    sobel_x = torch.tensor([[-1,0,1],
                            [-2,0,2],
                            [-1,0,1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],
                            [ 0, 0, 0],
                            [ 1, 2, 1]], dtype=img.dtype, device=img.device).view(1,1,3,3)

    # Repeat for each channel
    sobel_x = sobel_x.repeat(C, 1, 1, 1)  # [C,1,3,3]
    sobel_y = sobel_y.repeat(C, 1, 1, 1)  # [C,1,3,3]

    gx = F.conv2d(img, sobel_x, padding=1, groups=C)
    gy = F.conv2d(img, sobel_y, padding=1, groups=C)
    return gx, gy

class SimpleGradientL1Loss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_grad=0.05):
        """
        Args:
            lambda_l1: weight for pixel-wise L1 loss
            lambda_grad: weight for gradient loss
        """
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_grad = lambda_grad
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # --- Pixel L1 loss ---
        l1_loss = self.l1(pred, target)

        # --- Gradient L1 loss (first-order derivatives) ---
        gx_pred, gy_pred = sobel_filters(pred)
        gx_tgt, gy_tgt = sobel_filters(target)

        grad_loss = self.l1(gx_pred, gx_tgt) + self.l1(gy_pred, gy_tgt)

        # --- Weighted sum ---
        return self.lambda_l1 * l1_loss + self.lambda_grad * grad_loss