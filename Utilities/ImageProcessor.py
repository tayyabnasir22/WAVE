import torch
import torch.nn.functional as F

class ImageProcessor:
    @staticmethod
    def Upsample(x, scale = 2):
        return F.interpolate(x, scale_factor=scale, mode="nearest")

    @staticmethod
    def MergePatchesOverlap(patches, H, W, patch_size=100, overlap=20, shave=6):

        B, C = patches[0].shape[:2]

        merged = torch.zeros(
            B, C, H, W,
            device=patches[0].device,
            dtype=patches[0].dtype
        )

        weight = torch.zeros_like(merged)

        coords = ImageProcessor._get_patch_coords(H, W, patch_size, overlap)

        for patch, (y, x) in zip(patches, coords):

            h = min(patch_size, H - y)
            w = min(patch_size, W - x)

            # Determine shaving depending on boundary
            top = shave if y > 0 else 0
            left = shave if x > 0 else 0
            bottom = shave if (y + patch_size) < H else 0
            right = shave if (x + patch_size) < W else 0

            # Crop patch
            patch_crop = patch[:, :, top:h-bottom, left:w-right]

            # Destination coordinates
            y1 = y + top
            y2 = y + h - bottom
            x1 = x + left
            x2 = x + w - right

            merged[:, :, y1:y2, x1:x2] += patch_crop
            weight[:, :, y1:y2, x1:x2] += 1

        merged = merged / weight.clamp(min=1)

        return merged.contiguous()

    @staticmethod
    def _get_patch_coords(H, W, patch_size, overlap):

        stride = patch_size - overlap

        ys = list(range(0, max(H - patch_size + 1, 1), stride))
        xs = list(range(0, max(W - patch_size + 1, 1), stride))

        if ys[-1] != H - patch_size:
            ys.append(max(H - patch_size, 0))

        if xs[-1] != W - patch_size:
            xs.append(max(W - patch_size, 0))

        coords = []
        for y in ys:
            for x in xs:
                coords.append((y, x))

        return coords

    @staticmethod
    def GetPatchesOverlap(
        img,
        patch_size=100, 
        overlap=20
    ):
        patches = []
        
        B, C, H, W = img.shape

        coords = ImageProcessor._get_patch_coords(H, W, patch_size, overlap)

        for y, x in coords:

            h = min(patch_size, H - y)
            w = min(patch_size, W - x)

            # LR patch
            patch = img[:, :, y:y+h, x:x+w]
            patches.append(patch.contiguous())

        return patches
    
    @staticmethod
    def FirstOrderDerivativeSobel(img):
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
    
    @staticmethod
    def GetYChannelTensor(x):
        # X is normalized RGB image tensor
        # BT.601 luma coefficients (0–255 range → normalized)
        gray_coeffs = torch.tensor([65.738, 129.057, 25.064], dtype=x.dtype, device=x.device) / 256.0

        # (3,) × (3,H,W) → broadcasting over H,W
        Y = (x * gray_coeffs.view(3,1,1)).sum(dim=0, keepdim=True)  # → 1,H,W

        return Y
    
    @staticmethod
    def PadToMultiple(x, multiple=14):
        if x.dim() == 4:
            _, _, h, w = x.shape
        elif x.dim() == 3:
            _, h, w = x.shape
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        # F.pad format: (left, right, top, bottom)
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0), h, w
    
    @staticmethod
    def PadToSize(x, patch_size_h, patch_size_w):
        if x.dim() == 4:
            _, _, h, w = x.shape
        elif x.dim() == 3:
            _, h, w = x.shape
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

        pad_h = patch_size_h - h
        pad_w = patch_size_w - w

        # F.pad format: (left, right, top, bottom)
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0), h, w
    
    @staticmethod
    def CropFromTop(x, orig_h, orig_w):
        if x.dim() == 4:
            return x[:, :, :orig_h, :orig_w]
        elif x.dim() == 3:
            return x[:, :orig_h, :orig_w]
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

    @staticmethod
    def ModCrop(image, modulo):
        h, w = image.shape[:2]
        h = h - (h % modulo)
        w = w - (w % modulo)

        if image.ndim == 3:
            return image[:h, :w, :]
        else:
            return image[:h, :w]