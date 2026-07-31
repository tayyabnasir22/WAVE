from DataProcessors.SRDataProcessorBase import SRDataProcessorBase
import numpy as np
import torch
import torchvision.transforms.functional as TF
import random

class ExtendedSRImplicitDownSampled(SRDataProcessorBase):
    def __init__(self, rgb_path: str, depth_path: str, depth_norm_path: str, mask_path: str, min_max_path: str, patch_size: int, scale: int, repeat: int = 1, augment: bool = False, train: bool = False, token_size: int = None):
        self.depths = np.load(depth_path)#[:100]
        self.depths_norm = np.load(depth_norm_path)#[:100]
        self.images = np.load(rgb_path)#[:100]
        self.masks = np.load(mask_path)#[:100]
        self.minmax = np.load(min_max_path)#[:100]
        self.augment = augment
        self.repeat = repeat
        self.train = train

        self.base_len = self.depths.shape[0]

        self._patch_size = patch_size
        self.token_size = token_size
        self.scale = scale

    def get_patch(self, img, gt, gt_norm, mask, patch_size):
        i = random.randrange(0, img.shape[1] - patch_size)
        j = random.randrange(0, img.shape[2] - patch_size)
        return img[:, i:i+patch_size, j:j+patch_size], gt[:, i:i+patch_size, j:j+patch_size], gt_norm[:, i:i+patch_size, j:j+patch_size], mask[:, i:i+patch_size, j:j+patch_size]

    def augment_data(self, img, gt, gt_norm, mask):
        # RGB augmentation
        if random.random() < 0.5:
            img = TF.adjust_brightness(img, 1 + random.uniform(-0.1, 0.1))
        if random.random() < 0.5:
            img = TF.adjust_contrast(img, 1 + random.uniform(-0.1, 0.1))

        if random.random() < 0.5:
            # horizontal flip
            img = TF.hflip(img)
            gt = TF.hflip(gt)
            gt_norm = TF.hflip(gt_norm)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            # vertical flip
            img = TF.vflip(img)
            gt = TF.vflip(gt)
            gt_norm = TF.vflip(gt_norm)
            mask = TF.vflip(mask)
        if random.random() < 0.5:
            # diagonal flip
            img = img.transpose(-1, -2)
            gt = gt.transpose(-1, -2)
            gt_norm = gt_norm.transpose(-1, -2)
            mask = mask.transpose(-1, -2)
        if random.random() < 0.5:
            # Rotation applied to patch directly having same widht and height
            # Only roate by 90 or -90 as other variations covered above
            _, h, w = img.shape
            if h == w:
                # rotate by +90 or -90 degrees
                if random.random() < 0.5:
                    img = torch.rot90(img, k=1, dims=(-2, -1))   # +90°
                    gt = torch.rot90(gt,  k=1, dims=(-2, -1))
                    gt_norm = torch.rot90(gt_norm,  k=1, dims=(-2, -1))
                    mask = torch.rot90(mask,  k=1, dims=(-2, -1))
                else:
                    img = torch.rot90(img, k=-1, dims=(-2, -1))  # -90°
                    gt = torch.rot90(gt,  k=-1, dims=(-2, -1))
                    gt_norm = torch.rot90(gt_norm,  k=-1, dims=(-2, -1))
                    mask = torch.rot90(mask,  k=-1, dims=(-2, -1))
        return img, gt, gt_norm, mask

    def Resize(self, tensor, size):
        return TF.resize(tensor, size, interpolation=TF.InterpolationMode.BICUBIC)

    def __len__(self):
        return self.base_len * self.repeat

    def __getitem__(self, idx):
        idx = idx % self.base_len

        depth = torch.from_numpy(self.depths[idx]).float().unsqueeze(0)
        depth_norm = torch.from_numpy(self.depths_norm[idx]).float().unsqueeze(0)
        img = torch.from_numpy(self.images[idx]).float()
        mask = torch.from_numpy(self.masks[idx]).float().unsqueeze(0)

        _, h, w = img.shape

        mx, mn = self.minmax[idx]

        if self.train:
            img, depth, depth_norm, mask = self.get_patch(img, depth, depth_norm, mask, patch_size=self._patch_size)
            img, depth, depth_norm, mask = self.augment_data(img, depth, depth_norm, mask)
            h = self._patch_size
            w = self._patch_size
            
        scale = self.scale
        lr = TF.resize(depth_norm, (h//scale, w//scale), interpolation=TF.InterpolationMode.BICUBIC)

        mn = torch.tensor(self.minmax[idx][1], dtype=torch.float32).view(1,1,1)
        mx = torch.tensor(self.minmax[idx][0], dtype=torch.float32).view(1,1,1)

        return {
            'rgb': img, 
            'gt': depth,
            'gt_norm': depth_norm,
            'lr': lr,
            'mask': mask,
            'min': mn,
            'max': mx,
            'orig_h': h,
            'orig_w': w,
        }