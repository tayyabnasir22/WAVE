from DataProcessors.SRDataProcessorBase import SRDataProcessorBase
import numpy as np
import torch
import torchvision.transforms.functional as TF
import random

class RealWorldProcessor(SRDataProcessorBase):
    def __init__(self, rgb_path: str, depth_path: str, depth_norm_path: str, mask_path: str,
                 min_max_path: str, depth_lr_norm_path: str, repeat: int = 1, augment: bool = False,
                 train: bool = False):
        self.depths = np.load(depth_path)#[:10]
        self.depths_norm = np.load(depth_norm_path)#[:10]
        self.images = np.load(rgb_path)#[:10]
        self.masks = np.load(mask_path)#[:10]
        self.minmax = np.load(min_max_path)#[:10]

        # Real LR depth from the sensor, already normalized with its own min/max
        self.depths_lr_norm = np.load(depth_lr_norm_path)#[:10]

        self.augment = augment
        self.repeat = repeat
        self.train = train

        self.base_len = self.depths.shape[0]

    def augment_data(self, img, gt, gt_norm, mask, lr):
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
            lr = TF.hflip(lr)
        if random.random() < 0.5:
            # vertical flip
            img = TF.vflip(img)
            gt = TF.vflip(gt)
            gt_norm = TF.vflip(gt_norm)
            mask = TF.vflip(mask)
            lr = TF.vflip(lr)
        return img, gt, gt_norm, mask, lr

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
        lr_real = torch.from_numpy(self.depths_lr_norm[idx]).float().unsqueeze(0)

        _, h, w = img.shape

        if self.train and self.augment:
            img, depth, depth_norm, mask, lr_real = self.augment_data(
                img, depth, depth_norm, mask, lr_real)
      
        # minmax stores [max, min] of the LR depth (what is available at inference)
        mn = torch.tensor(self.minmax[idx][1], dtype=torch.float32).view(1, 1, 1)
        mx = torch.tensor(self.minmax[idx][0], dtype=torch.float32).view(1, 1, 1)

        clean_lr = self.Resize(depth, lr_real.shape[-2:])       # raw depth units
        clean_lr_norm = (clean_lr - mn) / (mx - mn + 1e-8)       # LR's minmax, same as lr_real

        return {
            'rgb': img,
            'gt': depth,
            'gt_norm': depth_norm,
            'lr': lr_real,
            'mask': mask,
            'min': mn,
            'max': mx,
            'orig_h': h,
            'orig_w': w,
            'clean_lr': clean_lr,
            'clean_lr_norm': clean_lr_norm
        }