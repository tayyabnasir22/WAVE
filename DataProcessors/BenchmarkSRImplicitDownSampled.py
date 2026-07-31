from DataProcessors.SRDataProcessorBase import SRDataProcessorBase
from Utilities.ImageProcessor import ImageProcessor
import numpy as np
import torch
import torchvision.transforms.functional as TF
import os
from PIL import Image
import torchvision.transforms as transforms

class BenchmarkSRImplicitDownSampled(SRDataProcessorBase):
    def __init__(self, root_dir, scale=8):
        self.scale = scale

        self.GTs = []
        self.RGBs = []

        list_dir = os.listdir(root_dir)
        for name in list_dir:
            if name.find('output_color') > -1:
                self.RGBs.append('%s/%s' % (root_dir, name))
            elif name.find('output_depth') > -1:
                self.GTs.append('%s/%s' % (root_dir, name))
        self.RGBs.sort()
        self.GTs.sort()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.GTs)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.RGBs[idx]))
        gt = np.array(Image.open(self.GTs[idx]))
        assert gt.shape[0] == image.shape[0] and gt.shape[1] == image.shape[1]
        s = self.scale  
        image = ImageProcessor.ModCrop(image, s)
        gt = ImageProcessor.ModCrop(gt, s)

        image = torch.from_numpy(image).float().permute(-1, 0, 1)
        gt = torch.from_numpy(gt).float().unsqueeze(0)

        h, w = gt.shape[1], gt.shape[2]
        scale = self.scale

        lr = TF.resize(gt, (h//scale, w//scale), interpolation=TF.InterpolationMode.BICUBIC)

        gt = gt / 255.0
        image = image / 255.0
        lr = lr / 255.0
        
        sample = {
            'rgb': image, 
            'lr': lr, 
            'gt_norm': gt,
            'orig_h': h,
            'orig_w': w,
            'gt': gt,  
        }

        return sample
    