from abc import ABC, abstractmethod
from Utilities.ImageProcessor import ImageProcessor
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

class ValidationHelperBase(ABC):    
    def __init__(self, patch_size: int, overlap: int, scale: int = 4):
        self.patch_size = 336#patch_size
        self.overlap = overlap
        self.scale = scale

    @abstractmethod
    def EvaluateForTrainigData(self, data_loader: DataLoader, model: nn.Module,):
        pass

    @abstractmethod
    def EvaluteForTesting(self, data_loader: DataLoader, model: nn.Module, 
    ):
        pass

    @abstractmethod
    def Generate(self, data_loader: DataLoader, model: nn.Module, output_dir: str):
        pass
        
    def GetInference16(self, batch, model):
        # Pad the iage to be divisible by and 16
        valids = [i for i in range(16, 1600, 16)]

        h, w = batch['rgb'].shape[-2], batch['rgb'].shape[-1]
        h_patch, w_patch = h, w
        if h % self.patch_size != 0:
            h_patch = [i for i in valids if i >= h][0]
        
        if w % self.patch_size != 0:
            w_patch = [i for i in valids if i >= w][0]
            
        img, h, w = ImageProcessor.PadToSize(batch['rgb'], h_patch, w_patch)

        depth_norm, h, w = ImageProcessor.PadToSize(batch['gt_norm'], h_patch, w_patch)

        for k, v in batch.items():
            if k not in ['orig_h', 'orig_w']:
                batch[k] = v.cuda(non_blocking=True)

        lr = TF.resize(depth_norm, (h_patch//self.scale, w_patch//self.scale), interpolation=TF.InterpolationMode.BICUBIC)

        out = model(img.cuda(non_blocking=True), lr.cuda(non_blocking=True))

        return ImageProcessor.CropFromTop(out, h, w), batch['gt_norm']
    