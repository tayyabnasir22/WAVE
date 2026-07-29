from pytorch_msssim import ssim
import torch

class Evaluation:
     # Note training eval metrics do not shave pixels off the edges
    @staticmethod
    def PSNRTrain(sr, hr, rgb_range=1):
        diff = (sr - hr) / rgb_range
        valid = diff
        mse = valid.pow(2).mean()
        return -10 * torch.log10(mse)

    @staticmethod
    def SSIMTrain(sr, hr, rgb_range=1):
        diff_sr = sr / rgb_range
        diff_hr = hr / rgb_range
        return ssim(
            diff_sr, 
            diff_hr, 
            data_range=1.0, 
            size_average=True
        )
    
    @staticmethod
    def DepthRMSE(a, b, mx, mn, shave_pixels: bool = False):
        if shave_pixels == True: # This is to be done for validation and testing not training
            # Crop 6 pixels from all sides to remove boundary effects.
            a = a[:, :, 6:-6, 6:-6]
            b = b[:, :, 6:-6, 6:-6]
        
        # it is a*(max-min) + min
        a = a*(mx-mn) + mn
        b = b*(mx-mn) + mn
        a = a * 100
        b = b * 100
        
        return torch.sqrt(torch.mean(torch.pow(a-b,2))).item()

    def DepthRMSEBenchmark(a, b, shave_pixels: bool = False):
        if shave_pixels == True: # This is to be done for validation and testing not training
            # Crop 6 pixels from all sides to remove boundary effects.
            a = a[:, :, 6:-6, 6:-6]
            b = b[:, :, 6:-6, 6:-6]
        
        # it is a*(max-min) + min
        a = a * 255.0
        b = b * 255.0
        
        return torch.sqrt(torch.mean(torch.pow(a-b,2))).item()

    def RMSE(a, b, mx = None, mn = None, shave_pixels: bool = False):
        if shave_pixels == True: # This is to be done for validation and testing not training
            # Crop 6 pixels from all sides to remove boundary effects.
            a = a[:, :, 6:-6, 6:-6]
            b = b[:, :, 6:-6, 6:-6]

        if mx is not None and mn is not None:
            # it is a*(max-min) + min
            a = a*(mx-mn) + mn
            b = b*(mx-mn) + mn
            a = a * 100
            b = b * 100
                
        return torch.sqrt(torch.mean(torch.pow(a-b,2))).item()
    
    def Threshold(pred, target):
        thresh = torch.max((target / pred), (pred / target))

        d1 = torch.sum(thresh < 1.25).float() / len(thresh).item()
        return d1
    
    def ABSRel(pred, target, mx = None, mn = None, shave_pixels: bool = False):
        if shave_pixels == True: # This is to be done for validation and testing not training
            # Crop 6 pixels from all sides to remove boundary effects.
            pred = pred[:, :, 6:-6, 6:-6]
            target = target[:, :, 6:-6, 6:-6]

        diff = pred - target        

        abs_rel = torch.mean(torch.abs(diff) / target).item()
        return abs_rel
    
    def SqRel(pred, target, mx = None, mn = None, shave_pixels: bool = False):
        if shave_pixels == True: # This is to be done for validation and testing not training
            # Crop 6 pixels from all sides to remove boundary effects.
            pred = pred[:, :, 6:-6, 6:-6]
            target = target[:, :, 6:-6, 6:-6]

        diff = pred - target        

        abs_rel = torch.mean(torch.pow(diff, 2) / target).item()
        return abs_rel
    
    def SiLog(pred, target):
        diff_log = torch.log(pred) - torch.log(target)
        silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2)).item()

        return silog
    
    def RMSEAbs(pred, target, mx = None, mn = None, shave_pixels: bool = False):
        if shave_pixels == True: # This is to be done for validation and testing not training
            # Crop 6 pixels from all sides to remove boundary effects.
            pred = pred[:, :, 6:-6, 6:-6]
            target = target[:, :, 6:-6, 6:-6]

        scale = torch.median(target) / torch.median(pred)
        pred_aligned = pred * scale

        return torch.sqrt(torch.mean((pred_aligned - target) ** 2)).item()
    
    def MAE(pred, target, mx = None, mn = None, shave_pixels: bool = False):
        if shave_pixels == True: # This is to be done for validation and testing not training
            # Crop 6 pixels from all sides to remove boundary effects.
            pred = pred[:, :, 6:-6, 6:-6]
            target = target[:, :, 6:-6, 6:-6]
        
        diff = pred - target

        mae = torch.mean(torch.abs(diff)).item()
        return mae
