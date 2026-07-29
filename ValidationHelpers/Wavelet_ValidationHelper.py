from Models.RunningAverage import RunningAverage
from Utilities.Evaluation import Evaluation
from ValidationHelpers.ValidationHelperBase import ValidationHelperBase
import torch
from tqdm import tqdm

class Wavelet_ValidationHelper(ValidationHelperBase):
    def __init__(self, patch_size: int, overlap: int, scale: int = 4):
        super().__init__(patch_size, overlap, scale)

    def _Evaluate(self, data_loader, model, shave = False):
        model.eval()
        with torch.no_grad():
            rmse_res = RunningAverage()
            
            pbar = tqdm(data_loader, leave=False, desc='val')
            for batch in pbar:
                # Conver the rgb image and the lr image into patches
                out, gt_norm = self.GetInference16(batch, model)

                rmse = Evaluation.DepthRMSE(out, gt_norm, batch['max'], batch['min'], shave_pixels=shave)
                rmse_res.SetItem(rmse, batch['rgb'].shape[0])

        return {
            'RMSE': rmse_res.GetItem()
        }

    # Unable to run this for now
    def EvaluateForTrainigData(self, data_loader, model):
        return self._Evaluate(data_loader, model, True)

    def EvaluteForTesting(self, data_loader, model):
        return self._Evaluate(data_loader, model, True)
    
    def Generate(self, data_loader, model, output_dir):
        pass
                