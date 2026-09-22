from Models.RunningAverage import RunningAverage
from Utilities.Evaluation import Evaluation
from ValidationHelpers.ValidationHelperBase import ValidationHelperBase
import torch
from tqdm import tqdm

class RealWorldValidationHelper(ValidationHelperBase):
    def __init__(self, scale: int = 4):
        super().__init__(
            448, 20, scale
        )

    def GetInference(self, batch, model):
        # Assumption is to be used with TOFDSR and RGBDD both of which have real data multiple of 16 for width and height
        img= batch['rgb']

        for k, v in batch.items():
            if k not in ['orig_h', 'orig_w']:
                batch[k] = v.cuda(non_blocking=True)

        lr = batch['lr']

        out = model(img.cuda(non_blocking=True), lr.cuda(non_blocking=True))

        return out, batch['gt_norm']

    def _Evaluate(self, data_loader, model, shave = False):
        model.eval()
        with torch.no_grad():
            rmse_res = RunningAverage()
            
            pbar = tqdm(data_loader, leave=False, desc='val')
            for batch in pbar:
                # Conver the rgb image and the lr image into patches
                out, gt_norm = self.GetInference(batch, model)

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