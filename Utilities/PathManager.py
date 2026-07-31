from Models.BenchmarkType import BenchmarkType
from Models.ModelType import ModelType
from Utilities.Logger import Logger
import os
from tensorboardX import SummaryWriter
import shutil

class PathManager:
    BASE_PATH = './'

    @staticmethod
    def CheckPathExists(path, remove=True):
        basename = os.path.basename(path.rstrip('/'))
        if os.path.exists(path):
            if remove and (basename.startswith('_')
                    or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
                shutil.rmtree(path)
                os.makedirs(path)
        else:
            os.makedirs(path)

    @staticmethod
    def SetModelSavePath(save_path, remove=True):
        PathManager.CheckPathExists(save_path, remove=remove)
        Logger.SetLogPath(save_path)
        writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
        return Logger.Log, writer
    
    @staticmethod
    def GetModelSavePath(
        model_name: ModelType,
        benchmark_type: BenchmarkType,
        input_patch: int = 256, 
        scale: int = 4
    ):
        return '_'.join(
            [
                PathManager.GetBasePath() + 'model_states',
                model_name.name,
                benchmark_type.name,
                'Patch',
                str(input_patch),
                'Scale',
                str(scale)
            ]
        )
    
    @staticmethod
    def GetBasePath():
        return PathManager.BASE_PATH
    