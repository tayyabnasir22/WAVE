from DataProcessors.RealWorldProcessor import RealWorldProcessor
from Models.BenchmarkType import BenchmarkType
from Pipelines.Validation.BaseTestingPipeline import BaseTestingPipeline
from Utilities.PathManager import PathManager
from Utilities.DataLoaders import DataLoaders
import os

class RealWorldTestingPipeline(BaseTestingPipeline):
    def __init__(
        self,
        valid_data_path: str = PathManager.GetBasePath() + BenchmarkType.NYUV2.name,
        model_load_path: str = PathManager.GetBasePath() + 'model_states',
        model_name: str = 'last.pth',
        total_example: int = 100,
        eval_scale: int = 4,
        patch_size_valid: int = None, # Does not matter
    ):
        self._valid_data_path = valid_data_path
        self._valid_data_pathScale = ''
        self._model_load_path = model_load_path
        self._model_name = model_name
        self._total_example = total_example
        self._eval_scale = eval_scale
        self._patch_size_valid = patch_size_valid

    def CreateDataLoaders(self,):
        self.validation_data_loader = DataLoaders.GetTestingDataLoader(
            RealWorldProcessor(
                rgb_path=os.path.join(self.configurations.data_configurations.base_folder, 'test_images_stand_split.npy'),
                depth_path=os.path.join(self.configurations.data_configurations.base_folder, 'test_depths_clipped_split.npy'),
                depth_norm_path=os.path.join(self.configurations.data_configurations.base_folder, 'test_depths_norm_split.npy'),
                mask_path=os.path.join(self.configurations.data_configurations.base_folder, 'test_mask_split.npy'),
                min_max_path=os.path.join(self.configurations.data_configurations.base_folder, 'test_minmax_split.npy'),
                depth_lr_norm_path=os.path.join(self.configurations.data_configurations.base_folder, 'test_depths_lr_norm_split.npy'),
                repeat=self.configurations.data_configurations.repeat,
                augment=False,
                train=False,
            ),
            self.configurations.data_configurations.batch_size
        )
