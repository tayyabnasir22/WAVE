from DataProcessors.BenchmarkSRImplicitDownSampled import BenchmarkSRImplicitDownSampled
from Models.BenchmarkType import BenchmarkType
from Pipelines.Validation.BaseTestingPipeline import BaseTestingPipeline
from Utilities.PathManager import PathManager
from Utilities.DataLoaders import DataLoaders

class BaseTestingPipelineBenchmark(BaseTestingPipeline):
    def __init__(
        self,
        valid_data_path: str = PathManager.GetBasePath() + BenchmarkType.MIDDLE.name,
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
            BenchmarkSRImplicitDownSampled(
                root_dir=self._valid_data_path,
                scale=self.configurations.data_configurations.scale,
            ),
            self.configurations.data_configurations.batch_size
        )
