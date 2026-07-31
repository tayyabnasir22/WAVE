from ModelFactories.BaseModelFactory import BaseModelFactory
from Models.BenchmarkType import BenchmarkType
from Models.ModelType import ModelType
from Models.SavedModelType import SavedModelType
from TrainingHelpers.TrainingHelperBase import TrainingHelperBase
from Utilities.PathManager import PathManager
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, model: ModelType, training_helper: TrainingHelperBase, benchmark_type: BenchmarkType, input_patch: int = 280, scale: int = 4, repeat: int = 4, batch_size: int = 2):
        self._model = model
        self._training_helper = training_helper
        self._input_patch = input_patch
        self._scale = scale
        self._repeat = repeat
        self._batch_size = batch_size
        self._benchmark_type = benchmark_type

        base_path = PathManager.GetModelSavePath(
            self._model,
            self._benchmark_type,
            self._input_patch,
            self._scale
        )

        self._model_save_path = base_path + ''
        self._model_load_path = base_path + '/' + SavedModelType.Last.value + '.pth'
    
    def _GetModelFactory(self):
        return BaseModelFactory()

    @abstractmethod
    def TrainModel(self,):
        pass
    