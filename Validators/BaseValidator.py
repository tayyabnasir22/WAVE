from abc import ABC, abstractmethod
from Models.BenchmarkType import BenchmarkType
from Models.ModelType import ModelType
from Pipelines.Validation.BaseTestingPipeline import BaseTestingPipeline
from ValidationHelpers.ValidationHelperBase import ValidationHelperBase

class BaseValidator(ABC):
    def __init__(self, model: ModelType, validation_helper: ValidationHelperBase, benchmark_type: BenchmarkType):
        self._model = model
        self._validation_helper = validation_helper
        self._benchmark_type = benchmark_type

    @abstractmethod
    def TestModel(self,):
        pass

    @abstractmethod
    def Generate(self, out_dir: str):
        pass

    def _GetPrediction(self, pipeline: BaseTestingPipeline):
        return self._validation_helper.EvaluteForTesting(
            pipeline.validation_data_loader, 
            pipeline.model
        )
