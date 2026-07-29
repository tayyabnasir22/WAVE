from ModelFactories.BaseModelFactory import BaseModelFactory
from Models.BenchmarkType import BenchmarkType
from Models.ModelType import ModelType
from Pipelines.Training.BaseTrainingPipeline import BaseTrainingPipeline
from Pipelines.Validation.BaseTestingPipeline import BaseTestingPipeline
from Pipelines.Validation.BaseTestingPipelineBenchmark import BaseTestingPipelineBenchmark
from Trainers.BaseTrainer import BaseTrainer
from TrainingHelpers.WAVE_TrainingHelper import WAVE_TrainingHelper
from Utilities.PathManager import PathManager
import os

class WAVETrainer(BaseTrainer):
    def __init__(self, model: ModelType, benchmark_type: BenchmarkType, input_patch: int = 256, scale: int = 4, repeat: int = 4, batch_size: int = 2):
        model = model
        super().__init__(
            model=model,
            benchmark_type=benchmark_type,
            training_helper=None,
            input_patch=input_patch,
            scale=scale,
            repeat=repeat,
            batch_size=batch_size
        )

    def _RunTrain(self, pipeline: BaseTrainingPipeline, factory: BaseModelFactory):
        factory.BuildModel(pipeline, self._model, use_pretrained = not os.path.exists(pipeline._model_save_path), num_feats=48, 
            img_size=self._input_patch, 
            patch_size=16, 
            scale=self._scale
        )

        # Call the training
        logger, writer = PathManager.SetModelSavePath(
            pipeline.configurations.save_path, False
        )

        self._training_helper.Train(logger, writer, 0, False)

    def _GetPipeline(self,):
        return BaseTrainingPipeline(
            train_data_path=PathManager.GetBasePath() + self._benchmark_type.name,
            valid_data_path=PathManager.GetBasePath() + self._benchmark_type.name,
            scale=self._scale,
            patch_size_train=self._input_patch,
            patch_size_valid=self._input_patch,
            model_save_path=self._model_save_path,
            model_load_path=self._model_load_path,
            train_repeat=self._repeat,
            total_examples=1000,
            epoch_val=1,
            epoch_save=5,
            batch_size=self._batch_size,
            epochs=200,
            start_learning_rate=0.0001,
            milestones_count=5,
        )

    def TrainModel(self,):
        # 1. Init thre required Pipeline
        pipeline = self._GetPipeline()

        validation_pipeline = BaseTestingPipeline(
            valid_data_path = PathManager.GetBasePath() + self._benchmark_type.name,
            eval_scale = self._scale,
            patch_size_valid = self._input_patch,
        )
        validation_pipeline.LoadConfigurations()
        validation_pipeline.CreateDataLoaders()

        benchmark_pipeline = BaseTestingPipelineBenchmark(
            valid_data_path = PathManager.GetBasePath() + BenchmarkType.MIDDLE.name, 
            eval_scale = self._scale, 
            patch_size_valid = self._input_patch
        )
        benchmark_pipeline.LoadConfigurations()
        benchmark_pipeline.CreateDataLoaders()

        self._training_helper = WAVE_TrainingHelper(
            pipeline, 
            validation_pipeline,
            benchmark_pipeline,
            self._scale,
            self._input_patch
        )

        # 2. Build the model
        factory = self._GetModelFactory()

        # 3. Train the model using the factory
        self._RunTrain(pipeline, factory)

    