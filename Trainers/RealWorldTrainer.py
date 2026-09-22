from Pipelines.Training.RealDataTrainingPipeline import RealDataTrainingPipeline
from Pipelines.Validation.RealWorldTestingPipeline import RealWorldTestingPipeline
from Trainers.WAVETrainer import WAVETrainer
# Change this to point to the real world trainer with degrdation L1 loss
from TrainingHelpers.RealWorldTrainingHelper import RealWorldTrainingHelper
from Utilities.PathManager import PathManager

class RealWorldTrainer(WAVETrainer):
    def _GetPipeline(self,):
        return RealDataTrainingPipeline(
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
    
            validation_pipeline = RealWorldTestingPipeline(
                valid_data_path = PathManager.GetBasePath() + self._benchmark_type.name,
                eval_scale = self._scale,
                patch_size_valid = self._input_patch,
            )
            validation_pipeline.LoadConfigurations()
            validation_pipeline.CreateDataLoaders()
    
            benchmark_pipeline = None
    
            self._training_helper = RealWorldTrainingHelper(
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