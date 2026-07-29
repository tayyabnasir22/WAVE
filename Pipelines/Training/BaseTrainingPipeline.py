import os
from Configurations.TrainingConfigurations import TrainingConfigurations
from Configurations.TrainingDataConfigurations import TrainingDataConfigurations
from Configurations.ValidationDataConfigurations import ValidationDataConfigurations
from DataProcessors.ExtendedSRImplicitDownSampled import ExtendedSRImplicitDownSampled
from Models.BenchmarkType import BenchmarkType
from Pipelines.PipelineBase import PipelineBase
from Utilities.DataLoaders import DataLoaders
from Utilities.Evaluation import Evaluation
from Utilities.ModelAttributesManager import ModelAttributesManager
from Utilities.PathManager import PathManager
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch
from Utilities.StatsHelpers import StatsHelpers

class BaseTrainingPipeline(PipelineBase):
    def __init__(
        self,
        train_data_path: str = PathManager.GetBasePath() + BenchmarkType.NYUV2.name,
        valid_data_path: str = PathManager.GetBasePath() + BenchmarkType.NYUV2.name,
        model_save_path: str = PathManager.GetBasePath() + 'model_states',
        model_load_path: str = PathManager.GetBasePath() + 'model_states/last.pth',
        batch_size: int = 2,
        train_repeat: int = 1,
        patch_size_train: int = 256,
        patch_size_valid: int = 256,
        start_learning_rate: float = 0.0001,
        scale: int = 4,
        total_examples: int = 1000,
        epochs: int = 100,
        milestones_count: int = 3,
        epoch_val: int = 10,
        epoch_save: int = 10,
        gamma_schedular: float = 0.3,
    ):
        self.start_epoch = -1
        
        self._train_data_path = train_data_path
        self._valid_data_path = valid_data_path
        self._model_save_path = model_save_path
        self._model_load_path = model_load_path
        self._batch_size = batch_size
        self._train_repeat = train_repeat
        self._patch_size_train = patch_size_train
        self._patch_size_valid = patch_size_valid
        self._start_learning_rate = start_learning_rate
        self._scale = scale
        self._total_examples = total_examples
        self._epochs = epochs
        self._milestones_count = milestones_count
        self._epoch_val = epoch_val
        self._epoch_save = epoch_save
        self._gamma_schedular = gamma_schedular
        
    def InitModel(self, model: nn.Module):
        self.model = model

    def LoadConfigurations(self,):
        self.configurations = TrainingConfigurations(
            optimizer={'learning_rate': self._start_learning_rate},
            data_configurations=TrainingDataConfigurations(
                patch_size=self._patch_size_train, 
                augment=True, 
                batch_size=self._batch_size, 
                base_folder=self._train_data_path, 
                repeat=self._train_repeat, 
                scale=self._scale,
                total_examples=self._total_examples,
            ),
            validation_data_configurations=ValidationDataConfigurations(
                patch_size=self._patch_size_valid, 
                augment=False, 
                batch_size=1, 
                base_folder=self._valid_data_path, 
                repeat=1, 
                scale=self._scale, 
                total_examples=100, 
                eval_scale=self._scale,
            ),
            lr_scheduler={'milestones': StatsHelpers.GetMulitStepMilestones(self._epochs, self._milestones_count), 'gamma': self._gamma_schedular},
            epochs=self._epochs,
            save_path=self._model_save_path,
            resume_path=self._model_load_path,
            epoch_val=self._epoch_val,
            epoch_save=self._epoch_save,
            monitor_metric='RMSE',
        )
        
    def CreateDataLoaders(self,):
        self.training_data_loader = DataLoaders.GetTrainingDataLoader(
            ExtendedSRImplicitDownSampled(
                rgb_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_images_stand_split.npy'),
                depth_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_depths_clipped_split.npy'),
                depth_norm_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_depths_norm_split.npy'),
                mask_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_mask_split.npy'),
                min_max_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_minmax_split.npy'),
                patch_size=self._patch_size_train,
                scale=self.configurations.data_configurations.scale,
                repeat=self.configurations.data_configurations.repeat,
                augment=self.configurations.data_configurations.augment,
                train=True,
                token_size=None,
            ),
            self.configurations.data_configurations.batch_size,
        )
        
    def InitModelObjectives(self, ):
        self.loss = nn.L1Loss()
        self.metrics = [Evaluation.RMSE, Evaluation.ABSRel, Evaluation.RMSEAbs]


    def LoadModelWeights(self, ):
        self.saved_model = None
        if os.path.exists(self.configurations.resume_path):
            self.saved_model = torch.load(self.configurations.resume_path)
            self.model.load_state_dict(self.saved_model['model'])

    def InitTrainingRecipe(self, ):
        # 1. Set the start epoch
        self.start_epoch = self.start_epoch if self.saved_model is None else self.saved_model['epoch']

        # 2. Create/Load the optimizer
        # Call optimizer.step() AFTER loss.backward() during every training iteration/step (every batch).
        if self.saved_model is not None:
            self.optimizer: Optimizer = ModelAttributesManager.CreateAdamOptimizer(
                self.model.parameters(), 
                self.saved_model['optimizer'], 
                self.configurations.optimizer['learning_rate'], 
                load_sd=True
            )
        else:
            self.optimizer: Optimizer = ModelAttributesManager.CreateAdamOptimizer(
                self.model.parameters(), 
                None,
                self.configurations.optimizer['learning_rate'], 
                load_sd=False
            )

        self.lr_scheduler = ModelAttributesManager.CreateMultiStepLRScheduler(
            self.optimizer, 
            self.configurations.lr_scheduler['milestones'], 
            self.configurations.lr_scheduler['gamma'], 
            self.start_epoch
        )

    def InitModelObjectives(self, ):
        self.loss = nn.L1Loss()
        self.metrics = [Evaluation.DepthRMSE]