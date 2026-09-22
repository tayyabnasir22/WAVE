from DataProcessors.RealWorldProcessor import RealWorldProcessor
from Pipelines.Training.BaseTrainingPipeline import BaseTrainingPipeline
from Utilities.DataLoaders import DataLoaders
import os

class RealDataTrainingPipeline(BaseTrainingPipeline):
    def CreateDataLoaders(self,):
        self.training_data_loader = DataLoaders.GetTrainingDataLoader(
            RealWorldProcessor(
                rgb_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_images_stand_split.npy'),
                depth_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_depths_clipped_split.npy'),
                depth_norm_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_depths_norm_split.npy'),
                mask_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_mask_split.npy'),
                min_max_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_minmax_split.npy'),
                depth_lr_norm_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_depths_lr_norm_split.npy'),
                repeat=self.configurations.data_configurations.repeat,
                augment=self.configurations.data_configurations.augment,
                train=True,
            ),
            self.configurations.data_configurations.batch_size,
        )