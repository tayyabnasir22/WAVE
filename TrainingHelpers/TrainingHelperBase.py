from abc import ABC, abstractmethod
from Models.Timer import Timer
from Pipelines.Training.BaseTrainingPipeline import BaseTrainingPipeline
from Pipelines.Validation.BaseTestingPipelineBenchmark import BaseTestingPipelineBenchmark
from Pipelines.Validation.BaseTestingPipeline import BaseTestingPipeline
from Utilities.Logger import Logger
from Utilities.ModelAttributesManager import ModelAttributesManager
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tensorboardX import SummaryWriter
from Models.SavedModelType import SavedModelType
import torch

class TrainingHelperBase(ABC):
    def __init__(self, pipeline: BaseTrainingPipeline, validation_pipeline: BaseTestingPipeline, benchmark_pipeline: BaseTestingPipelineBenchmark, validation_metric_seed: float):
        self._pipeline: BaseTrainingPipeline = pipeline
        self._validation_metric_seed = validation_metric_seed
        self._validation_pipeline = validation_pipeline
        self._benchmark_pipeline = benchmark_pipeline        

    def register_nan_hooks(self, model: nn.Module):
        hooks = []
        for name, module in model.named_modules():
            def make_hook(n):
                def hook(mod, inp, out):
                    t = out if isinstance(out, torch.Tensor) else out[0]
                    if torch.isnan(t).any():
                        raise RuntimeError(f"NaN first appeared in: {n}")
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))
        return hooks
        
    @abstractmethod
    def RunEpoch(
        self,
        model: nn.Module, 
        train_loader: DataLoader, 
        optimizer: Optimizer, 
        loss_fn: nn.Module, 
        metrics: list, 
        epoch: int, 
        writer = SummaryWriter
    ):
        pass

    @abstractmethod
    def Validation(
        self,
        model, 
        log_info: list, 
        epoch: int, 
        validation_metric_seed: float
    ):
        pass

    @abstractmethod
    def BenachmarkValidation(
        self,
        model, 
        log_info: list, 
        epoch: int, 
        validation_metric_seed: float
    ):
        pass

    def LogLoadingInformation(self, start_epoch: int, model: nn.Module, optimizer: Optimizer, lr_scheduler: LRScheduler, logger: Logger.Log):
        # Set that start to 1 if fresh start, else add +1 if resuming        
        logger('Current epoch: ', start_epoch)
        # Print LR(s)
        for i, group in enumerate(optimizer.param_groups):
            logger(f"Param group {i} -> current LR: {group['lr']} | initial LR: {group.get('initial_lr', 'N/A')}")

        # Print decay info
        if hasattr(lr_scheduler, "gamma"):
            logger(f"Decay factor (gamma): {lr_scheduler.gamma}")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        logger(f"Trainable parameters: {trainable_params:,}")
        logger(f"Non-trainable parameters: {non_trainable_params:,}")
        logger(f"Total parameters: {trainable_params + non_trainable_params:,}")

    def Train(
        self,
        logger: Logger.Log,
        writer: SummaryWriter, 
        n_gpus: int, 
        allow_multi_gpu: bool = True
    ):
        # 1. Check if multiple GPUs can be used for training
        if n_gpus > 1 and allow_multi_gpu:
            self._pipeline.model = nn.parallel.DataParallel(self._pipeline.model)

        # 2. Init the timer
        timer = Timer()
        

        logger('Last completed epoch: ', self._pipeline.start_epoch)
        self._pipeline.start_epoch = 1 if self._pipeline.start_epoch == -1 else self._pipeline.start_epoch + 1

        self.LogLoadingInformation(self._pipeline.start_epoch, self._pipeline.model, self._pipeline.optimizer, self._pipeline.lr_scheduler, logger)

        hooks = self.register_nan_hooks(self._pipeline.model)
        # 3. Train the model for the required epochs
        for epoch in range(self._pipeline.start_epoch, self._pipeline.configurations.epochs + 1):
            # 3.1. Start logging the epoch
            epoch_start = timer.Elapsed()
            log_info = ['epoch {}/{}'.format(epoch, self._pipeline.configurations.epochs)]

            writer.add_scalar('lr', self._pipeline.optimizer.param_groups[0]['lr'], epoch)

            # 3.2. Run the training steps for the epoch, and get loss
            loss = self.RunEpoch(
                self._pipeline.model, 
                self._pipeline.training_data_loader, 
                self._pipeline.optimizer, 
                self._pipeline.loss, 
                self._pipeline.metrics, 
                epoch, 
                self._pipeline.configurations.data_configurations, 
                writer
            )

            # 3.3. Adjust the learning rate
            self._pipeline.lr_scheduler.step()

            # 3.4. Log loss and lr info
            log_info.append('train: loss={:.4f} lr={:.8f}'.format(loss, self._pipeline.optimizer.param_groups[0]['lr']))

            # 3.5. Check if the model was paralellized across multiple gpus
            if n_gpus > 1 and allow_multi_gpu:
                model_ = self._pipeline.model.module
            else:
                model_ = self._pipeline.model

            # 3.6. Save the current epoch model
            ModelAttributesManager.SaveModel(
                model_, 
                self._pipeline.optimizer, 
                epoch, 
                self._pipeline.configurations.save_path, 
                SavedModelType.Last.value
            )

            # 3.7. Save model if required for this epoch
            if epoch % self._pipeline.configurations.epoch_save == 0:
                ModelAttributesManager.SaveModel(
                    model_, 
                    self._pipeline.optimizer, 
                    epoch, 
                    self._pipeline.configurations.save_path, 
                    'epoch_' + str(epoch)
                )

            # 3.8. Incase validation needs to be run for this epoch
            if epoch % self._pipeline.configurations.epoch_val == 0:
                self.Validation(
                    model_, 
                    log_info, 
                    epoch, 
                    self._validation_metric_seed
                )

                self.BenachmarkValidation(
                    model_,
                    log_info, 
                    epoch, 
                    self._validation_metric_seed
                )

            # 3.9. Print Epoch time, Total time spent so far, and time left for training completion
            progress = (epoch - self._pipeline.start_epoch + 1) / (self._pipeline.configurations.epochs - self._pipeline.start_epoch + 1)
            elapsed_total = timer.Elapsed()
            log_info.append('{} {}/{}'.format(
                    Timer.ConvertTimeToText(elapsed_total - epoch_start), 
                    Timer.ConvertTimeToText(elapsed_total),
                    Timer.ConvertTimeToText(elapsed_total / progress)
                )
            )

            logger(', '.join(log_info))
            writer.flush()
