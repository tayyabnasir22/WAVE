from dataclasses import dataclass
from Configurations.TrainingDataConfigurations import TrainingDataConfigurations

@dataclass
class ValidationDataConfigurations(TrainingDataConfigurations):
    eval_scale: int