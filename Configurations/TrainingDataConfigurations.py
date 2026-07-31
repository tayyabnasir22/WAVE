from dataclasses import dataclass

@dataclass
class TrainingDataConfigurations:
    patch_size: int
    augment: bool
    batch_size: int
    base_folder: str
    repeat: int
    scale: int
    total_examples: int