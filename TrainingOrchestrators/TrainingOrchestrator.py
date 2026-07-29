from Models.BenchmarkType import BenchmarkType
from Models.ModelType import ModelType
from Trainers.WAVETrainer import WAVETrainer

class TrainingOrchestrator:
    BATCH = 2
    REPEAT = 2
    SCALE = 4
    MODEL = ModelType.WAVE
    BENCHMARK = BenchmarkType.NYUV2
    PATCH = 448
    @staticmethod
    def Train():
        print('Batch, Repeat, Scale, Model, Benchmark: ', TrainingOrchestrator.BATCH, TrainingOrchestrator.REPEAT, TrainingOrchestrator.SCALE, TrainingOrchestrator.MODEL, TrainingOrchestrator.BENCHMARK)
    
        if TrainingOrchestrator.MODEL == ModelType.WAVE:
            WAVETrainer(
                model=TrainingOrchestrator.MODEL,
                benchmark_type=TrainingOrchestrator.BENCHMARK,
                input_patch=TrainingOrchestrator.PATCH,
                scale=TrainingOrchestrator.SCALE,
                repeat=TrainingOrchestrator.REPEAT,
                batch_size=TrainingOrchestrator.BATCH,
            ).TrainModel()
        else:
            raise Exception('Not implemented')
            

    