from Components.WAVE import WAVE
from Models.ModelType import ModelType
from Pipelines.PipelineBase import PipelineBase

class BaseModelFactory:
    def __init__(self):
        self._models = {
            ModelType.WAVE: WAVE,
        }

    def BuildModel(self, pipeline: PipelineBase, model: ModelType, **kwargs):
        # Create the ecnoder and decoder
        print(kwargs)
        model = self._models[model](**kwargs).cuda()

        # Call the Pipeline chains
        pipeline.LoadConfigurations()
        pipeline.InitModel(model)
        pipeline.CreateDataLoaders()
        pipeline.LoadModelWeights()
        pipeline.InitTrainingRecipe()
        pipeline.InitModelObjectives()