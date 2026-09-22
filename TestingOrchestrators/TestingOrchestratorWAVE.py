from Models.BenchmarkType import BenchmarkType
from Models.ModelType import ModelType
from ValidationHelpers.WAVE_ValidationHelper import WAVE_ValidationHelper
from ValidationHelpers.WAVE_ValidationHelperBenchmark import WAVE_ValidationHelperBenchmark
from ValidationHelpers.RealWorldValidationHelper import RealWorldValidationHelper
from Validators.WAVE_Validator import WAVE_Validator

class TestingOrchestratorWAVE:
    @staticmethod
    def WAVEValidator32x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(32), 'x')
        validator = WAVE_Validator(
            model=ModelType.WAVE, 
            validation_helper=WAVE_ValidationHelper(336, 64, 32), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './model_states_WAVE_NYUV2_Patch_448_Scale_32', 'last.pth', 449, 32)
        print('-'*33)

    @staticmethod
    def WAVEValidator16x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(16), 'x')
        validator = WAVE_Validator(
            model=ModelType.WAVE, 
            validation_helper=WAVE_ValidationHelper(336, 64, 16), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './model_states_WAVE_NYUV2_Patch_448_Scale_16', 'last.pth', 449, 16)
        print('-'*33)

    @staticmethod
    def WAVEValidator8x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(8), 'x')
        validator = WAVE_Validator(
            model=ModelType.WAVE, 
            validation_helper=WAVE_ValidationHelper(336, 64, 8), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './model_states_WAVE_HYPERSIM_Patch_448_Scale_8', 'last.pth', 449, 8)
        print('-'*33)

    @staticmethod
    def WAVEValidator4x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(4), 'x')
        validator = WAVE_Validator(
            model=ModelType.WAVE, 
            validation_helper=WAVE_ValidationHelper(336, 64, 4), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './model_states_WAVE_NYU_RGBDD_Merged_Patch_368_Scale_4', 'last.pth', 449, 4)
        print('-'*33)

    @staticmethod
    def WAVEValidatorBenchamrk32x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(32), 'x')
        validator = WAVE_Validator(
            model=ModelType.WAVE, 
            validation_helper=WAVE_ValidationHelperBenchmark(336, 64, 32), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './model_states_WAVE_NYUV2_Patch_448_Scale_32', 'last.pth', 449, 32)
        print('-'*33)
    
    @staticmethod
    def WAVEValidatorBenchamrk16x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(16), 'x')
        validator = WAVE_Validator(
            model=ModelType.WAVE, 
            validation_helper=WAVE_ValidationHelperBenchmark(336, 64, 16), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './model_states_WAVE_NYUV2_Patch_448_Scale_16', 'last.pth', 449, 16)
        print('-'*33)

    @staticmethod
    def WAVEValidatorBenchamrk8x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(8), 'x')
        validator = WAVE_Validator(
            model=ModelType.WAVE, 
            validation_helper=WAVE_ValidationHelperBenchmark(336, 64, 8), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './model_states_WAVE_HYPERSIM_Patch_448_Scale_8', 'last.pth', 449, 8)
        print('-'*33)

    @staticmethod
    def WAVEValidatorBenchamrk4x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(4), 'x')
        validator = WAVE_Validator(
            model=ModelType.WAVE, 
            validation_helper=WAVE_ValidationHelperBenchmark(336, 64, 4), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './model_states_WAVE_NYU_RGBDD_Merged_Patch_368_Scale_4', 'last.pth', 449, 4)
        print('-'*33)

    @staticmethod
    def ValidatorReal(data = BenchmarkType):
        print('Running ' + data.value + ', Real')
        validator = WAVE_Validator(
            model=ModelType.WAVEReal, 
            validation_helper=RealWorldValidationHelper(4), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './model_states_WAVEReal_RGBDDReal_Patch_448_Scale_1', 'last.pth', 449, 4)
        print('-'*33)