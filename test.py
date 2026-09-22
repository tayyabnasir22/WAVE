from TestingOrchestrators.TestingOrchestratorWAVE import TestingOrchestratorWAVE
from Models.BenchmarkType import BenchmarkType

def main():
      # TestingOrchestratorWAVE.WAVEValidator4x(BenchmarkType.DIML)
      # TestingOrchestratorWAVE.WAVEValidator4x(BenchmarkType.NYUV2)
      # TestingOrchestratorWAVE.WAVEValidator4x(BenchmarkType.RGBDD)
      # TestingOrchestratorWAVE.WAVEValidator4x(BenchmarkType.TOFDSRD)
      # TestingOrchestratorWAVE.WAVEValidatorBenchamrk4x(BenchmarkType.MIDDLE)
      # TestingOrchestratorWAVE.WAVEValidatorBenchamrk4x(BenchmarkType.LU)

      TestingOrchestratorWAVE.WAVEValidator8x(BenchmarkType.DIML)
      TestingOrchestratorWAVE.WAVEValidator8x(BenchmarkType.NYUV2)
      TestingOrchestratorWAVE.WAVEValidator8x(BenchmarkType.RGBDD)
      TestingOrchestratorWAVE.WAVEValidator8x(BenchmarkType.TOFDSRD)
      TestingOrchestratorWAVE.WAVEValidatorBenchamrk8x(BenchmarkType.MIDDLE)
      TestingOrchestratorWAVE.WAVEValidatorBenchamrk8x(BenchmarkType.LU)

      # TestingOrchestratorWAVE.WAVEValidator16x(BenchmarkType.DIDOE)
      TestingOrchestratorWAVE.WAVEValidator16x(BenchmarkType.DIML)
      TestingOrchestratorWAVE.WAVEValidator16x(BenchmarkType.NYUV2)
      TestingOrchestratorWAVE.WAVEValidator16x(BenchmarkType.RGBDD)
      TestingOrchestratorWAVE.WAVEValidator16x(BenchmarkType.TOFDSRD)
      TestingOrchestratorWAVE.WAVEValidatorBenchamrk16x(BenchmarkType.MIDDLE)
      TestingOrchestratorWAVE.WAVEValidatorBenchamrk16x(BenchmarkType.LU)

      TestingOrchestratorWAVE.WAVEValidator32x(BenchmarkType.DIML)
      TestingOrchestratorWAVE.WAVEValidator32x(BenchmarkType.NYUV2)
      TestingOrchestratorWAVE.WAVEValidator32x(BenchmarkType.RGBDD)
      TestingOrchestratorWAVE.WAVEValidator32x(BenchmarkType.TOFDSRD)
      TestingOrchestratorWAVE.WAVEValidatorBenchamrk32x(BenchmarkType.MIDDLE)
      TestingOrchestratorWAVE.WAVEValidatorBenchamrk32x(BenchmarkType.LU)

      TestingOrchestratorWAVE.ValidatorReal(BenchmarkType.RGBDDReal)
      TestingOrchestratorWAVE.ValidatorReal(BenchmarkType.TOFDSRDReal)

if __name__ == '__main__':
    main()