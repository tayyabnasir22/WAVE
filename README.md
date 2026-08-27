# WAVE: Reversing the Guidance Hierarchy for Coarse-to-Fine Guided Depth Super-Resolution

A centralized repository for **training** and **evaluating** the WAVE model at different scaling factors.

## Citation

If you use this repository or build upon it in your research, please cite:

> **WAVE: Reversing the Guidance Hierarchy for Coarse-to-Fine Guided Depth Super-Resolution**

```bibtex
@article{nasir2026wavereversingguidancehierarchy,
  title   = {WAVE: Reversing the Guidance Hierarchy for Coarse-to-Fine Guided Depth Super-Resolution},
  author  = {Nasir, Tayyab and Liu, Daochang and Mian, Ajmal},
  journal = {arXiv preprint arXiv:2601.17723},
  year    = {2026}
}
```

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Training

1. Open `train.py`.
2. Make the desired changes to scale, dataset, and other settings.
3. Run the script to begin model training:

```bash
python train.py [scale] wave
```

### Testing

1. Open `test.py`.
2. Uncomment the desired testing scale and datasets.
3. Run the script to evaluate your trained model:

```bash
python test.py
```

## Configurations

The repository includes several configurable components to enable flexible experimentation:

- **Training and evaluation strategies** — classes within the `Pipelines` folder
- **Datasets** — the `BenchmarkType` class

## Saving and Loading Models

The model saving and loading mechanism automatically constructs model paths based on the configuration parameters used during training. This ensures that each model checkpoint is uniquely identified by its scale, dataset, and other settings, allowing for seamless testing and reproducibility.

### How It Works

Each training run saves its model weights under a directory name that encodes the full configuration. When performing evaluation, the **same configuration** must be used to correctly locate and load the desired checkpoint.

## Checkpoints

Pretrained checkpoints are available [here](https://drive.google.com/drive/folders/1uY5uzU8AAKafeoN_hbuC3bxVN3WJXhin).

## Related Repositories

- **Data processing:** [GDSR-Data-Preperation](https://github.com/tayyabnasir22/GDSR-Data-Preperation)
- **Base code:** This code is built on top of [NAIMA-GDSR](https://github.com/tayyabnasir22/NAIMA-GDSR). If you find this code useful, please cite that work as well.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).