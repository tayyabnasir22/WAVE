# WAVE: Reversing the Guidance Hierarchy for Coarse-to-Fine Guided Depth Super-Resolution

A centralized repository for **training** and **evaluating** WAVE model for different scaling factors.


## Citation

If you use this repository or build upon it in your research, please cite the following paper:

> **WAVE: Reversing the Guidance Hierarchy for Coarse-to-Fine Guided Depth Super-Resolution**


---


## Getting Started

### Training

    1. Open train.py.  
    2. Make the desired changes to Scale, dataset, and other settings.  
    3. Run the script to begin model training.

```bash
python train.py
```

### Testing

    1. Open test.py.
    2. Uncomment the desired testing scale and datasets.
    3. Run the script to evaluate your trained model.

```bash
python test.py
```

## Configurations

The repository includes several configurable components to enable flexible experimentation.

    - Training and evaluation strategies
    - Datasets

## Saving and Loading Models

The **model saving and loading mechanism** automatically constructs model paths based on the configuration parameters used during training.  
This ensures that each model checkpoint is uniquely identified by its scale, dataset, and other settings allowing for seamless testing and reproducibility.

---

### How It Works

Each training run saves its model weights under a directory name that encodes the full configuration.  
When performing evaluation, the **same configuration** must be used to correctly locate and load the desired checkpoint.
